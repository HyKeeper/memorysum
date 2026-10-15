#!/usr/bin/env python3
"""本地基准测试 runner：在 200 题上评测检索质量，输出逐叶子指标。

用法：
    .venv/bin/python bench/run_benchmark.py
    .venv/bin/python bench/run_benchmark.py --questions bench/benchmark_v1.jsonl --top-k 10
    .venv/bin/python bench/run_benchmark.py --dump-failures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from memorysum.api.schemas import SearchRequest
from memorysum.providers.factory import build_provider, embedding_endpoint
from memorysum.services.search import SearchService
from memorysum.settings import RuntimeSettings, load_pipeline_config
from memorysum.storage.database import DatabaseManager

DEFAULT_TOP_K = 10


# ─── 评分逻辑 ─────────────────────────────────────────────────────────


def score_substring(results: list, answers: list[str], threshold: int = 10) -> tuple[bool, int | None]:
    """检查所有 answer_substrings 是否出现在搜索结果正文中。
    返回 (hit, first_rank)。"""
    if not answers:
        return False, None
    str_answers = [str(a) for a in answers if isinstance(a, (str, int, float)) and len(str(a).strip()) >= threshold]

    # 需要所有答案子串都出现在结果中
    found_at: list[int] = []
    for ans in str_answers:
        if len(ans.strip()) < threshold:
            continue
        for rank, item in enumerate(results, start=1):
            content = item.content or ""
            if ans in content:
                found_at.append(rank)
                break
        else:
            return False, None  # 有一个没找到

    return True, min(found_at) if found_at else None


def score_temporal(results: list, answers: list[str]) -> tuple[bool, int | None]:
    """时间题：检查结果中是否同时有时间锚点和事实。"""
    if not answers:
        return False, None
    str_answers = [str(a) for a in answers if isinstance(a, (str, int, float))]
    for rank, item in enumerate(results, start=1):
        content = item.content or ""
        # 检查是否包含日期格式和相对时间标记
        has_date = bool(re.search(r"\d{4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", content))
        has_relative = any(ans.lower() in content.lower() for ans in str_answers)
        if has_date and has_relative:
            return True, rank
    return False, None


def score_multi_evidence(results: list, answers: list[str]) -> tuple[bool, int | None]:
    """多跳题：检查所有证据片段是否都在结果中。"""
    if not answers:
        return False, None
    found_at = []
    for ans in answers:
        if len(ans.strip()) < 10:
            continue
        for rank, item in enumerate(results, start=1):
            if ans in (item.content or ""):
                found_at.append(rank)
                break
        else:
            return False, None
    return True, min(found_at) if found_at else None


def score_coverage(results: list, answers: list[str]) -> tuple[bool, int | None]:
    """覆盖率题：检查关键事实的覆盖率。"""
    if not answers:
        return False, None
    str_answers = [str(a) for a in answers if isinstance(a, (str, int, float))]
    all_content = " ".join(item.content or "" for item in results[:5])
    found = sum(1 for ans in str_answers if ans.lower() in all_content.lower())
    coverage = found / len(str_answers) if str_answers else 0
    return coverage >= 0.5, 1 if coverage >= 0.5 else None  # 50% 覆盖率算通过


SCORE_FNS = {
    "substring": score_substring,
    "temporal": score_temporal,
    "multi_evidence": score_multi_evidence,
    "coverage": score_coverage,
}


# ─── 主流程 ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="本地基准测试 runner")
    parser.add_argument("--questions", default="bench/benchmark_v1.jsonl", help="题库文件")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="每次搜索的 top_k")
    parser.add_argument("--dump-failures", action="store_true", help="打印失败题详情")
    parser.add_argument("--output", default=None, help="保存结果到 JSON 文件")
    parser.add_argument("--config", default=None, help="覆盖配置文件")
    parser.add_argument(
        "--scope-map",
        default=None,
        help="scope_hash → user_id 映射文件（JSON），用于将题库中的 hash 映射到实际 user_id",
    )
    args = parser.parse_args()

    # 加载题库
    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"错误：题库不存在：{questions_path}", file=sys.stderr)
        print("先运行 .venv/bin/python bench/generate_questions.py 生成题库", file=sys.stderr)
        sys.exit(1)

    questions = []
    with open(questions_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    print(f"题库: {len(questions)} 题")

    # 叶子覆盖
    leaf_counts = Counter(q["leaf"] for q in questions)
    print(f"叶子覆盖: {len(leaf_counts)}/20")

    # 加载 scope 映射（如果有）
    scope_map = {}
    if args.scope_map:
        with open(args.scope_map) as f:
            scope_map = json.load(f)

    # 初始化服务
    # 加载配置
    if args.config:
        from memorysum.settings import PipelineConfig
        pipeline = PipelineConfig.model_validate(json.loads(Path(args.config).read_text(encoding="utf-8")))
    else:
        pipeline = load_pipeline_config()
    runtime = RuntimeSettings.from_environment()
    space = hashlib.sha256(
        (pipeline.embedding_space_id + "\n" + embedding_endpoint(runtime, pipeline)).encode()
    ).hexdigest()
    database = DatabaseManager(runtime.data_root, pipeline, space)
    provider = build_provider(runtime, pipeline)
    service = SearchService(database, provider, pipeline)

    # 按 scope 分组执行
    scope_questions: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        scope_hash = q.get("scope_hash", "")
        scope_questions[scope_hash].append(q)

    print(f"涉及 {len(scope_questions)} 个 scope")
    print()

    # 执行评测
    all_results = []
    failed_scopes = []

    for scope_hash, scope_qs in sorted(scope_questions.items()):
        # 尝试找到 user_id
        user_id = scope_map.get(scope_hash, scope_hash)  # 默认用 scope_hash 作为 user_id
        if not user_id:
            # 尝试直接用 hash 作为 scope_id 查找
            try:
                reference = database.resolve_scope(scope_hash, create=False)
                if reference:
                    user_id = scope_hash
            except Exception:
                pass

        if not user_id:
            # 尝试从 registry 查找 scope_id 对应的 user_id
            import sqlite3
            try:
                conn = sqlite3.connect(str(runtime.data_root / "registry.sqlite3"))
                row = conn.execute(
                    "SELECT scope_id FROM scopes WHERE scope_id LIKE ?",
                    (scope_hash + "%",),
                ).fetchone()
                conn.close()
                if row:
                    user_id = scope_hash
            except Exception:
                pass

        if not user_id:
            failed_scopes.append(scope_hash)
            continue

        # 为这个 scope 的每道题跑搜索
        for q in scope_qs:
            try:
                response = service.search(
                    SearchRequest(query=q["question"], user_id=user_id, top_k=args.top_k)
                )
                results = response.data or []
            except Exception as e:
                all_results.append({"question": q, "hit": False, "rank": None, "error": str(e)})
                continue

            score_fn = SCORE_FNS.get(q.get("score_type", "substring"), score_substring)
            answers = q.get("answer_substrings", [])
            hit, rank = score_fn(results, answers)

            all_results.append({
                "question": q,
                "hit": hit,
                "rank": rank,
                "n_results": len(results),
            })

    if failed_scopes:
        print(f"⚠️  {len(failed_scopes)} 个 scope 未找到（已跳过）")

    # 汇总
    print("=" * 70)
    print("基准测试结果")
    print("=" * 70)
    print()

    # 按叶子汇总
    leaf_stats: dict[str, dict] = {}
    for result in all_results:
        q = result["question"]
        leaf = q["leaf"]
        if leaf not in leaf_stats:
            leaf_stats[leaf] = {"total": 0, "hits": 0, "ranks": []}
        leaf_stats[leaf]["total"] += 1
        if result["hit"]:
            leaf_stats[leaf]["hits"] += 1
        if result.get("rank"):
            leaf_stats[leaf]["ranks"].append(result["rank"])

    # 输出表格
    all_leaves = sorted(set(list(leaf_stats.keys()) + [
        "A1","A2","B1","B2","B4","C1","C2","C3","D1","D2","E1","E2","E3","F1","G1","G2","G4","H2","I1","I2"
    ]))

    total_questions = len(all_results)
    total_hits = sum(1 for r in all_results if r["hit"])
    all_ranks = [r["rank"] for r in all_results if r.get("rank")]

    print(f"{'叶子':5s} {'题数':>4s} {'命中':>4s} {'准确率':>8s} {'MRR':>8s} {'Top-3':>8s}")
    print("-" * 45)

    for leaf in all_leaves:
        stats = leaf_stats.get(leaf, {"total": 0, "hits": 0, "ranks": []})
        total = stats["total"]
        hits = stats["hits"]
        acc = hits / total if total > 0 else 0
        ranks = stats["ranks"]
        mrr = sum(1/r for r in ranks) / len(ranks) if ranks else 0
        top3 = sum(1 for r in ranks if r <= 3) / len(ranks) if ranks else 0

        bar = "█" * int(acc * 20)
        status = "✅" if acc >= 0.7 else ("⚠️" if acc >= 0.3 else "❌")
        print(f"{status} {leaf:4s} {total:>4d} {hits:>4d} {acc:>7.1%}  {mrr:>7.3f}  {top3:>7.1%}  {bar}")

    print("-" * 45)
    total_acc = total_hits / total_questions if total_questions > 0 else 0
    total_mrr = sum(1/r for r in all_ranks) / len(all_ranks) if all_ranks else 0
    total_top3 = sum(1 for r in all_ranks if r <= 3) / len(all_ranks) if all_ranks else 0
    print(f"{'TOTAL':5s} {total_questions:>4d} {total_hits:>4d} {total_acc:>7.1%}  {total_mrr:>7.3f}  {total_top3:>7.1%}")

    # 统计功效
    print()
    se = (total_acc * (1 - total_acc) / total_questions) ** 0.5 if total_questions > 0 else 0
    print(f"标准误: ±{se:.1%}（95% 置信区间 ±{1.96*se:.1%}）")
    print(f"可检测最小差异（80% 功效）: ±{1.96*2*se:.1%}")
    print(f"平台 smoke 赛道（~30 题）的标准误: ±{(0.7*0.3/30)**0.5:.1%}")

    # 失败题详情
    if args.dump_failures:
        print()
        print("=" * 70)
        print("失败题详情")
        print("=" * 70)
        failures = [r for r in all_results if not r["hit"]]
        for r in failures[:30]:
            q = r["question"]
            err = r.get("error", "")
            print(f"  [{q['leaf']}] {q['question'][:60]}")
            if err:
                print(f"        error: {err}")
            print(f"        expected: {q.get('answer_substrings', [])[:2]}")
            print()

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": pipeline.pipeline_id,
            "top_k": args.top_k,
            "total_questions": total_questions,
            "total_hits": total_hits,
            "accuracy": total_acc,
            "mrr": total_mrr,
            "top3": total_top3,
            "standard_error": se,
            "leaf_stats": {
                leaf: {
                    "total": s["total"],
                    "hits": s["hits"],
                    "accuracy": s["hits"] / s["total"] if s["total"] > 0 else 0,
                    "mrr": sum(1/r for r in s["ranks"]) / len(s["ranks"]) if s["ranks"] else 0,
                }
                for leaf, s in leaf_stats.items()
            },
        }
        with open(output_path, "w") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到 {output_path}")

    # 关闭
    provider.close()

    # 退出码
    if total_acc < 0.5:
        sys.exit(1)


if __name__ == "__main__":
    main()