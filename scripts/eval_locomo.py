#!/usr/bin/env python
"""LoCoMo 官方标注检索评测：用 evidence 判定"这次命中能否被平台认出来"。

数据：官方 `locomo10.json`（含 199 题/对话 的 question / answer / evidence / category）。
    curl -sSL -o locomo10.json https://cdn.jsdelivr.net/gh/snap-research/locomo@main/data/locomo10.json
再用 `ADD 示例`（README）把该对话灌成 scope，或复用已有 scope。

两个口径，都要看：
- **by_ids**：返回项的 `source_ids` 是否覆盖 gold evidence（`D1:3` 这样的 chunk 级 id）。
  这是平台 `_pick_ids` 的匹配方式，直接决定 Recall / F1。
- **by_text**：gold turn 的原文是否真的出现在返回正文里。衡量"LLM 有没有看到答案"，
  也是 accuracy 的上限。两者之差 = chunk 级 id 粒度掩盖的部分。

`--budget` 是一个**敏感性分析工具**：假设响应被截到 N token 时，同一次检索还剩多少条/
多少 gold（用来观察"条目大小 vs 条目数量"的权衡）。**平台侧没有关于响应长度限制的
任何口径**（2026-10-14 由项目方确认），所以这只是"如果我们猜错了会怎样"的对照，
**不是对平台行为的模拟**，不要拿它的结论当配置依据。

用法（仓库根目录）：
    .venv/bin/python scripts/eval_locomo.py --data /home/rain/bench/locomo10.json
    .venv/bin/python scripts/eval_locomo.py --budgets 0,4096 --tags baseline,direct384
离线运行（进程内直接调服务层），不经过 HTTP，不会干扰正在运行的 memorysum。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.getcwd(), ".env"), override=False)

from memorysum.api.schemas import SearchRequest  # noqa: E402
from memorysum.core.text import reader_token_count  # noqa: E402
from memorysum.providers.factory import build_provider, embedding_endpoint  # noqa: E402
from memorysum.services.search import SearchService  # noqa: E402
from memorysum.settings import RuntimeSettings, load_pipeline_config  # noqa: E402
from memorysum.storage.database import DatabaseManager  # noqa: E402

DEFAULT_USER = (
    "agentic_memory_http_memory_locomo_qa_chunk4096_run6d1f2b5a-632c-4b09-af80-e7f57390d0ae_exp0"
)
CAT_NAMES = {1: "multi-hop", 2: "temporal", 3: "open", 4: "single-hop", 5: "adversarial"}
_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def gold_turn_texts(conv: dict, evidence: list[str]) -> list[str]:
    """`D1:3` -> `session_1` 第 3 个 turn 的原文。"""
    out = []
    for ev in evidence:
        m = re.fullmatch(r"D(\d+):(\d+)", ev.strip().rstrip(";"))
        if not m:
            continue
        turns = conv.get(f"session_{int(m.group(1))}") or []
        idx = int(m.group(2))
        if 0 < idx <= len(turns):
            item = turns[idx - 1]
            text = item.get("text") if isinstance(item, dict) else str(item)
            if text:
                out.append(norm(text)[:120])
    return out


def score(items, case: dict, conv: dict, budget: int) -> tuple[bool, bool, int, int]:
    """Return (ids_all, txt_all, first_rank, kept_records) under an optional token budget."""
    got: set[str] = set()
    blob: list[str] = []
    used = kept = 0
    rank = 0
    for i, hit in enumerate(items):
        cost = reader_token_count(hit.content)
        if budget and used + cost > budget:
            break
        used += cost
        kept += 1
        got.update(hit.source_ids or [])
        blob.append(norm(hit.content))
        if rank == 0 and set(case["evidence"]) & set(hit.source_ids or []):
            rank = i + 1
    ev = set(case["evidence"])
    joined = " ".join(blob)
    texts = gold_turn_texts(conv, case["evidence"])
    return ev <= got, bool(texts) and all(t in joined for t in texts), rank, kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/rain/bench/locomo10.json")
    ap.add_argument("--sample-id", default="conv-26")
    ap.add_argument("--user-id", default=DEFAULT_USER)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="每类最多取这么多题；0 = 全量")
    ap.add_argument("--budgets", default="0,4096,8192")
    ap.add_argument("--tags", default="current", help="逗号分隔的标签，仅用于报告")
    ap.add_argument("--dump", default="", help="把每题明细写到 jsonl")
    args = ap.parse_args()

    data = json.load(open(args.data))
    sample = next((x for x in data if x["sample_id"] == args.sample_id), None)
    if sample is None:
        raise SystemExit(f"{args.data} 里没有 {args.sample_id}")
    conv = sample["conversation"]
    by_cat: dict[int, list[dict]] = {}
    for item in sample["qa"]:
        cat = int(item.get("category", 0))
        if cat in CAT_NAMES and cat != 5 and item.get("evidence"):
            by_cat.setdefault(cat, []).append(item)
    cases: list[dict] = []
    for i in range(max(len(v) for v in by_cat.values())):
        for cat in sorted(by_cat):
            if i < len(by_cat[cat]) and (not args.limit or len(cases) < args.limit):
                cases.append(by_cat[cat][i])

    runtime = RuntimeSettings.from_environment()
    pipeline = load_pipeline_config()
    space = hashlib.sha256(
        (pipeline.embedding_space_id + "\n" + embedding_endpoint(runtime, pipeline)).encode()
    ).hexdigest()
    db = DatabaseManager(runtime.data_root, pipeline, space)
    provider = build_provider(runtime, pipeline)
    service = SearchService(db, provider, pipeline)

    budgets = [int(x) for x in args.budgets.split(",")]
    started = time.time()
    results = []
    for case in cases:
        # 尾部空格绕过进程内结果缓存，保证每次评测都真跑检索
        resp = service.search(
            SearchRequest(query=case["question"] + " ", user_id=args.user_id, top_k=args.top_k)
        )
        results.append((resp.data, case))
    elapsed = time.time() - started

    print(
        f"\n=== sample={args.sample_id} user=...{args.user_id[-24:]} top_k={args.top_k} "
        f"n={len(cases)} {elapsed:.0f}s ==="
    )
    print(
        f"{'tag':<14}{'budget':>8}{'ids_all':>9}{'txt_all':>9}{'MRR':>8}{'avg_recs':>10}   per-cat ids_all"
    )
    for budget in budgets:
        ia = tx = recs = 0
        mrr = 0.0
        per_cat: dict[int, list[int]] = {}
        for items, case in results:
            a, b, rank, kept = score(items, case, conv, budget)
            ia += a
            tx += b
            recs += kept
            mrr += 1.0 / rank if rank else 0.0
            per_cat.setdefault(int(case["category"]), []).append(a)
        n = len(results)
        cats = {CAT_NAMES[c]: round(100 * sum(v) / len(v), 1) for c, v in sorted(per_cat.items())}
        label = args.tags.split(",")[0] if budgets.index(budget) == 0 else ""
        print(
            f"{label:<14}{budget or 'off':>8}{100 * ia / n:>8.1f}%{100 * tx / n:>8.1f}%"
            f"{mrr / n:>8.3f}{recs / n:>10.1f}   {cats}"
        )
        if args.dump:
            with open(args.dump, "a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "tag": args.tags,
                            "budget": budget,
                            "ids_all": 100 * ia / n,
                            "txt_all": 100 * tx / n,
                            "mrr": mrr / n,
                            "avg_recs": recs / n,
                            "n": n,
                            "by_cat": cats,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    provider.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
