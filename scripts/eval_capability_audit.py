#!/usr/bin/env python
"""按平台的能力树口径，审计我们这侧的证据覆盖（本地可复算，不需要 gold）。

平台给的能力树（teval 冒烟 v1.0）有 A–I 九个能力、20 个子能力，分数按实例加权。
**我们无法本地复现分数本身**——那需要每题的金答案与判据（专家不给）。能复现的是
「输入侧」：对每一道题，我们返回的证据是否覆盖了问题里的关键实体/数值；若连关键
实体都不在返回正文里，那这一题无论判据多宽松都不可能答对（我们的责任）。
反过来，证据在场而判 0 分，则属于下游（答题模型/判据）问题。

题目 → 子能力的映射是**推断**的（题面家族 + 能力命名），不是平台给的，脚本里逐条
标注了依据；有平台回灌的逐题判定时，`scripts/extract_qa_feedback.py` 的 score 可以
直接接进来做校准。

用法：
    .venv/bin/python scripts/eval_capability_audit.py [requests.jsonl] [--user-map B|A]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env", override=False)

from memorysum.api.schemas import SearchRequest  # noqa: E402
from memorysum.providers.factory import build_provider, embedding_endpoint  # noqa: E402
from memorysum.services.search import SearchService  # noqa: E402
from memorysum.settings import RuntimeSettings, load_pipeline_config  # noqa: E402
from memorysum.storage.database import DatabaseManager  # noqa: E402

#: 平台分数从 `backups/platform_tree.json` 读（`fetch_platform_results.py --save-tree` 生成），
#: 默认最新一轮——不再写死在脚本里（曾写死过 A/B 两轮 69.05/68.08，平台跑到第 8 轮后就过期了）。
PLATFORM_TREE = Path("backups/platform_tree.json")


def load_platform_leaves() -> tuple[dict[str, float], str, float | None]:
    """返回（叶子名→分数, 轮次 label, 总分）；树不存在时返回空表。"""
    if not PLATFORM_TREE.exists():
        return {}, "?", None
    tree = json.loads(PLATFORM_TREE.read_text(encoding="utf-8"))
    run = tree["runs"][-1]
    return run["leaves"], run["label"], run.get("total")


#: 题面里的独特片段 → 子能力（推断，按题面家族分组；同族题面共享一条规则）。
MAP_RULES: tuple[tuple[str, str], ...] = (
    ("When Jon has lost his job", "C1"),  # 时间点回忆
    ("Which year did Audrey adopt", "C1"),
    ("INR复查", "C1"),  # 中文医疗：时间点
    ("2024-01-12记录", "C1"),
    ("Can you list the order in which", "C2"),  # 事件顺序
    ("右膝晨僵时长从最初的10分钟，变化成当前", "C2"),  # 变化/阶段
    ("When did Caroline go to the LGBTQ", "C1"),  # 时间点回忆（曾漏映射，dump 里显示为 '?'）
    ("camera’s movement", "B2"),  # 因果/中间步骤（镜头→场景含义）
    ("Taken together, Billing", "B1"),  # 跨片段聚合（多次爆发→整体立场）
    ("courtroom demonstration", "B1"),
    ("Reach 24 using the numbers", "B4"),  # 受控开放域（算 24）
    ("NTSB preliminary excerpt", "B2"),  # 事件链复盘
    ("You are a document-grounded", "G1"),  # 文档grounded推理 ×2
    ("How would you describe the overall OTC uptake", "G4"),  # 经验数据发现
    ("Have I obtained an API key", "H2"),  # 隐私最小披露
    ("What kind of warm-up should I do", "E1"),  # 偏好/个性化
    ("weekend reading time feel more relaxing", "E1"),
    ("医生，最近膝盖又有点刺痛", "E3"),  # 健康关怀
    ("What fields would Caroline be likely to pursue", "A2"),  # 实体/角色归因
    ("What type of music does the author of Dune", "A1"),
    ("Which genre of music is played by the author", "A1"),
    ("Which political party did Joaquim Molins", "A1"),
    ("What is Sarah Garcia's birth date", "A1"),
    ("What is Sarah Garcia's middle name", "A1"),
    ("What is Sarah Garcia's highest education level", "A2"),
    ("What is Sarah Garcia's job title", "D1"),  # as-of 当前值
    ("On Sep 05, 2025, what was Sarah Garcia's job title", "D1"),
    ("What is the official language", "B4"),  # 受控开放域（多跳常识）×3
    ("Which language is officially spoken", "B4"),
    ("What continent", "B4"),
    ("On which continent", "B4"),
    ("Foreman’s behavior", "A2"),
    ("CONVERSATION TRANSCRIPT", "F1"),  # 长历史全局综合（30 天报告）
    ("task index", "I2"),  # 内部迁移
)


def capability_of(question: str) -> str:
    for needle, capability in MAP_RULES:
        if needle.lower() in question.lower():
            return capability
    return "?"


STOP = {
    "What",
    "When",
    "Which",
    "Where",
    "Who",
    "How",
    "Why",
    "Can",
    "You",
    "The",
    "And",
    "Please",
    "Provide",
    "Answer",
    "Given",
    "Taken",
    "Does",
    "Did",
    "Are",
    "Was",
    "Were",
    "Would",
    "Could",
    "Most",
    "About",
    "Over",
    "Into",
    "That",
    "This",
    "There",
    "Their",
    "With",
    "From",
    "Your",
    "Just",
    "Only",
    "Like",
    "Some",
    "List",
    "Order",
    "Based",
    "Document",
    "Question",
    "Option",
    "Options",
    "Correct",
    "Enclosed",
    "Parentheses",
}


def key_terms(question: str, limit: int = 12) -> list[str]:
    """问题里的高信息量词：专名、年份、中文实词。"""
    terms = [token for token in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", question) if token not in STOP]
    terms += re.findall(r"\b(?:19|20)\d{2}\b", question)
    terms += [t for t in re.findall(r"[\u4e00-\u9fff]{2,4}", question)][:6]
    seen: list[str] = []
    for term in terms:
        if term not in seen:
            seen.append(term)
    return seen[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests", nargs="?", default="backups/accuracy-2026-10-14/requests.jsonl")
    parser.add_argument(
        "--verdicts", help="extract_qa_feedback.py 产出的 jsonl（有则叠加真实判定）"
    )
    args = parser.parse_args()

    questions: dict[str, str] = {}
    for line in open(args.requests, encoding="utf-8"):
        row = json.loads(line)
        if row.get("kind") == "search" and row.get("payload"):
            questions.setdefault(row["payload"]["query"], row["payload"]["user_id"])
    verdicts: dict[str, float] = {}
    if args.verdicts and Path(args.verdicts).exists():
        for line in open(args.verdicts, encoding="utf-8"):
            row = json.loads(line)
            if row.get("question"):
                verdicts[" ".join(str(row["question"]).split())] = float(row.get("score") or 0.0)

    runtime = RuntimeSettings.from_environment()
    pipeline = load_pipeline_config()
    space = hashlib.sha256(
        (pipeline.embedding_space_id + "\n" + embedding_endpoint(runtime, pipeline)).encode()
    ).hexdigest()
    service = SearchService(
        DatabaseManager(runtime.data_root, pipeline, space),
        build_provider(runtime, pipeline),
        pipeline,
    )

    per_capability: dict[str, list[tuple[str, float, int]]] = {}
    print(f"{'#':>3} {'能力':<4}{'关键实体覆盖':>12}{'条数':>5}{'官方判定':>9}  题面")
    for index, (query, user_id) in enumerate(questions.items(), start=1):
        capability = capability_of(query)
        response = service.search(SearchRequest(query=query, user_id=user_id, top_k=10))
        haystack = " ".join(" ".join(item.content.split()) for item in response.data).lower()
        terms = key_terms(query)
        hit = sum(1 for term in terms if term.lower() in haystack)
        coverage = hit / len(terms) if terms else 1.0
        verdict = verdicts.get(" ".join(query.split()))
        per_capability.setdefault(capability, []).append((query, coverage, len(response.data)))
        flag = "  <<< 证据里连关键实体都缺" if coverage < 0.6 else ""
        print(
            f"{index:>3} {capability:<4}{coverage * 100:>11.0f}%{len(response.data):>5}"
            f"{('' if verdict is None else str(verdict)):>9}  {query[:44].replace(chr(10), ' ')}{flag}"
        )

    print(
        "\n按能力汇总（平台分数从 backups/platform_tree.json 的最新一轮读；我们的证据实体覆盖率）:"
    )
    raw_leaves, run_label, run_total = load_platform_leaves()
    print(
        f"  平台对照 = {run_label}（总分 {run_total}）"
        if raw_leaves
        else "  （找不到平台树，只列本地覆盖率）"
    )
    print(f"  {'能力':<5}{'平台':>8}{'题数':>5}{'覆盖率中位':>11}")
    for capability, rows in sorted(per_capability.items()):
        coverages = sorted(row[1] for row in rows)
        median = coverages[len(coverages) // 2]
        score = raw_leaves.get(capability)
        shown = f"{score:>8.2f}" if score is not None else f"{'-':>8}"
        print(f"  {capability:<5}{shown}{len(rows):>5}{median * 100:>10.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
