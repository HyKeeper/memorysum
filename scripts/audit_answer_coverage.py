#!/usr/bin/env python
"""对每道真实题做「相关内容有没有被交出去」的自动审计。

动机：2026-10-14 的抽查（11 题）显示答案基本都在第 1 条，但那是人工挑的题。
这个脚本把检查自动化：从题面抽出**实体词**（人名/专有名词/数字），在 scope 的消息里找
含这些词的消息，再看它们有没有出现在我们返回的记录里。

判据（宁可保守，避免误报）：
  · 若含实体词的消息**一条都没被交出** → `MISS`（真正值得查的）
  · 若交出了但**少于一半** → `PART`
  · 否则 `OK`

注意：这不是"答案有没有"的判据（那需要 gold，只有平台有）。它只回答"我们有没有把
scope 里与题相关的部分交出去"，用来把"我们的问题"和"平台侧问题"分开。

用法：
    .venv/bin/python scripts/audit_answer_coverage.py backups/round6/requests.jsonl [--top-k 100]
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

# 题面里常见的噪声词（不该当实体）
_STOP = {
    "what",
    "which",
    "when",
    "where",
    "who",
    "why",
    "how",
    "the",
    "and",
    "for",
    "with",
    "about",
    "from",
    "that",
    "this",
    "these",
    "those",
    "does",
    "did",
    "was",
    "were",
    "are",
    "you",
    "your",
    "his",
    "her",
    "their",
    "请",
    "什么",
    "为什么",
    "如何",
    "哪些",
    "是否",
}


def _sampled_in(text: str, joined: str, points: int = 4, width: int = 60) -> bool:
    """在消息的多个位置采样，任一采样片段出现在返回里就算交付。

    窗口可能只覆盖消息中段，只看开头会误判 MISS。
    """
    if not text:
        return False
    if len(text) <= width:
        return text in joined
    step = max(1, (len(text) - width) // max(1, points - 1))
    for start in range(0, len(text) - width + 1, step):
        if text[start : start + width] in joined:
            return True
    return False


def entities(query: str) -> list[str]:
    """抽可当锚点的实体：连续的大写开头词、数字、中文专有名词。"""
    out: list[str] = []
    # 连续 TitleCase / 全大写 词组（人名、机构名）
    for match in re.finditer(r"\b(?:[A-Z][a-z]{2,})(?:\s+[A-Z][a-z]{2,})*\b", query):
        token = match.group(0)
        if token.lower() not in _STOP and len(token) >= 4:
            out.append(token)
    # 4 位年份 / 明确的数字串
    out.extend(re.findall(r"\b(?:19|20)\d{2}\b", query))
    # 中文专名（2-6 个连续汉字，排除常见虚词）
    for match in re.finditer(r"[\u4e00-\u9fff]{2,6}", query):
        token = match.group(0)
        if token not in _STOP and len(token) >= 2:
            out.append(token)
    seen, uniq = set(), []
    for token in out:
        if token not in seen:
            seen.add(token)
            uniq.append(token)
    return uniq[:14]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests", nargs="?", default="backups/round6/requests.jsonl")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    questions: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in open(args.requests, encoding="utf-8"):
        row = json.loads(line)
        if row.get("kind") != "search" or not row.get("payload"):
            continue
        query, user_id = row["payload"]["query"], row["payload"]["user_id"]
        if query in seen:
            continue
        seen.add(query)
        questions.append((query, user_id))

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

    print(f"{'#':>3} {'相关消息':>8} {'交出':>5} {'条数':>5}  判读  实体样例 / 题面")
    print("-" * 118)
    verdicts = {"OK": 0, "PART": 0, "MISS": 0, "NO_ENTITY": 0}
    misses: list[str] = []
    for index, (query, user_id) in enumerate(questions, start=1):
        ents = entities(query)
        if not ents:
            verdicts["NO_ENTITY"] += 1
            print(f"{index:>3} {'-':>8} {'-':>5} {'-':>5}  无实体  {query[:60]!r}")
            continue
        reference = service._database.resolve_scope(user_id, create=False)  # noqa: SLF001
        if reference is None:
            continue
        connection = service._database.connect(reference, user_id)
        rows = list(connection.execute("select message_id, raw_content from messages"))
        connection.close()
        relevant = {mid for mid, text in rows if any(e in text for e in ents)}
        diagnostics: dict[str, object] = {}
        response = service.search(
            SearchRequest(query=query, user_id=user_id, top_k=args.top_k), diagnostics=diagnostics
        )
        joined = "\n".join(item.content for item in response.data)

        delivered = {mid for mid, text in rows if mid in relevant and _sampled_in(text, joined)}
        if not relevant:
            verdict = "NO_ENTITY"
        elif not delivered:
            verdict = "MISS"
        elif len(delivered) * 2 < len(relevant):
            verdict = "PART"
        else:
            verdict = "OK"
        verdicts[verdict] += 1
        if verdict in ("MISS", "PART"):
            misses.append(
                f"[{index}] {verdict} relevant={len(relevant)} delivered={len(delivered)} "
                f"| {query[:70]}"
            )
        print(
            f"{index:>3} {len(relevant):>8} {len(delivered):>5} {len(response.data):>5}  "
            f"{verdict:<5}  {', '.join(ents[:4])} | {query[:46]!r}"
        )

    print("\n=== 汇总 ===")
    for key, value in verdicts.items():
        print(f"  {key}: {value}")
    if misses:
        print("\n=== 需要人工看的 ===")
        for line in misses:
            print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
