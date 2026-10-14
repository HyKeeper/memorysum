#!/usr/bin/env python
"""逐题体检：「平台的题面」 vs 「我们收到的库」 vs 「我们交出的证据」。

动机（2026-10-14）：F1=0.00% 那道题（clbench 的 30 天报告）看起来像"我们回答得差"，
实际问题在**灌库**：平台只 Add 了一条 4630 字符的 `D1:1`，而题面是一份 9050 字符的
完整文档 —— 缺的那 4400 字符（artifacts/prior outputs/final task）从来没进过我们的库。
**我们完整存下、完整返回，丢掉的是平台没给的。** 这类失分改我们这边没有任何用。

所以先体检、再动手：对每道题量三个数
  · `query`       题面长度（平台让我们检索什么）
  · `scope`       我们库里有什么（消息数 / 窗口数 / 总字符）
  · `returned`    我们交了什么（条数 / 字符 / 占 scope 的比例）
并按 `returned_chars / scope_chars`（覆盖率）和 `scope_chars / query_chars`（够不够）分类：

  A. `scope ≪ query`      → **灌库缺口**（平台侧，改不了）—— F1 就是这类
  B. `scope 够但覆盖低`   → **我们的问题**（检索/选择没把内容交出去）← 唯一能改的
  C. `scope 够且覆盖高`   → 证据交够了，失分在平台侧（生成/判据）

用法：
    .venv/bin/python scripts/audit_question_coverage.py backups/round6/requests.jsonl --top-k 100
    .venv/bin/python scripts/audit_question_coverage.py <requests.jsonl> --json /tmp/audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def scope_stats(service: SearchService, user_id: str) -> dict:
    reference = service._database.resolve_scope(user_id, create=False)  # noqa: SLF001
    if reference is None:
        return {"messages": 0, "windows": 0, "chars": 0}
    connection = service._database.connect(reference, user_id)
    try:
        messages = connection.execute("select count(*) from messages").fetchone()[0]
        chars = connection.execute(
            "select coalesce(sum(length(raw_content)),0) from messages"
        ).fetchone()[0]
        try:
            windows = connection.execute("select count(*) from windows").fetchone()[0]
        except Exception:  # noqa: BLE001 - 表名随 schema 版本变化，拿不到就算了
            windows = -1
        return {"messages": int(messages), "windows": int(windows), "chars": int(chars)}
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests", nargs="?", default="backups/round6/requests.jsonl")
    parser.add_argument("--top-k", type=int, default=100, help="平台口径是 100")
    parser.add_argument("--json", dest="json_out", default="", help="把逐题明细写到这个文件")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
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
        if args.limit and len(questions) >= args.limit:
            break

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

    rows = []
    print(
        f"{'#':>3} {'query':>6} {'msg':>4} {'win':>5} {'scope_ch':>9} {'ret':>4} {'ret_ch':>7} "
        f"{'cov%':>5} {'scope/query':>11}  判读"
    )
    print("-" * 96)
    for index, (query, user_id) in enumerate(questions, start=1):
        diagnostics: dict[str, object] = {}
        response = service.search(
            SearchRequest(query=query, user_id=user_id, top_k=args.top_k), diagnostics=diagnostics
        )
        stats = scope_stats(service, user_id)
        returned_chars = sum(len(item.content) for item in response.data)
        coverage = returned_chars / stats["chars"] if stats["chars"] else 0.0
        supply = stats["chars"] / len(query) if query else 0.0
        if supply < 0.6:
            verdict = "A 灌库缺口(平台侧)"
        elif coverage < 0.35:
            verdict = "B 覆盖低(我们的问题!)"
        else:
            verdict = "C 交付够(平台侧)"
        rows.append(
            {
                "index": index,
                "query_chars": len(query),
                "scope": stats,
                "returned_records": len(response.data),
                "returned_chars": returned_chars,
                "coverage": coverage,
                "supply": supply,
                "strategy": diagnostics.get("strategy"),
                "query_head": query[:90],
                "user_id": user_id,
                "verdict": verdict,
            }
        )
        print(
            f"{index:>3} {len(query):>6} {stats['messages']:>4} {stats['windows']:>5} "
            f"{stats['chars']:>9} {len(response.data):>4} {returned_chars:>7} "
            f"{coverage * 100:>5.0f} {supply:>11.2f}  {verdict}"
        )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"\n明细写入 {args.json_out}")

    print("\n=== 分档统计 ===")
    from collections import Counter

    counts = Counter(row["verdict"] for row in rows)
    for verdict in sorted(counts):
        print(f"  {verdict}: {counts[verdict]} 题")
    print(
        "\n注：A=平台没把内容灌进来（改不了）；B=内容在库里但我们没交出去（**可改**）；C=交够了。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
