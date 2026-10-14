#!/usr/bin/env python
"""把每道题的「问题 + 我们检索返回的证据」dump 成可读块，供人或 agent 逐题判分。

设计：判分不需要专家给的金答案——**证据本身就足以判断**：
  ・若正确答案能从我们返回的正文里读出来 → 这题算我们对（1.0）；
  ・只能读出部分（例如问年份只给了月份区间、问顺序只给了片段）→ 0.5；
  ・返回的正文读不出答案（信息在我们没返回的窗口里，或压根不在库里）→ 0.0。
这测的是「我们交给平台的东西」这一环，与平台自己的答题模型无关。

用法：
    .venv/bin/python scripts/dump_question_evidence.py [requests.jsonl] [--per-record 400] [--records 8]
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
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env", override=False)

from eval_capability_audit import capability_of  # noqa: E402
from memorysum.api.schemas import SearchRequest  # noqa: E402
from memorysum.providers.factory import build_provider, embedding_endpoint  # noqa: E402
from memorysum.services.search import SearchService  # noqa: E402
from memorysum.settings import RuntimeSettings, load_pipeline_config  # noqa: E402
from memorysum.storage.database import DatabaseManager  # noqa: E402


def compact(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests", nargs="?", default="backups/accuracy-2026-10-14/requests.jsonl")
    parser.add_argument("--out", default="/tmp/evidence_dump.md")
    parser.add_argument("--per-record", type=int, default=420)
    parser.add_argument("--records", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10, help="平台口径用 100")
    args = parser.parse_args()

    questions: dict[str, str] = {}
    for line in open(args.requests, encoding="utf-8"):
        row = json.loads(line)
        if row.get("kind") == "search" and row.get("payload"):
            questions.setdefault(row["payload"]["query"], row["payload"]["user_id"])

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

    out = []
    for index, (query, user_id) in enumerate(questions.items(), start=1):
        capability = capability_of(query)
        diagnostics: dict[str, object] = {}
        response = service.search(
            SearchRequest(query=query, user_id=user_id, top_k=args.top_k), diagnostics=diagnostics
        )
        reference = service._database.resolve_scope(user_id, create=False)  # noqa: SLF001
        connection = service._database.connect(reference, user_id)
        messages = connection.execute("select count(*) from messages").fetchone()[0]
        windows = connection.execute("select count(*) from windows").fetchone()[0]
        connection.close()
        out.append(
            f"\n### [{index:>2}] {capability} | scope {messages} 条消息 / {windows} 窗口 | "
            f"strategy={diagnostics.get('strategy')} | 返回 {len(response.data)} 条 | "
            f"候选 {diagnostics.get('candidate_windows')} / 组 {diagnostics.get('rerank_groups')}\n"
            f"Q: {compact(query, 900)}\n"
        )
        for rank, item in enumerate(response.data[: args.records], start=1):
            out.append(f"  #{rank} {compact(item.content, args.per_record)}\n")

    Path(args.out).write_text("".join(out), encoding="utf-8")
    print(f"写入 {args.out}（{len(questions)} 题，{sum(len(x) for x in out)} 字符）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
