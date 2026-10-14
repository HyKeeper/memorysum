#!/usr/bin/env python
"""诊断 `_select_output` 到底把候选丢在哪一步（cap=1 与 cap=2 同题 A/B）。

背景：第 7 轮把 `per_source_cap` 1→2 之后，官方题的 `returned` 中位**一条都没变**
（11 → 11），分数逐叶子零变动。这说明 cap 不是瓶颈，我原来的判断错了。
这个脚本用同一批官方题跑两次，把 `_select_output` 的入参（ranked 的长度、unique source 数、
同 source 的条目分布）打出来，再用包装器统计每一步过滤掉多少。

用法：
    .venv/bin/python scripts/diagnose_output_budget.py [requests.jsonl] [--limit 10]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests", nargs="?", default="backups/round6/requests.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    questions: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in open(args.requests, encoding="utf-8"):
        row = json.loads(line)
        if row.get("kind") == "search" and row.get("payload"):
            q = row["payload"]["query"]
            if q in seen:
                continue
            seen.add(q)
            questions.append((q, row["payload"]["user_id"]))
        if len(questions) >= args.limit:
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

    from memorysum.services.search import _primary_message_id  # noqa: PLC0415

    captured: list[dict] = []
    original = SearchService._select_output

    def spy(
        self,
        ranked,
        top_k,
        mode,
        question_cost,
        query="",
    ):  # type: ignore[no-untyped-def]
        entries = list(ranked)
        sources = [_primary_message_id(entry) for entry, _ in entries]
        per_source = Counter(sources)
        captured.append(
            {
                "ranked_len": len(entries),
                "unique_sources": len(per_source),
                "top_source_repeat": per_source.most_common(3),
                "cap": self._pipeline.retrieval.per_source_cap,
                "output_records": mode.output_records,
                "output_tokens": mode.output_tokens,
            }
        )
        return original(self, ranked, top_k, mode, question_cost, query)

    SearchService._select_output = spy  # type: ignore[method-assign]

    print(f"题数 {len(questions)}；config per_source_cap={pipeline.retrieval.per_source_cap}")
    print(
        f"{'#':>3} {'cap':>3} {'ranked':>7} {'uniq_src':>8} {'returned':>8} {'chars':>7} "
        f"{'前3大source重复':>20}"
    )
    print("-" * 78)
    rows = []
    for cap in (1, 2):
        # pydantic 模型是 frozen 的，只能整份重建（SearchService 也在构造时抓住 pipeline）
        pipe = pipeline.model_copy(
            update={"retrieval": pipeline.retrieval.model_copy(update={"per_source_cap": cap})}
        )
        service = SearchService(
            DatabaseManager(runtime.data_root, pipe, space),
            build_provider(runtime, pipe),
            pipe,
        )
        for index, (query, user_id) in enumerate(questions, start=1):
            diagnostics: dict[str, object] = {}
            response = service.search(
                SearchRequest(query=query, user_id=user_id, top_k=100), diagnostics=diagnostics
            )
            info = captured[-1] if captured else {}
            chars = sum(len(item.content) for item in response.data)
            rows.append((index, cap, info, len(response.data), chars))
            print(
                f"{index:>3} {cap:>3} {info.get('ranked_len', 0):>7} "
                f"{info.get('unique_sources', 0):>8} {len(response.data):>8} {chars:>7} "
                f"{str(info.get('top_source_repeat')):>20}"
            )

    print("\n=== 按 cap 汇总 ===")
    for cap in (1, 2):
        sub = [r for r in rows if r[1] == cap]
        if not sub:
            continue
        rets = sorted(r[3] for r in sub)
        ranked = sorted(r[2].get("ranked_len", 0) for r in sub)
        uniq = sorted(r[2].get("unique_sources", 0) for r in sub)
        mid = len(sub) // 2
        print(
            f"cap={cap}: ranked 中位 {ranked[mid]}  unique_source 中位 {uniq[mid]}  "
            f"returned 中位 {rets[mid]}"
        )

    print("\n=== 逐题差异（cap2 - cap1）===")
    by_index = defaultdict(dict)
    for index, cap, _info, ret, chars in rows:
        by_index[index][cap] = (ret, chars)
    changed = 0
    for index, pair in sorted(by_index.items()):
        if 1 in pair and 2 in pair:
            d = pair[2][0] - pair[1][0]
            flag = "   <-- 变了" if d else ""
            if d:
                changed += 1
            print(
                f"  题{index:>3}: cap1={pair[1][0]:>3} 条/{pair[1][1]:>6} 字  "
                f"cap2={pair[2][0]:>3} 条/{pair[2][1]:>6} 字  Δ条数={d:+d}{flag}"
            )
    print(f"\n条数发生变化的题：{changed}/{len(by_index)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
