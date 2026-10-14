#!/usr/bin/env python
"""窗口粒度扫描：同一份数据用不同 chunking 配置重建，再用同一批 query 评测。

背景：AGENTS.md 第 11 条声称「窗口粒度已经是最优（512/640 或 256/320 都掉到 29/60）」，
但 git 历史里 `leaf 512` 从未出现过，且当前配置就是 256/320 —— 该结论没有可追溯的记录。
本脚本用影子 data_root 重建 + 评测，把这条结论重新钉一遍。

用法：
    .venv/bin/python scripts/window_sweep.py [--user-id <beam scope>]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env", override=False)

import eval_retrieval as ER  # noqa: E402
from memorysum.api.schemas import AddRequest, SearchRequest  # noqa: E402
from memorysum.core.identity import scope_id  # noqa: E402
from memorysum.providers.factory import build_provider, embedding_endpoint  # noqa: E402
from memorysum.services.add import AddService  # noqa: E402
from memorysum.services.search import SearchService  # noqa: E402
from memorysum.settings import RuntimeSettings, load_pipeline_config  # noqa: E402
from memorysum.storage.database import DatabaseManager  # noqa: E402

DEFAULT_USER = ER.DEFAULT_USER
SHADOW_ROOT = Path("/tmp/ms-window-sweep")

#: (标签, leaf target/max, context target/max, overlap)
CONFIGS = [
    ("base-256/320", 256, 320, 320, 384),
    ("up-384/480", 384, 480, 480, 560),
    ("up-512/640", 512, 640, 640, 768),
    ("down-128/160", 128, 160, 160, 192),
]


def shadow_pipeline(base, label: str, leaf: int, leaf_max: int, ctx: int, ctx_max: int):
    chunking = base.chunking.model_copy(
        update={
            "leaf": base.chunking.leaf.model_copy(
                update={"target": leaf, "max": leaf_max, "min": min(128, leaf), "overlap": 64}
            ),
            "context": base.chunking.context.model_copy(
                update={"target": ctx, "max": ctx_max, "min": min(128, ctx), "overlap": 64}
            ),
        }
    )
    # chunking 属于数据契约 → 影子 pipeline_id 也要区分，避免与生产 scope 混用
    return base.model_copy(
        update={"chunking": chunking, "pipeline_id": f"{base.pipeline_id}-exp-{label}"}
    )


def read_source(user_id: str) -> list[tuple[str, str, str, int | None]]:
    path = Path("data/scopes") / f"{scope_id(user_id)}.sqlite3"
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = [
        (r["request_id"], r["role"], r["raw_content"], r["source_timestamp_ms"])
        for r in connection.execute(
            "select request_id, role, raw_content, source_timestamp_ms, ordinal from messages "
            "order by request_id, ordinal"
        )
    ]
    connection.close()
    return rows


def rebuild(user_id: str, pipeline, source_rows):
    runtime = RuntimeSettings.from_environment()
    space = hashlib.sha256(
        (pipeline.embedding_space_id + "\n" + embedding_endpoint(runtime, pipeline)).encode()
    ).hexdigest()
    database = DatabaseManager(SHADOW_ROOT, pipeline, space)
    provider = build_provider(runtime, pipeline)
    service = AddService(database, provider, pipeline)
    started = time.perf_counter()
    batches: list[tuple[str, list]] = []
    for request_id, role, content, ts in source_rows:
        if not batches or batches[-1][0] != request_id:
            batches.append((request_id, []))
        message = {"role": role, "content": content}
        if ts is not None:
            message["timestamp"] = int(ts)
        batches[-1][1].append(message)
    for request_id, messages in batches:
        service.add(
            AddRequest(
                user_id="sweep-target", session_id="sweep", request_id=request_id, messages=messages
            )
        )
    provider.close()
    return database, time.perf_counter() - started


def evaluate(database, pipeline):
    runtime = RuntimeSettings.from_environment()
    provider = build_provider(runtime, pipeline)
    service = SearchService(database, provider, pipeline)
    gold = ER.load_gold(database, "sweep-target")
    ranks, id_ranks = [], []
    for query, gold_id, _marker in ER.CASES:
        response = service.search(SearchRequest(query=query, user_id="sweep-target", top_k=10))
        target, gold_ids = gold.get(gold_id, ("", ()))
        rank = id_rank = None
        for index, item in enumerate(response.data, start=1):
            if rank is None and ER.contains_verbatim(ER.normalise(item.content), target):
                rank = index
            if id_rank is None and ER.ids_hit(item, gold_ids):
                id_rank = index
        ranks.append(rank)
        id_ranks.append(id_rank)
    provider.close()
    return ranks, id_ranks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", default=DEFAULT_USER)
    args = parser.parse_args()
    base = load_pipeline_config()
    source_rows = read_source(args.user_id)
    print(f"源 scope: {len(source_rows)} 条消息（{args.user_id[:28]}…）")
    print(f"影子目录: {SHADOW_ROOT}\n")
    print(
        f"{'配置':<16}{'窗口':>7}{'重建s':>8}{'txt top1':>10}{'top3':>7}{'MRR':>8}{'ids top1':>10}{'前半':>7}{'后半':>7}"
    )
    results = []
    for label, leaf, leaf_max, ctx, ctx_max in CONFIGS:
        shutil.rmtree(SHADOW_ROOT, ignore_errors=True)
        SHADOW_ROOT.mkdir(parents=True, exist_ok=True)
        pipeline = shadow_pipeline(base, label, leaf, leaf_max, ctx, ctx_max)
        database, took = rebuild(args.user_id, pipeline, source_rows)
        ranks, id_ranks = evaluate(database, pipeline)
        t1, t3, mrr = ER.metrics(ranks)
        i1, _i3, _im = ER.metrics(id_ranks)
        half = len(ranks) // 2
        a = ER.metrics(ranks[:half])[0]
        b = ER.metrics(ranks[half:])[0]
        connection = sqlite3.connect(SHADOW_ROOT / "scopes" / f"{scope_id('sweep-target')}.sqlite3")
        windows = connection.execute("select count(*) from windows").fetchone()[0]
        connection.close()
        print(
            f"{label:<16}{windows:>7}{took:>8.0f}{t1:>7}/60{t3:>7}{mrr:>8.4f}{i1:>7}/60{a:>7}{b:>7}"
        )
        results.append((label, t1, t3, mrr, i1, windows))
    shutil.rmtree(SHADOW_ROOT, ignore_errors=True)
    print("\n汇总（txt_all top-1 / ids_all top-1）：")
    for label, t1, _t3, mrr, i1, windows in results:
        print(f"  {label:<16} txt {t1}/60  ids {i1}/60  MRR {mrr:.4f}  窗口 {windows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
