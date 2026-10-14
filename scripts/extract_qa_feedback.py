#!/usr/bin/env python
"""从 add 通道里榨出平台的"官方信号"（专家只给 add+query 时，这是我们唯一的判定来源）。

三类信号（全部嵌在 /add 的正文里，不需要专家额外配合）：

1. `Streaming online QA feedback: {qa_id, question, predicted_answer, score, method}` ——
   逐题判定（每轮数量不定，实测 4 条）。
2. `Streaming intra-transfer train trajectory: {dataset, task_index, eval_feedback:{metrics:{score},
   reward, status}}` —— 流式任务类（ALF/DB）的**平台自评**（实测 2 条，task_index 289 score=1、86 score=1）。
3. `[<suite>][session_N][D…]` 头部 —— 每个 scope 属于哪个赛道（用于约束"题 → 子能力"的映射）。

用法：
    .venv/bin/python scripts/extract_qa_feedback.py <日志> <out.jsonl>
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

_FEEDBACK_RE = re.compile(r"\{.*\}", re.DOTALL)
_DEBUG_RE = re.compile(r"event=api_add_debug_payload trace_id=(\S+) payload_json=(.*)$")
_HTTP_RE = re.compile(r"event=http_request timestamp=(\S+) method=(\w+) path=(\S+) status=(\d+)")
_KV_FIELDS = ("qa_id", "question", "predicted_answer", "score", "method", "schema_version")
_SUITE_RE = re.compile(r"\s*\[([A-Za-z0-9_]+)\]\[(session_\d+)\]")


def _record(kind: str, payload: dict, trace: str, item: dict) -> dict:
    record = {
        "kind": kind,
        "trace_id": trace,
        "user_id": payload.get("user_id"),
        "session_id": payload.get("session_id"),
        "request_id": payload.get("request_id"),
    }
    record.update(item)
    return record


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    source, dest = Path(sys.argv[1]), Path(sys.argv[2])

    records: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    suites: set[tuple[object, str]] = set()
    with source.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            debug = _DEBUG_RE.search(line)
            if debug:
                try:
                    payload = json.loads(debug.group(2))
                except ValueError:
                    continue
                trace = debug.group(1)
                for message in payload.get("messages", []):
                    text = str(message.get("content", ""))
                    suite = _SUITE_RE.match(text)
                    if suite and (payload.get("user_id"), suite.group(1)) not in suites:
                        suites.add((payload.get("user_id"), suite.group(1)))
                        pending.append(_record("suite", payload, trace, {"suite": suite.group(1)}))
                    if "train trajectory" in text:
                        try:
                            body = json.loads(_FEEDBACK_RE.search(text).group(0))
                        except (AttributeError, ValueError):
                            continue
                        feedback = body.get("eval_feedback") or {}
                        pending.append(
                            _record(
                                "trajectory",
                                payload,
                                trace,
                                {
                                    "dataset": body.get("dataset"),
                                    "task_index": body.get("task_index"),
                                    "task_score": (feedback.get("metrics") or {}).get("score"),
                                    "reward": feedback.get("reward"),
                                    "status": feedback.get("status"),
                                    "schema_version": body.get("schema_version"),
                                },
                            )
                        )
                    if "qa_id" not in text or "question" not in text:
                        continue
                    match = _FEEDBACK_RE.search(text)
                    if not match:
                        continue
                    try:
                        item = json.loads(match.group(0))
                    except ValueError:
                        continue
                    item = {key: item.get(key) for key in _KV_FIELDS}
                    item["role"] = message.get("role")
                    pending.append(_record("qa_feedback", payload, trace, item))
                continue
            http = _HTTP_RE.search(line)
            if http and pending:
                for record in pending:
                    record["timestamp"] = http.group(1)
                records.extend(pending)
                pending = []

    with dest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    kinds = Counter(str(record.get("kind")) for record in records)
    print(f"{len(records)} 条平台信号 -> {dest}")
    print(f"  分类: {dict(kinds)}")
    scores = Counter(
        str(record.get("score")) for record in records if record.get("kind") == "qa_feedback"
    )
    tasks = [
        f"{record.get('dataset')}#{record.get('task_index')}={record.get('task_score')}"
        for record in records
        if record.get("kind") == "trajectory"
    ]
    if scores:
        print(f"  QA 判分分布: {dict(scores)}")
    if tasks:
        print(f"  流式任务自评: {tasks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
