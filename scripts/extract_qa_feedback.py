#!/usr/bin/env python
"""从审计日志里提取平台回灌的逐题 QA 反馈（我们唯一能拿到的"题目 + 判定"来源）。

平台把答题结果作为 Add 载荷回灌，形如：

    Streaming online QA feedback:
    {"method":"locomo_llm_judge","predicted_answer":"Yesterday.","qa_id":"locomo1:q0000:step01",
     "question":"When Jon has lost his job as a banker?","score":0.0,
     "schema_version":"streaming-online-feedback-v1"}

专家不给答案与评分标准，所以这套回灌（`qa_id` / `question` / `predicted_answer` /
`score`）是唯一的官方信号。配合 `extract_loadtest_requests.py` 导出的 Search 清单，
就能离线重放同一批问题并归因：**证据里没有答案 = 召回问题；证据里有答案却判错 =
下游（答题模型/判据）问题**。

用法：
    .venv/bin/python scripts/extract_qa_feedback.py logs/memorysum.log qa.jsonl
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


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    source, dest = Path(sys.argv[1]), Path(sys.argv[2])

    records: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    with source.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            debug = _DEBUG_RE.search(line)
            if debug:
                try:
                    payload = json.loads(debug.group(2))
                except ValueError:
                    continue
                for message in payload.get("messages", []):
                    text = str(message.get("content", ""))
                    if "qa_id" not in text or "question" not in text:
                        continue
                    match = _FEEDBACK_RE.search(text)
                    if not match:
                        continue
                    try:
                        item = json.loads(match.group(0))
                    except ValueError:
                        continue
                    record = {
                        "trace_id": debug.group(1),
                        "user_id": payload.get("user_id"),
                        "session_id": payload.get("session_id"),
                        "request_id": payload.get("request_id"),
                    }
                    record.update({key: item.get(key) for key in _KV_FIELDS})
                    record["role"] = message.get("role")
                    pending.append(record)
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

    scores = Counter(str(record.get("score")) for record in records)
    methods = Counter(str(record.get("method")) for record in records)
    print(f"{len(records)} 条反馈 -> {dest}")
    print(f"  判分分布: {dict(scores)}")
    print(f"  判据:     {dict(methods)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
