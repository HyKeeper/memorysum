#!/usr/bin/env python
"""把 memorysum 的审计日志解析成结构化请求清单（用于压测复盘 / 准确率打分）。

用法:
    .venv/bin/python scripts/extract_loadtest_requests.py logs/memorysum.log out.jsonl

要点：结构化事件行（`api_add` / `api_search` / `*_debug_payload`）都带 `trace_id`，
**按 trace_id 关联**才能拿到正确的 status/latency 与请求体。不要用「就近的
http_request 行」去配对——并发下会串行错配（实测把 Add 的 payload 配到 Search 的
状态码上）。`http_request` 行只用来给 timestamp 定位，属于近似值。

输出每行一个 JSON 对象：
    {"trace_id", "kind": "search"|"add", "status", "latency_ms", "timestamp",
     "event": {结构化事件的其余字段}, "payload": {原始请求体}}
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_EVENT_RE = re.compile(r"event=(api_(?:add|search)) trace_id=(\S+) (.*)$")
_DEBUG_RE = re.compile(
    r"event=(api_(?:add|search)_debug_payload) trace_id=(\S+) payload_json=(.*)$"
)
_HTTP_RE = re.compile(
    r"event=http_request timestamp=(\S+) method=(\w+) path=(\S+) status=(\d+) latency_ms=([\d.]+)"
)
_KV_SPLIT_RE = re.compile(r"(?:^|\s)(\w+)=")


def parse_kv(text: str) -> dict[str, str]:
    """Split `k=v k=v` where values may contain spaces (e.g. channel_hits dicts)."""
    matches = list(_KV_SPLIT_RE.finditer(text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields[match.group(1)] = text[match.end() : end].strip()
    return fields


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    source, dest = Path(sys.argv[1]), Path(sys.argv[2])

    records: dict[str, dict[str, object]] = {}
    order: list[str] = []
    timestamps: list[tuple[str, str]] = []  # (path, timestamp) 按文件顺序
    pending_payload: dict[str, object] | None = None

    with source.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            debug = _DEBUG_RE.search(line)
            if debug:
                try:
                    payload: object = json.loads(debug.group(3))
                except ValueError:
                    payload = None
                pending_payload = payload
                trace = debug.group(2)
                record = records.setdefault(trace, {"trace_id": trace})
                record["payload"] = payload
                continue

            event = _EVENT_RE.search(line)
            if event:
                kind = event.group(1).split("_", 1)[1]
                trace = event.group(2)
                fields = parse_kv(event.group(3))
                record = records.setdefault(trace, {"trace_id": trace})
                record["kind"] = kind
                record["status"] = int(fields.get("status", 0))
                record["latency_ms"] = int(fields.get("latency_ms", 0))
                record["event"] = {
                    key: value
                    for key, value in fields.items()
                    if key not in {"status", "latency_ms"}
                }
                if pending_payload is not None and "payload" not in record:
                    record["payload"] = pending_payload
                pending_payload = None
                order.append(trace)
                continue

            http = _HTTP_RE.search(line)
            if http:
                timestamps.append((http.group(3), http.group(1)))

    # timestamp 用「文件顺序里同路径的下一条 http_request」近似
    cursor = 0
    for trace in order:
        record = records[trace]
        path = "/search" if record.get("kind") == "search" else "/add"
        while cursor < len(timestamps) and timestamps[cursor][0] != path:
            cursor += 1
        if cursor < len(timestamps):
            record["timestamp"] = timestamps[cursor][1]
            cursor += 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("w", encoding="utf-8") as fh:
        for trace in order:
            fh.write(json.dumps(records[trace], ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
    print(f"{written} 条请求 -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
