#!/usr/bin/env python
"""把平台回灌的赛道标签（suite）连到具体的 scope 与题面上。

为什么需要它：我们只能被动观测（AGENTS.md 约定 5）—— 评测方只发 `/add`、`/search`
并给分，不回答任何问题。所以每一点关于"平台在测什么"的知识都必须从流量里榨出来。

链条：`/add` 里嵌的 `[<suite>][session_N][D…]` 头部 → `extract_qa_feedback.py` 抽出的
`suite` 信号带 `session_id` → 该 session 的 add 请求带 `user_id` → 该 scope 的 `/search`
就是属于这个赛道的题。

用法：
    .venv/bin/python scripts/map_suites.py backups/<轮次>/platform_signals.jsonl \
        backups/<轮次>/requests.jsonl
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    signals = Path(sys.argv[1])
    requests = Path(sys.argv[2])

    rows = [json.loads(line) for line in requests.open(encoding="utf-8")]
    by_session: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        session = (row.get("payload") or {}).get("session_id")
        if session:
            by_session[session].append(row)

    scopes_by_suite: dict[str, set[str]] = collections.defaultdict(set)
    for line in signals.open(encoding="utf-8"):
        record = json.loads(line)
        if record.get("kind") != "suite":
            continue
        suite = str(record.get("suite"))
        for row in by_session.get(str(record.get("session_id")), []):
            who = (row.get("payload") or {}).get("user_id")
            if who:
                scopes_by_suite[suite].add(who)

    questions_by_scope: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        if row.get("kind") != "search":
            continue
        who = (row.get("payload") or {}).get("user_id")
        if who:
            questions_by_scope[who].append(str(row["payload"].get("query", "")))

    print(f"{'赛道':<22}{'scope':<14}{'题数':>5}")
    total = 0
    for suite in sorted(scopes_by_suite):
        for who in sorted(scopes_by_suite[suite]):
            questions = questions_by_scope.get(who, [])
            total += len(questions)
            print(f"\n[{suite}] scope {who[:14]}  {len(questions)} 题")
            for question in questions:
                print(f"    {question[:88]!r}")
    print(f"\n合计 {total} 条检索带有赛道标签")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
