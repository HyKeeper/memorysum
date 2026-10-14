#!/usr/bin/env python
"""检查/清理评测轮次的 scope —— 防止 smoke 与 full 复用 user_id 互相污染。

背景（机制）：scope 按 `sha256(user_id)` 落盘（`data/scopes/<hash>.sqlite3`），
**同一个 user_id 再次 Add 是追加，不是替换**（只有 request_id 与 payload 都相同时才幂等）。
所以「smoke 和 full 用同一个 user_id」会让 full 的检索空间里混进 smoke 的数据：
不只是变多，旧题的问答对还可能被直接命中。

用法：
    # 检查：本轮每个 user_id 的 scope 是否"就是本轮灌的那份"（跑分后立刻跑）
    .venv/bin/python scripts/scope_purity.py check --requests backups/<轮次>/requests.jsonl

    # 预览清理：打印会删哪些文件与 registry 行，不做任何修改
    .venv/bin/python scripts/scope_purity.py purge --requests backups/<轮次>/requests.jsonl

    # 真正清理（**先停 memorysum 服务**，否则连接缓存会继续写旧 inode）
    .venv/bin/python scripts/scope_purity.py purge --requests backups/<轮次>/requests.jsonl --yes

判定标准：
    ・消息数 == 本轮 add 的消息数 → 干净
    ・消息数 > 本轮 add 的消息数 → 被复用/重复灌（差值即污染量）
    ・first_seen_ms 早于本轮开始时间 → 该 scope 是旧轮次留下的
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import sqlite3
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from memorysum.core.identity import scope_id  # noqa: E402

DATA_ROOT = Path(os.getenv("MEMORYSUM_DATA_ROOT", "./data")).expanduser()


def ms_to_text(value: int | None) -> str:
    if not value:
        return "?"
    return datetime.datetime.fromtimestamp(value / 1000, datetime.UTC).strftime("%m-%d %H:%M")


def iso_to_ms(value: str | None) -> int | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return int(datetime.datetime.fromisoformat(text).timestamp() * 1000)
    except ValueError:
        return None


def load_round(requests_path: Path) -> tuple[dict[str, int], dict[str, str], str | None]:
    """返回 (每个 user_id 的 add 消息数, user_id → 首题, 本轮第一次请求时间)。"""
    add_messages: dict[str, int] = collections.defaultdict(int)
    first_question: dict[str, str] = {}
    earliest: str | None = None
    for line in requests_path.open(encoding="utf-8"):
        row = json.loads(line)
        who = row.get("payload", {}).get("user_id")
        if not who:
            continue
        stamp = row.get("timestamp")
        if stamp and (earliest is None or stamp < earliest):
            earliest = stamp
        if row.get("kind") == "add":
            add_messages[who] += len(row["payload"].get("messages", []))
        elif row.get("kind") == "search":
            first_question.setdefault(who, row["payload"].get("query", "")[:52])
    return add_messages, first_question, earliest


def scope_path(who: str) -> Path:
    return DATA_ROOT / "scopes" / f"{scope_id(who)}.sqlite3"


def inspect(who: str) -> dict[str, object]:
    path = scope_path(who)
    if not path.exists():
        return {"exists": False}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        messages = connection.execute("select count(*) from messages").fetchone()[0]
        windows = connection.execute("select count(*) from windows").fetchone()[0]
        adds = connection.execute("select count(*) from requests").fetchone()[0]
        row = connection.execute("select first_seen_ms from scope_manifest").fetchone()
    except sqlite3.Error as exc:
        return {"exists": True, "error": type(exc).__name__}
    finally:
        connection.close()
    return {
        "exists": True,
        "messages": messages,
        "windows": windows,
        "adds": adds,
        "first_seen_ms": row[0] if row else None,
    }


def command_check(args: argparse.Namespace) -> int:
    add_messages, questions, earliest = load_round(Path(args.requests))
    if not add_messages:
        print("本轮没有 /add，无法判断（只有检索的轮次不创建 scope）")
        return 0
    earliest_ms = iso_to_ms(earliest)
    print(f"数据目录 {DATA_ROOT}；本轮 {len(add_messages)} 个空间，最早请求 {earliest}")
    print(
        f"{'scope(短)':<18}{'预期消息':>9}{'实有':>7}{'窗口':>7}{'add次数':>8}  first_seen   判定"
    )
    dirty: list[tuple[str, int, int]] = []
    reused: list[str] = []
    for who, expected in sorted(add_messages.items(), key=lambda kv: -kv[1]):
        info = inspect(who)
        if not info.get("exists"):
            print(f"{who[:16]:<18}{expected:>9}{'不存在':>7}")
            continue
        got = int(info.get("messages", 0))
        seen = int(info.get("first_seen_ms") or 0)
        # 早于本轮最早请求（留 2 分钟容差）→ 这个 scope 是旧轮次留下的，本轮在它上面追加
        stale = bool(earliest_ms and seen and seen < earliest_ms - 120_000)
        if stale:
            verdict = "⚠ 旧 scope（早于本轮，结果会互相干扰）"
            reused.append(who)
        elif got == expected:
            verdict = "干净"
        elif got > expected:
            verdict = f"⚠ 多 {got - expected} 条（被复用/重复灌）"
            dirty.append((who, expected, got))
        else:
            verdict = f"⚠ 少 {expected - got} 条（本轮 add 未全部落库）"
            dirty.append((who, expected, got))
        print(
            f"{who[:16]:<18}{expected:>9}{got:>7}{int(info.get('windows', 0)):>7}"
            f"{int(info.get('adds', 0)):>8}  {ms_to_text(info.get('first_seen_ms')):<12} {verdict}"
        )
    print()
    if reused:
        print(f"发现 {len(reused)} 个 scope 早于本轮就已存在（user_id 被复用）：")
        for who in reused:
            print(f"  {who[:20]}  首题：{questions.get(who, '')!r}")
        print(
            "\n这批 user_id 的旧数据会和本轮叠加 → 跑分前应先清理"
            "（`scope_purity.py purge --requests <同一个文件> --yes`，先停服务）。"
        )
        return 2
    if dirty:
        print(f"发现 {len(dirty)} 个可疑空间：")
        for who, expected, got in dirty:
            print(f"  {who[:20]}  预期 {expected}，实有 {got}，首题：{questions.get(who, '')!r}")
        print(
            "\n若确认是被复用，用 `scope_purity.py purge --requests <同一个文件> --yes` 清理（先停服务）。"
        )
        return 1
    print("全部干净：每个 scope 都是本轮新建，消息数与本轮 /add 一致。")
    return 0


def command_purge(args: argparse.Namespace) -> int:
    add_messages, _, _ = load_round(Path(args.requests))
    targets = sorted(add_messages)
    print(f"将清理 {len(targets)} 个 user_id 对应的 scope（数据目录 {DATA_ROOT}）")
    total_bytes = 0
    planned: list[tuple[str, Path]] = []
    for who in targets:
        path = scope_path(who)
        if not path.exists():
            continue
        planned.append((who, path))
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(path) + suffix)
            if candidate.exists():
                total_bytes += candidate.stat().st_size
    for who, path in planned:
        print(f"  {who[:20]}  {path.name}")
    print(f"共 {len(planned)} 个 scope 文件，{total_bytes / 1e6:.1f} MB")
    if not args.yes:
        print("\n[dry-run] 未做任何修改。确认后加 --yes（并请先停 memorysum 服务）。")
        return 0
    registry = DATA_ROOT / "registry.sqlite3"
    connection = sqlite3.connect(registry)
    try:
        with connection:
            for who, path in planned:
                key = scope_id(who)
                connection.execute("DELETE FROM scopes WHERE scope_id=?", (key,))
                for suffix in ("", "-wal", "-shm"):
                    candidate = Path(str(path) + suffix)
                    if candidate.exists():
                        candidate.unlink()
        connection.execute("VACUUM")
    finally:
        connection.close()
    print(f"\n已清理 {len(planned)} 个 scope（文件与 registry 行）。服务重启后会按需重建空 scope。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("check", command_check), ("purge", command_purge)):
        node = sub.add_parser(name)
        node.add_argument("--requests", required=True, help="该轮的 requests.jsonl")
        if name == "purge":
            node.add_argument("--yes", action="store_true", help="真正执行删除（默认只预览）")
        node.set_defaults(handler=handler)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
