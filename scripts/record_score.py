#!/usr/bin/env python
"""把「一次平台跑分」与当时的代码/配置指纹一起记进台账。

为什么需要：分数只有在**能对上当时的环境**时才有意义。这个脚本自动抓：
  · git HEAD（含 src/ 是否脏、HEAD 里 src 树的哈希）；
  · 线上 memorysum 进程（pid / 启动时间）与「启动时间是否晚于最后一次改 src 的提交」；
  · 两个模型服务的完整启动参数（rerank 的 `-ub` 曾经就是隐形上限）；
  · config/champion-v1.json 的 sha256；
  · 本轮数据目录、平台总分、本地严格分与逐叶子偏差（可选）。
然后追加一行到 `docs/score_ledger.md`，并写一条结构化记录到 `docs/scores.json`。

用法：
    .venv/bin/python scripts/record_score.py --total 69.05 --round 2026-10-14T15:28 \
        --data backups/accuracy-2026-10-14T1528 --verdicts <verdicts.jsonl> --tree <platform_tree.json> --note "…"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

LEDGER = Path("docs/score_ledger.md")
SCORES = Path("docs/scores.json")
CONFIG = Path("src/memorysum/config/champion-v1.json")
PID_FILE = Path("logs/memorysum.pid")


def run(*args: str) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_info(pattern: str) -> str:
    output = run(
        "bash", "-c", f"ps -eo pid,args -w -w | grep -F -e '{pattern}' | grep -v grep | head -1"
    )
    if not output:
        return "未运行"
    pid = output.split()[0]
    started = run("ps", "-p", pid, "-o", "lstart=")
    return f"pid={pid} 启动={started.strip()} | {output.split(' ', 1)[1]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=float, required=True, help="平台总分")
    parser.add_argument("--round", required=True, help="轮次标识（如 2026-10-14T15:28 本地时间）")
    parser.add_argument("--data", help="本轮数据目录（backups/...）")
    parser.add_argument("--verdicts", help="本地判定文件（用于算本地严格分）")
    parser.add_argument("--tree", help="平台分数表 json")
    parser.add_argument("--note", default="")
    parser.add_argument(
        "--deployed-src",
        help="当时**线上真正在跑**的代码提交（补录历史轮次时用；默认取当前 HEAD，"
        "并提示线上进程启动时间是否晚于 src 最后一次改动）",
    )
    parser.add_argument(
        "--env-index",
        type=int,
        help="复用台账里第 N 条（0 起）的环境指纹——用于补录历史轮次，避免把今天的进程状态记到过去",
    )
    args = parser.parse_args()

    head = run("git", "rev-parse", "HEAD")
    subject = run("git", "log", "-1", "--pretty=%s")
    dirty = run("git", "status", "--porcelain")
    src_dirty = [line for line in dirty.splitlines() if "src/" in line]
    src_tree = run("git", "rev-parse", "HEAD:src")
    # 最后一次改动 src/ 的提交时间，用来判断线上进程是否已包含全部代码改动
    last_src_commit_iso = run("git", "log", "-1", "--format=%cI", "--", "src")
    last_src_commit_sha = run("git", "log", "-1", "--format=%h", "--", "src")

    pid = PID_FILE.read_text(encoding="utf-8").strip() if PID_FILE.exists() else ""
    service_started = run("ps", "-p", pid, "-o", "lstart=").strip() if pid else ""
    service_line = process_info("memorysum") or "未运行"

    local_score = None
    leaf_deltas: dict[str, float] = {}
    platform_leaves: dict[str, float] = {}
    if args.verdicts and Path(args.verdicts).exists():
        rows = [json.loads(line) for line in open(args.verdicts, encoding="utf-8")]
        per_leaf: dict[str, list[float]] = {}
        for row in rows:
            per_leaf.setdefault(row["cap"], []).append(float(row["v"]))
        local_score = round(
            100.0 * sum(sum(v) for v in per_leaf.values()) / sum(len(v) for v in per_leaf.values()),
            2,
        )
        if args.tree and Path(args.tree).exists():
            tree = json.loads(Path(args.tree).read_text(encoding="utf-8"))
            platform_leaves = tree["runs"][-1]["leaves"]
            for leaf, values in per_leaf.items():
                if leaf in platform_leaves:
                    leaf_deltas[leaf] = round(
                        100.0 * sum(values) / len(values) - platform_leaves[leaf], 2
                    )

    entry = {
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "round": args.round,
        "platform_total": args.total,
        "local_strict_total": local_score,
        "platform_leaves": platform_leaves,
        "leaf_deltas": leaf_deltas,
        "deployed_src": args.deployed_src or f"{head[:8]} {subject[:40]}",
        "git_head": head,
        "git_head_subject": subject,
        "src_tree": src_tree,
        "src_dirty": src_dirty,
        "last_src_commit": f"{last_src_commit_sha} @ {last_src_commit_iso}",
        "service": service_line,
        "service_started": service_started,
        "rerank_server": process_info("--port 7997"),
        "embedding_server": process_info("--port 7999"),
        "config_sha256": sha256(CONFIG),
        "data_dir": args.data,
        "note": args.note,
    }

    history = json.loads(SCORES.read_text(encoding="utf-8")) if SCORES.exists() else []
    if args.env_index is not None:
        source = history[args.env_index]
        for field in (
            "git_head",
            "git_head_subject",
            "src_tree",
            "last_src_commit",
            "service",
            "service_started",
            "rerank_server",
            "embedding_server",
            "config_sha256",
        ):
            if field in source:
                entry[field] = source[field]
        entry["env_reused_from"] = source["round"]
    history.append(entry)
    SCORES.write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")

    if not LEDGER.exists():
        LEDGER.write_text(
            "# 平台跑分台账\n\n每一行 = 一次专家跑分，并记录当时的代码/配置指纹。\n"
            "**分数必须能对上环境才有意义**：同一个 commit 在改过 `-ub`、`config` 或模型服务后不再等价。\n\n"
            "| 轮次 | 平台总分 | 本地严格分 | 差值 | 部署 src | 线上进程启动 | rerank `-ub` | 数据目录 |\n"
            "|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    ub_match = re.search(r"-ub (\d+)", entry["rerank_server"])
    rerank_ub = ub_match.group(1) if ub_match else "—"
    row = (
        f"| {args.round} | **{args.total:.2f}** | "
        f"{'' if local_score is None else f'{local_score:.2f}'} | "
        f"{'' if local_score is None else f'{local_score - args.total:+.2f}'} | "
        f"{entry['deployed_src']} | {service_started or '—'} | {rerank_ub or '—'} | "
        f"`{args.data or '—'}` |\n"
    )
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(row)

    print(f"已记入 {LEDGER} 与 {SCORES}（共 {len(history)} 轮）")
    print(f"  git HEAD      = {head[:8]} {subject[:50]}")
    print(f"  src 最后一次改动 = {last_src_commit_sha} @ {last_src_commit_iso}")
    print(f"  线上服务       = {service_line[:110]}")
    print(f"  config sha256 = {entry['config_sha256'][:16]}  数据目录 = {args.data}")
    if local_score is not None:
        print(f"  本地严格分 = {local_score:.2f}（差值 {local_score - args.total:+.2f}）")
        print(f"  最大叶子偏差 = {sorted(leaf_deltas.items(), key=lambda kv: -abs(kv[1]))[:4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
