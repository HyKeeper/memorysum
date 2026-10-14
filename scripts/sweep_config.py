#!/usr/bin/env python
"""批量扫描配置变体：改一个键 → 跑 eval_retrieval → 记录指标 → 还原。

自动化 `scripts/eval_retrieval.py` 的重复劳动，并强制保存**完整**结果（txt_all / ids_all /
split-half），因为这些单变量实验最容易犯的错就是只看全样本净值。

用法：
    .venv/bin/python scripts/sweep_config.py --set reranker.fusion_weight=1.5 \
        --set reranker.fusion_weight=3.0 --baseline
    # --set 可以给多个；--baseline 表示先跑一次当前配置作为对照

输出：追加到 backups/config-sweep.jsonl（每行一个配置的结果），并打印对比表。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

CONFIG = Path("src/memorysum/config/champion-v1.json")
LEDGER = Path("backups/config-sweep.jsonl")


def read_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def write_config(data: dict) -> None:
    CONFIG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def apply_override(data: dict, spec: str) -> None:
    """`a.b.c=value` —— 值按 JSON 解析（数字/布尔/null/数组），失败就当字符串。"""
    path, _, raw = spec.partition("=")
    keys = path.split(".")
    node = data
    for key in keys[:-1]:
        node = node[key]
    try:
        value = json.loads(raw)
    except ValueError:
        value = raw
    node[keys[-1]] = value


def run_eval(top_k: int = 10) -> dict:
    started = datetime.now(UTC)
    proc = subprocess.run(
        [".venv/bin/python", "scripts/eval_retrieval.py", "--top-k", str(top_k)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    text = proc.stdout + proc.stderr
    result: dict[str, object] = {"seconds": (datetime.now(UTC) - started).total_seconds()}
    # 按行解析（不要用跨行正则：`[^:]*` 会匹配换行，把 ids_all 的 MRR 误配给 txt_all）
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("txt_all"):
            numbers = re.findall(r"(top1|top3|MRR)=([\d.]+)", stripped)
            for key, value in numbers:
                result[f"txt_{key.lower()}"] = float(value) if "." in value else int(value)
        elif stripped.startswith("ids_all"):
            numbers = re.findall(r"(top1|top3|MRR)=([\d.]+)", stripped)
            for key, value in numbers:
                result[f"ids_{key.lower()}"] = float(value) if "." in value else int(value)
        elif stripped.startswith("前半") or stripped.startswith("后半"):
            half = "half1" if stripped.startswith("前半") else "half2"
            txt = re.search(r"txt_all top1=(\d+)/30", stripped)
            ids = re.search(r"ids_all top1=(\d+)/30", stripped)
            result[f"{half}_txt"] = int(txt.group(1)) if txt else None
            result[f"{half}_ids"] = int(ids.group(1)) if ids else None
    if result["txt_top1"] is None:
        result["error"] = text[-400:]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", action="append", default=[], metavar="a.b=value")
    parser.add_argument("--baseline", action="store_true", help="先测当前配置作对照")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="传给 eval_retrieval 的 top_k。测查询扩展（PRF）必须用 100，因为它在 top_k<30 时不触发",
    )
    args = parser.parse_args()

    import hashlib

    baseline_bytes = CONFIG.read_bytes()
    origin = json.loads(baseline_bytes)
    baseline_sha = hashlib.sha256(baseline_bytes).hexdigest()[:12]
    runs: list[tuple[str, dict]] = []

    if args.baseline:
        print("[baseline] 当前配置", flush=True)
        runs.append((f"baseline {baseline_sha}", run_eval(args.top_k)))

    try:
        for spec in args.set:
            data = json.loads(json.dumps(origin))  # deep copy
            apply_override(data, spec)
            write_config(data)
            sha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()[:12]
            print(f"[sweep] {spec}  ({sha})", flush=True)
            runs.append((f"{spec}  {sha}", run_eval(args.top_k)))
            write_config(origin)  # 每个变体之后都还原，避免叠加
    finally:
        # 无论正常结束、异常还是被 Ctrl-C/中断，都要把配置还原
        write_config(origin)
        sha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()[:12]
        print(f"[restore] 已还原配置 ({sha})", flush=True)

    print()
    header = f"{'配置':<38}{'txt1':>5}{'txt3':>6}{'txtMRR':>9}{'ids1':>6}{'ids3':>6}{'idsMRR':>9}{'half':>10}"
    print(header)
    print("-" * len(header))
    for label, row in runs:
        half = f"{row.get('half1_txt')}/{row.get('half2_txt')}"
        print(
            f"{label:<38}{row.get('txt_top1'):>5}{row.get('txt_top3'):>6}"
            f"{row.get('txt_mrr'):>9}{row.get('ids_top1'):>6}{row.get('ids_top3'):>6}"
            f"{row.get('ids_mrr'):>9}{half:>10}"
        )
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        for label, row in runs:
            handle.write(
                json.dumps(
                    {"at": datetime.now(UTC).isoformat(), "label": label, **row},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"\n已追加到 {LEDGER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
