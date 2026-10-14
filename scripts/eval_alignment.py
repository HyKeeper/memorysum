#!/usr/bin/env python
"""把本地判定与平台能力树对齐：逐叶子偏差表 + 校准后的总分预测。

为什么要它：本地严格分与平台总分的差（本轮 +0.21）是**误差抵消**的结果，不可靠；
平台的子能力分值跨轮极稳（两次跑分 20 个叶子里 18 个完全相同），所以真正该对齐的是
**逐叶子**。本脚本把每轮的逐叶子偏差记进 bias 表，下一轮用「本地叶子 + 历史偏差」
预测平台叶子，再与平台新公布的表比较——残差收敛才算我们的标准真的接近了。

用法：
    .venv/bin/python scripts/eval_alignment.py <verdicts.jsonl> <platform_tree.json> [--run B] [--record]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

BIAS_PATH = Path("backups/platform_alignment.json")


def load_local(path: Path) -> dict[str, list[float]]:
    per_leaf: dict[str, list[float]] = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        per_leaf.setdefault(row["cap"], []).append(float(row["v"]))
    return per_leaf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("verdicts")
    parser.add_argument("platform_tree")
    parser.add_argument("--run", default="B", help="用哪一次平台跑分做对照（默认最后一次）")
    parser.add_argument("--record", action="store_true", help="把本轮偏差写进 bias 表")
    args = parser.parse_args()

    per_leaf = load_local(Path(args.verdicts))
    tree = json.loads(Path(args.platform_tree).read_text(encoding="utf-8"))
    run = next((item for item in tree["runs"] if item["label"] == args.run), tree["runs"][-1])
    platform = run["leaves"]

    history = (
        json.loads(BIAS_PATH.read_text(encoding="utf-8")) if BIAS_PATH.exists() else {"rounds": []}
    )

    local_scores = {leaf: 100.0 * sum(values) / len(values) for leaf, values in per_leaf.items()}
    table = []
    for leaf, local in sorted(local_scores.items()):
        actual = platform.get(leaf)
        if actual is None:
            continue
        table.append((leaf, local, actual, local - actual))

    print(f"对照平台跑分 {run['label']}（{run['id']}，总分 {run['total']}）\n")
    print(f"{'子能力':<6}{'本地':>8}{'平台':>8}{'偏差':>8}   备注")
    for leaf, local, actual, delta in sorted(table, key=lambda row: -abs(row[3])):
        flag = "映射可能错位" if abs(delta) >= 30 else ("" if abs(delta) < 10 else "口径差异")
        print(f"{leaf:<6}{local:>8.2f}{actual:>8.2f}{delta:>+8.2f}   {flag}")

    total = 100.0 * sum(sum(v) for v in per_leaf.values()) / sum(len(v) for v in per_leaf.values())
    uncalibrated = total
    calibrated = total
    if history["rounds"]:
        deltas = [row["total_delta"] for row in history["rounds"]]
        bias = statistics.mean(deltas)
        calibrated = total - bias
        print(
            f"\n历史总偏差（{len(deltas)} 轮）: {[f'{d:+.2f}' for d in deltas]} → 平均 {bias:+.2f}"
        )
    print(f"本地严格总分          = {uncalibrated:.2f}")
    print(f"平台总分              = {run['total']:.2f}")
    print(
        f"校准后预测（本地−历史平均偏差）= {calibrated:.2f}   残差 {calibrated - run['total']:+.2f}"
    )

    if args.record:
        history["rounds"].append(
            {
                "platform_run": run["id"],
                "local_total": round(uncalibrated, 2),
                "platform_total": run["total"],
                "total_delta": round(uncalibrated - run["total"], 2),
                "leaf_deltas": {leaf: round(delta, 2) for leaf, _l, _a, delta in table},
            }
        )
        BIAS_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已记录到 {BIAS_PATH}（共 {len(history['rounds'])} 轮）")
        print("下一轮：用同样的判据跑 verdicts，再 --record，看 total_delta 与各叶子偏差是否收敛。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
