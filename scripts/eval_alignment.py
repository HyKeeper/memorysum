#!/usr/bin/env python
"""把本地判定与平台能力树对齐：逐叶子偏差表 + 校准后的总分预测。

为什么要它：本地严格分与平台总分的差有时是**误差抵消**的结果，不可靠；平台的子能力
分值跨轮极稳（多轮里 17/20 片逐字相同），所以真正该对齐的是**逐叶子**。本脚本把每轮的
逐叶子偏差记进 bias 表，下一轮用「本地叶子 + 历史偏差」预测平台叶子，再与平台新公布的
表比较——残差收敛才算我们的标准真的接近了。

平台树从哪来：`backups/platform_tree.json`（`scripts/fetch_platform_results.py --save-tree`
生成的全部轮次），默认取**最新一轮**；`--run <label>` 指定轮次（label 是北京时间 HH:MM）。
旧格式（`backups/round6/platform_tree.json` 那种单轮树）也能读。

用法：
    .venv/bin/python scripts/eval_alignment.py <verdicts.jsonl> [platform_tree.json]
        [--run 23:00] [--fit-on 18:38] [--record]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DEFAULT_TREE = Path("backups/platform_tree.json")
BIAS_PATH = Path("backups/platform_alignment.json")


def load_local(path: Path) -> dict[str, list[float]]:
    per_leaf: dict[str, list[float]] = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        per_leaf.setdefault(row["cap"], []).append(float(row["v"]))
    return per_leaf


def pick_run(tree: dict, label: str | None) -> dict:
    runs = tree["runs"]
    if label:
        run = next((item for item in runs if item["label"] == label), None)
        if run is None:
            labels = ", ".join(item["label"] for item in runs)
            raise SystemExit(f"树里没有 label={label} 的轮次（有：{labels}）")
        return run
    return runs[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("verdicts")
    parser.add_argument(
        "platform_tree",
        nargs="?",
        default=str(DEFAULT_TREE),
        help=f"平台分数树（默认 {DEFAULT_TREE}，取最新一轮）",
    )
    parser.add_argument("--run", help="指定平台轮次 label（北京时间 HH:MM；默认最新一轮）")
    parser.add_argument("--record", action="store_true", help="把本轮偏差写进 bias 表")
    parser.add_argument(
        "--fit-on",
        help="用指定平台轮次（如 18:38）的逐叶子偏差做校准，再对照 --run 评估（留出验证）",
    )
    args = parser.parse_args()

    per_leaf = load_local(Path(args.verdicts))
    tree = json.loads(Path(args.platform_tree).read_text(encoding="utf-8"))
    run = pick_run(tree, args.run)
    platform = run["leaves"]
    # 只对齐子能力叶子（树里还有 A-I 父能力与 I，属于展示层，不参与分数预测）。
    leaf_names = [name for name in platform if len(name) > 1 and name[0].isalpha()]

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

    print(f"对照平台跑分 {run['label']}（{run.get('id', '-')}，总分 {run['total']}）\n")
    print(f"{'子能力':<6}{'本地':>8}{'平台':>8}{'偏差':>8}   备注")
    for leaf, local, actual, delta in sorted(table, key=lambda row: -abs(row[3])):
        flag = "映射可能错位" if abs(delta) >= 30 else ("" if abs(delta) < 10 else "口径差异")
        print(f"{leaf:<6}{local:>8.2f}{actual:>8.2f}{delta:>+8.2f}   {flag}")

    if args.fit_on:
        fitted = pick_run(tree, args.fit_on)
        print(
            f"\n留出验证：用 {fitted['label']}（{fitted['total']}）的偏差校准，"
            f"预测 {run['label']}（{run['total']}）"
        )
        hits = misses = 0
        for leaf, local in sorted(local_scores.items()):
            if leaf not in platform or leaf not in fitted["leaves"]:
                continue
            predicted = local + (fitted["leaves"][leaf] - local)
            if abs(predicted - platform[leaf]) < 0.01:
                hits += 1
            else:
                misses += 1
                print(
                    f"   {leaf}: 预测 {predicted:.2f} 实际 {platform[leaf]:.2f} "
                    f"（{fitted['label']} 那次是 {fitted['leaves'][leaf]:.2f}）"
                )
        print(f"   叶子命中 {hits}/{hits + misses}（{100 * hits / max(1, hits + misses):.0f}%）")

    total = 100.0 * sum(sum(v) for v in per_leaf.values()) / sum(len(v) for v in per_leaf.values())
    uncalibrated = total
    calibrated = total
    if args.fit_on:
        fitted = pick_run(tree, args.fit_on)
        predicted = uncalibrated + (fitted["total"] - uncalibrated)
        print(
            f"   总分预测 = {predicted:.2f}，实际 = {run['total']:.2f}，"
            f"残差 = {predicted - run['total']:+.2f}（≈ 平台自身两轮的噪声）"
        )
    print(f"本地严格总分（按题等权）  = {uncalibrated:.2f}")
    print(f"平台总分                  = {run['total']:.2f}")
    # 能力树口径：实测叶 + 平台该轮未覆盖叶代入（这些叶子的实例不经过 /search，跨轮极稳）。
    covered = {leaf: local_scores[leaf] for leaf in local_scores if leaf in platform}
    for leaf in leaf_names:
        covered.setdefault(leaf, platform[leaf])
    backfilled = len(leaf_names) - sum(1 for leaf in leaf_names if leaf in local_scores)
    tree_score = sum(covered.values()) / len(covered) if covered else float("nan")
    passed = len(leaf_names) - backfilled
    print(
        f"本地能力树口径（{passed} 片实测 + {backfilled} 片按平台代入）= {tree_score:.2f}"
        f"   残差 {tree_score - run['total']:+.2f}    ← 这才是能和平台对话的数字"
    )
    # 校准只用**同一份判定文件**的历史轮次——判据/证据变了就不能混算平均偏差。
    verdicts_key = str(Path(args.verdicts))
    same = [row for row in history["rounds"] if row.get("verdicts") == verdicts_key]
    same_with_tree = [row for row in same if "tree_delta" in row]
    if not args.fit_on:
        if len(same_with_tree) >= 2:
            deltas = [row["tree_delta"] for row in same_with_tree]
            bias = statistics.mean(deltas)
            calibrated = tree_score - bias
            print(
                f"\n历史 tree 偏差（同判据 {len(deltas)} 轮）: "
                f"{[f'{d:+.2f}' for d in deltas]} → 平均 {bias:+.2f}"
            )
            print(
                f"校准后预测（本地树口径−平均偏差）= {calibrated:.2f} "
                f"  残差 {calibrated - run['total']:+.2f}"
            )
        elif same_with_tree:
            print(
                f"\n同判据只有 1 轮历史（tree 偏差 {same_with_tree[0]['tree_delta']:+.2f}），"
                f"尚无留出轮次可校准；下一轮用同一份判定再 --record 后才能算预测。"
            )
        else:
            print(
                f"\n历史校准：这是判定 {verdicts_key} 的首轮（无同判据历史），"
                f"本轮 tree 残差 {tree_score - run['total']:+.2f} 将作为基准。"
            )

    if args.record:
        history["rounds"].append(
            {
                "platform_run": run["label"],
                "platform_id": run.get("id"),
                "verdicts": verdicts_key,
                "local_total": round(uncalibrated, 2),
                "local_tree_total": round(tree_score, 2),
                "platform_total": run["total"],
                "total_delta": round(uncalibrated - run["total"], 2),
                "tree_delta": round(tree_score - run["total"], 2),
                "leaf_deltas": {leaf: round(delta, 2) for leaf, _l, _a, delta in table},
            }
        )
        BIAS_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已记录到 {BIAS_PATH}（共 {len(history['rounds'])} 轮）")
        print("下一轮：用同样的判据跑 verdicts，再 --record，看 tree_delta 与各叶子偏差是否收敛。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
