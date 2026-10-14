#!/usr/bin/env python
"""本地能力树评分器：把「逐题判定」聚合成平台同形状的能力树，并与平台数字并列。

判定从哪来：`backups/<轮次>/local_verdicts_strict.jsonl`，每行
    {"i": 题号, "cap": "C1", "v": 1.0|0.5|0.0, "blame": "...", "note": "..."}
判定由人或 agent **读我们返回的证据**得出（不需要专家的金答案）：
  1.0 = 正确答案能从返回正文里读出来；
  0.5 = 只能读出部分（常因为缺时间锚点、或需要跨记录判断顺序）；
  0.0 = 读不出（blame=retrieval 是我们漏召回；blame=memory 是库里就没有/灌错）。
所以本分数是「**证据充分性**」的上界，不是平台的端到端准确率；两者之差就是
「答题模型 + 判据」那一环。`--no-echo` 可剔除平台回灌的 QA 反馈记录再算一版。

平台对照从哪来：`backups/platform_tree.json`（`scripts/fetch_platform_results.py
--save-tree` 生成的全部轮次分数树），默认用**最新一轮**；`--run <label>` 可换轮次
（label 是北京时间 HH:MM）。**平台分数不再写死在脚本里**——每跑完一轮平台分，
先 `--save-tree` 更新树，再跑本脚本即可看到最新基准。

用法：
    .venv/bin/python scripts/eval_local_score.py [verdicts.jsonl] [--run 23:00] [--no-echo]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

DEFAULT_TREE = Path("backups/platform_tree.json")

#: 平台树的父能力 → 子能力（我们按同样规则算，便于对齐）。
PARENT = {
    "A": ("A1", "A2"),
    "B": ("B1", "B2", "B4"),
    "C": ("C1", "C2", "C3"),
    "D": ("D1", "D2"),
    "E": ("E1", "E2", "E3"),
    "F": ("F1",),
    "G": ("G1", "G2", "G4"),
    "H": ("H2",),
    "I": ("I1", "I2"),
}


def load_run(path: Path, label: str | None) -> tuple[dict, dict | None]:
    """读平台树，返回（选定轮, 上一轮或 None）。"""
    tree = json.loads(path.read_text(encoding="utf-8"))
    runs = tree["runs"]
    if label:
        run = next((row for row in runs if row["label"] == label), None)
        if run is None:
            labels = ", ".join(row["label"] for row in runs)
            raise SystemExit(f"树里没有 label={label} 的轮次（有：{labels}）")
    else:
        run = runs[-1]
    index = runs.index(run)
    return run, (runs[index - 1] if index > 0 else None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "verdicts", nargs="?", default="backups/accuracy-2026-10-14/local_verdicts_strict.jsonl"
    )
    parser.add_argument("--no-echo", action="store_true", help="剔除平台回灌记录相关的判定")
    parser.add_argument(
        "--tree",
        default=str(DEFAULT_TREE),
        help="平台分数树 json（默认 backups/platform_tree.json，取最新一轮）",
    )
    parser.add_argument("--run", help="指定平台轮次 label（北京时间 HH:MM；默认最新一轮）")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.verdicts, encoding="utf-8")]
    if args.no_echo:
        rows = [row for row in rows if "回灌" not in row.get("note", "")]

    per_cap: dict[str, list[float]] = collections.defaultdict(list)
    per_blame: collections.Counter = collections.Counter()
    for row in rows:
        per_cap[row["cap"]].append(float(row["v"]))
        per_blame[row["blame"]] += 1

    platform_run: dict | None = None
    previous_run: dict | None = None
    tree_path = Path(args.tree)
    if tree_path.exists():
        platform_run, previous_run = load_run(tree_path, args.run)
    else:
        print(f"（找不到平台树 {tree_path}，只打印本地分数；")
        print(
            "  更新方式：.venv/bin/python scripts/fetch_platform_results.py --save-tree backups/platform_tree.json）\n"
        )

    print(f"本地判定 {len(rows)} 题（1.0=证据够 / 0.5=部分 / 0.0=读不出）")
    print(f"失分归因分布: {dict(per_blame)}\n")

    local_scores: dict[str, float] = {}
    if platform_run:
        header = f"{'能力':<5}{'本地':>8}{'题数':>5}{'平台':>8}{'偏差':>9}"
        if previous_run:
            header += f"{'上轮':>8}{'轮间Δ':>8}"
        print(
            f"平台对照 = {platform_run['label']}（{platform_run.get('id', '-')}，"
            f"总分 {platform_run['total']}）"
        )
        print(header)
    else:
        print(f"{'能力':<5}{'本地':>8}{'题数':>5}")
    for cap, values in sorted(per_cap.items()):
        local = 100.0 * sum(values) / len(values)
        local_scores[cap] = local
        line = f"{cap:<5}{local:>8.2f}{len(values):>5}"
        if platform_run:
            actual = platform_run["leaves"].get(cap)
            if actual is None:
                line += f"{'-':>8}{'-':>9}"
            else:
                line += f"{actual:>8.2f}{local - actual:>+9.2f}"
            if previous_run and cap in platform_run["leaves"] and cap in previous_run["leaves"]:
                before = previous_run["leaves"][cap]
                line += f"{before:>8.2f}{platform_run['leaves'][cap] - before:>+8.2f}"
        print(line)

    if platform_run:
        missing = [leaf for leaf in platform_run["leaves"] if len(leaf) > 1 and leaf not in per_cap]
        leaf_names = [leaf for leaf in platform_run["leaves"] if len(leaf) > 1]
        covered = [leaf for leaf in leaf_names if leaf in per_cap]
        print(
            f"\n本批判定覆盖 {len(covered)}/{len(leaf_names)} 片叶子；未覆盖: "
            f"{', '.join(sorted(missing))}"
        )
        print("（未覆盖的叶子通常是按 step 计分的流式跑法，或题面家族没进 /search 的实例）")
    else:
        missing = []

    print(f"\n{'父能力':<6}{'本地':>8}{'平台':>8}   子能力")
    for parent, children in PARENT.items():
        values = [local_scores[c] for c in children if c in local_scores]
        platform_parent = None
        if platform_run:
            # 优先读平台自己的父能力值（I 在平台叫 streaming，树生成时已归一化为 I）；
            # 平台没有的（F）用子能力均值。
            platform_parent = platform_run["leaves"].get(parent)
            if platform_parent is None:
                parts = [platform_run["leaves"][c] for c in children if c in platform_run["leaves"]]
                platform_parent = sum(parts) / len(parts) if parts else None
        if not values and platform_parent is None:
            continue
        local_parent = sum(values) / len(values) if values else float("nan")
        shown = f"{platform_parent:>8.2f}" if platform_parent is not None else f"{'-':>8}"
        print(
            f"{parent:<6}{local_parent:>8.2f}{shown}   "
            + " ".join(f"{c}={local_scores.get(c, float('nan')):.0f}" for c in children)
        )

    # ── 两个口径 ────────────────────────────────────────────────────────────────
    # ① 按题等权：衡量"我们交出去的证据够不够"，是平台端到端分的**上界**。
    # ② 按子能力等权（能力树口径）：平台的树就是这样聚合的（实测他们的总分 ≈ 其叶子均值
    #    ±1.2），所以要做预测必须换成这个口径；我们观测不到的叶子（实例不经过 /search。
    #    的 5 片）用平台该轮的值代入——它们跨轮极稳（多轮里 17/20 逐字相同）。
    total = 100.0 * sum(float(row["v"]) for row in rows) / len(rows)
    print(f"\n① 证据充分性（按题等权，{len(rows)} 题）      = {total:.2f}")
    leaf_scores = {leaf: 100.0 * sum(v) / len(v) for leaf, v in per_cap.items()}
    if platform_run:
        filled = {
            leaf: platform_run["leaves"][leaf] for leaf in missing if leaf in platform_run["leaves"]
        }
        merged = {**leaf_scores, **filled}
        # 口径只数 20 个子能力叶子（树里还有 A-I 父能力与 I/streaming，那是展示层）。
        leaf_names = [name for name in merged if len(name) > 1]
        tree_score = sum(merged[name] for name in leaf_names) / len(leaf_names)
        print(
            f"② 能力树口径（{len(leaf_scores)} 片实测 + {len(filled)} 片按平台"
            f"『{platform_run['label']}』代入）= {tree_score:.2f}"
            f"   平台总分 = {platform_run['total']:.2f}（差 {tree_score - platform_run['total']:+.2f}）"
        )
        print(f"   代入的片: {filled}")
        print(
            f"   仅按我们实测的 {len(leaf_scores)} 片等权 = "
            f"{sum(leaf_scores.values()) / len(leaf_scores):.2f}"
        )
    else:
        print(
            f"② 能力树口径：需要平台树才能把缺失的 {len(missing)} 片补上；"
            f"仅按我们实测的 {len(leaf_scores)} 片等权 = "
            f"{sum(leaf_scores.values()) / len(leaf_scores):.2f}"
        )
    print(
        "\n说明：①是上界（只看证据）；②才是能和平台对话的数字。两者之差 = 平台答题/判据层 + 聚合口径差。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
