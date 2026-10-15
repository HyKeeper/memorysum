#!/usr/bin/env python
"""本地能力树评分器：把「逐题判定」聚合成平台同形状的能力树，并与平台数字并列。

判定从哪来：`backups/accuracy-2026-10-14/local_verdicts.jsonl`，每行
    {"i": 题号, "cap": "C1", "v": 1.0|0.5|0.0, "blame": "...", "note": "..."}
判定由人或 agent **读我们返回的证据**得出（不需要专家的金答案）：
  1.0 = 正确答案能从返回正文里读出来；
  0.5 = 只能读出部分（常因为缺时间锚点、或需要跨记录判断顺序）；
  0.0 = 读不出（blame=retrieval 是我们漏召回；blame=memory 是库里就没有/灌错）。
所以本分数是「**证据充分性**」的上界，不是平台的端到端准确率；两者之差就是
「答题模型 + 判据」那一环。`--no-echo` 可剔除平台回灌的 QA 反馈记录再算一版。

用法：
    .venv/bin/python scripts/eval_local_score.py [verdicts.jsonl]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

#: 平台两次跑分的子能力百分比（A=13:25 那次 69.05，B=14:27 那次 68.08）与总分。
PLATFORM = {
    "A1": (66.67, 66.67),
    "A2": (100.0, 100.0),
    "B1": (100.0, 100.0),
    "B2": (100.0, 100.0),
    "B4": (100.0, 100.0),
    "C1": (60.00, 53.33),
    "C2": (80.00, 60.00),
    "C3": (100.0, 100.0),
    "D1": (66.67, 66.67),
    "D2": (40.00, 40.00),
    "E1": (100.0, 100.0),
    "E2": (100.0, 100.0),
    "E3": (100.0, 100.0),
    "F1": (0.0, 0.0),
    "G1": (100.0, 100.0),
    "G2": (20.00, 20.00),
    "G4": (50.00, 50.00),
    "H2": (50.00, 50.00),
    "I1": (46.43, 46.43),
    "I2": (50.00, 50.00),
}
PLATFORM_TOTAL = (69.05, 68.08)
#: 平台树的父能力由子能力均值构成（我们按同样规则算，便于对齐）。
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "verdicts", nargs="?", default="backups/accuracy-2026-10-14/local_verdicts.jsonl"
    )
    parser.add_argument("--no-echo", action="store_true", help="剔除平台回灌记录相关的判定")
    parser.add_argument(
        "--tree",
        help="平台分数表 json：用于把本地分换算成能力树口径（缺失的子能力按平台历史值代入）",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.verdicts, encoding="utf-8")]
    if args.no_echo:
        rows = [row for row in rows if "回灌" not in row.get("note", "")]

    per_cap: dict[str, list[float]] = collections.defaultdict(list)
    per_blame: collections.Counter = collections.Counter()
    for row in rows:
        per_cap[row["cap"]].append(float(row["v"]))
        per_blame[row["blame"]] += 1

    print(f"本地判定 {len(rows)} 题（1.0=证据够 / 0.5=部分 / 0.0=读不出）")
    print(f"失分归因分布: {dict(per_blame)}\n")
    print(f"{'能力':<5}{'本地':>8}{'题数':>5}{'平台A':>8}{'平台B':>8}{'本地-平台B':>11}")
    local_scores: dict[str, float] = {}
    for cap, values in sorted(per_cap.items()):
        local = 100.0 * sum(values) / len(values)
        local_scores[cap] = local
        a, b = PLATFORM.get(cap, (float("nan"), float("nan")))
        print(f"{cap:<5}{local:>8.2f}{len(values):>5}{a:>8.2f}{b:>8.2f}{local - b:>+11.2f}")
    missing = [cap for cap in PLATFORM if cap not in per_cap]
    if missing:
        print(f"\n未被这 41 题覆盖的平台子能力: {', '.join(sorted(missing))}")
        print("（说明平台这些子能力的实例来自 41 条 /search 之外，例如按 step 计分的流式跑法）")

    print(f"\n{'父能力':<6}{'本地':>8}{'平台B':>8}   子能力")
    for parent, children in PARENT.items():
        values = [local_scores[c] for c in children if c in local_scores]
        platform = [PLATFORM[c][1] for c in children]
        local_parent = sum(values) / len(values) if values else float("nan")
        platform_parent = sum(platform) / len(platform)
        print(
            f"{parent:<6}{local_parent:>8.2f}{platform_parent:>8.2f}   "
            + " ".join(f"{c}={local_scores.get(c, float('nan')):.0f}" for c in children)
        )

    # ── 两个口径 ────────────────────────────────────────────────────────────────
    # ① 按题等权：衡量"我们交出去的证据够不够"，是平台端到端分的**上界**。
    # ② 按子能力等权（能力树口径）：平台的树就是这样聚合的（实测他们的总分 ≈ 其叶子均值
    #    ±1.2），所以要做预测必须换成这个口径；我们观测不到的 5 个子能力（C3/D2/E2/G2/I1
    #    的实例不经过 /search）用平台最近一次的值代入——它们跨轮极稳（4 轮里 18/20 逐字相同）。
    total = 100.0 * sum(float(row["v"]) for row in rows) / len(rows)
    print(f"\n① 证据充分性（按题等权，41 题）      = {total:.2f}")
    leaf_scores = {leaf: 100.0 * sum(v) / len(v) for leaf, v in per_cap.items()}
    missing = [leaf for leaf in PLATFORM if leaf not in leaf_scores]
    if args.tree:
        tree = (
            json.loads(Path(args.tree).read_text(encoding="utf-8"))
            if Path(args.tree).exists()
            else None
        )
        reference = tree["runs"][-1] if tree else None
        if reference:
            filled = {
                leaf: reference["leaves"][leaf] for leaf in missing if leaf in reference["leaves"]
            }
            merged = {**leaf_scores, **filled}
            tree_score = sum(merged.values()) / len(merged)
            print(
                f"② 能力树口径（{len(leaf_scores)} 片实测 + {len(filled)} 片按平台"
                f"『{reference['label']}』代入）= {tree_score:.2f}"
                f"   平台总分 = {reference['total']:.2f}（差 {tree_score - reference['total']:+.2f}）"
            )
            print(f"   代入的片: {filled}")
            print(
                f"   仅按我们实测的 {len(leaf_scores)} 片等权 = "
                f"{sum(leaf_scores.values()) / len(leaf_scores):.2f}"
            )
    else:
        print(f"② 能力树口径：需要 --tree <平台分数表> 才能把缺失的 {len(missing)} 片补上")
        print(
            f"   仅按我们实测的 {len(leaf_scores)} 片等权 = {sum(leaf_scores.values()) / len(leaf_scores):.2f}"
        )
    print(
        "\n说明：①是上界（只看证据）；②才是能和平台对话的数字。两者之差 = 平台答题/判据层 + 聚合口径差。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
