#!/usr/bin/env python
"""把某一轮专家的评测流量导出成 CSV，便于在别的评测平台上原样重放。

导出内容（`exp/`）：
  · `adds.csv`        一行一条 message：user_id/request_id/session_id/序号/role/timestamp/content
                      （content 是逐字原文，含平台的 META 头、QA 反馈、trajectory 等载荷）
  · `searches.csv`    一行一次检索：seq/timestamp/user_id/query/top_k/options/以及该题的平台判定（若有）
  · `questions.csv`   唯一题面清单：题面/能力（本地推断）/本地判定/失分归因/备注/平台叶子值
  · `platform_leaf_scores.csv`  平台四次跑分的 20 个子能力分值（用于复算聚合口径）
  · `scoring_rubric.csv`        本地判据规则与两个聚合口径的定义
  · `README.md`       格式说明与重放步骤

用法：
    .venv/bin/python scripts/export_exp_csv.py --round backups/accuracy-2026-10-14T1635 \
        --verdicts backups/accuracy-2026-10-14T1528/local_verdicts_strict_with_anchor.jsonl \
        --out exp
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RUBRIC: tuple[tuple[str, str, str, str], ...] = (
    (
        "判据-1",
        "精确值题（年份/日期/姓名/数值）",
        "答案必须原样出现在返回正文，且与 as-of 条件吻合",
        "1.0 / 0.0",
    ),
    (
        "判据-2",
        "时间题（when/what time/多久/顺序）",
        "只有相对时间（yesterday、半个月前）而无可换算的绝对锚点 → 0；有锚点则按能否读出答案判",
        "1.0 / 0.5 / 0.0",
    ),
    ("判据-3", "选择题（MCQ）", "需决定性证据；仅相关片段 → 0.5", "1.0 / 0.5 / 0.0"),
    (
        "判据-4",
        "不依赖记忆的题（建议/推理/长历史综合）",
        "按平台模型能否答对判：短建议题 1.0；长历史综合 0.0（用官方 E1=100 / F1=0 校准）",
        "1.0 / 0.0",
    ),
    (
        "口径-①",
        "证据充分性",
        "41 题按题等权平均 —— 我们交出的证据够不够的上界",
        "分数（偏高 5~8 分）",
    ),
    (
        "口径-②",
        "能力树口径",
        "按子能力等权，观测不到的 5 片（C3/D2/E2/G2/I1）按平台最近一次值代入（跨轮极稳）；平台总分 ≈ 其 20 片叶子均值 ±1.2",
        "分数（与平台可比，实测 ±2）",
    ),
    (
        "归因字段",
        "blame",
        "ok=证据够 / anchor=缺时间锚点 / retrieval=碎片化漏召回 / partial=需推理 / memory=库里没有或灌错 / judge=平台误判 / n/a=不依赖记忆",
        "分类",
    ),
    ("用法提醒", "跨轮对接", "题面顺序每轮会被打乱 → 判定必须按题面文本连接，不能按序号", "—"),
    (
        "用法提醒",
        "A/B 判定",
        "看实例级证据（平台回灌的 QA feedback），不要只看叶子或总分（平台自身两轮就差 ~1 分）",
        "—",
    ),
)


def load_requests(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True, help="轮次目录（含 requests.jsonl）")
    parser.add_argument("--verdicts", help="本地判定 jsonl（i 是该轮题面首次出现的序号）")
    parser.add_argument(
        "--verdicts-order",
        help="判定文件对应的轮次目录（用它的 requests.jsonl 定 i→题面 的映射；"
        "题面顺序每轮会被打乱，必须按题面文本连接）",
    )
    parser.add_argument("--since", help="只导出该 UTC 时间之后的请求（如 2026-10-14T08:30）")
    parser.add_argument("--until", help="只导出该 UTC 时间之前的请求")
    parser.add_argument("--tree", action="append", default=[], help="平台分数表 json（可多次传入）")
    parser.add_argument("--out", default="exp")
    args = parser.parse_args()

    round_dir = Path(args.round)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in load_requests(round_dir / "requests.jsonl")
        if (not args.since or str(row.get("timestamp") or "")[:16] >= args.since)
        and (not args.until or str(row.get("timestamp") or "")[:16] <= args.until)
    ]
    print(f"按时间过滤后 {len(rows)} 条请求" + (f"（since {args.since}）" if args.since else ""))

    # ① 逐条 message → adds.csv
    adds = []
    for row in rows:
        if row.get("kind") != "add" or not row.get("payload"):
            continue
        payload = row["payload"]
        for ordinal, message in enumerate(payload.get("messages", [])):
            adds.append(
                {
                    "timestamp": row.get("timestamp", ""),
                    "user_id": payload.get("user_id", ""),
                    "request_id": payload.get("request_id", ""),
                    "session_id": payload.get("session_id", ""),
                    "ordinal": ordinal,
                    "role": message.get("role", ""),
                    "message_timestamp_ms": message.get("timestamp", ""),
                    "content": message.get("content", ""),
                }
            )
    with (out / "adds.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(adds[0].keys()) if adds else ["timestamp"])
        writer.writeheader()
        writer.writerows(adds)

    # ② 逐次检索 → searches.csv（并挂上平台判定，若有）
    verdicts_by_question: dict[str, dict] = {}
    for line in (round_dir / "platform_signals.jsonl").open(encoding="utf-8"):
        item = json.loads(line)
        if item.get("kind") == "qa_feedback" and item.get("question"):
            verdicts_by_question[" ".join(str(item["question"]).split())] = item

    searches = []
    for row in rows:
        if row.get("kind") != "search" or not row.get("payload"):
            continue
        payload = row["payload"]
        question = " ".join(str(payload.get("query", "")).split())
        verdict = verdicts_by_question.get(question, {})
        searches.append(
            {
                "seq": len(searches) + 1,
                "timestamp": row.get("timestamp", ""),
                "user_id": payload.get("user_id", ""),
                "query": payload.get("query", ""),
                "top_k": payload.get("top_k", ""),
                "options_json": json.dumps(payload.get("options") or [], ensure_ascii=False),
                "qa_id": verdict.get("qa_id", ""),
                "platform_score": verdict.get("score", ""),
                "platform_answer": verdict.get("predicted_answer", ""),
            }
        )
    with (out / "searches.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(searches[0].keys()))
        writer.writeheader()
        writer.writerows(searches)

    # ③ 唯一题面 + 本地判定 → questions.csv
    local: dict[str, dict] = {}
    if args.verdicts:
        order_source = Path(args.verdicts_order or args.round) / "requests.jsonl"
        order = []
        for row in load_requests(order_source):
            if row.get("kind") == "search" and row.get("payload"):
                q = row["payload"]["query"]
                if q not in order:
                    order.append(q)
        raw = [json.loads(line) for line in Path(args.verdicts).open(encoding="utf-8")]
        for index, question in enumerate(order, start=1):
            match = next((r for r in raw if r["i"] == index), None)
            if match:
                local[" ".join(question.split())] = match
    questions = []
    seen: set[str] = set()
    for row in searches:
        key = " ".join(row["query"].split())
        if key in seen:
            continue
        seen.add(key)
        verdict = local.get(key, {})
        questions.append(
            {
                "n": len(questions) + 1,
                "user_id": row["user_id"],
                "question": row["query"],
                "capability_local": verdict.get("cap", ""),
                "verdict_local": verdict.get("v", ""),
                "blame_local": verdict.get("blame", ""),
                "note_local": verdict.get("note", ""),
                "qa_id": row["qa_id"],
                "platform_score": row["platform_score"],
            }
        )
    with (out / "questions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(questions[0].keys()))
        writer.writeheader()
        writer.writerows(questions)

    # ④ 平台各轮叶子分 → platform_leaf_scores.csv
    trees = []
    for path in args.tree or ["backups/accuracy-2026-10-14T1635/platform_tree.json"]:
        if Path(path).exists():
            trees.append(json.loads(Path(path).read_text(encoding="utf-8")))
    if trees:
        labels: list[str] = []
        for tree in trees:
            for run in tree["runs"]:
                labels.append(f"{run['label']}({run['id'][11:16]})")
        leaves: list[str] = []
        for tree in trees:
            for run in tree["runs"]:
                for leaf in run["leaves"]:
                    if leaf not in leaves:
                        leaves.append(leaf)
        with (out / "platform_leaf_scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["leaf", "capability", *labels])
            names = {
                "A": "显式事实召回",
                "B": "关系与多跳推理",
                "C": "时间与事件序列",
                "D": "记忆治理",
                "E": "个性化与关怀",
                "F": "摘要/长历史",
                "G": "上下文学习与执行",
                "H": "认识论安全与隐私",
                "I": "流式记忆",
            }
            for leaf in leaves:
                writer.writerow(
                    [
                        leaf,
                        names.get(leaf[0], ""),
                        *[run["leaves"].get(leaf, "") for tree in trees for run in tree["runs"]],
                    ]
                )
            writer.writerow(
                ["TOTAL", "总分", *[run["total"] for tree in trees for run in tree["runs"]]]
            )

    # ⑤ 判据与口径 → scoring_rubric.csv
    with (out / "scoring_rubric.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["类别", "适用", "规则", "输出"])
        writer.writerows(RUBRIC)

    print(f"导出到 {out}/：")
    for name in (
        "adds.csv",
        "searches.csv",
        "questions.csv",
        "platform_leaf_scores.csv",
        "scoring_rubric.csv",
    ):
        path = out / name
        if path.exists():
            lines = sum(1 for _ in path.open(encoding="utf-8"))
            print(f"  {name:<28}{lines - 1:>5} 行数据")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
