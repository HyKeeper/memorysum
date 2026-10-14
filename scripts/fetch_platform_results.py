#!/usr/bin/env python
"""直接读平台的评测 API（不用等 Add 回灌），拿每轮的权威分数与逐叶子分解。

背景（2026-10-14 发现）：平台的 Evaluation 页面背后是公开的 JSON API，用我们自己的
Leaderboard API Key 就能查：
  POST /access/validate                     -> 会话信息（system_name / 版本 / 配额）
  GET  /api/v1/evaluations?limit=100        -> 全部评测轮次（含 result.capabilities）
  GET  /api/v1/evaluations/<id>             -> 单轮详情（与列表项同构）
  GET  /api/v1/evaluation-tracks            -> 赛道清单（textual/coding/multimodal）
  GET  /api/v1/evaluation-tracks/<t>/contract
  GET  /datasets                            -> smoke/full/light 各自的数据集成员
认证头：X-Leaderboard-Api-Key: <key>（key 放 .env 的 AML_LEADERBOARD_API_KEY）。

这把"平台分数只能从 /add 正文里的 qa_feedback 反推"变成"直接读"——
`result.capabilities` / `result.subcapabilities` 就是能力树的逐叶子得分（20 个叶子：
A1..H2 + streaming.online / streaming.intra_transfer），不需要本地推断映射。

用法：
    .venv/bin/python scripts/fetch_platform_results.py                    # 台账（UTC + 北京时间）
    .venv/bin/python scripts/fetch_platform_results.py --leaves           # 逐叶子明细（全部轮次）
    .venv/bin/python scripts/fetch_platform_results.py --diff            # 相邻轮次的叶子变化
    .venv/bin/python scripts/fetch_platform_results.py --diff <id_a> <id_b>
    .venv/bin/python scripts/fetch_platform_results.py --save backups/platform-api

注意：这是"平台文档声明 + 我们自己取数"的混合口径 —— 取数方式是本脚本，可复现；
文档口径见 docs/platform-api-notes.md（原文摘录，非我们实测）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv 在 dev 依赖里；缺了也能跑（靠环境变量）
    load_dotenv = None  # type: ignore[assignment]

BASE_URL = os.environ.get("AML_BASE_URL", "https://agentmemoryleaderboard.ai")
CST = timezone(timedelta(hours=8))
LEAF_ORDER = [
    "A1",
    "A2",
    "A3",
    "B1",
    "B2",
    "B3",
    "B4",
    "C1",
    "C2",
    "C3",
    "D1",
    "D2",
    "D3",
    "F1",
    "E1",
    "E2",
    "E3",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "H1",
    "H2",
    "streaming.online",
    "streaming.intra_transfer",
]


def load_key() -> str:
    if load_dotenv is not None:
        load_dotenv(".env", override=False)
    key = os.environ.get("AML_LEADERBOARD_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "缺少 AML_LEADERBOARD_API_KEY。请在 .env 里写一行：\n"
            "AML_LEADERBOARD_API_KEY=ldbd_sk_..."
        )
    return key


def api_get(path: str, key: str, *, base: str = BASE_URL, timeout: float = 60.0):
    # Cloudflare 会按浏览器指纹拦截：urllib 默认的 "Python-urllib/3.x" 直接 403
    # （error code 1010）。带一个正常 UA 即可（curl 默认 UA 也能过）。
    request = Request(
        f"{base}{path}",
        headers={
            "X-Leaderboard-Api-Key": key,
            "Content-Type": "application/json",
            "User-Agent": "memorysum-eval/1.0 (+https://github.com/HyKeeper/memorysum)",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")[:400]
        raise SystemExit(f"HTTP {error.code} on {path}: {body}") from None
    except URLError as error:
        raise SystemExit(f"网络失败 {path}: {error.reason}") from None


def fetch_versions(key: str) -> dict:
    return api_get("/access/versions", key)


def fetch_evaluations(key: str) -> list[dict]:
    jobs: list[dict] = []
    offset = 0
    while True:
        page = api_get(f"/api/v1/evaluations?limit=100&offset={offset}", key)
        batch = page.get("evaluations") or []
        jobs.extend(batch)
        next_offset = page.get("next_offset")
        if not batch or next_offset is None:
            break
        offset = int(next_offset)
    return jobs


def _leaf_rows(job: dict) -> dict[str, float]:
    result = job.get("result") or {}
    rows: dict[str, float] = {}
    for item in result.get("subcapabilities") or []:
        rows[str(item.get("id"))] = float(item.get("score") or 0.0)
    for item in result.get("capabilities") or []:
        if item.get("id") == "streaming":
            continue
        rows.setdefault(str(item.get("id")), float(item.get("score") or 0.0))
    return rows


def _stamp(iso: str) -> tuple[str, str]:
    try:
        moment = datetime.fromisoformat(iso).astimezone(CST)
    except (TypeError, ValueError):
        return iso or "-", "-"
    return moment.strftime("%Y-%m-%dT%H:%M:%S%z")[:-2], moment.strftime("%H:%M")


def cmd_table(jobs: list[dict]) -> None:
    ordered = sorted(jobs, key=lambda job: job.get("created_at") or "")
    print(f"{'北京时间':>5}  {'UTC':22s} {'phase':6s} {'status':9s} {'score':>6s}  eval_id")
    for job in ordered:
        cst, hhmm = _stamp(job.get("created_at") or "")
        result = job.get("result") or {}
        score = result.get("score")
        shown = f"{score * 100:.2f}" if isinstance(score, (int, float)) else "-"
        print(
            f"{hhmm:>5}  {job.get('created_at', '-'):22s} "
            f"{job.get('phase', '-'):6s} {job.get('status', '-'):9s} {shown:>6s}  "
            f"{job.get('evaluation_id')}"
        )
    scores = [
        (job.get("result") or {}).get("score")
        for job in ordered
        if isinstance((job.get("result") or {}).get("score"), (int, float))
    ]
    if scores:
        tail = [f"{value * 100:.2f}" for value in scores[-3:]]
        print(f"\n最近 {len(tail)} 轮：{', '.join(tail)}    （共 {len(scores)} 轮有分数）")


def cmd_leaves(jobs: list[dict], *, only: str | None = None) -> None:
    ordered = sorted(jobs, key=lambda job: job.get("created_at") or "")
    if only:
        ordered = [job for job in ordered if job.get("evaluation_id") == only]
    if not ordered:
        print("没有匹配的轮次")
        return
    leaves = [_leaf_rows(job) for job in ordered]
    names = [name for name in LEAF_ORDER if any(name in row for row in leaves)]
    for row in leaves:  # 补上能力树里出现但我们没列到的（防止平台加叶子）
        for key in row:
            if key not in names:
                names.append(key)
    header = "leaf".ljust(24) + "".join(
        f"{_stamp(job.get('created_at') or '')[1]:>7s}" for job in ordered
    )
    print(header)
    print("-" * len(header))
    for name in names:
        cells = "".join(
            f"{row[name] * 100:>7.2f}" if name in row else f"{'-':>7s}" for row in leaves
        )
        print(name.ljust(24) + cells)
    print(
        "\n列 = 轮次（北京时间 HH:MM）。分数是平台的逐叶子百分比；"
        "'-' 表示那一轮该叶子没有题（或没上报）。"
    )


def cmd_diff(jobs: list[dict], pair: list[str] | None = None) -> None:
    ordered = sorted(jobs, key=lambda job: job.get("created_at") or "")
    if pair:
        picked = [job for job in ordered if job.get("evaluation_id") in set(pair)]
        picked.sort(key=lambda job: job.get("created_at") or "")
    else:
        picked = ordered[-2:]
    if len(picked) != 2:
        print(f"需要两轮才能比较（当前匹配 {len(picked)} 轮）")
        return
    older, newer = picked
    before, after = _leaf_rows(older), _leaf_rows(newer)
    names = [name for name in LEAF_ORDER if name in before or name in after]
    for key in list(before) + list(after):
        if key not in names:
            names.append(key)

    def score(job: dict) -> float:
        return float((job.get("result") or {}).get("score") or 0.0) * 100

    print(
        f"旧：{_stamp(older.get('created_at') or '')[1]} {older.get('evaluation_id')} "
        f"总分 {score(older):.2f}"
    )
    print(
        f"新：{_stamp(newer.get('created_at') or '')[1]} {newer.get('evaluation_id')} "
        f"总分 {score(newer):.2f}  (Δ {score(newer) - score(older):+.2f})"
    )
    print()
    print(f"{'leaf':24s}{'旧':>8s}{'新':>8s}{'Δ':>8s}")
    changed = 0
    for name in names:
        old = before.get(name, 0.0) * 100
        new = after.get(name, 0.0) * 100
        delta = new - old
        mark = "" if abs(delta) < 1e-9 else ("  <-- 变了" if abs(delta) >= 0.005 else "")
        if abs(delta) >= 0.005:
            changed += 1
        print(f"{name:24s}{old:>8.2f}{new:>8.2f}{delta:>+8.2f}{mark}")
    print(f"\n变动叶子数：{changed}（0 = 两轮在这一层完全一致）")


#: 平台 subcapability id → 本地判定口径 (本地把 streaming.* 叫 I1/I2，见 AGENTS.md)。
PLATFORM_TO_LOCAL_LEAF = {
    "streaming.online": "I1",
    "streaming.intra_transfer": "I2",
    "streaming": "I",
}


def _local_leaf_rows(job: dict) -> dict[str, float]:
    """与 `_leaf_rows` 相同，但把平台叫法映射成本地判定口径的名字。

    平台的 20 个叶子里有两片叫 `streaming.online` / `streaming.intra_transfer`，
    本地逐题判定与旧工具一直用 `I1` / `I2`；树文件是给本地对齐工具读的，
    所以在这里统一，避免每次读的时候各自记一套映射。`_leaf_rows` 会跳过
    `capabilities` 里的 `streaming` 聚合项（它是 I1/I2 的父，不是叶子），
    但对齐表需要它的 62.5 作为 I 的值，所以这里单独补回来。
    """
    rows = {
        PLATFORM_TO_LOCAL_LEAF.get(name, name): value for name, value in _leaf_rows(job).items()
    }
    for item in (job.get("result") or {}).get("capabilities") or []:
        if item.get("id") == "streaming":
            rows["I"] = float(item.get("score") or 0.0)
    return rows


def cmd_save(jobs: list[dict], key: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(CST).strftime("%Y-%m-%dT%H%M%S%z")[:-2]
    payload = {
        "fetched_at": datetime.now(CST).isoformat(),
        "base_url": BASE_URL,
        "evaluations": jobs,
        "access_versions": fetch_versions(key),
        "datasets": api_get("/datasets", key),
        "tracks": api_get("/api/v1/evaluation-tracks", key),
        "textual_contract": api_get("/api/v1/evaluation-tracks/textual/contract", key),
    }
    dest = directory / f"platform-results-{stamp}.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已保存 {dest}（{len(jobs)} 轮）")
    return None


def cmd_tree(jobs: list[dict], dest: Path) -> None:
    """把全部轮次落成 `eval_local_score.py` / `eval_alignment.py` / `record_score.py`
    能直接读的树文件（`{"runs": [{"label", "total", "leaves"}]}`，分数为 0-100）。

    只保留**真正产出能力树**的轮（失败轮 score=0 且 caps 为空，收进来会把「最新轮」
    污染成 0 分）；分数一律换算成百分比，与本地判定同口径。这样以后每跑一轮平台分，
    一条命令就让本地对齐工具看到最新基准，不需要再手写 JSON 或改脚本常量。
    """
    ordered = sorted(
        [job for job in jobs if _local_leaf_rows(job)],
        key=lambda job: job.get("created_at") or "",
    )
    runs: list[dict] = []
    for job in ordered:
        cst, hhmm = _stamp(job.get("created_at") or "")
        score = (job.get("result") or {}).get("score")
        runs.append(
            {
                "id": job.get("evaluation_id"),
                "label": hhmm,
                "phase": job.get("phase"),
                "created_at": job.get("created_at"),
                "local_time": cst,
                "total": round(float(score) * 100, 2) if isinstance(score, (int, float)) else None,
                "leaves": {
                    leaf: round(value * 100, 2) for leaf, value in _local_leaf_rows(job).items()
                },
            }
        )
    if not runs:
        print("没有可写成树的轮次（都不含能力树）")
        return None
    payload = {
        "note": (
            "平台逐轮分数树：由 scripts/fetch_platform_results.py --save-tree 生成，"
            "eval_local_score.py / eval_alignment.py / record_score.py 读这份文件做对照。"
            "分数为百分比（0-100）；平台叶子名 streaming.online / streaming.intra_transfer "
            "已在生成时映射为本地口径 I1 / I2（streaming→I），保持 20 叶子 + 父能力的原名。"
        ),
        "generated_at": datetime.now(CST).isoformat(),
        "base_url": BASE_URL,
        "runs": runs,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写入 {dest}（{len(runs)} 轮；最新 {runs[-1]['label']} 总分 {runs[-1]['total']:.2f}）")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="读取 AML 评测 API 的历史结果")
    parser.add_argument("--leaves", action="store_true", help="打印逐叶子明细表")
    parser.add_argument("--diff", nargs="*", metavar="EVAL_ID", help="比较两轮（默认最后两轮）")
    parser.add_argument("--only", metavar="EVAL_ID", help="只看某一轮（配合 --leaves）")
    parser.add_argument("--save", metavar="DIR", help="把完整 JSON 快照写到目录")
    parser.add_argument(
        "--save-tree",
        metavar="FILE",
        help="把全部轮次写成本地对齐工具能读的树文件（如 backups/platform_tree.json）",
    )
    parser.add_argument("--json", action="store_true", help="直接打印原始 JSON")
    args = parser.parse_args(argv)

    key = load_key()
    jobs = fetch_evaluations(key)
    if not jobs:
        print("这个 key 名下没有评测记录")
        return 1

    if args.json:
        print(json.dumps(jobs, ensure_ascii=False, indent=1))
        return 0
    if args.save:
        cmd_save(jobs, key, Path(args.save))
        return 0
    if args.save_tree:
        cmd_tree(jobs, Path(args.save_tree))
        return 0
    if args.leaves:
        cmd_leaves(jobs, only=args.only)
        return 0
    if args.diff is not None:
        cmd_diff(jobs, args.diff or None)
        return 0
    cmd_table(jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
