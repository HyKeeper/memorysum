#!/usr/bin/env python
"""机械判据：把能自动判的题先判掉，只把歧义题留给人读。

动机：本地评分要与平台收敛，前提是**跨轮用同一把尺**。人工逐轮重读 41 题会漂移，
所以把已在上一轮验证过的规则固化下来：

R1 时间题无绝对锚点 → 0.0
   问题问时间/顺序，而返回正文里只有 "yesterday / 半个月前" 这类相对表达、
   一个绝对日期都没有（官方 locomo1 给 0.0，与此规则一致）。
R2 选择题（MCQ）→ 由选项词覆盖决定
   从题面解析 A/B/C... 选项，统计每个选项的独特词在返回正文里的覆盖；
   只有一个选项覆盖显著 → 1.0；都不覆盖 → 0.0；多个并列 → 0.5（留给人判）。
R3 UPDATE 覆盖题 → 标注（不直接判分）
   返回正文含 "UPDATE: replace the prior value" 时，答案应以最后一条为准。

用法：
    .venv/bin/python scripts/eval_mechanical_rules.py [requests.jsonl] [--compare verdicts.jsonl]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env", override=False)

from eval_capability_audit import capability_of  # noqa: E402
from memorysum.api.schemas import SearchRequest  # noqa: E402
from memorysum.providers.factory import build_provider, embedding_endpoint  # noqa: E402
from memorysum.services.search import SearchService  # noqa: E402
from memorysum.settings import RuntimeSettings, load_pipeline_config  # noqa: E402
from memorysum.storage.database import DatabaseManager  # noqa: E402

TEMPORAL_RE = re.compile(
    r"\bwhen\b|\bwhat year\b|\bwhich year\b|\bwhat date\b|\bhow long\b|\bhow many (?:minutes|days|"
    r"months|years)\b|\border\b|\bbefore\b|\bafter\b|什么时候|哪一年|多久|多长时间|顺序|先后|"
    r"变化成|几分钟|几天|几周",
    re.IGNORECASE,
)
RELATIVE_RE = re.compile(
    r"\byesterday\b|\btoday\b|\btomorrow\b|\blast (?:week|month|year|night)\b|\bnext (?:week|month|"
    r"year)\b|\bthis (?:week|month|year)\b|\brecently\b|\bthe other day\b|昨天|今天|明天|上周|上个月|"
    r"前几天|这几天|最近|半个月|一个月|前几天|那天",
    re.IGNORECASE,
)
ABSOLUTE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\d{4}-\d{2}-\d{2}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"[a-z]*\.? \d{1,2}|\d{1,2}月\d{1,2}日|Session \d+ \(|\[\w+_META[^\]]*chunk_id=[^\s\]]*session\d+",
    re.IGNORECASE,
)
OPTION_RE = re.compile(r"(?:^|\s)([A-H])[.、)]\s*([^A-H]{3,400}?)(?=(?:\s[A-H][.、)])|$)")

STOPWORDS = frozenset(
    """the and for with that this from have has are was were will would your you our their his her
    its not but can could should about into over more most only other some such than then there
    these they what when which where who why how does did doing during under until very also been
    being both each few just like made make many much must same because before after again
    """.split()
)


def evidence_text(service, question: str, user_id: str, limit: int = 10) -> str:
    response = service.search(SearchRequest(query=question, user_id=user_id, top_k=limit))
    return " ".join(" ".join(item.content.split()) for item in response.data)


def option_terms(question: str) -> dict[str, set[str]]:
    options: dict[str, set[str]] = {}
    for label, text in OPTION_RE.findall(question + " "):
        words = {
            w for w in re.findall(r"[A-Za-z\u4e00-\u9fff]{4,}", text.lower()) if w not in STOPWORDS
        }
        if len(words) >= 2:
            options[label.upper()] = words
    return options


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests", nargs="?", default="backups/accuracy-2026-10-14/requests.jsonl")
    parser.add_argument("--compare", help="与人工判定比对（verdicts.jsonl）")
    args = parser.parse_args()

    questions: dict[str, str] = {}
    for line in open(args.requests, encoding="utf-8"):
        row = json.loads(line)
        if row.get("kind") == "search" and row.get("payload"):
            questions.setdefault(row["payload"]["query"], row["payload"]["user_id"])
    hand: dict[int, float] = {}
    if args.compare:
        for row in (json.loads(line) for line in open(args.compare, encoding="utf-8")):
            hand[int(row["i"])] = float(row["v"])

    runtime = RuntimeSettings.from_environment()
    pipeline = load_pipeline_config()
    space = hashlib.sha256(
        (pipeline.embedding_space_id + "\n" + embedding_endpoint(runtime, pipeline)).encode()
    ).hexdigest()
    service = SearchService(
        DatabaseManager(runtime.data_root, pipeline, space),
        build_provider(runtime, pipeline),
        pipeline,
    )

    print(f"{'#':>3} {'能力':<4}{'机械':>6}{'人工':>6}  规则说明")
    agree = decided = total = 0
    for index, (question, user_id) in enumerate(questions.items(), start=1):
        total += 1
        text = evidence_text(service, question, user_id)
        mechanical: float | None = None
        reason = ""
        if (
            TEMPORAL_RE.search(question)
            and RELATIVE_RE.search(text)
            and not ABSOLUTE_RE.search(text)
        ):
            mechanical, reason = 0.0, "R1 时间题且正文只有相对时间、无绝对锚点"
        else:
            options = option_terms(question)
            if len(options) >= 3:
                covered = {
                    label: sum(1 for word in words if word in text.lower()) / len(words)
                    for label, words in options.items()
                }
                best = max(covered.values())
                winners = [
                    label for label, value in covered.items() if value >= max(0.6, best - 0.1)
                ]
                if best >= 0.6 and len(winners) == 1:
                    mechanical, reason = 1.0, f"R2 仅 {winners[0]} 选项词覆盖 {best:.0%}"
                elif best < 0.2:
                    mechanical, reason = 0.0, f"R2 选项词最高覆盖仅 {best:.0%}"
                else:
                    reason = f"R2 选项覆盖并列/不足（最高 {best:.0%}）→ 留人判"
            elif "UPDATE: replace the prior value" in text:
                reason = "R3 含 UPDATE 覆盖记录 → 以最后一条为准（留人判）"
        if mechanical is not None:
            decided += 1
            if index in hand:
                agree += mechanical == hand[index] or (
                    hand[index] != 0 and mechanical == 0
                )  # 机械规则只判 0/1
        print(
            f"{index:>3} {capability_of(question):<4}"
            f"{('' if mechanical is None else f'{mechanical:.1f}'):>6}"
            f"{(f'{hand[index]:.1f}' if index in hand else ''):>6}  {reason or '无线索 → 留人判'}"
        )
    print(
        f"\n机械可判 {decided}/{total} 题；与人判一致 {agree}/{decided}"
        f"（{100 * agree / max(1, decided):.0f}%）"
    )
    print("下一轮流程：dump → 机械规则先判 → 人只补余下部分 → 聚合 → eval_alignment 记偏差。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
