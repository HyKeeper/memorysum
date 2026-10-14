"""Deterministic query-mode detection from the frozen v1 word lists."""

from __future__ import annotations

import re

_WIDE_ENGLISH = re.compile(
    r"\b(?:all|list|every|enumerate|compare|difference|order|sequence|before|after|latest|current|changed|changes|history|steps)\b",
    re.IGNORECASE,
)
_WIDE_CHINESE = (
    "所有",
    "全部",
    "列出",
    "有哪些",
    "分别",
    "比较",
    "区别",
    "顺序",
    "先后",
    "之前",
    "之后",
    "最新",
    "现在",
    "变化",
    "历史",
    "步骤",
)
_CONTEXT_ENGLISH = re.compile(
    r"\b(?:according to|given|above|provided context|this document|these rules|tables?|examples?|procedures?)\b",
    re.IGNORECASE,
)
_CONTEXT_CHINESE = ("根据", "上述", "文中", "规则", "表格", "示例", "流程")
_FIRST_PERSON_ENGLISH = re.compile(r"\b(?:my|mine|for me|suit me)\b", re.IGNORECASE)
_PREFERENCE_ENGLISH = re.compile(
    r"\b(?:prefer|preference|favorite|favourite|taste|style|recommend)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_CHINESE = ("我的", "适合我")
_PREFERENCE_CHINESE = ("偏好", "喜欢", "口味", "风格", "推荐")


def is_wide_query(query: str) -> bool:
    return bool(_WIDE_ENGLISH.search(query)) or any(term in query for term in _WIDE_CHINESE)


def is_context_query(query: str) -> bool:
    return bool(_CONTEXT_ENGLISH.search(query)) or any(term in query for term in _CONTEXT_CHINESE)


def is_personal_query(query: str) -> bool:
    first_person = bool(_FIRST_PERSON_ENGLISH.search(query)) or any(
        term in query for term in _FIRST_PERSON_CHINESE
    )
    preference = bool(_PREFERENCE_ENGLISH.search(query)) or any(
        term in query for term in _PREFERENCE_CHINESE
    )
    return first_person and preference


#: 时间/顺序类问题：这些题必须能在正文里读到绝对时间，否则"昨天/半个月前"无法解析，
#: 而平台记录的平台侧证据是：`locomo1` 问失业时间，答题模型答 "Yesterday."，被判 0.0。
_TEMPORAL_ENGLISH = re.compile(
    r"\b(?:when|what (?:year|date|day|time)|which (?:year|date|day)|how long|how many "
    r"(?:minutes|hours|days|weeks|months|years)|order|before|after|earlier|later|as of|"
    r"by when|since|duration|interval)\b",
    re.IGNORECASE,
)
_TEMPORAL_CHINESE = (
    "什么时候",
    "何时",
    "哪一年",
    "哪一天",
    "几号",
    "多久",
    "多长时间",
    "顺序",
    "先后",
    "之前",
    "之后",
    "变化成",
    "从最初",
    "现在是多少",
    "复查",
    "记录的",
    "当时",
)


def is_temporal_query(query: str) -> bool:
    return bool(_TEMPORAL_ENGLISH.search(query)) or any(term in query for term in _TEMPORAL_CHINESE)
