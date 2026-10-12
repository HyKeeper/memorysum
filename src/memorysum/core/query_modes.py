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
