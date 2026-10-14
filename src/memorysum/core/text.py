"""Text normalization helpers shared by Add, retrieval, and rendering."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC
from functools import lru_cache

import jieba
import tiktoken

_TOKEN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[A-Za-z0-9_]+|[^\w\s]", re.UNICODE
)
_ENGLISH_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+", re.UNICODE)


def proxy_tokens(text: str) -> list[re.Match[str]]:
    """Return deterministic approximate embedding-token spans, not provider tokens."""
    return list(_TOKEN_RE.finditer(text))


def english_terms(text: str, *, limit: int = 128) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for match in _ENGLISH_RE.finditer(text.lower()):
        term = match.group(0)
        if term not in seen:
            seen.add(term)
            terms.append(term)
            if len(terms) >= limit:
                break
    return terms


def chinese_terms(text: str, *, limit: int = 128) -> list[str]:
    """Use Jieba terms plus character bigrams for robust Chinese FTS recall."""
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        term = term.strip().lower()
        if term and term not in seen and len(term) > 1:
            seen.add(term)
            terms.append(term)

    for token in jieba.cut(text, HMM=False):
        add(token)
        if len(terms) >= limit:
            return terms[:limit]
    for match in _CJK_RE.finditer(text):
        run = match.group(0)
        for index in range(len(run) - 1):
            add(run[index : index + 2])
            if len(terms) >= limit:
                return terms[:limit]
    return terms


def safe_fts_expression(terms: Iterable[str], *, limit: int = 32) -> str | None:
    """Build a MATCH expression from quoted terms; user syntax is never executed."""
    quoted: list[str] = []
    seen: set[str] = set()
    for raw_term in terms:
        term = raw_term.strip()
        if not term or term in seen:
            continue
        seen.add(term)
        quoted.append('"' + term.replace('"', '""') + '"')
        if len(quoted) >= limit:
            break
    return " OR ".join(quoted) if quoted else None


@lru_cache(maxsize=2)
def _encoding(name: str):
    return tiktoken.get_encoding(name)


def reader_token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count reader-estimated tokens. This is not the platform's exact tokenizer."""
    return len(_encoding(encoding_name).encode(text, disallowed_special=()))


def query_options_text(query: str, options: list[str] | None) -> str:
    if not options:
        return query
    return query + "\n\nOptions:\n" + "\n".join(options)


def utc_timestamp(timestamp_ms: int | None) -> str:
    if timestamp_ms is None:
        return "unknown"
    from datetime import datetime

    try:
        value = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return f"unix-ms:{timestamp_ms}"
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


_WS_RUN = re.compile(r"[ \t\r\f\v]*\n[ \t\r\f\v\n]*")
_TAB_RUN = re.compile(r"\t+")
_SPACE_RUN = re.compile(r" {2,}")
_ESCAPED = ((r"\\\"", '"'), (r"\\/", "/"), (r"\\n", "\n"), (r"\\t", " "))


def collapse_whitespace(text: str, *, unescape: bool = False) -> str:
    """把病态空白（\n×N、\t×N、连续空格）压到最小，可选解掉 JSON 转义。

    实测（2026-10-14，XLM-R /tokenize 与 tiktoken 两种口径）：
      · 连续 \n/\t 省 **0.0%** 真实 token（sentencepiece 把空白运行当单个标记），
        我们自己的 tiktoken 预算口径也只省 1.4%；
      · `\n` 直接接单词**不影响分词**（`code:\n```javascript` 两种情况都是 6 token，
        词项都是 ['code','javascrip']）；
      · 只有 JSON 反斜杠转义重的 trajectory 记录省 7%。
    结论：这是**卫生性**清理，不是分数杠杆；而且返回正文按约定必须保持原文切片，
    所以只在显式开启 `groups.normalize_whitespace` 时才用于响应正文。
    """
    text = _WS_RUN.sub("\n", text)
    text = _TAB_RUN.sub(" ", text)
    text = _SPACE_RUN.sub(" ", text)
    if unescape:
        for pattern, replacement in _ESCAPED:
            text = re.sub(pattern, replacement, text)
    return text
