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


#: Marker between the head and the tail of a clipped query. Retrieval-only input: it never
#: reaches the response body or the stored原文.
CLIP_SEPARATOR = "\n…\n"

#: Characters that end a text unit; a clip prefers to cut right after one of these so the
#: embedded copy never starts or ends mid-word (CJK text has few of them, so the plain
#: character cut is the fallback).
_CLIP_BOUNDARIES = ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", "; ", " ")


def _snap_inward(piece: str, *, from_end: bool) -> str:
    """Cut an already-sized slice to a text-unit boundary (only ever shortens it).

    The boundary search stays inside the slice, so a caller's token budget is never
    exceeded by the snap; the caller re-measures anyway because token counts are not
    linear in characters.
    """
    if not piece:
        return piece
    window = min(200, len(piece))
    probe = piece[:window] if from_end else piece[-window:]
    for boundary in _CLIP_BOUNDARIES:
        index = probe.find(boundary) if from_end else probe.rfind(boundary)
        if index >= 0:
            cut = index + len(boundary)
            piece = piece[cut:] if from_end else piece[: len(piece) - len(probe) + cut]
            break
    return piece.strip()


def clip_middle(text: str, max_proxy_tokens: int, *, head_ratio: float = 0.5) -> str:
    """Trim an over-long text to a proxy-token budget, keeping both its head and its tail.

    Document-grounded questions arrive with the whole document glued to the question
    (measured 13k-186k chars) and the question sits at either end, so keeping one side
    only silently drops it. Both sides are trimmed conservatively: token counts are not
    linear in characters (CJK is ~0.6 tokens/char, URL-heavy English ~1.5), so each side
    is re-measured and shrunk until it fits its share. The unit snapping happens last so
    that neither shrink step can leave a half word or a half sentence in the clip.
    """
    if max_proxy_tokens <= 0 or not text:
        return text
    total = reader_token_count(text)
    if total <= max_proxy_tokens:
        return text
    head_budget = max(1, int(max_proxy_tokens * head_ratio))
    tail_budget = max(1, max_proxy_tokens - head_budget)
    head = text[: int(len(text) * head_budget / total)]
    tail = text[-int(len(text) * tail_budget / total) :]
    while head and reader_token_count(head) > head_budget:
        head = head[: int(len(head) * 0.9)]
    while tail and reader_token_count(tail) > tail_budget:
        tail = tail[-int(len(tail) * 0.9) :]
    head = _snap_inward(head, from_end=False)
    tail = _snap_inward(tail, from_end=True)
    if not head:
        return tail
    if not tail:
        return head
    return head + CLIP_SEPARATOR + tail


def query_options_text(query: str, options: list[str] | None) -> str:
    if not options:
        return query
    return query + "\n\nOptions:\n" + "\n".join(options)


def compose_embedding_input(instruction: str | None, text: str) -> str:
    """Join an optional retrieval instruction with the text that gets embedded.

    Providers that need no instruction (e.g. DashScope ``text-embedding-v4``) leave
    ``query_instruction`` empty; in that case the text must be embedded verbatim instead
    of gaining a leading ``"\n\n"`` from a blind join.
    """
    prefix = (instruction or "").strip()
    if not prefix:
        return text
    return f"{prefix}\n\n{text}"


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
