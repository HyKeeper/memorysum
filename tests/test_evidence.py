"""core/evidence.py 的严格限制：每条都必须"失败即原样返回"。"""

from __future__ import annotations

import json
from unittest.mock import patch

from memorysum.core.evidence import PREFIX, extract_citation, purify
from memorysum.settings import EvidencePurifyConfig

CONTENT = (
    "Session 2 (1:14 pm on 25 May, 2023)\n"
    "Melanie: I ran a charity race for mental health last Saturday - it was really rewarding.\n"
    "Caroline: That's amazing!"
)
QUESTION = "When did Melanie run a charity race?"


def _cfg(**kw) -> EvidencePurifyConfig:
    base = {"enabled": True, "min_probability": 0.85, "min_chars": 15}
    base.update(kw)
    return EvidencePurifyConfig(**base)


class _Response:
    def __init__(self, body: dict, status: int = 200) -> None:
        self._body = json.dumps(body).encode()
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _body(*citations) -> dict:
    return {"answers": {"a": {"citations": list(citations)}}}


def test_disabled_never_calls_service_and_returns_input():
    with patch("urllib.request.urlopen") as call:
        assert purify(CONTENT, QUESTION, _cfg(enabled=False)) == CONTENT
        call.assert_not_called()


def test_verified_span_is_prefixed_and_original_kept():
    span = "I ran a charity race for mental health last Saturday - it was really rewarding."
    with patch(
        "urllib.request.urlopen", return_value=_Response(_body({"text": span, "probability": 0.97}))
    ):
        out = purify(CONTENT, QUESTION, _cfg())
    assert out.startswith(PREFIX)
    assert span in out
    assert out.endswith(CONTENT)  # 原文完整保留
    assert CONTENT in out


def test_hallucinated_span_is_rejected():
    """引用不在原文里 —— 这是防幻觉的关键一环。"""
    with patch(
        "urllib.request.urlopen",
        return_value=_Response(_body({"text": "I ran a marathon in Berlin.", "probability": 0.99})),
    ):
        assert purify(CONTENT, QUESTION, _cfg()) == CONTENT


def test_low_probability_is_rejected():
    span = "I ran a charity race for mental health last Saturday - it was really rewarding."
    with patch(
        "urllib.request.urlopen", return_value=_Response(_body({"text": span, "probability": 0.5}))
    ):
        assert purify(CONTENT, QUESTION, _cfg()) == CONTENT


def test_too_short_span_is_rejected():
    with patch(
        "urllib.request.urlopen", return_value=_Response(_body({"text": "Session", "probability": 0.99}))
    ):
        assert purify(CONTENT, QUESTION, _cfg()) == CONTENT


def test_bare_time_anchor_is_skipped_but_real_span_used():
    """'Session 2 (…)' 已在正文里，前置它没有增量，应继续找下一条。"""
    anchor = "Session 2 (1:14 pm on 25 May, 2023)"
    real = "I ran a charity race for mental health last Saturday"
    with patch(
        "urllib.request.urlopen",
        return_value=_Response(
            _body({"text": anchor, "probability": 0.99}, {"text": real, "probability": 0.9})
        ),
    ):
        out = purify(CONTENT, QUESTION, _cfg())
    assert real in out
    assert not out.startswith(f"{PREFIX}{anchor}")


def test_transport_failure_returns_input():
    from urllib.error import URLError

    with patch("urllib.request.urlopen", side_effect=URLError("down")):
        assert purify(CONTENT, QUESTION, _cfg()) == CONTENT


def test_malformed_body_returns_input():
    with patch("urllib.request.urlopen", return_value=_Response({"unexpected": True})):
        assert purify(CONTENT, QUESTION, _cfg()) == CONTENT
    with patch("urllib.request.urlopen", return_value=_Response({}, status=500)):
        assert extract_citation(CONTENT, QUESTION, _cfg()) is None


def test_non_numeric_probability_is_skipped():
    span = "I ran a charity race for mental health last Saturday"
    with patch(
        "urllib.request.urlopen",
        return_value=_Response(_body({"text": span, "probability": "high"})),
    ):
        assert purify(CONTENT, QUESTION, _cfg()) == CONTENT
