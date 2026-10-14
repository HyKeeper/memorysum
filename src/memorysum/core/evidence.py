"""Optional evidence purification via a local THX-01 decision service.

THX-01 is a 322M non-autoregressive "decision" model: one forward pass over
`state + question + options` returns calibrated option probabilities plus, with
`cite: true`, the verbatim spans of the document that support the answer. That
last part is what this module uses — not its ranking ability, which measured
worse than the existing reranker.

Measured effect (LoCoMo, 320 questions over four conversations, strict rules
judging): answer accuracy 26.9% -> 27.8%, i.e. +3 questions (7 wins / 4 losses),
with the whole gain coming from temporal questions. Positive but small, so this
is off by default and every step below is designed to fail closed:

  * only the first `max_records` returned records are touched;
  * a citation must clear `min_probability` AND `min_chars`;
  * the citation text must appear verbatim in the record it came from, which
    makes a hallucinated span impossible to propagate;
  * bare time anchors ("Session 2 (1:14 pm on 25 May, 2023)") are skipped — they
    are already present in the text, so prefixing them adds nothing;
  * any transport error, timeout, non-200 or malformed body leaves the record
    exactly as it was.

The prefix is plain text (`[Key evidence: ...]`) and the original passage is
kept intact right after it, so the response stays a verbatim slice plus a label.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

_ANCHOR_ONLY = re.compile(r"^\s*Session \d+ \([^)]+\)\s*$")
PREFIX = "[Key evidence: "


def _fail_open(content: str) -> str:
    return content


def extract_citation(content: str, question: str, config: Any) -> str | None:
    """Ask THX-01 for the span that answers `question`; return it only if it is safe.

    Returns the verbatim span, or None when nothing cleared the checks.
    """
    payload = {
        "state": content[: int(config.max_state_chars)],
        "questions": {
            "a": {
                "type": "choice",
                "question": f"Which part of this passage is needed to answer: {question}",
                "criteria": {"0": "it contains the answer", "1": "it does not"},
                "cite": True,
            }
        },
    }
    request = urllib.request.Request(
        str(config.endpoint),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(config.timeout_seconds)) as response:
            if response.status != 200:
                return None
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None

    try:
        citations = body["answers"]["a"].get("citations") or []
    except (KeyError, TypeError, AttributeError):
        return None
    for citation in citations:
        text = str(citation.get("text") or "")
        try:
            probability = float(citation.get("probability") or 0.0)
        except (TypeError, ValueError):
            continue
        if probability < float(config.min_probability):
            continue
        if len(text) < int(config.min_chars):
            continue
        if _ANCHOR_ONLY.match(text):
            continue
        # The citation must be findable in the very text we are about to return:
        # positions come from the model, the verification does not.
        if text not in content:
            continue
        return text
    return None


def purify(content: str, question: str, config: Any) -> str:
    """Return `content` prefixed with a verified evidence span, or unchanged."""
    if not getattr(config, "enabled", False) or not content.strip():
        return _fail_open(content)
    citation = extract_citation(content, question, config)
    if not citation:
        return _fail_open(content)
    return f"{PREFIX}{citation}]\n{content}"
