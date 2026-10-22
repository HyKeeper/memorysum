"""Bounded-retry JSON POST shared by HTTP model providers.

`local_openai` and `dashscope` both need the same failure taxonomy: timeouts,
connection failures, retryable 5xx/429, deterministic "input too large" answers
and a stable `ProviderError` code for every branch. Keeping one implementation
means a new provider cannot silently drift into different retry semantics.

The helpers never log request bodies or credentials; only the provider's own
error message is quoted.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import Any

import httpx

from memorysum.core.errors import ProviderError

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: llama.cpp rejects a prompt that does not fit in one physical batch, or in the slot
#: context, with HTTP 500. The prompt is simply too long for the model - retrying cannot
#: help, and waiting 1 s + 2 s for the same rejection burns the caller's deadline (measured
#: on 2026-10-14: every failed Search stalled for 3027 ms and then returned 503).
#: The pipeline's own budget (`reranker.single_pair_proxy_tokens`) is a *proxy* estimate and
#: deliberately conservative; `reranker -ub` must cover its worst-case real length (see the
#: calibration table in README「服务拓扑」).
CAPACITY_ERROR_MARKERS = (
    "too large to process",
    "larger than the max context size",
    "exceeds the available context size",
)


def error_message(response: httpx.Response, *, limit: int = 200) -> str:
    """Best-effort read of a provider's error text; never the request payload we sent."""
    decoded: object = None
    with contextlib.suppress(ValueError):
        decoded = response.json()
    message = ""
    if isinstance(decoded, dict):
        error = decoded.get("error", decoded)
        if isinstance(error, dict):
            message = str(error.get("message") or "")
        elif isinstance(error, str):
            message = error
        if not message:
            message = str(decoded.get("message") or "")
    if not message:
        with contextlib.suppress(Exception):
            message = response.text
    return " ".join(message.split())[:limit]


def post_json(
    client: httpx.Client,
    path: str,
    payload: dict[str, Any],
    *,
    attempts: int,
    headers: dict[str, str],
    what: str,
) -> dict[str, Any] | list[Any]:
    """POST with bounded retries; shared by HTTP embedding and rerank clients."""
    last_timeout = False

    for attempt in range(attempts):
        try:
            response = client.post(path, json=payload, headers=headers)
        except httpx.TimeoutException:
            last_timeout = True
            if attempt + 1 >= attempts:
                break
            time.sleep(1 if attempt == 0 else 2)
            continue
        except httpx.RequestError as exc:
            if attempt + 1 >= attempts:
                logger.warning("%s connection failure (%s)", what, type(exc).__name__)
                raise ProviderError(
                    "model_provider_unavailable",
                    f"The {what} server is temporarily unavailable.",
                    status_code=503,
                    retry_after=5,
                ) from None
            time.sleep(1 if attempt == 0 else 2)
            continue

        detail = error_message(response) if response.is_error else ""
        if response.is_error and any(marker in detail for marker in CAPACITY_ERROR_MARKERS):
            # Deterministic rejection: fail immediately instead of retrying a prompt that
            # no number of attempts can shrink. The message carries the server's own limit.
            logger.warning("%s server rejected an oversized input: %s", what, detail)
            raise ProviderError(
                "model_provider_input_too_large",
                f"The {what} server rejected an input that is too long for the model.",
                status_code=503,
            )
        if response.status_code in RETRYABLE_STATUS and attempt + 1 < attempts:
            time.sleep(1 if attempt == 0 else 2)
            continue
        if response.is_error:
            status = response.status_code
            logger.warning("%s server returned HTTP %d: %s", what, status, detail)
            if status == 429:
                raise ProviderError(
                    "model_provider_rate_limited",
                    f"The {what} server is rate limiting requests.",
                    status_code=429,
                    retry_after=5,
                )
            if status in {408, 504}:
                raise ProviderError(
                    "model_provider_timeout",
                    f"The {what} server request timed out.",
                    status_code=504,
                )
            if status in {401, 403}:
                raise ProviderError(
                    "model_provider_unauthorized",
                    f"The {what} provider rejected the configured credentials.",
                    status_code=503,
                )
            raise ProviderError(
                "model_provider_error",
                f"The {what} server rejected or failed a request.",
                status_code=503,
                retry_after=5 if status >= 500 else None,
            )
        try:
            decoded = response.json()
        except (ValueError, json.JSONDecodeError):
            raise ProviderError(
                "model_provider_invalid_response",
                f"The {what} server returned an invalid response.",
                status_code=503,
            ) from None
        if not isinstance(decoded, (dict, list)):
            raise ProviderError(
                "model_provider_invalid_response",
                f"The {what} server returned an invalid response.",
                status_code=503,
            )
        return decoded

    if last_timeout:
        raise ProviderError(
            "model_provider_timeout",
            f"The {what} server request timed out.",
            status_code=504,
        )
    raise ProviderError(
        "model_provider_unavailable",
        f"The {what} server is temporarily unavailable.",
        status_code=503,
        retry_after=5,
    )
