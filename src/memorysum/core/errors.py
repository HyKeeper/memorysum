"""Service-level errors translated into stable HTTP responses."""

from __future__ import annotations


class ServiceError(Exception):
    """An expected service failure safe to expose to the API caller."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 503,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retry_after = retry_after


class ProviderError(ServiceError):
    """A model-provider failure; never includes request text or credentials."""


class ScopeRetired(ServiceError):
    def __init__(self) -> None:
        super().__init__(
            "scope_retired",
            "This memory scope is no longer available.",
            status_code=403,
        )


class RequestConflict(ServiceError):
    def __init__(self) -> None:
        super().__init__(
            "request_id_payload_conflict",
            "request_id was already used with a different payload.",
            status_code=422,
        )


class AddInProgress(ServiceError):
    def __init__(self) -> None:
        super().__init__(
            "add_in_progress",
            "An Add request for this memory scope is still being processed.",
            status_code=503,
            retry_after=5,
        )


class PipelineMismatch(ServiceError):
    def __init__(self) -> None:
        super().__init__(
            "pipeline_mismatch",
            "Stored memory was created by a different pipeline configuration.",
            status_code=503,
        )


class StorageError(ServiceError):
    def __init__(self) -> None:
        super().__init__(
            "storage_unavailable",
            "The memory store is temporarily unavailable.",
            status_code=503,
            retry_after=5,
        )
