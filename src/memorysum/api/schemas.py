"""Strict textual Add/Search wire schemas for the AML API."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from memorysum.core.identity import payload_hash
from memorysum.settings import load_pipeline_config

#: Wire limits come from the pipeline config so that the audited algorithm
#: configuration stays the single source of truth.
_CAPACITY = load_pipeline_config().capacity


class StrictRequest(BaseModel):
    # `extra="ignore"`，不是 `forbid`：平台文档说它只读声明的字段，而**我们**对入参也要宽容 ——
    # 实测（2026-10-14）平台会带未声明字段（`session_id`/`filters`/`metadata`/`request_id` …）来调
    # `/search`，`forbid` 会直接 422，而文档明确规定 422 "不自动重试" 且"当前阶段立即失败"。
    # 全天日志里有 **163 次** 422，全部是 `extra_forbidden`。忽略未知字段不会改变语义。
    model_config = ConfigDict(extra="ignore", strict=True)


class AddMessage(StrictRequest):
    role: Literal["user", "assistant"]
    timestamp: StrictInt | None = None
    content: StrictStr

    @model_validator(mode="before")
    @classmethod
    def timestamp_must_be_omitted_or_integer(cls, value: object) -> object:
        if isinstance(value, dict) and "timestamp" in value and value["timestamp"] is None:
            raise ValueError("timestamp must be omitted or an integer Unix-millisecond value")
        return value

    @field_validator("content")
    @classmethod
    def validate_original_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be empty or whitespace-only")
        if "\x00" in value:
            raise ValueError("content must not contain NUL")
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError("content must be valid UTF-8") from exc
        return value


class AddRequest(StrictRequest):
    request_id: StrictStr
    messages: list[AddMessage] = Field(min_length=1, max_length=_CAPACITY.max_add_messages)
    user_id: StrictStr
    session_id: StrictStr

    @field_validator("request_id", "user_id", "session_id")
    @classmethod
    def validate_opaque_identifier(cls, value: str) -> str:
        if value == "":
            raise ValueError("identifier must not be empty")
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError("identifier must be valid UTF-8") from exc
        return value

    def canonical_payload(self) -> dict[str, object]:
        """Return exactly the fields received, preserving missing timestamps."""
        serialized_messages: list[dict[str, object]] = []
        for message in self.messages:
            item: dict[str, object] = {"role": message.role}
            if "timestamp" in message.model_fields_set:
                item["timestamp"] = message.timestamp
            item["content"] = message.content
            serialized_messages.append(item)
        return {
            "request_id": self.request_id,
            "messages": serialized_messages,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }

    def payload_digest(self) -> bytes:
        return payload_hash(self.canonical_payload())


class SearchRequest(StrictRequest):
    query: StrictStr
    options: list[StrictStr] | None = Field(default=None, max_length=_CAPACITY.max_search_options)
    user_id: StrictStr
    top_k: StrictInt = Field(ge=0)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError("query must be valid UTF-8") from exc
        return value

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        for option in value:
            if not option.strip():
                raise ValueError("options must not contain empty strings")
            try:
                option.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise ValueError("options must be valid UTF-8") from exc
        return value

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        if value == "":
            raise ValueError("user_id must not be empty")
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError("user_id must be valid UTF-8") from exc
        return value


class AddResponse(BaseModel):
    success: Literal[True]
    request_id: str
    user_id: str
    session_id: str


class SearchItem(BaseModel):
    id: str
    content: str
    score: float
    created_at: str
    #: Harness identifiers of the originating messages (LoCoMo "D1:3", BEAM
    #: "1:msg7", PersonaMem "m3"), lifted from each chunk's metadata header at Add
    #: time. Empty when the stored message carried no header. The platform resolves
    #: a hit to its gold turn from these ids, so omitting them costs recall even
    #: when retrieval is correct.
    source_ids: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    data: list[SearchItem]
