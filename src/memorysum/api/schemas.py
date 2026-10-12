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


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


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
    messages: list[AddMessage] = Field(min_length=1, max_length=1024)
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
    options: list[StrictStr] | None = Field(default=None, max_length=32)
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


class SearchResponse(BaseModel):
    data: list[SearchItem]
