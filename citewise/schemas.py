import json
import re
import uuid
from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field, field_validator

from citewise.models import Conversation, Document, Message

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class Credentials(BaseModel):
    # Syntax check only: self-hosted installs often use internal domains (.local, .corp,
    # .test) that strict validators reject as non-deliverable.
    email: str = Field(max_length=320)
    password: str = Field(min_length=8, max_length=256)

    @field_validator("email")
    @classmethod
    def _normalize(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL.match(v):
            raise ValueError("Enter a valid email address")
        return v


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=20, max_length=200)


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    created_at: datetime


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    media_type: str
    size_bytes: int
    status: str
    error: str | None
    page_count: int | None
    chunk_count: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, d: Document) -> "DocumentOut":
        return cls.model_validate(d, from_attributes=True)


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, c: Conversation) -> "ConversationOut":
        return cls.model_validate(c, from_attributes=True)


class SourceOut(BaseModel):
    n: int
    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None
    heading: str | None
    excerpt: str


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    sources: list[SourceOut] = []
    model: str | None
    created_at: datetime

    @classmethod
    def of(cls, m: Message) -> "MessageOut":
        return cls(
            id=m.id,
            role=m.role,
            content=m.content,
            sources=json.loads(m.citations_json) if m.citations_json else [],
            model=m.model,
            created_at=m.created_at,
        )


class AskRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=100)
