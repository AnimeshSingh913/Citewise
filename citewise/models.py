import uuid
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import Column, DateTime, Index, Text, UniqueConstraint
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


def _ts(**kwargs) -> Column:
    return Column(DateTime(timezone=True), **kwargs)


class DocumentStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"
    DELETING = "DELETING"


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class User(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    email: str = Field(index=True, unique=True, max_length=320)
    password_hash: str
    created_at: datetime = Field(default_factory=utcnow, sa_column=_ts(nullable=False))


class RefreshToken(SQLModel, table=True):
    """Opaque refresh tokens, stored hashed. Tokens in one login chain share a family_id,
    so reuse of an already-rotated token revokes the whole chain."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="user.id", ondelete="CASCADE", index=True)
    family_id: uuid.UUID = Field(index=True)
    token_hash: str = Field(index=True, unique=True)
    expires_at: datetime = Field(sa_column=_ts(nullable=False))
    revoked_at: datetime | None = Field(default=None, sa_column=_ts(nullable=True))
    created_at: datetime = Field(default_factory=utcnow, sa_column=_ts(nullable=False))


class Document(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("user_id", "filename", name="uq_document_user_filename"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="user.id", ondelete="CASCADE", index=True)
    filename: str = Field(max_length=255)
    media_type: str = Field(max_length=100)
    size_bytes: int
    sha256: str = Field(max_length=64)
    storage_key: str
    status: str = Field(default=DocumentStatus.PENDING, index=True)
    error: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    page_count: int | None = None
    chunk_count: int = 0
    embedding_model: str | None = None
    created_at: datetime = Field(default_factory=utcnow, sa_column=_ts(nullable=False))
    updated_at: datetime = Field(default_factory=utcnow, sa_column=_ts(nullable=False))


class Chunk(SQLModel, table=True):
    """A parent chunk: the unit handed to Claude as one citable content block."""

    __table_args__ = (Index("ix_chunk_document_order", "document_id", "chunk_index"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    document_id: uuid.UUID = Field(foreign_key="document.id", ondelete="CASCADE", index=True)
    chunk_index: int
    page_number: int | None = None
    heading: str | None = Field(default=None, max_length=500)
    content: str = Field(sa_column=Column(Text, nullable=False))


class Job(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    document_id: uuid.UUID = Field(foreign_key="document.id", ondelete="CASCADE", index=True)
    status: str = Field(default=JobStatus.PENDING, index=True)
    attempts: int = 0
    max_attempts: int = 3
    run_after: datetime = Field(default_factory=utcnow, sa_column=_ts(nullable=False, index=True))
    lease_expires_at: datetime | None = Field(default=None, sa_column=_ts(nullable=True))
    worker_id: str | None = None
    last_error: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(default_factory=utcnow, sa_column=_ts(nullable=False))
    updated_at: datetime = Field(default_factory=utcnow, sa_column=_ts(nullable=False))


class Conversation(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="user.id", ondelete="CASCADE", index=True)
    title: str | None = Field(default=None, max_length=200)
    created_at: datetime = Field(default_factory=utcnow, sa_column=_ts(nullable=False))
    updated_at: datetime = Field(default_factory=utcnow, sa_column=_ts(nullable=False, index=True))


class Message(SQLModel, table=True):
    __table_args__ = (Index("ix_message_conversation_created", "conversation_id", "created_at"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    conversation_id: uuid.UUID = Field(foreign_key="conversation.id", ondelete="CASCADE")
    role: str  # "user" | "assistant"
    content: str = Field(sa_column=Column(Text, nullable=False))
    # JSON-encoded list of citations for assistant messages
    citations_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    model: str | None = None
    created_at: datetime = Field(default_factory=utcnow, sa_column=_ts(nullable=False))
