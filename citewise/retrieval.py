import time
import uuid
from dataclasses import dataclass

from sqlalchemy import select

from citewise.config import get_settings
from citewise.db import session_scope
from citewise.metrics import RETRIEVAL_SECONDS
from citewise.models import Chunk, Document, DocumentStatus
from citewise.vectorstore import get_vector_store


@dataclass
class Passage:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int | None
    heading: str | None
    text: str
    score: float

    @property
    def title(self) -> str:
        return f"{self.filename} (page {self.page_number})" if self.page_number else self.filename

    def public(self) -> dict:
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "filename": self.filename,
            "page_number": self.page_number,
            "heading": self.heading,
            "excerpt": self.text[:280],
        }


async def retrieve(
    query: str,
    user_id: uuid.UUID,
    document_ids: list[uuid.UUID] | None = None,
    limit: int | None = None,
) -> list[Passage]:
    settings = get_settings()
    limit = limit or settings.context_parent_limit
    started = time.perf_counter()

    hits = await get_vector_store().search(query, user_id, document_ids, settings.retrieval_top_k)

    # Collapse child hits to their parents, keeping the best fused rank per parent.
    ranked: dict[uuid.UUID, float] = {}
    for hit in hits:
        ranked.setdefault(hit.parent_id, hit.score)
    if not ranked:
        RETRIEVAL_SECONDS.observe(time.perf_counter() - started)
        return []

    # Authorization is re-checked against the source of truth: a vector payload alone is
    # never trusted to decide what a user may read.
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(Chunk, Document)
                .join(Document, Chunk.document_id == Document.id)
                .where(
                    Chunk.id.in_(list(ranked)),
                    Document.user_id == user_id,
                    Document.status == DocumentStatus.READY,
                )
            )
        ).all()

    passages = [
        Passage(
            chunk_id=chunk.id,
            document_id=doc.id,
            filename=doc.filename,
            page_number=chunk.page_number,
            heading=chunk.heading,
            text=chunk.content,
            score=ranked[chunk.id],
        )
        for chunk, doc in rows
    ]
    passages.sort(key=lambda p: p.score, reverse=True)
    RETRIEVAL_SECONDS.observe(time.perf_counter() - started)
    return passages[:limit]
