import asyncio
import time
import uuid

import structlog
from sqlalchemy import delete, select, update

from citewise.chunking import chunk_sections
from citewise.config import get_settings
from citewise.db import session_scope
from citewise.metrics import INGESTION_SECONDS
from citewise.models import Chunk, Document, DocumentStatus, utcnow
from citewise.parsing import UnsupportedFile, parse
from citewise.storage import get_storage
from citewise.vectorstore import ChildPoint, get_vector_store

log = structlog.get_logger(__name__)

CHILD_BATCH = 128


class DocumentGone(Exception):
    """The document was deleted while it was being processed."""


async def ingest_document(document_id: uuid.UUID) -> None:
    """Parse, chunk, persist and index one document. Idempotent: always starts by wiping
    whatever a previous attempt left behind, so retries can't leave duplicates."""
    settings = get_settings()
    store = get_vector_store()
    started = time.perf_counter()

    async with session_scope() as session:
        doc = await session.get(Document, document_id)
        if doc is None or doc.status == DocumentStatus.DELETING:
            raise DocumentGone(str(document_id))
        doc.status = DocumentStatus.PROCESSING
        doc.error = None
        doc.updated_at = utcnow()
        user_id, storage_key, media_type = doc.user_id, doc.storage_key, doc.media_type

    await _wipe_index(document_id)

    data = await get_storage().get(storage_key)
    parsed = await asyncio.to_thread(parse, data, media_type)

    pending_chunks: list[Chunk] = []
    pending_points: list[ChildPoint] = []
    count = 0

    async def flush() -> None:
        # Parents are committed before their children are indexed, so a search hit can
        # always be resolved to a row (and orphans are impossible on the DB side).
        if pending_chunks:
            async with session_scope() as session:
                session.add_all(pending_chunks)
        await store.upsert(pending_points)
        pending_chunks.clear()
        pending_points.clear()

    for parent in chunk_sections(
        parsed.sections, settings.parent_chunk_chars, settings.parent_chunk_overlap, settings.child_chunk_chars
    ):
        chunk = Chunk(
            document_id=document_id,
            chunk_index=parent.index,
            page_number=parent.page_number,
            heading=parent.heading,
            content=parent.text,
        )
        pending_chunks.append(chunk)
        count += 1
        pending_points.extend(
            ChildPoint(user_id=user_id, document_id=document_id, parent_id=chunk.id, text=child)
            for child in parent.children
        )
        if len(pending_points) >= CHILD_BATCH:
            await flush()
    await flush()

    if count == 0:
        raise UnsupportedFile("The document contains no extractable text")

    async with session_scope() as session:
        result = await session.execute(
            update(Document)
            .where(Document.id == document_id, Document.status == DocumentStatus.PROCESSING)
            .values(
                status=DocumentStatus.READY,
                chunk_count=count,
                page_count=parsed.page_count,
                embedding_model=store.embedder.name,
                updated_at=utcnow(),
            )
        )
        if result.rowcount == 0:
            raise DocumentGone(str(document_id))

    INGESTION_SECONDS.observe(time.perf_counter() - started)
    log.info("document_ingested", document_id=str(document_id), chunks=count, pages=parsed.page_count)


async def _wipe_index(document_id: uuid.UUID) -> None:
    await get_vector_store().delete_document(document_id)
    async with session_scope() as session:
        await session.execute(delete(Chunk).where(Chunk.document_id == document_id))


async def mark_failed(document_id: uuid.UUID, error: str, retrying: bool) -> None:
    """Record a failure. Partial chunks/vectors are removed so a FAILED document is clean."""
    try:
        await _wipe_index(document_id)
    except Exception:  # cleanup is best effort; the next attempt wipes again anyway
        log.warning("wipe_after_failure_failed", document_id=str(document_id), exc_info=True)
    async with session_scope() as session:
        await session.execute(
            update(Document)
            .where(Document.id == document_id, Document.status != DocumentStatus.DELETING)
            .values(
                status=DocumentStatus.PENDING if retrying else DocumentStatus.FAILED,
                error=error[:2000],
                updated_at=utcnow(),
            )
        )


async def delete_document(document_id: uuid.UUID) -> None:
    """Deletion saga. Every step is idempotent and the DELETING status is durable, so an
    interrupted deletion is resumed by the worker's startup sweep."""
    async with session_scope() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            return
        doc.status = DocumentStatus.DELETING
        doc.updated_at = utcnow()
        storage_key = doc.storage_key

    await get_vector_store().delete_document(document_id)
    await get_storage().delete(storage_key)
    async with session_scope() as session:
        await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        doc = await session.get(Document, document_id)
        if doc is not None:
            await session.delete(doc)
    log.info("document_deleted", document_id=str(document_id))


async def resume_interrupted_deletions() -> int:
    async with session_scope() as session:
        ids = (
            (await session.execute(select(Document.id).where(Document.status == DocumentStatus.DELETING)))
            .scalars()
            .all()
        )
    for doc_id in ids:
        try:
            await delete_document(doc_id)
        except Exception:
            log.error("resume_deletion_failed", document_id=str(doc_id), exc_info=True)
    return len(ids)
