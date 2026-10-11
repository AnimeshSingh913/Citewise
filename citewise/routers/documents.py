import hashlib
import re
import unicodedata
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from citewise.config import get_settings
from citewise.db import get_session
from citewise.deps import Pagination, current_user, not_found
from citewise.ingestion import delete_document
from citewise.models import Document, DocumentStatus, Job, JobStatus, User, utcnow
from citewise.parsing import UnsupportedFile, detect_media_type
from citewise.schemas import DocumentOut, Page
from citewise.storage import get_storage, make_storage_key
from citewise.worker import notify_new_job

router = APIRouter(prefix="/api/documents", tags=["documents"])


def clean_filename(name: str) -> str:
    name = unicodedata.normalize("NFC", name).replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip().strip(".")
    if not name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if len(name) > 200:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: 200 - len(ext) - 1] + dot + ext) if dot else name[:200]
    return name


async def _unique_filename(session: AsyncSession, user_id: uuid.UUID, filename: str) -> str:
    taken = set((await session.execute(select(Document.filename).where(Document.user_id == user_id))).scalars().all())
    if filename not in taken:
        return filename
    stem, dot, ext = filename.rpartition(".")
    if not dot:
        stem, ext = filename, ""
    n = 2
    while (candidate := f"{stem} ({n}){dot}{ext}") in taken:
        n += 1
    return candidate


async def _owned(session: AsyncSession, user: User, document_id: uuid.UUID) -> Document:
    doc = await session.get(Document, document_id)
    if doc is None or doc.user_id != user.id or doc.status == DocumentStatus.DELETING:
        raise not_found("Document")
    return doc


def _enqueue(session: AsyncSession, document_id: uuid.UUID) -> None:
    session.add(Job(document_id=document_id, max_attempts=get_settings().job_max_attempts))


@router.post("", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
async def upload(
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    filename = clean_filename(file.filename or "")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The file is empty")
    try:
        media_type = detect_media_type(filename, data[:8])
    except UnsupportedFile as e:
        raise HTTPException(status_code=415, detail=str(e)) from e

    sha256 = hashlib.sha256(data).hexdigest()
    duplicate = (
        (
            await session.execute(
                select(Document).where(
                    Document.user_id == user.id, Document.sha256 == sha256, Document.status != DocumentStatus.DELETING
                )
            )
        )
        .scalars()
        .first()
    )
    if duplicate is not None:
        raise HTTPException(
            status_code=409, detail={"message": "You already uploaded this file", "document_id": str(duplicate.id)}
        )

    doc_id = uuid.uuid4()
    key = make_storage_key(user.id, doc_id)
    await get_storage().put(key, data)
    doc = Document(
        id=doc_id,
        user_id=user.id,
        filename=await _unique_filename(session, user.id, filename),
        media_type=media_type,
        size_bytes=len(data),
        sha256=sha256,
        storage_key=key,
    )
    session.add(doc)
    await session.flush()
    _enqueue(session, doc_id)
    await session.commit()
    notify_new_job()
    return DocumentOut.of(doc)


@router.get("", response_model=Page[DocumentOut])
async def list_documents(
    page: Pagination = Depends(),
    status_filter: DocumentStatus | None = Query(None, alias="status"),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    conditions = [Document.user_id == user.id, Document.status != DocumentStatus.DELETING]
    if status_filter:
        conditions.append(Document.status == status_filter)
    total = (await session.execute(select(func.count()).select_from(Document).where(*conditions))).scalar_one()
    rows = (
        (
            await session.execute(
                select(Document)
                .where(*conditions)
                .order_by(Document.created_at.desc(), Document.id)
                .limit(page.limit)
                .offset(page.offset)
            )
        )
        .scalars()
        .all()
    )
    return Page(items=[DocumentOut.of(d) for d in rows], total=total, limit=page.limit, offset=page.offset)


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: uuid.UUID, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    return DocumentOut.of(await _owned(session, user, document_id))


@router.get("/{document_id}/file")
async def download(
    document_id: uuid.UUID, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    doc = await _owned(session, user, document_id)
    data = await get_storage().get(doc.storage_key)
    return Response(
        content=data,
        media_type=doc.media_type,
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{_rfc5987(doc.filename)}",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "sandbox",
        },
    )


def _rfc5987(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


@router.post("/{document_id}/reprocess", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
async def reprocess(
    document_id: uuid.UUID, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    """Re-run ingestion from the stored original: retries a FAILED document, or rebuilds a
    READY one (e.g. after changing the embedding model or chunking settings)."""
    doc = await _owned(session, user, document_id)
    if doc.status not in (DocumentStatus.FAILED, DocumentStatus.READY):
        raise HTTPException(status_code=409, detail=f"Document is {doc.status.lower()}; wait for it to finish")
    active = (
        await session.execute(
            select(func.count())
            .select_from(Job)
            .where(Job.document_id == doc.id, Job.status.in_([JobStatus.PENDING, JobStatus.RUNNING]))
        )
    ).scalar_one()
    if active:
        raise HTTPException(status_code=409, detail="This document is already queued")
    doc.status = DocumentStatus.PENDING
    doc.error = None
    doc.updated_at = utcnow()
    _enqueue(session, doc.id)
    await session.commit()
    notify_new_job()
    return DocumentOut.of(doc)


@router.delete("/{document_id}", status_code=status.HTTP_202_ACCEPTED)
async def delete(
    document_id: uuid.UUID,
    background: BackgroundTasks,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    doc = await _owned(session, user, document_id)
    # Mark first so the document disappears from lists and search immediately; the saga
    # runs after the response and is resumed on restart if interrupted.
    doc.status = DocumentStatus.DELETING
    doc.updated_at = utcnow()
    await session.commit()
    background.add_task(delete_document, document_id)
    return {"id": str(document_id), "status": DocumentStatus.DELETING}
