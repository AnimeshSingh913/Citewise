import asyncio
import uuid
from datetime import timedelta

from sqlalchemy import select

from citewise import ingestion, worker
from citewise.db import session_scope
from citewise.models import Document, DocumentStatus, Job, JobStatus, User, utcnow
from citewise.storage import get_storage


async def make_job(
    text: bytes = b"hello world, this is a document", max_attempts: int = 3
) -> tuple[uuid.UUID, uuid.UUID]:
    async with session_scope() as session:
        user = User(email=f"{uuid.uuid4().hex}@example.com", password_hash="x")
        session.add(user)
        await session.flush()
        doc = Document(
            user_id=user.id,
            filename=f"{uuid.uuid4().hex}.txt",
            media_type="text/plain",
            size_bytes=len(text),
            sha256=uuid.uuid4().hex,
            storage_key=f"{user.id}/{uuid.uuid4()}",
        )
        session.add(doc)
        await session.flush()
        job = Job(document_id=doc.id, max_attempts=max_attempts)
        session.add(job)
    await get_storage().put(doc.storage_key, text)
    return job.id, doc.id


async def load(model, id_):
    async with session_scope() as session:
        return await session.get(model, id_)


async def test_concurrent_claims_never_double_claim():
    await make_job()
    results = await asyncio.gather(*(worker.claim_job(f"w{i}") for i in range(8)))
    assert sum(r is not None for r in results) == 1


async def test_claims_each_job_once_across_workers():
    ids = {(await make_job())[0] for _ in range(5)}
    claimed = await asyncio.gather(*(worker.claim_job(f"w{i}") for i in range(10)))
    got = [c[0] for c in claimed if c]
    assert sorted(got) == sorted(ids) and len(set(got)) == len(got)


async def test_expired_lease_is_reclaimed():
    job_id, _ = await make_job()
    assert (await worker.claim_job("crashed-worker"))[0] == job_id
    assert await worker.claim_job("other") is None  # lease still valid

    async with session_scope() as session:
        job = await session.get(Job, job_id)
        job.lease_expires_at = utcnow() - timedelta(seconds=1)

    claimed = await worker.claim_job("rescuer")
    assert claimed is not None and claimed[0] == job_id and claimed[2] == 2  # second attempt


async def test_transient_failure_retries_with_backoff_then_fails(monkeypatch):
    job_id, doc_id = await make_job(max_attempts=2)
    calls = 0

    async def flaky(document_id):
        nonlocal calls
        calls += 1
        raise ConnectionError("vector store unavailable")

    monkeypatch.setattr(worker, "ingest_document", flaky)

    assert await worker.run_one("w")
    job = await load(Job, job_id)
    assert job.status == JobStatus.PENDING and job.attempts == 1
    assert (await load(Document, doc_id)).status == DocumentStatus.PENDING
    assert await worker.run_one("w") is False  # backoff: not runnable yet

    async with session_scope() as session:
        (await session.get(Job, job_id)).run_after = utcnow()
    assert await worker.run_one("w")
    job = await load(Job, job_id)
    assert job.status == JobStatus.FAILED and job.attempts == 2 and "vector store" in job.last_error
    assert (await load(Document, doc_id)).status == DocumentStatus.FAILED
    assert calls == 2


async def test_unsupported_file_fails_without_retry():
    job_id, doc_id = await make_job(text=b"   ")
    assert await worker.run_one("w")
    assert (await load(Job, job_id)).status == JobStatus.FAILED
    assert (await load(Job, job_id)).attempts == 1
    assert (await load(Document, doc_id)).status == DocumentStatus.FAILED


async def test_document_deleted_mid_processing_is_cancelled(monkeypatch):
    job_id, doc_id = await make_job()

    async def deleted_during(document_id):
        await ingestion.delete_document(document_id)
        raise ingestion.DocumentGone(str(document_id))

    monkeypatch.setattr(worker, "ingest_document", deleted_during)
    assert await worker.run_one("w")
    assert await load(Document, doc_id) is None


async def test_exhausted_lease_fails_document():
    job_id, doc_id = await make_job(max_attempts=1)
    await worker.claim_job("dies")
    async with session_scope() as session:
        (await session.get(Job, job_id)).lease_expires_at = utcnow() - timedelta(seconds=1)
    assert await worker.claim_job("next") is None
    assert await worker.fail_exhausted_leases() == 1
    assert (await load(Document, doc_id)).status == DocumentStatus.FAILED


async def test_interrupted_deletion_is_resumed():
    _, doc_id = await make_job()
    async with session_scope() as session:
        (await session.get(Document, doc_id)).status = DocumentStatus.DELETING
    assert await ingestion.resume_interrupted_deletions() == 1
    async with session_scope() as session:
        assert (await session.execute(select(Document).where(Document.id == doc_id))).first() is None
