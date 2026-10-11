"""Durable job queue on the relational DB.

Claiming is a single atomic UPDATE ... RETURNING. On Postgres the candidate row is picked
with FOR UPDATE SKIP LOCKED, so any number of worker processes can share the queue; on
SQLite writes are serialized, so the same statement is safe for one process.

Each claim takes a time-limited lease that a heartbeat keeps extending. A worker that
crashes simply stops renewing, and its job becomes claimable again when the lease runs
out: no restart-time recovery step is needed, and it works across machines."""

import asyncio
import os
import socket
import traceback
import uuid
from datetime import timedelta

import structlog
from sqlalchemy import DateTime, String, bindparam, func, select, text, update

from citewise.config import get_settings
from citewise.db import get_engine, session_scope
from citewise.ingestion import DocumentGone, ingest_document, mark_failed, resume_interrupted_deletions
from citewise.metrics import INGESTION_JOBS, QUEUE_DEPTH
from citewise.models import Job, JobStatus, utcnow
from citewise.parsing import UnsupportedFile

log = structlog.get_logger(__name__)

_wake = asyncio.Event()


def notify_new_job() -> None:
    _wake.set()


def _claim_sql(postgres: bool) -> str:
    lock = "FOR UPDATE SKIP LOCKED" if postgres else ""
    claimable = """
        (status = 'PENDING' AND run_after <= :now)
        OR (status = 'RUNNING' AND lease_expires_at < :now)
    """
    return f"""
        UPDATE job
           SET status = 'RUNNING',
               attempts = attempts + 1,
               worker_id = :worker_id,
               lease_expires_at = :lease,
               updated_at = :now
         WHERE id = (
                SELECT id FROM job
                 WHERE ({claimable}) AND attempts < max_attempts
                 ORDER BY run_after
                 LIMIT 1 {lock}
               )
           AND (({claimable}) AND attempts < max_attempts)
     RETURNING id, document_id, attempts, max_attempts
    """


async def claim_job(worker_id: str) -> tuple[uuid.UUID, uuid.UUID, int, int] | None:
    settings = get_settings()
    now = utcnow()
    params = {"now": now, "lease": now + timedelta(seconds=settings.job_lease_seconds), "worker_id": worker_id}
    # Typed bind params so datetimes are serialized by the dialect exactly like ORM writes.
    stmt = text(_claim_sql(settings.is_postgres)).bindparams(
        bindparam("now", type_=DateTime(timezone=True)),
        bindparam("lease", type_=DateTime(timezone=True)),
        bindparam("worker_id", type_=String),
    )
    async with get_engine().begin() as conn:
        row = (await conn.execute(stmt, params)).first()
    if row is None:
        return None
    job_id, doc_id = (uuid.UUID(str(v)) if not isinstance(v, uuid.UUID) else v for v in row[:2])
    return job_id, doc_id, int(row[2]), int(row[3])


async def fail_exhausted_leases() -> int:
    """Jobs whose worker died on the last allowed attempt can't be claimed again; fail them
    so their documents don't sit in PROCESSING forever."""
    now = utcnow()
    async with session_scope() as session:
        jobs = (
            (
                await session.execute(
                    select(Job).where(
                        Job.status == JobStatus.RUNNING,
                        Job.lease_expires_at < now,
                        Job.attempts >= Job.max_attempts,
                    )
                )
            )
            .scalars()
            .all()
        )
        for job in jobs:
            job.status = JobStatus.FAILED
            job.last_error = "Worker stopped responding on the final attempt"
            job.updated_at = now
        doc_ids = [j.document_id for j in jobs]
    for doc_id in doc_ids:
        await mark_failed(doc_id, "Processing timed out repeatedly", retrying=False)
    return len(doc_ids)


async def _heartbeat(job_id: uuid.UUID, worker_id: str, lost: asyncio.Event) -> None:
    settings = get_settings()
    interval = max(1.0, settings.job_lease_seconds / 3)
    while True:
        await asyncio.sleep(interval)
        async with session_scope() as session:
            result = await session.execute(
                update(Job)
                .where(Job.id == job_id, Job.worker_id == worker_id, Job.status == JobStatus.RUNNING)
                .values(lease_expires_at=utcnow() + timedelta(seconds=settings.job_lease_seconds))
            )
        if result.rowcount == 0:
            lost.set()
            return


async def _finish(job_id: uuid.UUID, worker_id: str, **values) -> bool:
    """Write the job outcome only if we still hold the lease (fencing)."""
    async with session_scope() as session:
        result = await session.execute(
            update(Job)
            .where(Job.id == job_id, Job.worker_id == worker_id)
            .values(updated_at=utcnow(), lease_expires_at=None, **values)
        )
    return result.rowcount > 0


def _backoff(attempt: int) -> timedelta:
    return timedelta(seconds=min(300, 10 * 2 ** (attempt - 1)))


async def run_one(worker_id: str) -> bool:
    """Claim and process a single job. Returns False when the queue is empty."""
    claimed = await claim_job(worker_id)
    if claimed is None:
        return False
    job_id, doc_id, attempt, max_attempts = claimed
    log.info("job_claimed", job_id=str(job_id), document_id=str(doc_id), attempt=attempt, worker=worker_id)

    lost = asyncio.Event()
    heartbeat = asyncio.create_task(_heartbeat(job_id, worker_id, lost))
    try:
        await ingest_document(doc_id)
    except DocumentGone:
        await _finish(job_id, worker_id, status=JobStatus.SUCCEEDED, last_error="document deleted")
        INGESTION_JOBS.labels("cancelled").inc()
    except UnsupportedFile as e:
        # The file itself is the problem; retrying can't help.
        if await _finish(job_id, worker_id, status=JobStatus.FAILED, last_error=str(e)):
            await mark_failed(doc_id, str(e), retrying=False)
        INGESTION_JOBS.labels("rejected").inc()
    except asyncio.CancelledError:
        # Shutdown: hand the job back immediately rather than waiting for the lease to expire.
        await asyncio.shield(
            _finish(job_id, worker_id, status=JobStatus.PENDING, attempts=attempt - 1, run_after=utcnow())
        )
        raise
    except Exception as e:
        detail = "".join(traceback.format_exception_only(e)).strip()
        retrying = attempt < max_attempts
        log.error("job_failed", job_id=str(job_id), attempt=attempt, retrying=retrying, error=detail)
        values = {"status": JobStatus.PENDING if retrying else JobStatus.FAILED, "last_error": detail}
        if retrying:
            values["run_after"] = utcnow() + _backoff(attempt)
        if await _finish(job_id, worker_id, **values):
            await mark_failed(
                doc_id, f"Attempt {attempt} failed, retrying" if retrying else "Processing failed", retrying
            )
        INGESTION_JOBS.labels("retry" if retrying else "failed").inc()
    else:
        if lost.is_set():
            log.warning("lease_lost", job_id=str(job_id))
        await _finish(job_id, worker_id, status=JobStatus.SUCCEEDED, last_error=None)
        INGESTION_JOBS.labels("succeeded").inc()
    finally:
        heartbeat.cancel()
    return True


async def queue_depth() -> int:
    async with session_scope() as session:
        return (
            await session.execute(select(func.count()).select_from(Job).where(Job.status == JobStatus.PENDING))
        ).scalar_one()


async def worker_loop(concurrency: int | None = None) -> None:
    settings = get_settings()
    concurrency = concurrency or settings.worker_concurrency
    host = f"{socket.gethostname()}-{os.getpid()}"

    resumed = await resume_interrupted_deletions()
    if resumed:
        log.info("resumed_deletions", count=resumed)

    async def lane(n: int) -> None:
        worker_id = f"{host}-{n}"
        while True:
            try:
                if await run_one(worker_id):
                    continue
                await fail_exhausted_leases()
                QUEUE_DEPTH.set(await queue_depth())
            except asyncio.CancelledError:
                raise
            except Exception:
                log.error("worker_iteration_failed", worker=worker_id, exc_info=True)
            _wake.clear()
            try:
                await asyncio.wait_for(_wake.wait(), timeout=settings.worker_poll_seconds)
            except TimeoutError:
                pass

    log.info("worker_started", concurrency=concurrency)
    await asyncio.gather(*(lane(i) for i in range(concurrency)))
