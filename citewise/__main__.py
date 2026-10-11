"""Command line: `python -m citewise <command>`.

serve        Run the API (and, unless CITEWISE_RUN_WORKER_IN_API=false, the worker).
worker       Run only the ingestion worker; start as many as you like.
reindex      Queue every document whose index was built with a different embedding model.
"""

import argparse
import asyncio


async def _worker() -> None:
    from citewise import db
    from citewise.main import configure_logging
    from citewise.vectorstore import get_vector_store
    from citewise.worker import worker_loop

    configure_logging()
    db.init_engine()
    await get_vector_store().ensure_collection()
    try:
        await worker_loop()
    finally:
        await db.dispose_engine()


async def _reindex() -> None:
    from sqlalchemy import or_, select

    from citewise import db
    from citewise.config import get_settings
    from citewise.models import Document, DocumentStatus, Job
    from citewise.vectorstore import get_vector_store

    db.init_engine()
    current = get_vector_store().embedder.name
    async with db.session_scope() as session:
        docs = (
            (
                await session.execute(
                    select(Document).where(
                        Document.status == DocumentStatus.READY,
                        or_(Document.embedding_model.is_(None), Document.embedding_model != current),
                    )
                )
            )
            .scalars()
            .all()
        )
        for doc in docs:
            doc.status = DocumentStatus.PENDING
            session.add(Job(document_id=doc.id, max_attempts=get_settings().job_max_attempts))
    await db.dispose_engine()
    print(f"Queued {len(docs)} document(s) for re-indexing with {current}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="citewise")
    parser.add_argument("command", choices=["serve", "worker", "reindex"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        uvicorn.run("citewise.main:app", host=args.host, port=args.port, proxy_headers=True)
    elif args.command == "worker":
        asyncio.run(_worker())
    else:
        asyncio.run(_reindex())


if __name__ == "__main__":
    main()
