import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text

from citewise import db
from citewise.config import get_settings
from citewise.middleware import MaxBodySizeMiddleware
from citewise.routers import auth, conversations, documents
from citewise.vectorstore import get_vector_store
from citewise.worker import worker_loop

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    db.get_engine()
    if settings.env != "production":
        await db.create_all()  # production schema is managed by `alembic upgrade head`
    await get_vector_store().ensure_collection()

    worker = asyncio.create_task(worker_loop()) if settings.run_worker_in_api else None
    log.info("startup_complete", env=settings.env, worker=bool(worker))
    yield
    if worker:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker
    await db.dispose_engine()


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    app = FastAPI(
        title="Citewise",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.limiter = auth.limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.max_upload_bytes + 64 * 1024)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        if not request.url.path.startswith("/api/docs"):
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            )
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.error("unhandled_exception", path=request.url.path, exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    app.include_router(auth.router)
    app.include_router(documents.router)
    app.include_router(conversations.router)

    @app.get("/api/health", tags=["ops"])
    async def health():
        return {"status": "ok"}

    @app.get("/api/ready", tags=["ops"])
    async def ready():
        checks = {}
        try:
            async with db.get_engine().connect() as conn:
                await conn.execute(text("SELECT 1"))
            checks["database"] = "ok"
        except Exception:
            checks["database"] = "error"
        try:
            store = get_vector_store()
            await store.client.collection_exists(store.collection)
            checks["vector_store"] = "ok"
        except Exception:
            checks["vector_store"] = "error"
        ok = all(v == "ok" for v in checks.values())
        return JSONResponse(status_code=200 if ok else 503, content={"status": "ok" if ok else "degraded", **checks})

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
