import os

os.environ.update(
    {
        "CITEWISE_ENV": "testing",
        "CITEWISE_EMBEDDING_BACKEND": "hashing",
        "CITEWISE_RUN_WORKER_IN_API": "false",
        "CITEWISE_QDRANT_PATH": ":memory:",
        "CITEWISE_JOB_LEASE_SECONDS": "30",
        "CITEWISE_MAX_UPLOAD_MB": "2",
    }
)

import io  # noqa: E402

import httpx  # noqa: E402
import pymupdf  # noqa: E402
import pytest  # noqa: E402
from qdrant_client import AsyncQdrantClient  # noqa: E402

from citewise import db  # noqa: E402
from citewise.embeddings import HashingEmbedder, set_embedder  # noqa: E402
from citewise.generation import set_generator  # noqa: E402
from citewise.main import app  # noqa: E402
from citewise.storage import LocalStorage, set_storage  # noqa: E402
from citewise.vectorstore import VectorStore, set_vector_store  # noqa: E402


@pytest.fixture(autouse=True)
async def env(tmp_path):
    db.init_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await db.create_all()
    embedder = HashingEmbedder()
    set_embedder(embedder)
    store = VectorStore(AsyncQdrantClient(location=":memory:"), "test", embedder)
    await store.ensure_collection()
    set_vector_store(store)
    set_storage(LocalStorage(tmp_path / "uploads"))
    yield store
    set_generator(None)
    set_vector_store(None)
    set_storage(None)
    set_embedder(None)
    await store.client.close()
    await db.dispose_engine()


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def register_and_login(client: httpx.AsyncClient, email: str = "ada@example.com") -> dict:
    password = "correct horse battery staple"
    r = await client.post("/api/auth/register", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    r = await client.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    tokens = r.json()
    return {"Authorization": f"Bearer {tokens['access_token']}", "_tokens": tokens}


def auth_headers(session: dict) -> dict:
    return {"Authorization": session["Authorization"]}


@pytest.fixture
async def alice(client):
    return await register_and_login(client, "alice@example.com")


@pytest.fixture
async def bob(client):
    return await register_and_login(client, "bob@example.com")


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        overflow = page.insert_textbox(pymupdf.Rect(40, 40, 570, 820), text, fontsize=8)
        assert overflow >= 0, "test text does not fit on one PDF page"  # PyMuPDF drops it silently
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def parse_sse(body: str) -> list[tuple[str, object]]:
    import json

    events = []
    for block in body.strip().split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if name:
            events.append((name, data))
    return events
