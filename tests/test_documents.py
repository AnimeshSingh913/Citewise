from qdrant_client import models

from citewise.worker import run_one
from tests.conftest import auth_headers, make_pdf


async def upload(client, session, name="report.pdf", data=None, mime="application/pdf"):
    data = data if data is not None else make_pdf(["Revenue in 2024 was 12 million.", "Headcount reached 340."])
    return await client.post("/api/documents", headers=auth_headers(session), files={"file": (name, data, mime)})


async def drain():
    while await run_one("test-worker"):
        pass


async def test_upload_ingest_ready_with_pages(client, alice):
    r = await upload(client, alice)
    assert r.status_code == 202, r.text
    doc = r.json()
    assert doc["status"] == "PENDING"

    await drain()
    r = await client.get(f"/api/documents/{doc['id']}", headers=auth_headers(alice))
    body = r.json()
    assert body["status"] == "READY", body
    assert body["page_count"] == 2
    assert body["chunk_count"] == 2


async def test_duplicate_content_is_rejected_and_names_deduplicated(client, alice):
    data = make_pdf(["Same bytes twice."])
    first = (await upload(client, alice, data=data)).json()
    r = await upload(client, alice, data=data)
    assert r.status_code == 409
    assert r.json()["detail"]["document_id"] == first["id"]

    other = await upload(client, alice, data=make_pdf(["Different content entirely."]))
    assert other.json()["filename"] == "report (2).pdf"


async def test_unsupported_and_spoofed_files(client, alice):
    assert (await upload(client, alice, "virus.exe", b"MZ\x90\x00", "application/octet-stream")).status_code == 415
    assert (await upload(client, alice, "fake.pdf", b"<html>hi</html>")).status_code == 415
    assert (await upload(client, alice, "empty.txt", b"", "text/plain")).status_code == 400


async def test_path_traversal_filename_is_neutralized(client, alice):
    r = await upload(client, alice, "../../etc/passwd.txt", b"just text", "text/plain")
    assert r.status_code == 202
    assert r.json()["filename"] == "passwd.txt"


async def test_oversized_upload_rejected(client, alice):
    big = b"%PDF-" + b"0" * (3 * 1024 * 1024)
    assert (await upload(client, alice, data=big)).status_code == 413


async def test_documents_are_private(client, alice, bob):
    doc = (await upload(client, alice)).json()
    assert (await client.get(f"/api/documents/{doc['id']}", headers=auth_headers(bob))).status_code == 404
    assert (await client.delete(f"/api/documents/{doc['id']}", headers=auth_headers(bob))).status_code == 404
    assert (await client.get("/api/documents", headers=auth_headers(bob))).json()["total"] == 0


async def test_pagination(client, alice):
    for i in range(5):
        await upload(client, alice, f"doc{i}.txt", f"unique text number {i}".encode(), "text/plain")
    r = await client.get("/api/documents?limit=2&offset=0", headers=auth_headers(alice))
    page = r.json()
    assert page["total"] == 5 and len(page["items"]) == 2
    r = await client.get("/api/documents?limit=2&offset=4", headers=auth_headers(alice))
    assert len(r.json()["items"]) == 1
    await drain()
    r = await client.get("/api/documents?status=READY", headers=auth_headers(alice))
    assert r.json()["total"] == 5


async def test_failed_document_can_be_reprocessed(client, alice):
    # A .txt containing only whitespace has no extractable text -> permanent failure.
    doc = (await upload(client, alice, "blank.txt", b"   \n  ", "text/plain")).json()
    await drain()
    body = (await client.get(f"/api/documents/{doc['id']}", headers=auth_headers(alice))).json()
    assert body["status"] == "FAILED" and body["error"]

    r = await client.post(f"/api/documents/{doc['id']}/reprocess", headers=auth_headers(alice))
    assert r.status_code == 202
    assert r.json()["status"] == "PENDING"
    # Can't double-queue.
    assert (await client.post(f"/api/documents/{doc['id']}/reprocess", headers=auth_headers(alice))).status_code == 409


async def test_reprocess_ready_document_does_not_duplicate_chunks(client, alice, env):
    doc = (await upload(client, alice)).json()
    await drain()
    await client.post(f"/api/documents/{doc['id']}/reprocess", headers=auth_headers(alice))
    await drain()
    body = (await client.get(f"/api/documents/{doc['id']}", headers=auth_headers(alice))).json()
    assert body["status"] == "READY" and body["chunk_count"] == 2
    count = await env.client.count(
        env.collection,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=doc["id"]))]
        ),
    )
    first_count = count.count
    await client.post(f"/api/documents/{doc['id']}/reprocess", headers=auth_headers(alice))
    await drain()
    again = await env.client.count(
        env.collection,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=doc["id"]))]
        ),
    )
    assert again.count == first_count > 0


async def test_delete_removes_everything(client, alice, env):
    doc = (await upload(client, alice)).json()
    await drain()
    r = await client.delete(f"/api/documents/{doc['id']}", headers=auth_headers(alice))
    assert r.status_code == 202
    assert (await client.get(f"/api/documents/{doc['id']}", headers=auth_headers(alice))).status_code == 404
    remaining = await env.client.count(env.collection)
    assert remaining.count == 0


async def test_download_original(client, alice):
    data = b"plain text body"
    doc = (await upload(client, alice, "n.txt", data, "text/plain")).json()
    r = await client.get(f"/api/documents/{doc['id']}/file", headers=auth_headers(alice))
    assert r.status_code == 200 and r.content == data
