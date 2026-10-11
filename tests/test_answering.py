from citewise.answering import retrieval_query
from citewise.generation import ExtractiveGenerator, GenEvent, set_generator
from tests.conftest import auth_headers, make_pdf, parse_sse
from tests.test_documents import drain, upload


class ScriptedGenerator:
    """Cites whichever passage mentions the word it is told to look for."""

    def __init__(self):
        self.calls = []

    async def stream(self, question, history, passages):
        self.calls.append({"question": question, "history": history, "passages": passages})
        target = next((i for i, p in enumerate(passages) if "340" in p.text), 0)
        yield GenEvent("token", "Headcount reached 340")
        yield GenEvent("cite", passage_indices=[target])
        yield GenEvent("token", ".")
        yield GenEvent("done", model="scripted")


async def new_conversation(client, session):
    r = await client.post("/api/conversations", headers=auth_headers(session), json={})
    assert r.status_code == 201
    return r.json()["id"]


async def ask(client, session, conv_id, content, **extra):
    r = await client.post(
        f"/api/conversations/{conv_id}/messages", headers=auth_headers(session), json={"content": content, **extra}
    )
    assert r.status_code == 200, r.text
    return parse_sse(r.text)


async def test_answer_streams_with_numbered_page_citations(client, alice):
    gen = ScriptedGenerator()
    set_generator(gen)
    await upload(client, alice)
    await drain()
    conv = await new_conversation(client, alice)

    events = await ask(client, alice, conv, "How many people work there?")
    names = [e[0] for e in events]
    assert names[0] == "retrieved" and names[-1] == "done"
    cite = next(d for n, d in events if n == "cite")
    assert cite["numbers"] == [1]
    source = cite["sources"][0]
    assert source["filename"] == "report.pdf" and source["page_number"] == 2

    msgs = (await client.get(f"/api/conversations/{conv}/messages", headers=auth_headers(alice))).json()
    assert [m["role"] for m in msgs["items"]] == ["user", "assistant"]
    assistant = msgs["items"][1]
    assert assistant["content"] == "Headcount reached 340[1]."
    assert assistant["sources"][0]["page_number"] == 2
    assert assistant["model"] == "scripted"

    conv_meta = (await client.get(f"/api/conversations/{conv}", headers=auth_headers(alice))).json()
    assert conv_meta["title"] == "How many people work there?"


async def test_follow_up_gets_history(client, alice):
    gen = ScriptedGenerator()
    set_generator(gen)
    await upload(client, alice)
    await drain()
    conv = await new_conversation(client, alice)
    await ask(client, alice, conv, "How many people work there?")
    await ask(client, alice, conv, "And revenue?")
    history = gen.calls[1]["history"]
    assert [m["role"] for m in history] == ["user", "assistant"]


async def test_retrieval_is_isolated_between_users(client, alice, bob):
    gen = ScriptedGenerator()
    set_generator(gen)
    await upload(client, alice, data=make_pdf(["Alice's secret launch code is 7781."]))
    await drain()
    conv = await new_conversation(client, bob)
    await ask(client, bob, conv, "What is the secret launch code?")
    assert gen.calls[-1]["passages"] == []


async def test_document_filter_restricts_retrieval(client, alice):
    gen = ScriptedGenerator()
    set_generator(gen)
    a = (await upload(client, alice, "a.txt", b"apples are red and sweet", "text/plain")).json()
    await upload(client, alice, "b.txt", b"apples can also be green", "text/plain")
    await drain()
    conv = await new_conversation(client, alice)
    await ask(client, alice, conv, "what color are apples", document_ids=[a["id"]])
    assert {p.filename for p in gen.calls[-1]["passages"]} == {"a.txt"}


async def test_error_event_does_not_save_partial_answer(client, alice):
    class Failing:
        async def stream(self, question, history, passages):
            yield GenEvent("token", "partial")
            yield GenEvent("error", "boom")

    set_generator(Failing())
    conv = await new_conversation(client, alice)
    events = await ask(client, alice, conv, "anything")
    assert ("error", {"message": "boom"}) in events
    msgs = (await client.get(f"/api/conversations/{conv}/messages", headers=auth_headers(alice))).json()
    assert [m["role"] for m in msgs["items"]] == ["user"]


async def test_extractive_mode_cites_passages(client, alice):
    set_generator(ExtractiveGenerator())
    await upload(client, alice)
    await drain()
    conv = await new_conversation(client, alice)
    events = await ask(client, alice, conv, "revenue 2024")
    done = events[-1]
    assert done[0] == "done" and done[1]["sources"]


async def test_conversations_are_private_and_manageable(client, alice, bob):
    conv = await new_conversation(client, alice)
    assert (await client.get(f"/api/conversations/{conv}", headers=auth_headers(bob))).status_code == 404
    r = await client.post(f"/api/conversations/{conv}/messages", headers=auth_headers(bob), json={"content": "hi"})
    assert r.status_code == 404
    r = await client.patch(f"/api/conversations/{conv}", headers=auth_headers(alice), json={"title": "Renamed"})
    assert r.json()["title"] == "Renamed"
    assert (await client.delete(f"/api/conversations/{conv}", headers=auth_headers(alice))).status_code == 204
    assert (await client.get("/api/conversations", headers=auth_headers(alice))).json()["total"] == 0


def test_retrieval_query_folds_in_previous_question():
    history = [{"role": "user", "content": "What was ACME's revenue?"}, {"role": "assistant", "content": "12M"}]
    assert retrieval_query("and in 2023?", history) == "What was ACME's revenue?\nand in 2023?"
    long_q = "Explain in detail how the company allocates its research budget across divisions"
    assert retrieval_query(long_q, history) == long_q


async def test_unconfigured_engine_shows_actionable_message(client, alice, monkeypatch):
    from citewise import answering
    from citewise.generation import GeneratorNotConfigured

    def broken():
        raise GeneratorNotConfigured("No Gemini API key found. Set GEMINI_API_KEY in .env and restart the server.")

    monkeypatch.setattr(answering, "get_generator", broken)
    conv = await new_conversation(client, alice)
    events = await ask(client, alice, conv, "anything")
    assert events[-1] == (
        "error",
        {"message": "No Gemini API key found. Set GEMINI_API_KEY in .env and restart the server."},
    )
