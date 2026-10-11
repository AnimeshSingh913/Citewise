import uuid
from types import SimpleNamespace as NS

import anthropic
import httpx2

from citewise.generation import NO_CONTEXT_ANSWER, ClaudeGenerator, build_messages
from citewise.retrieval import Passage


def passage(text: str, page: int | None = 1, heading: str | None = None) -> Passage:
    return Passage(uuid.uuid4(), uuid.uuid4(), "report.pdf", page, heading, text, 1.0)


class FakeStream:
    def __init__(self, events, final):
        self.events, self.final = events, final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def gen():
            for e in self.events:
                yield e

        return gen()

    async def get_final_message(self):
        return self.final


class FakeClient:
    def __init__(self, events=(), final=None, error=None):
        self.requests = []
        outer = self

        class Messages:
            def stream(self, **kwargs):
                outer.requests.append(kwargs)
                if error:
                    raise error
                return FakeStream(list(events), final)

        self.beta = NS(messages=Messages())


def text_delta(index, text):
    return NS(type="content_block_delta", index=index, delta=NS(type="text_delta", text=text))


def cite_delta(index, doc_index):
    citation = NS(type="content_block_location", document_index=doc_index, cited_text="x")
    return NS(type="content_block_delta", index=index, delta=NS(type="citations_delta", citation=citation))


def stop(index):
    return NS(type="content_block_stop", index=index)


async def collect(gen):
    return [e async for e in gen]


def test_build_messages_sends_each_passage_as_citable_document():
    history = [
        {"role": "assistant", "content": "orphan"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    msgs = build_messages("q2", history, [passage("Alpha", 3, "Intro"), passage("Beta", None)])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    docs = msgs[-1]["content"][:2]
    assert docs[0]["type"] == "document"
    assert docs[0]["source"] == {"type": "content", "content": [{"type": "text", "text": "Alpha"}]}
    assert docs[0]["title"] == "report.pdf (page 3)" and docs[0]["context"] == "Section: Intro"
    assert docs[0]["citations"] == {"enabled": True}
    assert "context" not in docs[1] and docs[1]["title"] == "report.pdf"
    assert msgs[-1]["content"][-1] == {"type": "text", "text": "q2"}


async def test_request_uses_configured_model_effort_and_fallbacks():
    client = FakeClient(events=[], final=NS(stop_reason="end_turn", model="claude-opus-5-5"))
    await collect(ClaudeGenerator(client).stream("q", [], [passage("a")]))
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"] == {"effort": "medium"}
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    assert "thinking" not in req and "temperature" not in req


async def test_citations_are_grouped_per_text_block():
    events = [
        NS(type="content_block_start", index=0, content_block=NS(type="text")),
        text_delta(0, "Revenue was 12M"),
        cite_delta(0, 1),
        cite_delta(0, 1),
        cite_delta(0, 0),
        cite_delta(0, 9),  # out of range: ignored
        stop(0),
        text_delta(1, " overall."),
        stop(1),
    ]
    client = FakeClient(events, NS(stop_reason="end_turn", model="claude-opus-5-5"))
    out = await collect(ClaudeGenerator(client).stream("q", [], [passage("a"), passage("b")]))
    assert [(e.kind, e.text, e.passage_indices) for e in out] == [
        ("token", "Revenue was 12M", []),
        ("cite", "", [1, 0]),
        ("token", " overall.", []),
        ("done", "", []),
    ]
    assert out[-1].model == "claude-opus-5-5"


async def test_fallback_block_keeps_partial_and_continues():
    events = [
        text_delta(0, "Part one"),
        NS(type="content_block_start", index=1, content_block=NS(type="fallback", to=NS(model="claude-opus-5"))),
        text_delta(2, " part two"),
    ]
    out = await collect(
        ClaudeGenerator(FakeClient(events, NS(stop_reason="end_turn", model="claude-opus-5"))).stream(
            "q", [], [passage("a")]
        )
    )
    assert "".join(e.text for e in out if e.kind == "token") == "Part one part two"
    assert out[-1].model == "claude-opus-5"


async def test_refusal_resets_partial_answer():
    out = await collect(
        ClaudeGenerator(FakeClient([text_delta(0, "partial")], NS(stop_reason="refusal", model="m"))).stream(
            "q", [], [passage("a")]
        )
    )
    assert [e.kind for e in out] == ["token", "reset", "error"]


async def test_no_passages_skips_the_api_call():
    client = FakeClient()
    out = await collect(ClaudeGenerator(client).stream("q", [], []))
    assert out[0].text == NO_CONTEXT_ANSWER and client.requests == []


async def test_api_errors_become_friendly_error_events():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(429, request=request, headers={"retry-after": "1"})
    err = anthropic.RateLimitError("rate limited", response=response, body=None)
    out = await collect(ClaudeGenerator(FakeClient(error=err)).stream("q", [], [passage("a")]))
    assert out[-1].kind == "error" and "rate limited" in out[-1].text
