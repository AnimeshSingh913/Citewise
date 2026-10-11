import uuid
from types import SimpleNamespace as NS

from google.genai import errors

from citewise.gemini import CitationParser, GeminiGenerator, build_contents, format_passages
from citewise.generation import NO_CONTEXT_ANSWER
from citewise.retrieval import Passage


def passage(text, page=1, heading=None, filename="report.pdf"):
    return Passage(uuid.uuid4(), uuid.uuid4(), filename, page, heading, text, 1.0)


def chunk(text, finish=None):
    return NS(text=text, candidates=[NS(finish_reason=NS(name=finish) if finish else None)])


class FakeGemini:
    def __init__(self, chunks=(), error=None):
        self.calls = []
        outer = self

        class Models:
            async def generate_content_stream(self, **kwargs):
                outer.calls.append(kwargs)
                if error:
                    raise error

                async def gen():
                    for c in chunks:
                        yield c

                return gen()

        self.aio = NS(models=Models())


async def collect(gen):
    return [e async for e in gen]


def summarize(events):
    return [(e.kind, e.text, e.passage_indices) for e in events]


def test_parser_handles_markers_split_across_chunks():
    p = CitationParser(passage_count=3)
    events = p.feed("Revenue grew 31% [") + p.feed("2] and headcount hit 340 [1,") + p.feed(" 3][9].") + p.flush()
    assert summarize(events) == [
        ("token", "Revenue grew 31% ", []),
        ("cite", "", [1]),
        ("token", " and headcount hit 340 ", []),
        ("cite", "", [0, 2]),
        ("token", ".", []),  # [9] doesn't exist: dropped, never shown as a citation
    ]


def test_parser_leaves_ordinary_brackets_alone():
    p = CitationParser(passage_count=2)
    events = p.feed("In [2024] the [draft] said [") + p.feed("note]") + p.flush()
    assert "".join(e.text for e in events) == "In [2024] the [draft] said [note]"
    assert not [e for e in events if e.kind == "cite"]


def test_prompt_numbers_passages_and_strips_old_markers():
    history = [
        {"role": "assistant", "content": "orphan"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "It was 12M[1][2]."},
    ]
    contents = build_contents("q2", history, [passage("Alpha", 3), passage("Beta", None, "Pricing", "p.md")])
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[1]["parts"][0]["text"] == "It was 12M."
    prompt = contents[-1]["parts"][0]["text"]
    assert "[1] report.pdf (page 3)\nAlpha" in prompt
    assert "[2] p.md (section “Pricing”)\nBeta" in prompt
    assert prompt.endswith("Question: q2")
    assert format_passages([]) == ""


async def test_stream_maps_citations_and_reports_model():
    client = FakeGemini([chunk("Headcount was 340 ["), chunk("1]."), chunk("", finish="STOP")])
    out = await collect(GeminiGenerator(client).stream("q", [], [passage("Headcount 340")]))
    assert summarize(out) == [
        ("token", "Headcount was 340 ", []),
        ("cite", "", [0]),
        ("token", ".", []),
        ("done", "", []),
    ]
    assert out[-1].model == "gemini-flash-latest"
    config = client.calls[0]["config"]
    assert "system_instruction" in config and config["temperature"] == 0.2


async def test_safety_block_discards_partial_answer():
    client = FakeGemini([chunk("partial"), chunk("", finish="SAFETY")])
    out = await collect(GeminiGenerator(client).stream("q", [], [passage("x")]))
    assert [e.kind for e in out] == ["token", "reset", "error"]


async def test_rate_limit_becomes_friendly_error():
    err = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    out = await collect(GeminiGenerator(FakeGemini(error=err)).stream("q", [], [passage("x")]))
    assert out[-1].kind == "error" and "rate limit" in out[-1].text


async def test_no_passages_skips_the_api_call():
    client = FakeGemini()
    out = await collect(GeminiGenerator(client).stream("q", [], []))
    assert out[0].text == NO_CONTEXT_ANSWER and client.calls == []


class SequencedGemini:
    """Fails with the given errors in order, then streams `chunks`."""

    def __init__(self, failures, chunks):
        self.models_called = []
        outer = self

        class Models:
            async def generate_content_stream(self, model, **kwargs):
                outer.models_called.append(model)
                if failures:
                    raise failures.pop(0)

                async def gen():
                    for c in chunks:
                        yield c

                return gen()

        self.aio = NS(models=Models())


def server_error(code=503):
    return errors.ServerError(code, {"error": {"code": code, "message": "high demand", "status": "UNAVAILABLE"}})


async def test_overloaded_model_falls_back_to_lighter_model():
    client = SequencedGemini([server_error()], [chunk("Green [1]."), chunk("", finish="STOP")])
    out = await collect(GeminiGenerator(client).stream("q", [], [passage("x")]))
    assert client.models_called == ["gemini-flash-latest", "gemini-flash-lite-latest"]
    assert out[-1].kind == "done" and out[-1].model == "gemini-flash-lite-latest"
    assert [e.passage_indices for e in out if e.kind == "cite"] == [[0]]


async def test_both_models_overloaded_gives_friendly_error():
    client = SequencedGemini([server_error(), server_error()], [])
    out = await collect(GeminiGenerator(client).stream("q", [], [passage("x")]))
    assert out[-1].kind == "error" and "overloaded" in out[-1].text


async def test_auth_error_does_not_fall_back():
    err = errors.ClientError(403, {"error": {"code": 403, "message": "bad key", "status": "PERMISSION_DENIED"}})
    client = SequencedGemini([err], [])
    out = await collect(GeminiGenerator(client).stream("q", [], [passage("x")]))
    assert client.models_called == ["gemini-flash-latest"] and "invalid" in out[-1].text
