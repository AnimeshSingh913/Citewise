"""Answer generation with Google Gemini.

Gemini has no native citation API, so passages are numbered in the prompt and the model
is asked to cite them as [n]. A streaming parser turns those markers into the same "cite"
events the Claude path produces, and accepts only numbers that match a passage we actually
sent, so a citation can never point at a document the model wasn't given."""

import re
from collections.abc import AsyncIterator

import structlog

from citewise.config import get_settings
from citewise.generation import NO_CONTEXT_ANSWER, GeneratorNotConfigured, GenEvent
from citewise.metrics import LLM_ERRORS
from citewise.retrieval import Passage

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are Citewise, an assistant that answers questions using only the numbered source passages provided with the user's question.

Rules:
- Base every factual statement on the passages. Quote numbers, dates, and names exactly as written.
- Cite every claim immediately after it with the passage number in square brackets, like [2]. For several sources use [1][3]. Only cite passage numbers that exist.
- If the passages don't contain the answer, say so plainly. Never use outside knowledge.
- If passages disagree, point out the conflict and cite each side.
- Write clear, well-organized prose; use short lists only when they genuinely help."""

# Earlier answers contain [n] markers that referred to *their* sources; strip them so the
# model doesn't confuse them with this turn's numbering.
_OLD_MARKERS = re.compile(r"\[\d{1,3}(?:\s*,\s*\d{1,3})*\]")
_MARKER = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
_PARTIAL = re.compile(r"\[[\d,\s]{0,15}$")


def format_passages(passages: list[Passage]) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        where = f"page {p.page_number}" if p.page_number else (f"section “{p.heading}”" if p.heading else "")
        header = f"[{i}] {p.filename}" + (f" ({where})" if where else "")
        blocks.append(f"{header}\n{p.text}")
    return "\n\n".join(blocks)


def build_contents(question: str, history: list[dict], passages: list[Passage]) -> list[dict]:
    contents = []
    for m in history:
        text = _OLD_MARKERS.sub("", m["content"]).strip()
        if text:
            contents.append({"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": text}]})
    while contents and contents[0]["role"] != "user":
        contents.pop(0)
    prompt = f"Source passages:\n\n{format_passages(passages)}\n\nQuestion: {question}"
    contents.append({"role": "user", "parts": [{"text": prompt}]})
    return contents


class CitationParser:
    """Splits streamed text into plain text and [n] citation markers, even when a marker
    is split across chunks ("…grew [" + "2] in…")."""

    def __init__(self, passage_count: int):
        self.count = passage_count
        self.buffer = ""

    def feed(self, text: str) -> list[GenEvent]:
        self.buffer += text
        events: list[GenEvent] = []
        while True:
            match = _MARKER.search(self.buffer)
            if match is None:
                break
            self._emit_text(self.buffer[: match.start()], events)
            indices = []
            for raw in match.group(1).split(","):
                n = int(raw)
                if 1 <= n <= self.count and n - 1 not in indices:
                    indices.append(n - 1)
            if indices:
                events.append(GenEvent("cite", passage_indices=indices))
            self.buffer = self.buffer[match.end() :]
        # Hold back a possible incomplete marker at the end; release everything else.
        partial = _PARTIAL.search(self.buffer)
        cut = partial.start() if partial else len(self.buffer)
        self._emit_text(self.buffer[:cut], events)
        self.buffer = self.buffer[cut:]
        return events

    def flush(self) -> list[GenEvent]:
        events: list[GenEvent] = []
        self._emit_text(self.buffer, events)
        self.buffer = ""
        return events

    @staticmethod
    def _emit_text(text: str, events: list[GenEvent]) -> None:
        if text:
            events.append(GenEvent("token", text))


class GeminiGenerator:
    def __init__(self, client=None):
        if client is None:
            from google import genai

            key = get_settings().gemini_api_key
            if not key:
                raise GeneratorNotConfigured(
                    "No Gemini API key found. Set GEMINI_API_KEY in .env and restart the server."
                )
            from google.genai import types

            client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=60_000,  # ms; without it a stalled request can hang the answer indefinitely
                    # Short retries for transient server trouble. 429 (quota) is not retried here:
                    # waiting won't help within a single request on the free tier.
                    retry_options=types.HttpRetryOptions(
                        attempts=3, initial_delay=1.0, max_delay=4.0, http_status_codes=[500, 502, 503, 504]
                    ),
                ),
            )
        self.client = client

    def _config(self) -> dict:
        s = get_settings()
        return {
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.2,
            "max_output_tokens": s.gemini_max_output_tokens,
        }

    async def stream(self, question: str, history: list[dict], passages: list[Passage]) -> AsyncIterator[GenEvent]:
        if not passages:
            yield GenEvent("token", NO_CONTEXT_ANSWER)
            yield GenEvent("done")
            return

        from google.genai import errors

        s = get_settings()
        models = [s.gemini_model] + [m for m in [s.gemini_fallback_model] if m and m != s.gemini_model]
        contents = build_contents(question, history, passages)
        for attempt, model in enumerate(models):
            parser = CitationParser(len(passages))
            finish_reason = None
            emitted = False
            try:
                stream = await self.client.aio.models.generate_content_stream(
                    model=model, contents=contents, config=self._config()
                )
                async for chunk in stream:
                    for candidate in chunk.candidates or []:
                        if candidate.finish_reason is not None:
                            finish_reason = str(getattr(candidate.finish_reason, "name", candidate.finish_reason))
                    if chunk.text:
                        for event in parser.feed(chunk.text):
                            emitted = True
                            yield event
                break
            except errors.APIError as e:
                kind = {429: "rate_limit", 400: "bad_request", 401: "auth", 403: "auth", 404: "not_found"}.get(e.code)
                LLM_ERRORS.labels(f"gemini_{kind or e.code}").inc()
                log.error("gemini_api_error", model=model, code=e.code, message=str(e)[:500])
                # Overloaded or rate-limited: try the lighter fallback model, unless we already
                # streamed part of an answer (switching mid-answer would garble it).
                if e.code in (429, 500, 502, 503, 504) and not emitted and attempt + 1 < len(models):
                    log.info("gemini_fallback", to=models[attempt + 1])
                    continue
                yield GenEvent("reset")
                yield GenEvent("error", _friendly(e.code, model))
                return

        for event in parser.flush():
            yield event
        if finish_reason in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION"):
            LLM_ERRORS.labels("gemini_blocked").inc()
            yield GenEvent("reset")
            yield GenEvent("error", "Gemini declined to answer this request.")
            return
        if finish_reason == "MAX_TOKENS":
            yield GenEvent("token", "\n\n*(Answer truncated: reached the length limit.)*")
        yield GenEvent("done", model=model)


def _friendly(code: int, model: str) -> str:
    if code == 429:
        return "Gemini's free-tier rate limit was reached. Wait a minute and try again."
    if code in (401, 403):
        return "The Gemini API key is missing or invalid. Check GEMINI_API_KEY in .env."
    if code == 400:
        return "Gemini rejected the request. If this keeps happening, check that GEMINI_API_KEY is valid."
    if code in (500, 502, 503, 504):
        return "Gemini is overloaded right now (this is common on the free tier). Please try again in a moment."
    if code == 404:
        return f"Gemini model “{model}” wasn't found. Set CITEWISE_GEMINI_MODEL in .env to an available model."
    return "The answer service returned an error. Please try again."
