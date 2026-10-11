"""Answer generation with Claude's native citations.

Each retrieved passage is sent as its own citable document. Claude's response comes back
as text blocks whose citations name the exact document (passage) they rely on, so every
claim maps to a known file and page — no regex over "[Source: ...]" strings, and the
model cannot cite a passage it wasn't given."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol

import anthropic
import structlog

from citewise.config import get_settings
from citewise.metrics import LLM_ERRORS, LLM_FALLBACKS
from citewise.retrieval import Passage

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are Citewise, an assistant that answers questions using only the documents the user has uploaded. The documents for each question are attached to the user's message.

Ground every factual statement in the attached documents and cite them. Quote numbers, dates, and names exactly as written. If the documents don't contain the answer, say so plainly and, if useful, mention what related information they do contain. If documents disagree, point out the conflict and cite each side.

Write in clear, well-organized prose; use short lists when they genuinely help. Earlier turns of the conversation are context for follow-up questions, but only the documents attached to the latest message are evidence."""

NO_CONTEXT_ANSWER = (
    "I couldn't find anything in your documents that answers this. "
    "Try rephrasing, or check that the relevant document has finished processing."
)


class GeneratorNotConfigured(RuntimeError):
    """The selected answer engine can't run (e.g. missing API key). The message is shown to the user."""


@dataclass
class GenEvent:
    kind: Literal["token", "cite", "reset", "error", "done"]
    text: str = ""
    passage_indices: list[int] = field(default_factory=list)
    model: str | None = None


class Generator(Protocol):
    async def stream(self, question: str, history: list[dict], passages: list[Passage]) -> AsyncIterator[GenEvent]: ...


def _document_block(p: Passage) -> dict:
    block = {
        "type": "document",
        "source": {"type": "content", "content": [{"type": "text", "text": p.text}]},
        "title": p.title,
        "citations": {"enabled": True},
    }
    if p.heading:
        block["context"] = f"Section: {p.heading}"
    return block


def build_messages(question: str, history: list[dict], passages: list[Passage]) -> list[dict]:
    messages = [{"role": m["role"], "content": m["content"]} for m in history if m["content"].strip()]
    # The API requires the first turn to be the user's; drop a leading assistant turn
    # left over from history truncation.
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    content = [_document_block(p) for p in passages]
    content.append({"type": "text", "text": question})
    messages.append({"role": "user", "content": content})
    return messages


class ClaudeGenerator:
    def __init__(self, client: anthropic.AsyncAnthropic | None = None):
        if client is None:
            key = get_settings().anthropic_api_key
            try:
                client = anthropic.AsyncAnthropic(api_key=key) if key else anthropic.AsyncAnthropic()
            except anthropic.AnthropicError as e:
                raise GeneratorNotConfigured(
                    "No Anthropic API key found. Set ANTHROPIC_API_KEY in .env, or choose another engine "
                    "with CITEWISE_GENERATOR (gemini or extractive)."
                ) from e
        self.client = client

    def _request(self, question: str, history: list[dict], passages: list[Passage]) -> dict:
        s = get_settings()
        kwargs: dict = {
            "model": s.anthropic_model,
            "max_tokens": s.anthropic_max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": build_messages(question, history, passages),
            "output_config": {"effort": s.anthropic_effort},
        }
        if s.anthropic_fallbacks:
            # If a safety classifier declines, the API re-runs the request on Anthropic's
            # recommended fallback model inside the same stream.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        return kwargs

    async def stream(self, question, history, passages):
        if not passages:
            yield GenEvent("token", NO_CONTEXT_ANSWER)
            yield GenEvent("done")
            return

        block_citations: dict[int, list[int]] = {}
        try:
            async with self.client.beta.messages.stream(**self._request(question, history, passages)) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "fallback":
                            LLM_FALLBACKS.inc()
                            log.info("llm_fallback", to=getattr(event.content_block.to, "model", None))
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield GenEvent("token", delta.text)
                        elif delta.type == "citations_delta":
                            citation = delta.citation
                            if citation.type == "content_block_location" and 0 <= citation.document_index < len(
                                passages
                            ):
                                cited = block_citations.setdefault(event.index, [])
                                if citation.document_index not in cited:
                                    cited.append(citation.document_index)
                    elif event.type == "content_block_stop":
                        if block_citations.get(event.index):
                            yield GenEvent("cite", passage_indices=block_citations.pop(event.index))
                final = await stream.get_final_message()
        except anthropic.RateLimitError:
            LLM_ERRORS.labels("rate_limit").inc()
            yield GenEvent("error", "Claude is rate limited right now. Please try again in a moment.")
            return
        except anthropic.AuthenticationError:
            LLM_ERRORS.labels("auth").inc()
            log.error("anthropic_auth_failed")
            yield GenEvent("error", "The server's Anthropic API key is missing or invalid.")
            return
        except anthropic.APIStatusError as e:
            LLM_ERRORS.labels(f"http_{e.status_code}").inc()
            log.error("anthropic_api_error", status=e.status_code, request_id=e.request_id, message=e.message)
            yield GenEvent("error", "The answer service returned an error. Please try again.")
            return
        except anthropic.APIConnectionError:
            LLM_ERRORS.labels("connection").inc()
            yield GenEvent("error", "Couldn't reach the answer service. Please try again.")
            return

        if final.stop_reason == "refusal":
            # The whole fallback chain declined; a partial answer must not be shown as complete.
            LLM_ERRORS.labels("refusal").inc()
            yield GenEvent("reset")
            yield GenEvent("error", "This request was declined by the model's safety system.")
            return
        if final.stop_reason == "max_tokens":
            yield GenEvent("token", "\n\n*(Answer truncated: reached the length limit.)*")
        yield GenEvent("done", model=final.model)


class ExtractiveGenerator:
    """Offline mode: no LLM. Returns the best-matching passages verbatim, each cited.
    Useful for demos without an API key and for checking retrieval quality on its own."""

    async def stream(self, question, history, passages):
        if not passages:
            yield GenEvent("token", NO_CONTEXT_ANSWER)
            yield GenEvent("done", model="extractive")
            return
        yield GenEvent("token", "Offline mode (no language model configured). The most relevant passages:\n\n")
        for i, p in enumerate(passages[:3]):
            flat = " ".join(p.text.split())
            snippet = flat if len(flat) <= 500 else flat[:500].rsplit(" ", 1)[0] + "…"
            yield GenEvent("token", f"> {snippet}")
            yield GenEvent("cite", passage_indices=[i])
            yield GenEvent("token", "\n\n")
        yield GenEvent("done", model="extractive")


_generator: Generator | None = None


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        kind = get_settings().generator
        if kind == "extractive":
            _generator = ExtractiveGenerator()
        elif kind == "gemini":
            from citewise.gemini import GeminiGenerator  # avoids a circular import

            _generator = GeminiGenerator()
        else:
            _generator = ClaudeGenerator()
    return _generator


def set_generator(generator: Generator | None) -> None:
    global _generator
    _generator = generator
