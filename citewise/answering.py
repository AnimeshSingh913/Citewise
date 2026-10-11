"""Orchestrates one question: retrieve -> generate -> number citations -> persist."""

import json
import time
import uuid
from collections.abc import AsyncIterator

import structlog
from sqlalchemy import select

from citewise.config import get_settings
from citewise.db import session_scope
from citewise.generation import GeneratorNotConfigured, get_generator
from citewise.metrics import GENERATION_SECONDS
from citewise.models import Conversation, Message, utcnow
from citewise.retrieval import Passage, retrieve

log = structlog.get_logger(__name__)


def retrieval_query(question: str, history: list[dict]) -> str:
    """Follow-ups like "what about 2023?" carry little on their own; folding in the previous
    user question gives the retriever the missing subject without an extra LLM call."""
    previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    if previous and len(question.split()) < 12:
        return f"{previous}\n{question}"
    return question


async def load_history(conversation_id: uuid.UUID, limit: int) -> list[dict]:
    async with session_scope() as session:
        rows = (
            (
                await session.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
    return [{"role": m.role, "content": m.content} for m in reversed(rows)]


class SourceRegistry:
    """Assigns stable [n] numbers to passages in the order they are first cited."""

    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self.number_by_index: dict[int, int] = {}
        self.sources: list[dict] = []

    def number(self, passage_index: int) -> int:
        if passage_index not in self.number_by_index:
            n = len(self.sources) + 1
            self.number_by_index[passage_index] = n
            self.sources.append({"n": n, **self.passages[passage_index].public()})
        return self.number_by_index[passage_index]


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def answer_stream(
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    question: str,
    document_ids: list[uuid.UUID] | None,
) -> AsyncIterator[str]:
    settings = get_settings()
    history = await load_history(conversation_id, settings.history_messages)

    async with session_scope() as session:
        session.add(Message(conversation_id=conversation_id, role="user", content=question))
        conv = await session.get(Conversation, conversation_id)
        if conv is not None:
            conv.updated_at = utcnow()
            if not conv.title:
                conv.title = question.strip().splitlines()[0][:80]

    started = time.perf_counter()
    answer_parts: list[str] = []
    registry: SourceRegistry | None = None
    model: str | None = None
    failed = False
    completed = False  # only a fully streamed answer is saved; a disconnect mid-answer isn't
    try:
        passages = await retrieve(retrieval_query(question, history), user_id, document_ids)
        registry = SourceRegistry(passages)
        yield sse("retrieved", {"count": len(passages)})

        async for ev in get_generator().stream(question, history, passages):
            if ev.kind == "token":
                answer_parts.append(ev.text)
                yield sse("token", ev.text)
            elif ev.kind == "cite":
                numbers = [registry.number(i) for i in ev.passage_indices]
                marker = "".join(f"[{n}]" for n in numbers)
                answer_parts.append(marker)
                yield sse("cite", {"numbers": numbers, "sources": registry.sources})
            elif ev.kind == "reset":
                answer_parts.clear()
                registry = SourceRegistry(passages)
                yield sse("reset", {})
            elif ev.kind == "error":
                failed = True
                yield sse("error", {"message": ev.text})
            elif ev.kind == "done":
                model = ev.model
                completed = True
    except GeneratorNotConfigured as e:
        failed = True
        log.error("generator_not_configured", error=str(e))
        yield sse("error", {"message": str(e)})
    except Exception:
        failed = True
        log.error("answer_failed", conversation_id=str(conversation_id), exc_info=True)
        yield sse("error", {"message": "Something went wrong while answering."})
    finally:
        GENERATION_SECONDS.observe(time.perf_counter() - started)
        answer = "".join(answer_parts).strip()
        message_id = None
        if answer and completed and not failed:
            msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                citations_json=json.dumps(registry.sources if registry else []),
                model=model,
            )
            async with session_scope() as session:
                session.add(msg)
            message_id = str(msg.id)
    if not failed:
        yield sse("done", {"message_id": message_id, "model": model, "sources": registry.sources if registry else []})
