import uuid

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from citewise.answering import answer_stream
from citewise.db import get_session
from citewise.deps import Pagination, current_user, not_found
from citewise.models import Conversation, Message, User, utcnow
from citewise.schemas import AskRequest, ConversationCreate, ConversationOut, ConversationUpdate, MessageOut, Page

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


async def _owned(session: AsyncSession, user: User, conversation_id: uuid.UUID) -> Conversation:
    conv = await session.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        raise not_found("Conversation")
    return conv


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
async def create(
    body: ConversationCreate | None = None,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    conv = Conversation(user_id=user.id, title=body.title if body else None)
    session.add(conv)
    await session.commit()
    return ConversationOut.of(conv)


@router.get("", response_model=Page[ConversationOut])
async def list_conversations(
    page: Pagination = Depends(), user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    where = Conversation.user_id == user.id
    total = (await session.execute(select(func.count()).select_from(Conversation).where(where))).scalar_one()
    rows = (
        (
            await session.execute(
                select(Conversation)
                .where(where)
                .order_by(Conversation.updated_at.desc(), Conversation.id)
                .limit(page.limit)
                .offset(page.offset)
            )
        )
        .scalars()
        .all()
    )
    return Page(items=[ConversationOut.of(c) for c in rows], total=total, limit=page.limit, offset=page.offset)


@router.get("/{conversation_id}", response_model=ConversationOut)
async def get(
    conversation_id: uuid.UUID, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    return ConversationOut.of(await _owned(session, user, conversation_id))


@router.patch("/{conversation_id}", response_model=ConversationOut)
async def rename(
    conversation_id: uuid.UUID,
    body: ConversationUpdate,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    conv = await _owned(session, user, conversation_id)
    conv.title = body.title
    conv.updated_at = utcnow()
    await session.commit()
    return ConversationOut.of(conv)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(
    conversation_id: uuid.UUID, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    conv = await _owned(session, user, conversation_id)
    await session.delete(conv)
    await session.commit()


@router.get("/{conversation_id}/messages", response_model=Page[MessageOut])
async def messages(
    conversation_id: uuid.UUID,
    page: Pagination = Depends(),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    """Oldest-first, so offset=0 is the start of the conversation."""
    await _owned(session, user, conversation_id)
    where = Message.conversation_id == conversation_id
    total = (await session.execute(select(func.count()).select_from(Message).where(where))).scalar_one()
    rows = (
        (
            await session.execute(
                select(Message)
                .where(where)
                .order_by(Message.created_at, Message.id)
                .limit(page.limit)
                .offset(page.offset)
            )
        )
        .scalars()
        .all()
    )
    return Page(items=[MessageOut.of(m) for m in rows], total=total, limit=page.limit, offset=page.offset)


@router.post("/{conversation_id}/messages")
async def ask(
    conversation_id: uuid.UUID,
    body: AskRequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    """Server-Sent Events: retrieved -> token* / cite* -> done (or error)."""
    await _owned(session, user, conversation_id)
    return StreamingResponse(
        answer_stream(conversation_id, user.id, body.content, body.document_ids),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
