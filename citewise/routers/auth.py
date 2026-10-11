import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from citewise.config import get_settings
from citewise.db import get_session
from citewise.deps import current_user
from citewise.models import RefreshToken, User, utcnow
from citewise.schemas import Credentials, RefreshRequest, TokenPair, UserOut
from citewise.security import (
    DUMMY_PASSWORD_HASH,
    create_access_token,
    hash_password,
    hash_refresh_token,
    new_refresh_token,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])
limiter = Limiter(key_func=get_remote_address, enabled=get_settings().env != "testing")


async def _issue(session: AsyncSession, user_id: uuid.UUID, family_id: uuid.UUID | None = None) -> TokenPair:
    settings = get_settings()
    access, expires_in = create_access_token(user_id)
    refresh, refresh_hash = new_refresh_token()
    session.add(
        RefreshToken(
            user_id=user_id,
            family_id=family_id or uuid.uuid4(),
            token_hash=refresh_hash,
            expires_at=utcnow() + timedelta(days=settings.refresh_token_days),
        )
    )
    await session.commit()
    return TokenPair(access_token=access, refresh_token=refresh, expires_in=expires_in)


async def _revoke_family(session: AsyncSession, family_id: uuid.UUID) -> None:
    await session.execute(
        update(RefreshToken)
        .where(RefreshToken.family_id == family_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=utcnow())
    )


def _aware(dt: datetime) -> datetime:
    # SQLite drops tzinfo on read; values are always written in UTC.
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
async def register(request: Request, body: Credentials, session: AsyncSession = Depends(get_session)):
    user = User(email=body.email, password_hash=hash_password(body.password))
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists") from None
    return UserOut.model_validate(user, from_attributes=True)


@router.post("/login", response_model=TokenPair)
@limiter.limit("10/minute")
async def login(request: Request, body: Credentials, session: AsyncSession = Depends(get_session)):
    user = (await session.execute(select(User).where(User.email == body.email))).scalar_one_or_none()
    # Always run bcrypt so response time doesn't reveal whether the email is registered.
    valid = verify_password(body.password, user.password_hash if user else DUMMY_PASSWORD_HASH)
    if user is None or not valid:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    return await _issue(session, user.id)


@router.post("/refresh", response_model=TokenPair)
@limiter.limit("30/minute")
async def refresh(request: Request, body: RefreshRequest, session: AsyncSession = Depends(get_session)):
    token = (
        await session.execute(
            select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(body.refresh_token))
        )
    ).scalar_one_or_none()
    invalid = HTTPException(status_code=401, detail="Invalid or expired refresh token")
    if token is None:
        raise invalid
    if token.revoked_at is not None:
        # A rotated token was presented again: it was stolen or replayed. Kill the whole chain.
        await _revoke_family(session, token.family_id)
        await session.commit()
        raise invalid
    if _aware(token.expires_at) <= utcnow():
        raise invalid
    token.revoked_at = utcnow()
    return await _issue(session, token.user_id, family_id=token.family_id)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(body: RefreshRequest, session: AsyncSession = Depends(get_session)):
    token = (
        await session.execute(
            select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(body.refresh_token))
        )
    ).scalar_one_or_none()
    if token is not None:
        await _revoke_family(session, token.family_id)
        await session.commit()


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(current_user)):
    return UserOut.model_validate(user, from_attributes=True)
