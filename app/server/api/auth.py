import time
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.infrastructure.config.settings import settings
from app.infrastructure.database.models import User
from app.infrastructure.database.orm import get_sessionmaker
from app.infrastructure.utils.security import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])

AUTH_COOKIE_NAME = "agframe_access_token"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class UserCreate(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    username: str
    role: str
    is_active: bool

    class Config:
        from_attributes = True


def get_db():
    SessionLocal = get_sessionmaker()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _request_is_secure(request: Request) -> bool:
    forwarded_proto = str(request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    if forwarded_proto:
        return forwarded_proto == "https"
    return request.url.scheme == "https"


def _set_auth_cookie(response: Response, *, request: Request, access_token: str) -> None:
    max_age_seconds = int(settings.auth.access_token_expire_minutes) * 60
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=access_token,
        max_age=max_age_seconds,
        expires=max_age_seconds,
        httponly=True,
        secure=_request_is_secure(request),
        samesite="lax",
        path="/",
    )


def _clear_auth_cookie(response: Response, *, request: Request) -> None:
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        httponly=True,
        secure=_request_is_secure(request),
        samesite="lax",
        path="/",
    )


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    db: Session = Depends(get_db),
    cookie_token: Annotated[str | None, Cookie(alias=AUTH_COOKIE_NAME)] = None,
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    resolved_token = str(token or cookie_token or "").strip()
    if not resolved_token:
        raise credentials_exception
    payload = decode_access_token(resolved_token)
    if payload is None:
        raise credentials_exception
    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise credentials_exception
    username = subject

    stmt = select(User).where(User.username == username)
    user = db.execute(stmt).scalar_one_or_none()

    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_current_admin_user(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized",
        )
    return current_user


def _require_bootstrap_token(first_user_missing: bool, provided_bootstrap_token: str | None) -> None:
    if not first_user_missing:
        return
    configured_token = str(settings.auth.bootstrap_admin_token or "").strip()
    if not configured_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Bootstrap admin token is not configured",
        )
    if str(provided_bootstrap_token or "").strip() != configured_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bootstrap admin token is required to create the first admin user",
        )


@router.post("/token", response_model=Token)
async def login_for_access_token(
    response: Response,
    request: Request,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Session = Depends(get_db),
):
    stmt = select(User).where(User.username == form_data.username)
    user = db.execute(stmt).scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=timedelta(minutes=settings.auth.access_token_expire_minutes),
    )
    _set_auth_cookie(response, request=request, access_token=access_token)
    return {"access_token": access_token, "token_type": "bearer"}  # nosec B105


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response, request: Request):
    _clear_auth_cookie(response, request=request)
    response.status_code = status.HTTP_204_NO_CONTENT


@router.post("/register", response_model=UserResponse)
async def register_user(
    user_in: UserCreate,
    db: Session = Depends(get_db),
    x_bootstrap_admin_token: str | None = Header(default=None),
):
    stmt = select(User).where(User.username == user_in.username)
    existing_user = db.execute(stmt).scalar_one_or_none()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_password = get_password_hash(user_in.password)
    first_user = db.execute(select(User).limit(1)).scalar_one_or_none()
    is_first_user = first_user is None

    _require_bootstrap_token(is_first_user, x_bootstrap_admin_token)

    if not is_first_user and not bool(settings.auth.allow_open_registration):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Open registration is disabled",
        )

    role = "admin" if is_first_user else "user"
    new_user = User(
        username=user_in.username,
        hashed_password=hashed_password,
        role=role,
        is_active=True,
        created_at=int(time.time()),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.get("/users/me", response_model=UserResponse)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    return current_user
