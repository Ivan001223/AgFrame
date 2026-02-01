from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import select
from pydantic import BaseModel
import time

from app.infrastructure.database.orm import get_sessionmaker
from app.infrastructure.database.models import User
from app.infrastructure.utils.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    decode_access_token,
)
from app.infrastructure.config.config_manager import config_manager

router = APIRouter(prefix="/auth", tags=["auth"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


# --- Pydantic Schemas ---
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


# --- Dependencies ---
def get_db():
    SessionLocal = get_sessionmaker()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)], db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception

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


# --- Routes ---


@router.post("/token", response_model=Token)
async def login_for_access_token(
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

    auth_config = config_manager.get_config().get("auth", {})
    access_token_expires_minutes = auth_config.get("access_token_expire_minutes", 30)

    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=None,  # uses default in utils
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/register", response_model=UserResponse)
async def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    stmt = select(User).where(User.username == user_in.username)
    existing_user = db.execute(stmt).scalar_one_or_none()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_password = get_password_hash(user_in.password)
    # 默认第一个注册的是 admin (简化逻辑)，或者硬编码
    # 这里简单起见，默认 user，如果想测试 admin，需手动改数据库或增加 admin_token
    # 为了演示，如果数据库是空的，或者是第一个用户，设为 admin?
    # 不，先默认 user。可通过 header 或 secret code 注册 admin (ToDo)

    new_user = User(
        username=user_in.username,
        hashed_password=hashed_password,
        role="user",
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
