from __future__ import annotations

from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy ORM 基类"""

    pass


class User(Base):
    """
    用户表
    存储系统用户及其权限信息。
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        String(32), default="user", nullable=False
    )  # "admin" or "user"
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (Index("idx_users_username", "username", unique=True),)


class UserProfile(Base):
    """
    用户画像表
    存储用户的结构化画像数据（基本信息、技术栈、偏好、事实等）。
    """

    __tablename__ = "user_profile"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    profile_json: Mapped[dict] = mapped_column(
        JSON, nullable=False
    )  # 完整的画像 JSON 数据
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)  # 版本控制
    updated_at: Mapped[int] = mapped_column(BigInteger, nullable=False)


class UserMemoryItem(Base):
    __tablename__ = "user_memory_item"

    item_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    subkind: Mapped[str | None] = mapped_column(String(64), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    item_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_verified_at: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    updated_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_user_memory_user_kind_updated", "user_id", "kind", "updated_at"),
        Index(
            "uq_user_memory_user_kind_hash", "user_id", "kind", "item_hash", unique=True
        ),
        Index("idx_user_memory_user_session", "user_id", "session_id"),
    )


class UserMemoryEmbedding(Base):
    __tablename__ = "user_memory_embedding"

    item_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("user_memory_item.item_id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)


class ChatSession(Base):
    """
    聊天会话表
    管理用户的对话会话元数据。
    """

    __tablename__ = "chat_session"

    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    updated_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # 进度标记：记录已处理用于摘要和画像更新的消息位置
    last_summarized_msg_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )
    last_profiled_msg_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    messages: Mapped[list[ChatHistory]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (Index("idx_chat_session_user_updated", "user_id", "updated_at"),)


class ChatHistory(Base):
    """
    聊天记录表
    存储具体的对话消息内容。
    """

    __tablename__ = "chat_history"

    msg_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("chat_session.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    session: Mapped[ChatSession] = relationship(back_populates="messages")

    __table_args__ = (
        Index(
            "idx_chat_history_user_session_time", "user_id", "session_id", "created_at"
        ),
        Index("idx_chat_history_session_msg", "session_id", "msg_id"),
    )


class Document(Base):
    """
    文档表
    管理知识库中上传的文件记录。
    """

    __tablename__ = "document"

    doc_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_path: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    checksum: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )  # 文件哈希，用于去重
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    parents: Mapped[list[DocContent]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    knowledge_base_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    updated_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (
        Index("idx_knowledge_base_user_updated", "user_id", "updated_at"),
    )


class KnowledgeBaseDocument(Base):
    __tablename__ = "knowledge_base_document"

    doc_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document.doc_id", ondelete="CASCADE"),
        primary_key=True,
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("knowledge_base.knowledge_base_id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (
        Index("idx_knowledge_base_document_kb", "knowledge_base_id"),
    )


class DocContent(Base):
    """
    文档内容表 (Parent Chunks)
    存储文档的父级切片内容，用于 Parent Retrieval 策略。
    """

    __tablename__ = "doc_content"

    parent_chunk_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    doc_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("document.doc_id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)  # 较大的文本块
    page_num: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    document: Mapped[Document] = relationship(back_populates="parents")

    __table_args__ = (Index("idx_doc_content_doc", "doc_id"),)


class DocEmbedding(Base):
    __tablename__ = "doc_embedding"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    doc_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    parent_chunk_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    child_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (
        Index("idx_doc_embedding_doc", "doc_id"),
    )


class HarnessRun(Base):
    __tablename__ = "harness_run"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    policy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    input_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    current_step: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    resume_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    approval_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verification_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    updated_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    finished_at: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    __table_args__ = (
        Index("idx_harness_run_user_created", "user_id", "created_at"),
        Index("idx_harness_run_status_updated", "status", "updated_at"),
    )


class HarnessRunRuntimeState(Base):
    __tablename__ = "harness_run_runtime_state"

    run_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("harness_run.run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    review_state_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    continuation_state_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    research_state_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    updated_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (
        Index("idx_harness_runtime_state_updated", "updated_at"),
    )


class HarnessRunRuntimeStateHistory(Base):
    __tablename__ = "harness_run_runtime_state_history"

    history_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("harness_run.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    transition_type: Mapped[str] = mapped_column(String(64), nullable=False)
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    runtime_state_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (
        Index("idx_harness_runtime_history_run_version", "run_id", "version"),
        Index("idx_harness_runtime_history_run_created", "run_id", "created_at"),
    )


class HarnessAgentProject(Base):
    __tablename__ = "harness_agent_project"

    project_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    graph_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    updated_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (
        Index("idx_harness_agent_project_user_updated", "user_id", "updated_at"),
    )


class HarnessApproval(Base):
    __tablename__ = "harness_approval"

    approval_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("harness_run.run_id", ondelete="CASCADE"), nullable=False
    )
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    resolved_at: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    __table_args__ = (
        Index("idx_harness_approval_run", "run_id"),
        Index("idx_harness_approval_status_created", "status", "created_at"),
    )


class HarnessVerification(Base):
    __tablename__ = "harness_verification"

    verification_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("harness_run.run_id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    checks_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    artifacts_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (
        Index("idx_harness_verification_run", "run_id"),
        Index("idx_harness_verification_status_created", "status", "created_at"),
    )


class HarnessEvent(Base):
    __tablename__ = "harness_event"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    event_source: Mapped[str] = mapped_column(String(32), nullable=False)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    run_id: Mapped[str | None] = mapped_column(
        String(128), ForeignKey("harness_run.run_id", ondelete="CASCADE"), nullable=True
    )
    actor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    details_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (
        Index("idx_harness_event_run_created", "run_id", "created_at"),
        Index("idx_harness_event_session_created", "session_id", "created_at"),
        Index("idx_harness_event_user_created", "user_id", "created_at"),
        Index("idx_harness_event_type_created", "event_type", "created_at"),
    )


class HarnessModelProvider(Base):
    __tablename__ = "harness_model_provider"

    provider_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    base_url: Mapped[str] = mapped_column(String(512), nullable=False)
    api_key_encrypted: Mapped[str] = mapped_column(String(512), nullable=False)
    models_json: Mapped[list] = mapped_column(JSON, nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    updated_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (
        Index("idx_harness_model_provider_user_updated", "user_id", "updated_at"),
    )
