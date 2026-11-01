from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.mysql import MEDIUMTEXT

from app.core.database import Base


def now() -> datetime:
    return datetime.utcnow()


class UserAccount(Base):
    __tablename__ = "user_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(128))
    password_hash: Mapped[str] = mapped_column(String(128))
    roles_csv: Mapped[str] = mapped_column(String(256), default="ROLE_USER")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)

    sessions: Mapped[list["ChatSession"]] = relationship(back_populates="user")

    @property
    def roles(self) -> list[str]:
        return [role for role in self.roles_csv.split(",") if role]

    @roles.setter
    def roles(self, value: list[str] | set[str]) -> None:
        self.roles_csv = ",".join(sorted(value))


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(160))
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)

    user: Mapped[UserAccount] = relationship(back_populates="sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan")

    def touch(self) -> None:
        self.updated_at = now()


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        UniqueConstraint("user_id", "client_message_id", name="uq_chat_messages_user_client_message"),
        UniqueConstraint("response_dedupe_key", name="uq_chat_messages_response_dedupe_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"))
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"))
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    client_message_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    reply_to_message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    agent_trace_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    completion_status: Mapped[str] = mapped_column(String(32), default="COMPLETE", index=True)
    response_dedupe_key: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"
    __table_args__ = (UniqueConstraint("source", "version", name="uq_knowledge_document_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(256), index=True)
    version: Mapped[str] = mapped_column(String(64), default="1")
    title: Mapped[str] = mapped_column(String(256))
    domain: Mapped[str] = mapped_column(String(32), default="policy", index=True)
    school: Mapped[str] = mapped_column(String(128), default="本校")
    department: Mapped[str] = mapped_column(String(128), default="")
    source_url: Mapped[str] = mapped_column(Text, default="")
    published_on: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    effective_on: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    expires_on: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE", index=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    chunking_version: Mapped[str] = mapped_column(String(96), default="")
    content: Mapped[str] = mapped_column(Text().with_variant(MEDIUMTEXT(), "mysql"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(256), index=True)
    source_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    document_id: Mapped[Optional[int]] = mapped_column(ForeignKey("knowledge_documents.id"), nullable=True, index=True)
    parent_id: Mapped[Optional[int]] = mapped_column(ForeignKey("knowledge_chunks.id"), nullable=True, index=True)
    chunk_type: Mapped[str] = mapped_column(String(16), default="legacy", index=True)
    heading: Mapped[str] = mapped_column(String(512), default="")
    page_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    domain: Mapped[str] = mapped_column(String(32), default="support", index=True)
    school: Mapped[str] = mapped_column(String(128), default="本校")
    version: Mapped[str] = mapped_column(String(64), default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)

    @property
    def embedding_text(self) -> str:
        return f"{self.heading}\n{self.content}".strip()


class PsychologicalReport(Base):
    __tablename__ = "psychological_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"))
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"))
    message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    content: Mapped[str] = mapped_column(Text)
    intent: Mapped[str] = mapped_column(String(32))
    emotion: Mapped[str] = mapped_column(String(32))
    emotion_score: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    summary: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class RiskCase(Base):
    __tablename__ = "risk_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    risk_level: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    owner: Mapped[str] = mapped_column(String(128), default="unassigned")
    summary: Mapped[str] = mapped_column(Text)
    handoff_summary: Mapped[str] = mapped_column(Text, default="")
    acknowledged_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class CaseNote(Base):
    __tablename__ = "case_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(Integer, index=True)
    actor: Mapped[str] = mapped_column(String(128))
    note: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class AlertRecord(Base):
    __tablename__ = "alert_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(Integer, index=True)
    channel: Mapped[str] = mapped_column(String(64))
    recipient: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class ExcelRecord(Base):
    __tablename__ = "excel_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(Integer, index=True)
    file_path: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class ToolJob(Base):
    __tablename__ = "tool_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(Integer, index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    depends_on_job_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    run_after: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)
    last_error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class MemoryMaintenanceJob(Base):
    """持久化的普通记忆维护任务；安全状态写入绝不进入此异步队列。"""

    __tablename__ = "memory_maintenance_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    run_after: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)
    last_error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class DeadLetterRecord(Base):
    __tablename__ = "dead_letter_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    report_id: Mapped[int] = mapped_column(Integer, index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    reason: Mapped[str] = mapped_column(Text)
    payload: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class AgentRunTrace(Base):
    __tablename__ = "agent_run_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"), index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    report_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    intent: Mapped[str] = mapped_column(String(32), index=True)
    risk_level: Mapped[str] = mapped_column(String(32), default="LOW", index=True)
    current_risk: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    carryover_risk: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    effective_risk: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    response_mode: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    summary_version: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    covered_through_message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    safety_state_version: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    snapshot_degraded: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    original_input: Mapped[str] = mapped_column(Text)
    sanitized_input: Mapped[str] = mapped_column(Text)
    memory_brief: Mapped[str] = mapped_column(Text, default="")
    agent_steps_json: Mapped[str] = mapped_column(Text, default="[]")
    retrieved_knowledge_json: Mapped[str] = mapped_column(Text, default="[]")
    response_messages_json: Mapped[str] = mapped_column(Text, default="[]")
    assessment_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class ToolAuditRecord(Base):
    __tablename__ = "tool_audit_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    report_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    tool_name: Mapped[str] = mapped_column(String(64), index=True)
    policy: Mapped[str] = mapped_column(String(128), default="")
    allowed: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class ChatMemorySummary(Base):
    """每会话一条的持久滚动摘要投影（普通对话记忆的权威存储）。"""

    __tablename__ = "chat_memory_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"), index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), unique=True, index=True)
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    summary_text: Mapped[str] = mapped_column(Text, default="")
    covered_through_message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    summary_schema_version: Mapped[int] = mapped_column(Integer, default=1)
    version: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class RiskEvent(Base):
    """不可变风险事件流：检测、复核、状态转换、人工覆盖，只追加不原地改写。"""

    __tablename__ = "risk_events"
    __table_args__ = (UniqueConstraint("dedupe_key", name="uq_risk_events_dedupe_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    session_id: Mapped[int] = mapped_column(Integer, index=True)
    message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    report_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    event_kind: Mapped[str] = mapped_column(String(32), index=True)
    risk_type: Mapped[str] = mapped_column(String(32), default="OTHER")
    risk_level: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_summary: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(16), default="MODEL")
    previous_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    new_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    dedupe_key: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)


class SafetyState(Base):
    """会话级安全投影；原 safety_states 表保留为旧用户级状态审计。"""

    __tablename__ = "session_safety_states"
    __table_args__ = (UniqueConstraint("user_id", "session_id", name="uq_session_safety_owner"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    history_rebuilt: Mapped[bool] = mapped_column(Boolean, default=True)
    latest_session_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="RESOLVED", index=True)
    effective_risk: Mapped[str] = mapped_column(String(32), default="LOW")
    latest_event_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    last_high_risk_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_assessed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_safety_confirmation_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    next_review_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    reassessment_required: Mapped[bool] = mapped_column(Boolean, default=False)
    normal_turn_count_since_review: Mapped[int] = mapped_column(Integer, default=0)
    safe_reassessment_count: Mapped[int] = mapped_column(Integer, default=0)
    consecutive_low_assessments: Mapped[int] = mapped_column(Integer, default=0)
    latest_event_severity: Mapped[str] = mapped_column(String(64), default="")
    human_review_required: Mapped[bool] = mapped_column(Boolean, default=False)
    resolution_reason: Mapped[str] = mapped_column(Text, default="")
    version: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)
