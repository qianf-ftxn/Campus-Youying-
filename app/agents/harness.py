from __future__ import annotations

import logging
import json
import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.agents.factory import create_agent_runtime
from app.agents.result import AgentRunResult, AgentStep
from app.core.config import Settings
from app.core.enums import IntentType, MessageRole, ResponseMode, RiskLevel
from app.models.entities import AgentRunTrace, ChatMessage, ChatSession, PsychologicalReport, UserAccount
from app.schemas.dtos import AiMessage, ChatRequest
from app.services.assessment import PsychologyAssessment
from app.services.knowledge import SearchResult
from app.services.mcp_client import MindBridgeMcpToolClient
from app.services.memory import RedisShortTermMemoryStore
from app.services.memory_context import MemoryContextService
from app.services.memory_maintenance import MemoryMaintenanceQueue
from app.services.privacy import PrivacySanitizer
from app.services.safety_memory import SafetyMemoryService, SafetyTurnDecision
from app.services.tool_queue import ToolQueueService
from app.services.trace import AgentTraceService


logger = logging.getLogger(__name__)


@dataclass
class AgentToolPlan:
    report_id: int | None
    risk_level: str | None

    @property
    def requires_tools(self) -> bool:
        return self.report_id is not None


@dataclass
class AgentHarnessOutcome:
    session: ChatSession
    original_input: str
    model_input: str
    intent: IntentType
    risk_level: str | None
    assessment: PsychologyAssessment | None
    response_messages: list[AiMessage]
    response_text: str
    agent_steps: list[AgentStep]
    retrieved_knowledge: list[SearchResult]
    report_id: int | None
    tool_plan: AgentToolPlan
    trace_id: int | None
    current_message_id: int | None = None
    safety_decision: SafetyTurnDecision | None = None
    effective_risk: str = "LOW"
    response_mode: str = "support"
    memory_snapshot_metadata: dict | None = None
    cached: bool = False


class MindBridgeAgentHarness:
    """Runtime harness for one MindBridge agent turn.

    The harness owns business orchestration around the agent runtime. HTTP/SSE
    code can stay thin while this class manages input preparation, persistence,
    risk report creation, tool planning, and trace data.
    """

    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.privacy = PrivacySanitizer()
        self.memory = RedisShortTermMemoryStore(settings)
        self.memory_context = MemoryContextService(db, settings)

    def run(self, user: UserAccount, request: ChatRequest, stream_event=None) -> AgentHarnessOutcome:
        original_input = request.message.strip()
        model_input = self.privacy.sanitize(original_input)
        # clientMessageId 重试幂等（规格第 9 节）：用户维度全局查重，
        # 网络重试即使没带 sessionId 也会复用原会话，不会写入重复消息。
        user_message = None
        session = None
        if request.clientMessageId:
            user_message = (
                self.db.query(ChatMessage)
                .filter(
                    ChatMessage.user_id == user.id,
                    ChatMessage.client_message_id == request.clientMessageId,
                    ChatMessage.role == MessageRole.USER.value,
                )
                .first()
            )
            if user_message is not None:
                session = self.db.query(ChatSession).filter(ChatSession.id == user_message.session_id).first()
                cached_reply = (
                    self.db.query(ChatMessage)
                    .filter(
                        ChatMessage.reply_to_message_id == user_message.id,
                        ChatMessage.role == MessageRole.ASSISTANT.value,
                        ChatMessage.completion_status == "COMPLETE",
                    )
                    .order_by(ChatMessage.id.desc())
                    .first()
                )
                if cached_reply is not None and session is not None:
                    return self._cached_outcome(session, user_message, cached_reply)
        if session is None:
            session = self._resolve_session(user, request.sessionId, original_input)
        if user_message is None:
            try:
                user_message = self.save_message(
                    user, session, MessageRole.USER, original_input, client_message_id=request.clientMessageId
                )
            except IntegrityError:
                self.db.rollback()
                if not request.clientMessageId:
                    raise
                user_message = (
                    self.db.query(ChatMessage)
                    .filter(
                        ChatMessage.user_id == user.id,
                        ChatMessage.client_message_id == request.clientMessageId,
                        ChatMessage.role == MessageRole.USER.value,
                    )
                    .first()
                )
                if user_message is None:
                    raise
                session = self.db.query(ChatSession).filter(ChatSession.id == user_message.session_id).first()
        if stream_event:
            stream_event({"type": "meta", "sessionId": session.public_id, "_messageId": user_message.id})
        # 下一轮模型调用前的强制兜底：常规压缩 + token 高水位（规格 6.8）。
        budget_actions: list[str] = []
        try:
            from app.services.memory_context import estimate_tokens

            _, budget_actions = self.memory_context.rolling.ensure_prompt_budget(
                session.id,
                fixed_tokens=800,
                rag_tokens=self.settings.rag_skill_budget_tokens,
                current_input_tokens=estimate_tokens(original_input),
            )
        except Exception as exc:
            logger.warning("pre-call prompt budget check failed for session=%s: %s", session.public_id, exc)
            budget_actions = [f"prompt budget check failed: {type(exc).__name__}"]
        agent_run = create_agent_runtime(self.db, self.settings).run(
            user, session, original_input, model_input, current_message_id=user_message.id, stream_event=stream_event
        )
        agent_run.memory_snapshot_metadata["promptBudgetActions"] = budget_actions
        # SafetyAgent 已在候选正文生成前完成持久化；这里按 message_id 幂等读取本轮 decision。
        safety_decision = self._persist_safety_memory(user, session, user_message, agent_run)
        report = self._create_report(user, session, original_input, agent_run, message_id=user_message.id)
        risk_level = agent_run.effective_risk.value
        trace = AgentTraceService(self.db).save_run(
            user=user,
            session=session,
            original_input=original_input,
            sanitized_input=model_input,
            memory_brief=agent_run.memory_brief,
            agent_run=agent_run,
            report_id=report.id if report is not None else None,
            message_id=user_message.id,
        )
        tool_plan = AgentToolPlan(report_id=report.id if report is not None else None, risk_level=risk_level)
        return AgentHarnessOutcome(
            session=session,
            original_input=original_input,
            model_input=model_input,
            intent=agent_run.intent,
            risk_level=risk_level,
            assessment=agent_run.assessment,
            response_messages=agent_run.response_messages,
            response_text=agent_run.response_text,
            agent_steps=agent_run.steps,
            retrieved_knowledge=agent_run.retrieved_knowledge,
            report_id=report.id if report is not None else None,
            tool_plan=tool_plan,
            trace_id=trace.id,
            current_message_id=user_message.id,
            safety_decision=safety_decision,
            effective_risk=risk_level,
            response_mode=agent_run.response_mode.value,
            memory_snapshot_metadata=agent_run.memory_snapshot_metadata,
        )

    def save_assistant_message(
        self,
        user: UserAccount,
        session: ChatSession,
        content: str,
        reply_to_message_id: int | None = None,
        completion_status: str = "COMPLETE",
    ) -> None:
        try:
            self.save_message(
                user,
                session,
                MessageRole.ASSISTANT,
                content,
                reply_to_message_id=reply_to_message_id,
                completion_status=completion_status,
                response_dedupe_key=(
                    f"message:{reply_to_message_id}:complete"
                    if reply_to_message_id is not None and completion_status == "COMPLETE"
                    else None
                ),
            )
        except IntegrityError:
            self.db.rollback()
            # 同一请求的并发重试已经保存了完整回复，视为幂等成功。
            if reply_to_message_id is None or completion_status != "COMPLETE":
                raise
        # assistant 保存成功后执行普通记忆维护（12→8 批量压缩，规格 6.8）。
        if completion_status == "COMPLETE":
            if self.settings.long_term_memory_enabled and reply_to_message_id is not None:
                try:
                    from app.services.long_term_memory import LongTermMemoryService
                    LongTermMemoryService(self.db, self.settings).enqueue(user.id, reply_to_message_id)
                except Exception as exc:
                    self.db.rollback()
                    logger.warning("Personal memory enqueue failed code=%s", type(exc).__name__)
            try:
                if self.settings.memory_maintenance_queue_enabled:
                    MemoryMaintenanceQueue(self.db, self.settings).enqueue(session.id)
                else:
                    self.memory_context.compact_after_turn(session.id)
            except Exception as exc:
                # 入队失败时同步维护，确保普通记忆不会因队列故障永久遗漏。
                logger.warning("memory maintenance enqueue failed; running synchronously session=%s: %s", session.public_id, exc)
                self.db.rollback()
                try:
                    self.memory_context.compact_after_turn(session.id)
                except Exception as fallback_exc:
                    logger.error(
                        "synchronous memory maintenance fallback failed session=%s: %s",
                        session.public_id,
                        fallback_exc,
                        exc_info=True,
                    )

    async def dispatch_tools(self, tool_plan: AgentToolPlan) -> list[str]:
        if tool_plan.report_id is None:
            return []
        if self.settings.tool_queue_enabled:
            ToolQueueService(self.db, self.settings).enqueue_report(tool_plan.report_id, tool_plan.risk_level)
            return ["queued"]
        return await MindBridgeMcpToolClient(self.settings).handle_report(tool_plan.report_id, tool_plan.risk_level)

    def save_message(
        self,
        user: UserAccount,
        session: ChatSession,
        role: MessageRole,
        content: str,
        client_message_id: str | None = None,
        reply_to_message_id: int | None = None,
        completion_status: str = "COMPLETE",
        response_dedupe_key: str | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            user_id=user.id,
            session_id=session.id,
            role=role.value,
            content=content,
            client_message_id=client_message_id,
            reply_to_message_id=reply_to_message_id,
            completion_status=completion_status,
            response_dedupe_key=response_dedupe_key,
        )
        if role == MessageRole.ASSISTANT and reply_to_message_id is not None:
            parent = self.db.get(ChatMessage, reply_to_message_id)
            if parent and parent.user_id == user.id and parent.session_id == session.id:
                message.agent_trace_id = parent.agent_trace_id
        self.db.add(message)
        session.touch()
        self.db.add(session)
        self.db.commit()
        self.db.refresh(message)
        self.memory.append(session.public_id, role.value, content)
        return message

    def _persist_safety_memory(
        self,
        user: UserAccount,
        session: ChatSession,
        message: ChatMessage,
        agent_run: AgentRunResult,
    ) -> SafetyTurnDecision | None:
        if not self.settings.safety_memory_enabled:
            return None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                assessment = agent_run.assessment
                current_risk = assessment.risk if assessment is not None else agent_run.current_risk
                return SafetyMemoryService(self.db, self.settings).record_turn_assessment(
                    user_id=user.id,
                    session_id=session.id,
                    message_id=message.id,
                    report_id=None,
                    current_risk=current_risk,
                    confidence=assessment.confidence if assessment is not None else 0.6,
                    summary=assessment.summary if assessment is not None else "",
                    safety_reassessment=assessment.safety_reassessment if assessment is not None else False,
                    protective_factors=assessment.protective_factors if assessment is not None else "",
                    has_plan_or_intent=assessment.has_plan_or_intent if assessment is not None else False,
                    risk_type=assessment.risk_type if assessment is not None else "OTHER",
                    evidence_summary=assessment.evidence_summary if assessment is not None else message.content,
                    source="MODEL",
                )
            except Exception as exc:
                self.db.rollback()
                last_error = exc
                logger.warning(
                    "safety memory persistence attempt %s/3 failed session=%s user=%s: %s",
                    attempt + 1,
                    session.public_id,
                    user.id,
                    exc,
                )
        # 权威安全状态不可写时不能继续普通回复，否则下一轮会遗忘本轮风险。
        logger.error(
            "safety memory persistence exhausted retries session=%s user=%s",
            session.public_id,
            user.id,
            exc_info=(type(last_error), last_error, last_error.__traceback__) if last_error else False,
        )
        raise RuntimeError("安全状态暂时无法可靠保存，请稍后重试") from last_error

    def _cached_outcome(
        self,
        session: ChatSession,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
    ) -> AgentHarnessOutcome:
        # 重试复用已输出正文，也复用当时的证据，避免政策更新后引用错版本。
        trace_id = assistant_message.agent_trace_id or user_message.agent_trace_id
        trace = self.db.query(AgentRunTrace).filter_by(
            id=trace_id, user_id=session.user_id, session_id=session.id,
        ).first() if trace_id else None
        if trace is None:
            trace = self.db.query(AgentRunTrace).filter(
                AgentRunTrace.user_id == session.user_id,
                AgentRunTrace.session_id == session.id, AgentRunTrace.original_input == user_message.content,
                AgentRunTrace.created_at >= user_message.created_at,
                AgentRunTrace.created_at <= assistant_message.created_at,
            ).order_by(AgentRunTrace.id.desc()).first()
        knowledge = []
        if trace:
            from dataclasses import fields
            allowed = {f.name for f in fields(SearchResult)}
            try:
                knowledge = [SearchResult(**{k: v for k, v in row.items() if k in allowed})
                             for row in json.loads(trace.retrieved_knowledge_json or "[]")]
            except (ValueError, TypeError):
                knowledge = []
        return AgentHarnessOutcome(
            session=session,
            original_input=user_message.content,
            model_input=self.privacy.sanitize(user_message.content),
            intent=IntentType(trace.intent) if trace else IntentType.CHAT,
            risk_level=(trace.risk_level or RiskLevel.LOW.value) if trace else RiskLevel.LOW.value,
            assessment=None,
            response_messages=[],
            response_text=assistant_message.content,
            agent_steps=[],
            retrieved_knowledge=knowledge,
            report_id=None,
            tool_plan=AgentToolPlan(report_id=None, risk_level=None),
            trace_id=trace.id if trace else None,
            current_message_id=user_message.id,
            effective_risk=(trace.effective_risk or trace.risk_level or RiskLevel.LOW.value) if trace else RiskLevel.LOW.value,
            response_mode=(trace.response_mode or ResponseMode.SUPPORT.value) if trace else ResponseMode.NORMAL_CHAT.value,
            memory_snapshot_metadata=None,
            cached=True,
        )

    def _resolve_session(self, user: UserAccount, public_id: str | None, text: str) -> ChatSession:
        if public_id:
            session = self.db.query(ChatSession).filter(ChatSession.public_id == public_id, ChatSession.user_id == user.id).first()
            if session is None:
                raise ValueError("Session not found")
            return session
        session = ChatSession(public_id=uuid.uuid4().hex, user_id=user.id, title=text[:36])
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def _create_report(
        self,
        user: UserAccount,
        session: ChatSession,
        text: str,
        agent_run: AgentRunResult,
        message_id: int | None = None,
    ) -> PsychologicalReport | None:
        if not agent_run.requires_report or agent_run.assessment is None:
            return None
        if message_id is not None:
            existing = self.db.query(PsychologicalReport).filter_by(user_id=user.id, message_id=message_id).first()
            if existing:
                return existing
        report = PsychologicalReport(
            user_id=user.id,
            session_id=session.id,
            message_id=message_id,
            content=text,
            intent=agent_run.intent.value,
            emotion=agent_run.assessment.emotion.value,
            emotion_score=agent_run.assessment.emotion_score,
            risk_level=agent_run.effective_risk.value,
            confidence=agent_run.assessment.confidence,
            summary=agent_run.assessment.summary,
        )
        self.db.add(report)
        self.db.commit()
        self.db.refresh(report)
        return report
