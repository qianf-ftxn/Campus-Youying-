from __future__ import annotations

import uuid
from time import perf_counter
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from app.agents.autonomous import (
    AgentPrivateMemory,
    AgentRuntimeServices,
    ContextAgent,
    CoordinatorAgent,
    ResponseAgent,
    SafetyAgent,
    UnderstandingAgent,
)
from app.agents.coordinator import EventDrivenCoordinator
from app.agents.events import AgentEvent, AgentEventType, CollaborationBlackboard
from app.agents.registry import AgentRegistry
from app.agents.result import AgentRunResult, AgentStep
from app.core.config import Settings
from app.core.enums import IntentType, ResponseMode, RiskLevel
from app.models.entities import ChatSession, UserAccount
from app.schemas.dtos import AiMessage
from app.services.agent_models import AgentModelRegistry
from app.services.ai import AiClient, PromptTemplates
from app.services.knowledge import KnowledgeService, SearchResult
from app.services.memory import RedisShortTermMemoryStore
from app.services.memory_context import MemoryContextService, MemorySnapshot


class EventDrivenAgentRuntimeService:
    """Actor-style multi-agent runtime.

    Agents observe open tasks, claim work independently, and return the shared
    AgentRunResult contract consumed by the rest of the app.
    """

    framework_name = "event_driven_multi_agent"
    max_steps = 8

    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.ai = AiClient(settings)
        self.knowledge = KnowledgeService(db, settings)
        self.memory = RedisShortTermMemoryStore(settings)
        self.model_registry = AgentModelRegistry(settings)
        self.private_memory = AgentPrivateMemory(settings)
        self.memory_context = MemoryContextService(db, settings)

    def run(
        self,
        user: UserAccount,
        session: ChatSession,
        original_input: str,
        model_input: str,
        current_message_id: int | None = None,
        stream_event=None,
    ) -> AgentRunResult:
        started = perf_counter()
        services = AgentRuntimeServices(
            db=self.db,
            settings=self.settings,
            user=user,
            session=session,
            ai=self.ai,
            model_registry=self.model_registry,
            memory=self.memory,
            private_memory=self.private_memory,
            knowledge=self.knowledge,
            stream_event=stream_event,
        )
        coordinator_agent = CoordinatorAgent(services)
        agents = [
            UnderstandingAgent(services),
            SafetyAgent(services),
            ContextAgent(services),
            ResponseAgent(services),
        ]
        # Runtime 在 Coordinator 之前前置加载不可变 MemorySnapshot（规格 8.1/8.2）。
        snapshot: MemorySnapshot = self.memory_context.load_snapshot(
            user_id=user.id,
            session_id=session.id,
            current_message_id=current_message_id,
        )
        board = CollaborationBlackboard(
            turn_id=uuid.uuid4().hex,
            user_id=user.id,
            session_id=session.public_id,
            user_input=original_input,
            model_input=model_input,
            memory_snapshot=snapshot,
            current_message_id=current_message_id,
        )
        board = board.append_event(
            AgentEvent(
                type=AgentEventType.MEMORY_SNAPSHOT_LOADED,
                actor=coordinator_agent.name,
                message="memory snapshot loaded before coordination",
                metadata=snapshot.audit_metadata(),
            )
        )
        board = board.append_event(
            AgentEvent(
                type=AgentEventType.TURN_STARTED,
                actor=coordinator_agent.name,
                message="user turn published to shared task board",
            )
        )
        registry = AgentRegistry(agents)
        final_board = EventDrivenCoordinator(registry, coordinator_agent, self.settings).run(board)
        result = self._to_result(final_board, user, snapshot)
        result.memory_snapshot_metadata.update({"modelCalls": self.model_registry.calls,
            "modelCallCount": len(self.model_registry.calls), "elapsedMs": round((perf_counter()-started)*1000, 2)})
        return result

    def _to_result(self, board: CollaborationBlackboard, user: UserAccount, snapshot: MemorySnapshot | None = None) -> AgentRunResult:
        intent = self._select_intent(board)
        risk = self._select_risk(board)
        context = board.latest_artifact("context")
        risk_artifact = board.latest_artifact("risk")
        # 未经匹配 SafetyAgent review + Coordinator accept 的 proposal 不能作为最终输出。
        accepted = board.accepted_artifact()
        memory_brief = "无相关历史记忆。"
        retrieved: list[SearchResult] = []
        response_messages: list[AiMessage] = []
        response_text = ""
        if context:
            memory_brief = context.payload.get("memoryBrief") or memory_brief
        if accepted:
            # 引用绑定实际参与该候选生成的上下文，不能附上稍后才到达的检索结果。
            consumed_context = next((item for item in board.artifacts
                if item.kind == "context" and item.id == accepted.payload.get("knowledgeContextId")), None)
            if consumed_context:
                retrieved = consumed_context.payload.get("retrievedKnowledge") or []
            response_messages = accepted.payload.get("messages") or []
            response_text = str(accepted.payload.get("candidateText", "") or "")
            response_mode_raw = str(accepted.payload.get("mode", "support"))
        else:
            response_mode_raw = "support"
        if not response_messages:
            response_mode_raw = "safety_support" if risk == RiskLevel.HIGH else response_mode_raw
            response_messages = self._fallback_messages(intent, risk, user.display_name, board.model_input)
        if not response_text:
            response_text = self._fallback_text(intent, risk, user.display_name)
        assessment = risk_artifact.payload.get("assessment") if risk_artifact else None
        try:
            response_mode = ResponseMode(response_mode_raw)
        except ValueError:
            response_mode = ResponseMode.SUPPORT
        try:
            current_risk = RiskLevel(str((risk_artifact or {}).payload.get("currentRisk", risk.value)).upper())
            carryover_risk = RiskLevel(str((risk_artifact or {}).payload.get("carryoverRisk", RiskLevel.LOW.value)).upper())
            effective_risk = RiskLevel(str((risk_artifact or {}).payload.get("effectiveRisk", risk.value)).upper())
        except (ValueError, AttributeError):
            current_risk, carryover_risk, effective_risk = risk, RiskLevel.LOW, risk
        safety_meta = (risk_artifact or {}).payload if risk_artifact else {}
        return AgentRunResult(
            intent=intent,
            business_intent=IntentType(str(board.latest_artifact("intent").payload["intent"])) if board.latest_artifact("intent") else intent,
            risk_level=effective_risk,
            assessment=assessment,
            retrieved_knowledge=retrieved,
            response_messages=response_messages,
            response_text=response_text,
            steps=self._events_to_steps(board),
            memory_brief=memory_brief,
            current_risk=current_risk,
            carryover_risk=carryover_risk,
            effective_risk=effective_risk,
            previous_safety_status=str(safety_meta.get("previousStatus", snapshot.safety_state.status if snapshot and snapshot.safety_state else "RESOLVED")),
            new_safety_status=str(safety_meta.get("newStatus", snapshot.safety_state.status if snapshot and snapshot.safety_state else "RESOLVED")),
            response_mode=response_mode,
            memory_snapshot_metadata=snapshot.audit_metadata() if snapshot else {},
            collaboration_events=list(board.events),
            collaboration_tasks=list(board.tasks.values()),
            collaboration_artifacts=list(board.artifacts),
        )

    def _select_intent(self, board: CollaborationBlackboard) -> IntentType:
        if any(event.type == AgentEventType.SAFETY_OVERRIDE for event in board.events):
            return IntentType.RISK
        artifact = board.latest_artifact("intent")
        if not artifact:
            return IntentType.CHAT
        try:
            return IntentType(str(artifact.payload.get("intent", IntentType.CHAT.value)).upper())
        except ValueError:
            return IntentType.CHAT

    def _select_risk(self, board: CollaborationBlackboard) -> RiskLevel:
        order = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}
        highest = RiskLevel.LOW
        for artifact in board.artifacts_by_kind("risk"):
            try:
                risk = RiskLevel(str(artifact.payload.get("risk", RiskLevel.LOW.value)).upper())
            except ValueError:
                risk = RiskLevel.LOW
            if order[risk] > order[highest]:
                highest = risk
        if any(event.type == AgentEventType.SAFETY_OVERRIDE for event in board.events):
            return RiskLevel.HIGH
        return highest

    def _fallback_messages(self, intent: IntentType, risk: RiskLevel, display_name: str, model_input: str) -> list[AiMessage]:
        return [
            PromptTemplates.answer_system_prompt(intent, risk, "", display_name),
            AiMessage(role="user", content=model_input),
        ]

    def _fallback_text(self, intent: IntentType, risk: RiskLevel, display_name: str) -> str:
        if risk == RiskLevel.HIGH:
            return (
                "我很在意你现在的安全。请先不要一个人待着，也先远离可能伤害自己的物品，"
                "马上联系一位身边可信任的人、辅导员或学校心理中心陪着你。"
                "如果危险正在发生，请立即联系当地紧急服务。你现在身边有没有可以马上联系的人？"
            )
        if intent == IntentType.CONSULT and risk == RiskLevel.LOW:
            return "暂时无法可靠检索适用政策，不能确认具体标准。请稍后重试或向学校相关部门核实。"
        if risk == RiskLevel.MEDIUM or intent == IntentType.RISK:
            return "我听到了你现在的难受。我们可以先确认你此刻是否安全，再一起把最困扰你的部分慢慢拆开。"
        return "抱歉，我刚才没能可靠地完成回复。请再试一次，我会继续帮你处理这个问题。"

    def _events_to_steps(self, board: CollaborationBlackboard) -> list[AgentStep]:
        steps = []
        for index, event in enumerate(board.events, start=1):
            detail = event.message or _compact_json(event.metadata)
            if event.artifact_id:
                detail = f"{detail}; artifact={event.artifact_id}" if detail else f"artifact={event.artifact_id}"
            steps.append(AgentStep(index, event.actor, event.type.value, detail))
        return steps


def _compact_json(value: Any) -> str:
    jsonable = _to_jsonable(value)
    if not jsonable:
        return ""
    return str(jsonable)[:240]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value
