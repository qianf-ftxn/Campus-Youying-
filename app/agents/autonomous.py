from __future__ import annotations

import uuid
import re
import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from app.agents.events import (
    AgentArtifact,
    AgentEvent,
    AgentEventType,
    AgentMessage,
    AgentTask,
    AgentTurnResult,
    CollaborationBlackboard,
    TaskPriority,
)
from app.agents.registry import AgentCapability, AgentDecision, AgentProfile
from app.core.config import Settings
from app.core.enums import IntentType, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.agent_models import AgentModelRegistry
from app.services.ai import AiClient, PromptTemplates, has_high_risk_signal
from app.services.assessment import PsychologicalAssessmentService
from app.services.intent_rules import classify_intent_rules, is_general_request, parse_intent_label
from app.services.streaming import (INTERNAL_SAFETY_LABELS, PHONE_PATTERN, SAFETY_OPENING,
    MONITORING_OPENING, collect_guarded_stream, verified_contact_numbers)

if TYPE_CHECKING:
    from app.models.entities import ChatSession, UserAccount
    from app.services.knowledge import KnowledgeService, SearchResult
    from app.services.memory import RedisShortTermMemoryStore


@dataclass
class AgentRuntimeServices:
    db: Session
    settings: Settings
    user: UserAccount
    session: ChatSession
    ai: AiClient
    model_registry: AgentModelRegistry
    memory: RedisShortTermMemoryStore
    private_memory: "AgentPrivateMemory"
    knowledge: KnowledgeService
    stream_event: Any = None


class AgentPrivateMemory:
    """Per-agent memory facade backed by isolated Redis keys."""

    def __init__(self, settings: Settings):
        from app.services.memory import RedisShortTermMemoryStore

        self.store = RedisShortTermMemoryStore(settings)

    def load(self, agent_name: str, session_public_id: str) -> list[AiMessage]:
        return self.store.load_recent(self._key(agent_name, session_public_id))

    def append(self, agent_name: str, session_public_id: str, content: str) -> None:
        self.store.append(self._key(agent_name, session_public_id), "system", content)

    def _key(self, agent_name: str, session_public_id: str) -> str:
        return f"agent:{agent_name}:{session_public_id}"


class BaseAutonomousAgent:
    profile: AgentProfile

    def __init__(self, services: AgentRuntimeServices):
        self.services = services

    @property
    def name(self) -> str:
        return self.profile.name

    def client(self) -> AiClient:
        return self.services.model_registry.client_for(self.name)

    def private_memory(self) -> list[AiMessage]:
        return self.services.private_memory.load(self.name, self.services.session.public_id)

    def remember(self, content: str) -> None:
        self.services.private_memory.append(self.name, self.services.session.public_id, content)

    def _artifact(
        self,
        kind: str,
        payload: dict[str, Any],
        task: AgentTask,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentArtifact:
        return AgentArtifact(
            id=f"{self.name}:{kind}:{uuid.uuid4().hex[:10]}",
            owner=self.name,
            kind=kind,
            payload=payload,
            confidence=confidence,
            task_id=task.id,
            metadata=metadata or {},
        )


class UnderstandingAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="UnderstandingAgent",
        capabilities=frozenset({AgentCapability.UNDERSTANDING}),
        system_prompt=(
            "你是 UnderstandingAgent。你只负责理解用户当前请求，不生成最终回复，不做风险处置。"
            "只输出 CHAT、CONSULT、RISK 三个标签之一，不输出 JSON、解释、主题或置信度。"
        ),
        memory_policy="private_intent_history",
        model_profile="understanding",
        tool_permissions=frozenset({"llm.intent"}),
    )

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        if board.latest_artifact("intent"):
            return AgentDecision(False, reason="intent artifact already exists")
        if self._is_directed(task, board):
            return AgentDecision(True, 0.82, "open user-turn task needs understanding")
        return AgentDecision(False, reason="task does not need understanding")

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        intent = self._classify(board.model_input or board.user_input, board)
        confidence = 0.92 if intent == IntentType.RISK else 0.78
        payload = {
            "intent": intent.value,
            "topic": self._topic(board.model_input or board.user_input, intent),
            "reason": getattr(self, "routing_reason", "autonomous intent proposal"),
            "requiresPolicyRetrieval": intent == IntentType.CONSULT,
            "routingSource": self.routing_source,
            "ruleId": self.routing_rule_id,
            "modelLabel": self.routing_model_label,
            "fallbackReason": self.routing_fallback_reason,
            "privateMemoryKey": self.services.private_memory._key(self.name, self.services.session.public_id),
        }
        self.remember(f"intent={intent.value}; topic={payload['topic']}")
        return AgentTurnResult(
            artifacts=(self._artifact("intent", payload, task, confidence),),
            messages=(
                AgentMessage(
                    id=f"msg:{uuid.uuid4().hex[:10]}",
                    sender=self.name,
                    recipient="*",
                    task_id=task.id,
                    kind="PROPOSAL",
                    content=f"我判断本轮意图是 {intent.value}",
                ),
            ),
        )

    def _is_directed(self, task: AgentTask, board: CollaborationBlackboard) -> bool:
        if AgentCapability.UNDERSTANDING.value in task.required_capabilities:
            return True
        return bool(board.user_input and task.metadata.get("kind") in {"root", "understanding"})

    def _classify(self, text: str, board: CollaborationBlackboard) -> IntentType:
        self.routing_source = "rule"
        self.routing_model_label = ""
        self.routing_fallback_reason = ""
        if has_high_risk_signal(text):
            self.routing_rule_id = "personal_danger"
            self.routing_reason = "current personal danger signal"
            return IntentType.RISK
        snapshot = board.memory_snapshot
        previous = [m.content for m in (snapshot.recent_messages if snapshot else ()) if m.role == "user"]
        decision = classify_intent_rules(text, previous)
        self.routing_rule_id = decision.rule_id
        if decision.intent is not None:
            self.routing_reason = decision.rule_id
            return decision.intent
        try:
            history = _snapshot_context_messages(board, limit=8)
            memory_context = "\n".join(item.content for item in self.private_memory()[-6:])
            messages = [
                *PromptTemplates.intent_prompt(history, text),
                AiMessage(role="system", content=f"{self.profile.system_prompt}\n私有记忆：\n{memory_context or '无'}"),
            ]
            intent = parse_intent_label(self.client().complete(messages))
            # 规则 UNKNOWN 后不再覆盖合法语义结果；不从解释文字中猜测意图。
            if intent is not None:
                self.routing_source = "model"
                self.routing_model_label = intent.value
                self.routing_reason = f"semantic classification: {intent.value}"
                return intent
            self.routing_fallback_reason = "invalid_model_label"
        except Exception as exc:
            self.routing_fallback_reason = f"model_error:{type(exc).__name__}"
        self.routing_source = "fallback"
        self.routing_reason = "CHAT fallback: intent model unavailable or invalid; independent safety assessment still applies"
        return IntentType.CHAT

    def _topic(self, text: str, intent: IntentType) -> str:
        lowered = text.lower()
        if has_high_risk_signal(lowered):
            return "safety"
        if intent == IntentType.CONSULT:
            return "campus_policy"
        if intent == IntentType.RISK:
            return "mental_health_support"
        if is_general_request(lowered):
            return "general_task"
        return "conversation"


class SafetyAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="SafetyAgent",
        capabilities=frozenset({AgentCapability.SAFETY}),
        system_prompt=(
            "你是 SafetyAgent。你独立评估风险，并审查候选回复是否安全。"
            "你可以发布 SAFETY_OVERRIDE；你不生成最终回复。"
        ),
        memory_policy="private_safety_ledger",
        model_profile="safety",
        tool_permissions=frozenset({"llm.risk", "rules.high_risk", "response.review"}),
    )

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        latest_response = board.latest_artifact("response_proposal")
        latest_review = board.latest_artifact("safety_review")
        if latest_response and (latest_review is None or latest_review.metadata.get("responseArtifactId") != latest_response.id):
            return AgentDecision(True, 0.95, "candidate response needs safety critique")
        if not board.latest_artifact("risk") and board.user_input:
            confidence = 0.98 if has_high_risk_signal(board.user_input) else 0.84
            return AgentDecision(True, confidence, "user input needs independent risk assessment")
        if AgentCapability.SAFETY.value in task.required_capabilities:
            return AgentDecision(True, 0.8, "task explicitly asks for safety")
        return AgentDecision(False, reason="no safety work needed")

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        response = board.latest_artifact("response_proposal")
        review = board.latest_artifact("safety_review")
        if response and (review is None or review.metadata.get("responseArtifactId") != response.id):
            return self._review_response(task, board, response)
        return self._assess_risk(task, board)

    def _assess_risk(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        snapshot = board.memory_snapshot
        assessment = PsychologicalAssessmentService(self.client()).assess(
            board.model_input or board.user_input, _snapshot_context_messages(board, limit=10)
        )
        from app.services.safety_memory import SafetyMemoryService

        safety_dto = snapshot.safety_state if snapshot else None
        current = assessment.risk
        # 安全事实必须在任何候选回复生成之前同步持久化。message_id 幂等键保证
        # Harness 后续读取 decision 时不会重复追加事件或重复推进状态。
        decision = SafetyMemoryService(self.services.db, self.services.settings).record_turn_assessment(
            user_id=self.services.user.id,
            session_id=self.services.session.id,
            message_id=board.current_message_id,
            report_id=None,
            current_risk=current,
            confidence=assessment.confidence,
            summary=assessment.summary,
            safety_reassessment=assessment.safety_reassessment,
            protective_factors=assessment.protective_factors,
            has_plan_or_intent=assessment.has_plan_or_intent,
            risk_type=assessment.risk_type,
            evidence_summary=assessment.evidence_summary,
            source="MODEL",
        )
        carryover = decision.carryover_risk
        effective = decision.effective_risk
        previous_status = decision.previous_status.value
        payload = {
            # 兼容旧消费方：risk 字段承载 effectiveRisk（本轮回复约束的最终等级）。
            "risk": effective.value,
            "currentRisk": current.value,
            "carryoverRisk": carryover.value,
            "effectiveRisk": effective.value,
            "previousStatus": previous_status,
            "newStatus": decision.new_status.value,
            "emotion": assessment.emotion.value,
            "emotionScore": assessment.emotion_score,
            "confidence": assessment.confidence,
            "summary": assessment.summary,
            "assessment": assessment,
            "safetyReassessment": assessment.safety_reassessment,
            "protectiveFactors": assessment.protective_factors,
            "hasPlanOrIntent": assessment.has_plan_or_intent,
            "reassessmentRequired": safety_dto.reassessment_required if safety_dto else False,
            "safetyStateVersion": safety_dto.version if safety_dto else 0,
            "privateMemoryKey": self.services.private_memory._key(self.name, self.services.session.public_id),
        }
        events: tuple[AgentEvent, ...] = ()
        if effective == RiskLevel.HIGH:
            events = (
                AgentEvent(
                    type=AgentEventType.SAFETY_OVERRIDE,
                    actor=self.name,
                    task_id=task.id,
                    message="effectiveRisk is HIGH (current + unresolved carryover)",
                    metadata={
                        "risk": RiskLevel.HIGH.value,
                        "currentRisk": current.value,
                        "carryoverRisk": carryover.value,
                        "effectiveRisk": effective.value,
                    },
                ),
            )
        self.remember(
            f"current={current.value}; carryover={carryover.value}; effective={effective.value}; summary={assessment.summary}"
        )
        return AgentTurnResult(
            artifacts=(self._artifact("risk", payload, task, assessment.confidence),),
            events=events,
            messages=(
                AgentMessage(
                    id=f"msg:{uuid.uuid4().hex[:10]}",
                    sender=self.name,
                    recipient="CoordinatorAgent",
                    task_id=task.id,
                    kind="SAFETY_ASSESSMENT",
                    content=f"current={current.value}; carryover={carryover.value}; effective={effective.value}",
                ),
            ),
        )

    def _review_response(self, task: AgentTask, board: CollaborationBlackboard, response: AgentArtifact) -> AgentTurnResult:
        risk_artifact = board.latest_artifact("risk")
        effective_raw = (risk_artifact.payload.get("effectiveRisk") if risk_artifact else None) or _risk_level(board).value
        try:
            effective = RiskLevel(str(effective_raw).upper())
        except ValueError:
            effective = _risk_level(board)
        mode = str(response.payload.get("mode", "support"))
        combined = str(response.payload.get("candidateText", "")).strip()
        approved = True
        reason = "response proposal satisfies current safety constraints"
        if effective == RiskLevel.HIGH and not (
            any(word in combined for word in ["安全", "危险", "伤害"])
            and any(word in combined for word in ["可信任", "信任的人", "身边的人", "陪", "紧急", "心理中心", "辅导员"])):
            approved = False
            reason = "high-risk response proposal lacks immediate safety guidance"
        if effective == RiskLevel.HIGH:
            context = next((item for item in board.artifacts if item.kind == "context"
                and item.id == response.payload.get("knowledgeContextId")), None)
            evidence = (context.payload.get("retrievedKnowledge") or []) if context else []
            phones = re.findall(PHONE_PATTERN, combined)
            verified_phones = verified_contact_numbers(evidence)
            if any(re.sub(r"\D", "", phone) not in verified_phones for phone in phones):
                approved = False
                reason = "回复包含未经来源核实的联系电话，请删去该号码，改为鼓励联系身边可信任的人或当地紧急服务。"
        if any(label in combined for label in INTERNAL_SAFETY_LABELS):
            approved = False
            reason = "response leaks internal safety labels to student"
        if mode == "monitored_chat" and effective >= RiskLevel.MEDIUM and not any(
            word in combined for word in ["最近", "近况", "还好", "怎么样", "安全"]
        ):
            approved = False
            reason = "monitored_chat response should include a brief check-in"
        payload = {
            "approved": approved,
            "reason": reason,
            "responseArtifactId": response.id,
            "risk": effective.value,
            "effectiveRisk": effective.value,
            "responseMode": mode,
            "privateMemoryKey": self.services.private_memory._key(self.name, self.services.session.public_id),
        }
        kind = "safety_review" if approved else "critique"
        events = ()
        follow_up_tasks = ()
        if not approved:
            events = (
                AgentEvent(
                    type=AgentEventType.REVISION_REQUESTED,
                    actor=self.name,
                    task_id=task.id,
                    artifact_id=response.id,
                    message=reason,
                ),
            )
            follow_up_tasks = (
                AgentTask(
                    id=f"task:revise-response:{uuid.uuid4().hex[:8]}",
                    title="Revise unsafe response proposal",
                    description=reason,
                    priority=TaskPriority.CRITICAL,
                    required_capabilities=frozenset({AgentCapability.RESPONSE.value}),
                    created_by=self.name,
                    metadata={"kind": "response", "revisionOf": response.id},
                ),
            )
        self.remember(f"review approved={approved}; reason={reason}")
        return AgentTurnResult(
            artifacts=(self._artifact(kind, payload, task, 0.95, {"responseArtifactId": response.id}),),
            tasks=follow_up_tasks,
            events=events,
        )


class ContextAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="ContextAgent",
        capabilities=frozenset({AgentCapability.CONTEXT}),
        system_prompt=(
            "你是 ContextAgent。你只负责为本轮协作提供上下文，包括私有记忆、会话摘要、RAG 证据和 skill 约束。"
            "你不判断最终答案是否可采纳。"
        ),
        memory_policy="private_context_memory",
        model_profile="context",
        tool_permissions=frozenset({"memory_snapshot.read", "rag.retrieve", "skills.read"}),
    )

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        if board.latest_artifact("context"):
            return AgentDecision(False, reason="context artifact already exists")
        # 规格要求 safety 之后才 gather-context，不依赖任务优先级碰运气。
        if not board.latest_artifact("risk") or not board.latest_artifact("intent"):
            return AgentDecision(False, reason="context waits for safety and intent")
        if AgentCapability.CONTEXT.value in task.required_capabilities:
            return AgentDecision(True, 0.86, "task explicitly asks for context")
        risk = _risk_level(board)
        intent = _intent(board)
        if risk in {RiskLevel.MEDIUM, RiskLevel.HIGH} or intent in {IntentType.CONSULT, IntentType.RISK}:
            return AgentDecision(True, 0.82, "support path needs memory, RAG, and skill context")
        return AgentDecision(False, reason="context not necessary for current artifacts")

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        from app.services.ai import PromptTemplates
        from app.services.memory import summarize_history_for_memory
        from app.services.skills import MindBridgeSkillLibrary

        snapshot = board.memory_snapshot
        intent = _intent(board)
        risk_artifact = board.latest_artifact("risk")
        effective_raw = (risk_artifact.payload.get("effectiveRisk") if risk_artifact else None) or _risk_level(board).value
        try:
            effective = RiskLevel(str(effective_raw).upper())
        except ValueError:
            effective = _risk_level(board)
        safety_dto = snapshot.safety_state if snapshot else None
        safety_status = _safety_status(board)

        # 模型历史组装顺序（规格第 10 节）：安全约束 → 历史摘要 → 最近原文 → 当前输入。
        model_history: list[AiMessage] = []
        if effective != RiskLevel.LOW or safety_status in {"ACTIVE", "MONITORING"}:
            model_history.append(
                PromptTemplates.safety_constraint_message(
                    effective,
                    safety_status,
                    safety_dto.reassessment_required if safety_dto else False,
                )
            )
        if snapshot and snapshot.summary_text and snapshot.summary_text != "暂无摘要内容。":
            model_history.append(AiMessage(
                role="system",
                content=(
                    "较早对话的结构化滚动摘要（仅供 MindBridge 内部上下文使用；不要向学生展示；"
                    "不要据此输出诊断、风险等级或后台标签）：\n" + snapshot.summary_text
                ),
            ))
        recent = list(snapshot.recent_messages) if snapshot else []
        model_history.extend(recent)
        model_history.append(AiMessage(role="user", content=board.model_input))
        model_history = self._bounded_model_history(model_history)

        # memoryBrief 仅作为本轮临时概览（RAG 改写、审计），不承担历史压缩职责。
        # 回复 Prompt 中当前输入只能出现一次；memoryBrief 只概括此前消息。
        memory_brief = summarize_history_for_memory(recent, "")
        if snapshot and snapshot.summary_text and snapshot.summary_text != "暂无摘要内容。":
            memory_brief = snapshot.summary_text + "\n" + memory_brief

        retrieved: list["SearchResult"] = []
        query = ""
        skill_context = ""
        if intent != IntentType.CHAT or effective != RiskLevel.LOW:
            domain = "policy" if intent == IntentType.CONSULT and effective != RiskLevel.HIGH else "support"
            query = self._rewrite_query(memory_brief, board.model_input, domain)
            try:
                retrieved = self.services.knowledge.retrieve(query, self.services.settings.knowledge_top_k,
                    domain=domain, school=self.services.settings.campus_name)
            except Exception as exc:
                # 安全支持和资料不足说明不能被检索服务故障中断。
                self.services.knowledge.last_retrieval = {"query": query, "domain": domain, "error": str(exc), "path": "unavailable"}
                retrieved = []
            retrieved = self._bounded_knowledge(retrieved)
            skill_context = MindBridgeSkillLibrary.response_skill_context(intent, effective, board.user_input)
        payload = {
            "memoryBrief": memory_brief,
            "modelHistory": model_history,
            "knowledgeQuery": query,
            "retrievalTrace": {**self.services.knowledge.last_retrieval,
                               "budgetOmitted": getattr(self, "budget_omitted", []),
                               "providedParents": [r.chunk_id for r in retrieved]},
            "retrievedKnowledge": retrieved,
            "skillContext": skill_context,
            "effectiveRisk": effective.value,
            "privateMemoryKey": self.services.private_memory._key(self.name, self.services.session.public_id),
        }
        self.remember(f"context intent={intent.value}; effective={effective.value}; retrieved={len(retrieved)}")
        return AgentTurnResult(
            artifacts=(self._artifact("context", payload, task, 0.88),),
            messages=(
                AgentMessage(
                    id=f"msg:{uuid.uuid4().hex[:10]}",
                    sender=self.name,
                    recipient="ResponseAgent",
                    task_id=task.id,
                    kind="CONTEXT_READY",
                    content=f"context ready; retrieved={len(retrieved)}",
                ),
            ),
        )

    def _rewrite_query(self, memory_brief: str, model_input: str, domain: str = "support") -> str:
        try:
            query = self.client().complete([
                AiMessage(role="system", content=(
                    f"{self.profile.system_prompt}\n检索查询改写：将当前问题补全为适合{domain}知识库的查询。"
                    "结合历史中的身份、业务主题和已确认条件；当前明确更正优先。不要把历史回答的金额当作检索条件。"
                    "当前问题已经完整时保留原意，不引入无关历史。只输出查询，不回答问题。")),
                AiMessage(role="user", content=f"记忆摘要：\n{memory_brief}\n\n当前输入：\n{model_input}"),
            ]).strip()
            return (query or model_input)[:240]
        except Exception:
            return model_input[:240]

    def _bounded_model_history(self, history: list[AiMessage]) -> list[AiMessage]:
        from app.services.memory_context import estimate_tokens

        count_limit = max(2, self.services.settings.chat_history_limit * 2)
        usable = self.services.settings.model_context_window_tokens - self.services.settings.reserved_output_tokens
        token_limit = max(512, int(usable * self.services.settings.memory_prompt_target_ratio))
        leading_systems: list[AiMessage] = []
        conversation = list(history)
        while conversation and conversation[0].role == "system":
            leading_systems.append(conversation.pop(0))

        # 安全约束必须保留；较早摘要只有在预算允许时加入。
        required_systems = [item for item in leading_systems if item.content.startswith("安全约束")]
        optional_systems = [item for item in leading_systems if item not in required_systems]
        selected: list[AiMessage] = list(required_systems)
        used = sum(estimate_tokens(item.content) for item in selected)

        chosen_reversed: list[AiMessage] = []
        for item in reversed(conversation[-count_limit:]):
            remaining = token_limit - used
            if remaining <= 0:
                break
            tokens = estimate_tokens(item.content)
            if tokens <= remaining:
                chosen_reversed.append(item)
                used += tokens
            elif not chosen_reversed:  # 当前输入过长时保留受控截断，而不是让模型请求超窗。
                ratio = max(0.05, remaining / max(1, tokens))
                chosen_reversed.append(AiMessage(role=item.role, content=item.content[: max(80, int(len(item.content) * ratio))]))
                used = token_limit
        for item in optional_systems:
            tokens = estimate_tokens(item.content)
            if used + tokens <= token_limit:
                selected.append(item)
                used += tokens
        return [*selected, *reversed(chosen_reversed)]

    def _bounded_knowledge(self, items: list["SearchResult"]) -> list["SearchResult"]:
        from app.services.memory_context import estimate_tokens

        selected = []
        self.budget_omitted = []
        used = 0
        budget = self.services.settings.rag_skill_budget_tokens
        for item in items:
            tokens = estimate_tokens(item.content)
            if used + tokens > budget:
                self.budget_omitted.append({"chunkId": item.chunk_id, "reason": "parent exceeds remaining token budget"})
                continue
            selected.append(item)
            used += tokens
        return selected


class ResponseAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="ResponseAgent",
        capabilities=frozenset({AgentCapability.RESPONSE}),
        system_prompt=(
            "你是 ResponseAgent。你根据黑板上的意图、风险、上下文和安全约束提出候选回复 prompt，"
            "但最终是否采纳由 CoordinatorAgent 决定。"
        ),
        memory_policy="private_response_strategy",
        model_profile="response",
        tool_permissions=frozenset({"llm.response_plan"}),
    )

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        if board.latest_artifact("response_proposal") and "revisionOf" not in task.metadata:
            return AgentDecision(False, reason="response proposal already exists")
        if not board.latest_artifact("intent") or not board.latest_artifact("risk"):
            return AgentDecision(False, reason="response needs intent and risk artifacts")
        intent = _intent(board)
        effective = _effective_risk(board)
        safety_status = _safety_status(board)
        if not board.latest_artifact("context") and effective != RiskLevel.HIGH:
            return AgentDecision(False, reason="response waits for shared context, including ordinary chat history")
        if intent == IntentType.CHAT and effective == RiskLevel.LOW and safety_status not in {"ACTIVE", "MONITORING"}:
            return AgentDecision(True, 0.78, "normal chat response can be proposed")
        if board.latest_artifact("context") or effective == RiskLevel.HIGH:
            return AgentDecision(True, 0.84, "support response has enough artifacts")
        if AgentCapability.RESPONSE.value in task.required_capabilities:
            return AgentDecision(True, 0.65, "explicit response task")
        return AgentDecision(False, reason="waiting for context")

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        intent = _intent(board)
        effective = _effective_risk(board)
        context = board.latest_artifact("context")
        context_payload = context.payload if context else {}
        model_history = context_payload.get("modelHistory") or [AiMessage(role="user", content=board.model_input)]
        memory_brief = context_payload.get("memoryBrief") or "无相关历史记忆。"
        knowledge = context_payload.get("retrievedKnowledge") or []
        skill_context = context_payload.get("skillContext") or ""
        knowledge_context = "\n\n".join(
            f"[资料{i+1}] {item.title or item.source} | 版本 {item.version} | {item.heading}"
            + (f" | 第{item.page_start}页" if item.page_start else "")
            + (" | 演示资料，非正式校规" if item.is_demo else "")
            + f"\n{item.content}" for i, item in enumerate(knowledge))
        safety_status = _safety_status(board)
        mode = self._select_mode(intent, effective, safety_status)
        prompt_intent = IntentType.CHAT if mode == "normal_chat" else (intent if intent != IntentType.CHAT else IntentType.RISK)
        messages = [
            PromptTemplates.answer_system_prompt(prompt_intent, effective, knowledge_context, self.services.user.display_name, skill_context),
            AiMessage(
                role="system",
                content=(
                    f"{self.profile.system_prompt}\n"
                    f"当前由 ResponseAgent 以 {mode} mode 提出回复方案。\n"
                    f"{_mode_prompt_note(mode)}\n"
                    f"私有记忆：\n{_format_private_memory(self.private_memory())}\n"
                    f"记忆摘要：\n{memory_brief}"
                ),
            ),
            *model_history,
        ]
        if "revisionOf" in task.metadata:
            critique = board.latest_artifact("critique")
            if critique:
                messages.insert(-1, AiMessage(
                    role="system",
                    content=f"上一候选回复未通过安全审查，请按以下原因重写，不要解释审查过程：{critique.payload.get('reason', '')}",
                ))
        generation_error = ""
        streamed = False
        try:
            if intent == IntentType.CONSULT and effective != RiskLevel.HIGH and not knowledge:
                retrieval = context_payload.get("retrievalTrace") or {}
                if retrieval.get("path") == "unavailable":
                    candidate_text = "资料检索服务暂时不可用，当前无法核实具体规则，请稍后重试。"
                else:
                    candidate_text = "当前尚未接入或未找到与你的问题对应的有效文件，暂时无法确认具体规则。请先添加对应资料后再查询。"
                if mode == "monitored_chat":
                    candidate_text = "先关心一下，你最近还好吗，现在是否安全？" + candidate_text
            else:
                can_stream = self.services.stream_event is not None
                if can_stream:
                    risk_payload = board.latest_artifact("risk").payload
                    self.services.stream_event({"type": "_response_plan", "messages": messages,
                        "knowledge": knowledge, "memoryBrief": memory_brief, "intent": intent.value,
                        "mode": mode, "assessment": risk_payload.get("assessment"),
                        "currentRisk": risk_payload.get("currentRisk", effective.value),
                        "carryoverRisk": risk_payload.get("carryoverRisk", "LOW"),
                        "effectiveRisk": effective.value,
                        "previousStatus": risk_payload.get("previousStatus", "RESOLVED"),
                        "newStatus": risk_payload.get("newStatus", "RESOLVED"),
                        "snapshotMetadata": board.memory_snapshot.audit_metadata() if board.memory_snapshot else {}})
                    opening = SAFETY_OPENING if effective == RiskLevel.HIGH else (
                        MONITORING_OPENING if mode == "monitored_chat" else "")
                    if opening:
                        self.services.stream_event({"type": "token", "content": opening})
                        messages.insert(-1, AiMessage(role="system", content=
                            "界面已经向用户展示以下开场，不要重复这段话，请直接继续提供支持：" + opening))
                    fallback = "暂时无法可靠生成回答，请稍后重试。"
                    result = asyncio.run(collect_guarded_stream(self.client(), messages,
                        lambda text: self.services.stream_event({"type": "token", "content": text}), fallback,
                        is_cancelled=getattr(self.services.stream_event, "is_cancelled", None),
                        verified_phones=verified_contact_numbers(knowledge) if effective == RiskLevel.HIGH else None))
                    candidate_text, generation_error = opening + result.text, result.error
                    streamed = True
                else:
                    candidate_text = self.client().complete(messages).strip()
            if not candidate_text:
                raise RuntimeError("Model returned empty answer; check thinking/output token configuration")
        except Exception as exc:
            generation_error = f"{type(exc).__name__}: {exc}"
            if effective == RiskLevel.HIGH:
                candidate_text = "我很在意你现在的安全。请先远离可能伤害自己的物品，联系身边可信任的人陪伴；如果危险正在发生，请立即联系当地紧急服务。你现在身边有人可以陪着你吗？"
            elif mode == "monitored_chat":
                candidate_text = "先关心一下，你最近还好吗，现在是否安全？暂时无法可靠生成回答，请稍后再试。"
            elif intent == IntentType.CONSULT:
                candidate_text = "暂时无法可靠生成政策解释，不能确认具体标准。请稍后重试或向学校相关部门核实。"
            else:
                candidate_text = "暂时无法可靠生成回答，请稍后再试；如果需要情绪支持，可以先告诉我你当前最需要什么帮助。"
        payload = {
            "messages": messages,
            "candidateText": candidate_text,
            "generationError": generation_error,
            "streamed": streamed,
            "knowledgeContextId": context.id if context else None,
            "mode": mode,
            "intent": intent.value,
            "risk": effective.value,
            "effectiveRisk": effective.value,
            "responseAgent": self.name,
            "privateMemoryKey": self.services.private_memory._key(self.name, self.services.session.public_id),
        }
        self.remember(f"response mode={mode}; intent={intent.value}; effective={effective.value}")
        return AgentTurnResult(
            artifacts=(self._artifact("response_proposal", payload, task, 0.86),),
            messages=(
                AgentMessage(
                    id=f"msg:{uuid.uuid4().hex[:10]}",
                    sender=self.name,
                    recipient="SafetyAgent",
                    task_id=task.id,
                    kind="REVIEW_REQUEST",
                    content="请审查候选回复方案。",
                ),
            ),
        )

    def _select_mode(self, intent: IntentType, effective: RiskLevel, safety_status: str) -> str:
        # 回复模式选择（规格 7.9）：安全约束高于普通意图。
        if effective == RiskLevel.HIGH:
            return "safety_support"
        if safety_status == "MONITORING":
            return "monitored_chat"
        if intent == IntentType.CONSULT:
            return "policy_answer"
        if intent != IntentType.CHAT or effective >= RiskLevel.MEDIUM:
            return "support"
        return "normal_chat"


class CoordinatorAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="CoordinatorAgent",
        capabilities=frozenset({AgentCapability.COORDINATION}),
        system_prompt=(
            "你是 CoordinatorAgent。你不规定固定 Agent 顺序；你只维护任务板、预算、安全门槛、冲突仲裁和最终采纳。"
        ),
        memory_policy="private_coordination_trace",
        model_profile="coordinator",
        tool_permissions=frozenset({"taskboard.write", "blackboard.accept"}),
    )

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        return AgentDecision(False, reason="CoordinatorAgent is driven by the event loop, not by fixed workflow slots")

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        return AgentTurnResult(close_task=False)

    def root_task(self, board: CollaborationBlackboard) -> AgentTask:
        return AgentTask(
            id="task:root",
            title="Resolve user turn",
            description=board.user_input,
            priority=TaskPriority.CRITICAL if has_high_risk_signal(board.user_input) else TaskPriority.NORMAL,
            created_by=self.name,
            metadata={"kind": "root"},
        )

    def remember_acceptance(self, artifact_id: str, reason: str) -> None:
        self.remember(f"accepted={artifact_id}; reason={reason}")


def _intent(board: CollaborationBlackboard) -> IntentType:
    artifact = board.latest_artifact("intent")
    if artifact:
        try:
            return IntentType(str(artifact.payload.get("intent", IntentType.CHAT.value)).upper())
        except ValueError:
            return IntentType.CHAT
    if has_high_risk_signal(board.user_input):
        return IntentType.RISK
    return classify_intent_rules(board.user_input, []).intent or IntentType.CHAT


def _risk_level(board: CollaborationBlackboard) -> RiskLevel:
    highest = RiskLevel.LOW
    order = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}
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


def _context_history(board: CollaborationBlackboard) -> list[AiMessage]:
    context = board.latest_artifact("context")
    if not context:
        return [AiMessage(role="user", content=board.model_input or board.user_input)]
    return context.payload.get("modelHistory") or [AiMessage(role="user", content=board.model_input or board.user_input)]


def _effective_risk(board: CollaborationBlackboard) -> RiskLevel:
    artifact = board.latest_artifact("risk")
    raw = artifact.payload.get("effectiveRisk") if artifact else None
    if raw:
        try:
            return RiskLevel(str(raw).upper())
        except ValueError:
            pass
    return _risk_level(board)


def _safety_status(board: CollaborationBlackboard) -> str:
    risk = board.latest_artifact("risk")
    if risk and risk.payload.get("newStatus") in {"ACTIVE", "MONITORING", "RESOLVED", "FALSE_POSITIVE"}:
        return risk.payload["newStatus"]
    snapshot = board.memory_snapshot
    if snapshot and snapshot.safety_state:
        return snapshot.safety_state.status
    return "RESOLVED"


def _snapshot_context_messages(board: CollaborationBlackboard, limit: int = 10) -> list[AiMessage]:
    """评估/分类用上下文：摘要概览 + 最近原文。"""
    snapshot = board.memory_snapshot
    messages: list[AiMessage] = []
    if snapshot and snapshot.summary_text and snapshot.summary_text != "暂无摘要内容。":
        messages.append(AiMessage(role="system", content=f"较早对话摘要：\n{snapshot.summary_text}"))
    if snapshot and snapshot.active_risk_events:
        event_lines = [
            f"- {event.risk_type}/{event.risk_level}: {event.summary}"
            for event in snapshot.active_risk_events[:5]
        ]
        messages.append(AiMessage(
            role="system",
            content="尚未解除的安全事件证据（仅供内部复核，不得向用户展示标签）：\n" + "\n".join(event_lines),
        ))
    if snapshot and snapshot.recent_messages:
        messages.extend(list(snapshot.recent_messages)[-limit:])
    if not any(message.role == "user" for message in messages):
        messages.append(AiMessage(role="user", content=board.model_input or board.user_input))
    return messages


def _mode_prompt_note(mode: str) -> str:
    if mode == "safety_support":
        return "当前安全优先：先回应情绪并确认当前安全，鼓励联系可信任的人或专业资源；普通请求在安全确认之后简短回应。"
    if mode == "monitored_chat":
        return PromptTemplates.monitored_chat_prefix()
    if mode == "support":
        return "以共情、支持、非诊断的方式回应，不向学生暴露任何后台标签。"
    return "正常回答，不主动做心理测评，不输出风险等级或后台标签。"


def _format_private_memory(items: list[AiMessage]) -> str:
    if not items:
        return "无"
    return "\n".join(f"- {item.content}" for item in items[-5:])
