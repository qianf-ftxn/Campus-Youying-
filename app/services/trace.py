
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from app.agents.result import AgentRunResult
from app.models.entities import AgentRunTrace, ChatSession, UserAccount, ChatMessage


class AgentTraceService:
    def __init__(self, db: Session):
        self.db = db

    def save_run(
        self,
        user: UserAccount,
        session: ChatSession,
        original_input: str,
        sanitized_input: str,
        memory_brief: str,
        agent_run: AgentRunResult,
        report_id: int | None,
        message_id: int | None = None,
    ) -> AgentRunTrace:
        snapshot_meta = agent_run.memory_snapshot_metadata or {}
        safety_meta = snapshot_meta.get("safetyState") or {}
        trace = AgentRunTrace(
            user_id=user.id,
            session_id=session.id,
            report_id=report_id,
            intent=agent_run.intent.value,
            risk_level=agent_run.risk_level.value,
            current_risk=agent_run.current_risk.value,
            carryover_risk=agent_run.carryover_risk.value,
            effective_risk=agent_run.effective_risk.value,
            response_mode=agent_run.response_mode.value,
            summary_version=snapshot_meta.get("summaryVersion"),
            covered_through_message_id=snapshot_meta.get("coveredThroughMessageId"),
            safety_state_version=safety_meta.get("version"),
            snapshot_degraded=bool(snapshot_meta.get("degraded", False)),
            original_input=original_input,
            sanitized_input=sanitized_input,
            memory_brief=memory_brief,
            agent_steps_json=_json(_agent_steps_with_collaboration(agent_run)),
            retrieved_knowledge_json=_json(agent_run.retrieved_knowledge),
            response_messages_json=_json([
                *agent_run.response_messages,
                {"role": "assistant", "content": agent_run.response_text,
                 "kind": "interrupted_candidate" if snapshot_meta.get("completionStatus") == "INTERRUPTED" else "approved_candidate"},
            ]),
            assessment_json=_json({
                "assessment": agent_run.assessment or {},
                "businessIntent": (agent_run.business_intent or agent_run.intent).value,
                "memorySnapshot": snapshot_meta,
            }),
        )
        self.db.add(trace)
        self.db.flush()
        if message_id is not None:
            message = self.db.get(ChatMessage, message_id)
            if message and message.user_id == user.id and message.session_id == session.id:
                message.agent_trace_id = trace.id
        self.db.commit()
        self.db.refresh(trace)
        return trace


def _json(value: Any) -> str:
    return json.dumps(_to_jsonable(value), ensure_ascii=False, default=str)


def _agent_steps_with_collaboration(agent_run: AgentRunResult) -> list[Any]:
    entries: list[Any] = [*agent_run.steps]
    entries.extend(
        {
            "kind": "agent_event",
            "type": getattr(event.type, "value", event.type),
            "actor": event.actor,
            "taskId": event.task_id,
            "artifactId": event.artifact_id,
            "message": event.message,
            "metadata": event.metadata,
        }
        for event in agent_run.collaboration_events
    )
    entries.extend(
        {
            "kind": "agent_task",
            "id": task.id,
            "title": task.title,
            "status": getattr(task.status, "value", task.status),
            "priority": getattr(task.priority, "value", task.priority),
            "requiredCapabilities": sorted(task.required_capabilities),
            "claimedBy": list(task.claimed_by),
            "createdBy": task.created_by,
            "metadata": task.metadata,
        }
        for task in agent_run.collaboration_tasks
    )
    entries.extend(
        {
            "kind": "agent_artifact",
            "id": artifact.id,
            "owner": artifact.owner,
            "artifactKind": artifact.kind,
            "confidence": artifact.confidence,
            "taskId": artifact.task_id,
            "metadata": artifact.metadata,
            # payload 里的候选 prompt/正文与 response_messages_json 重复且巨大，落库时替换为占位。
            "payload": _slim_artifact_payload(artifact.kind, artifact.payload),
        }
        for artifact in agent_run.collaboration_artifacts
    )
    return entries


def _slim_artifact_payload(kind: str, payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    slimmed = dict(payload)
    for key in ("messages", "modelHistory"):
        if key in slimmed:
            slimmed[key] = f"...omitted ({len(slimmed[key]) if isinstance(slimmed[key], list) else 0} messages, stored separately)"
    if "candidateText" in slimmed:
        slimmed["candidateText"] = "...omitted (approved candidate stored in response_messages_json)"
    slimmed.pop("assessment", None)  # assessment_json 已单独存储
    return slimmed


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
