from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.enums import IntentType, ResponseMode, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.assessment import PsychologyAssessment
from app.services.knowledge import SearchResult


@dataclass
class AgentStep:
    step: int
    agent: str
    action: str
    observation: str


@dataclass
class AgentRunResult:
    intent: IntentType
    risk_level: RiskLevel
    assessment: PsychologyAssessment | None
    retrieved_knowledge: list[SearchResult]
    response_messages: list[AiMessage]
    response_text: str
    steps: list[AgentStep]
    memory_brief: str
    current_risk: RiskLevel = RiskLevel.LOW
    carryover_risk: RiskLevel = RiskLevel.LOW
    effective_risk: RiskLevel = RiskLevel.LOW
    previous_safety_status: str = "RESOLVED"
    new_safety_status: str = "RESOLVED"
    response_mode: ResponseMode = ResponseMode.NORMAL_CHAT
    memory_snapshot_metadata: dict[str, Any] = field(default_factory=dict)
    collaboration_events: list[Any] = field(default_factory=list)
    collaboration_tasks: list[Any] = field(default_factory=list)
    collaboration_artifacts: list[Any] = field(default_factory=list)
    business_intent: IntentType | None = None

    @property
    def requires_report(self) -> bool:
        # carryover HIGH 约束回复，但不能把每一轮普通聊天都当作“新高风险事件”
        # 重复创建报告和告警；报告由本轮新风险或真实咨询意图触发。
        return self.current_risk >= RiskLevel.MEDIUM
