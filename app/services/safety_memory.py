"""安全事件记忆：RiskEvent 不可变事件流 + SafetyState 当前状态投影。

生命周期沿用原设计；作用范围已按 docs/session-history-and-risk-isolation.md 改为会话：
- MySQL 是权威存储，Redis 只做可重建缓存；
- 生命周期 ACTIVE / MONITORING / RESOLVED / FALSE_POSITIVE 与 RiskCaseStatus 严格分离；
- carryoverRisk 只继承同一用户、同一会话未解除（ACTIVE/MONITORING）状态的风险下限；
- 时间和普通轮数只触发复核，不单独解除风险；
- 状态更新使用 version 乐观锁。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from importlib import import_module
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.enums import (
    RiskEventKind,
    RiskEventType,
    RiskLevel,
    SafetyLifecycleStatus,
    max_risk,
)
from app.models.entities import RiskEvent, SafetyState, ChatSession
from app.services.privacy import PrivacySanitizer

logger = logging.getLogger(__name__)

SAFETY_STATE_CACHE_PREFIX = "mindbridge:safety-state:session:v2:"
SAFETY_STATE_CACHE_SCHEMA = 2

@dataclass(frozen=True)
class SafetyStateDTO:
    """不可变安全状态 DTO（供 Blackboard / Agent 使用，不绑定 ORM Session）。"""

    status: str = "RESOLVED"
    effective_risk: str = "LOW"
    version: int = 0
    latest_event_id: Optional[int] = None
    last_high_risk_at: Optional[str] = None
    last_assessed_at: Optional[str] = None
    next_review_at: Optional[str] = None
    reassessment_required: bool = False
    normal_turn_count_since_review: int = 0
    safe_reassessment_count: int = 0
    consecutive_low_assessments: int = 0
    human_review_required: bool = False
    latest_event_severity: str = ""
    degraded: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "effectiveRisk": self.effective_risk,
            "version": self.version,
            "latestEventId": self.latest_event_id,
            "lastHighRiskAt": self.last_high_risk_at,
            "lastAssessedAt": self.last_assessed_at,
            "nextReviewAt": self.next_review_at,
            "reassessmentRequired": self.reassessment_required,
            "normalTurnCountSinceReview": self.normal_turn_count_since_review,
            "safeReassessmentCount": self.safe_reassessment_count,
            "consecutiveLowAssessments": self.consecutive_low_assessments,
            "humanReviewRequired": self.human_review_required,
            "latestEventSeverity": self.latest_event_severity,
            "degraded": self.degraded,
        }


@dataclass(frozen=True)
class RiskEventDTO:
    id: Optional[int]
    kind: str
    risk_type: str
    risk_level: str
    status: str
    summary: str = ""
    created_at: Optional[str] = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "riskType": self.risk_type,
            "riskLevel": self.risk_level,
            "status": self.status,
            "summary": self.summary,
            "createdAt": self.created_at,
        }


@dataclass(frozen=True)
class SafetyTurnDecision:
    current_risk: RiskLevel
    carryover_risk: RiskLevel
    effective_risk: RiskLevel
    previous_status: SafetyLifecycleStatus
    new_status: SafetyLifecycleStatus
    latest_event_id: Optional[int] = None
    transitioned: bool = False
    reason: str = ""


def _now() -> datetime:
    return datetime.utcnow()


def parse_level(value: str | RiskLevel) -> RiskLevel:
    try:
        return RiskLevel(str(value).upper())
    except ValueError:
        return RiskLevel.LOW


def carryover_risk(state: SafetyStateDTO | None, settings: Settings) -> RiskLevel:
    if state is None or not settings.safety_memory_enabled:
        return RiskLevel.LOW
    try:
        status = SafetyLifecycleStatus(state.status)
    except ValueError:
        return RiskLevel.LOW
    if status == SafetyLifecycleStatus.ACTIVE:
        return parse_level(state.effective_risk)
    if status == SafetyLifecycleStatus.MONITORING:
        # MONITORING 初始下限由配置决定，但完成该阶段的连续 LOW 复核后，
        # SafetyState.effective_risk 可以显式降为 LOW。
        return parse_level(state.effective_risk)
    return RiskLevel.LOW


class SafetyMemoryService:
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.privacy = PrivacySanitizer()
        self._redis = self._connect_redis()
        self._replay_event = None

    # ------------------------------------------------------------------ load
    def _require_session(self, user_id: int, session_id: int) -> None:
        if not self.db.query(ChatSession.id).filter_by(id=session_id, user_id=user_id).first():
            raise ValueError("会话不存在或不属于该用户。")

    def restore_legacy_sessions(self) -> int:
        """按原始当前风险重新投影旧事件，忽略旧用户级 carryover；不修改事件流。

        完成标记允许启动中断后重新恢复该会话。已完成和新建会话不重复回放。
        """
        if not self.settings.safety_memory_enabled:
            return 0
        pairs = self.db.query(RiskEvent.user_id, RiskEvent.session_id).join(ChatSession,
            (ChatSession.id == RiskEvent.session_id) & (ChatSession.user_id == RiskEvent.user_id)).distinct().all()
        restored = 0
        for user_id, session_id in pairs:
            row = self.db.query(SafetyState).filter_by(user_id=user_id, session_id=session_id).first()
            if row and row.history_rebuilt:
                continue
            if row:
                self.db.delete(row)
                self.db.commit()
            events = self.db.query(RiskEvent).filter_by(user_id=user_id, session_id=session_id).filter(
                RiskEvent.event_kind.in_(["RISK_DETECTED", "REASSESSMENT", "MANUAL_OVERRIDE"])).order_by(RiskEvent.id).all()
            try:
                for event in events:
                    self._replay_event = event
                    payload = json.loads(event.payload_json or "{}")
                    if event.event_kind == "MANUAL_OVERRIDE":
                        self.manual_override(user_id, "历史记录恢复", event.new_status,
                            payload.get("reason", event.evidence_summary), session_id)
                    else:
                        self.record_turn_assessment(user_id=user_id, session_id=session_id,
                            message_id=event.message_id, report_id=event.report_id,
                            current_risk=parse_level(event.risk_level), confidence=event.confidence,
                            summary=event.evidence_summary,
                            safety_reassessment=bool(payload.get("validSafetyReassessment", False)),
                            protective_factors=payload.get("protectiveFactors", ""),
                            has_plan_or_intent=bool(payload.get("hasPlanOrIntent", False)),
                            risk_type=event.risk_type, source=event.source)
                row = self.db.query(SafetyState).filter_by(user_id=user_id, session_id=session_id).first()
                if row:
                    row.history_rebuilt = True
                    self.db.commit()
                    restored += 1
            finally:
                self._replay_event = None
            if row:
                self._write_cache(user_id, session_id, self._to_dto(row))
        return restored

    def get_state(self, user_id: int, session_id: int) -> SafetyStateDTO | None:
        # SafetyState 是安全关键权威数据。正常路径始终校验 MySQL，Redis 仅在
        # 数据库读取失败时提供带 degraded 标记的保守回退，避免旧缓存降级 ACTIVE。
        try:
            row = self.db.query(SafetyState).filter(SafetyState.user_id == user_id, SafetyState.session_id == session_id).first()
            if row is None:
                return None
            dto = self._to_dto(row)
            self._write_cache(user_id, session_id, dto)
            return dto
        except Exception:
            cached = self._read_cache(user_id, session_id)
            if cached is None:
                raise
            if cached.status not in {SafetyLifecycleStatus.ACTIVE.value, SafetyLifecycleStatus.MONITORING.value}:
                return SafetyStateDTO(
                    status=SafetyLifecycleStatus.ACTIVE.value,
                    effective_risk=RiskLevel.HIGH.value,
                    version=cached.version,
                    latest_event_id=cached.latest_event_id,
                    last_high_risk_at=cached.last_high_risk_at,
                    last_assessed_at=cached.last_assessed_at,
                    next_review_at=cached.next_review_at,
                    reassessment_required=True,
                    human_review_required=True,
                    degraded=True,
                )
            return SafetyStateDTO(**{**cached.__dict__, "degraded": True})

    def active_events(self, user_id: int, session_id: int, limit: int = 8) -> list[RiskEventDTO]:
        state = self.db.query(SafetyState).filter(SafetyState.user_id == user_id, SafetyState.session_id == session_id).first()
        active_status = state.status if state else ""
        rows = (
            self.db.execute(
                select(RiskEvent)
                .where(RiskEvent.user_id == user_id, RiskEvent.session_id == session_id)
                .order_by(RiskEvent.id.desc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        if state and state.status in {SafetyLifecycleStatus.ACTIVE.value, SafetyLifecycleStatus.MONITORING.value}:
            latest_high = (
                self.db.query(RiskEvent)
                .filter(RiskEvent.user_id == user_id, RiskEvent.session_id == session_id, RiskEvent.risk_level == RiskLevel.HIGH.value)
                .order_by(RiskEvent.id.desc())
                .first()
            )
            if latest_high is not None and all(row.id != latest_high.id for row in rows):
                rows = [*rows[: max(0, limit - 1)], latest_high]
                rows.sort(key=lambda item: item.id, reverse=True)
        return [
            RiskEventDTO(
                id=row.id,
                kind=row.event_kind,
                risk_type=row.risk_type,
                risk_level=row.risk_level,
                status=active_status,
                summary=row.evidence_summary[:160],
                created_at=row.created_at.isoformat() if row.created_at else None,
            )
            for row in rows
        ]

    # ------------------------------------------------------- turn assessment
    def record_turn_assessment(
        self,
        user_id: int,
        session_id: int,
        message_id: int | None,
        report_id: int | None,
        current_risk: RiskLevel,
        confidence: float,
        summary: str,
        safety_reassessment: bool = False,
        protective_factors: str = "",
        has_plan_or_intent: bool = False,
        risk_type: str = "OTHER",
        evidence_summary: str = "",
        source: str = "MODEL",
    ) -> SafetyTurnDecision:
        """一轮评估结束后合并 carryover 并按状态机持久化。升级即时，降级保守。"""
        self._require_session(user_id, session_id)
        if message_id is not None and self._replay_event is None:
            existing = (
                self.db.query(RiskEvent)
                .filter(RiskEvent.dedupe_key == f"message:{message_id}:assessment",
                        RiskEvent.user_id == user_id, RiskEvent.session_id == session_id)
                .first()
            )
            if existing is not None:
                state = self.db.query(SafetyState).filter(SafetyState.user_id == user_id, SafetyState.session_id == session_id).first()
                dto = self._to_dto(state) if state else None
                payload = json.loads(existing.payload_json or "{}")
                effective_existing = parse_level(payload.get("effectiveRisk", dto.effective_risk if dto else current_risk.value))
                status = SafetyLifecycleStatus(dto.status) if dto else SafetyLifecycleStatus.RESOLVED
                if payload.get("scope") != "session":
                    # 旧事件中的 effectiveRisk 可能包含其他会话的历史下限。
                    effective_existing = max_risk(parse_level(existing.risk_level), carryover_risk(dto, self.settings))
                return SafetyTurnDecision(
                    current_risk=parse_level(existing.risk_level),
                    carryover_risk=carryover_risk(dto, self.settings),
                    effective_risk=effective_existing,
                    previous_status=SafetyLifecycleStatus(existing.previous_status) if existing.previous_status else status,
                    new_status=SafetyLifecycleStatus(existing.new_status) if existing.new_status and payload.get("scope") == "session" else status,
                    latest_event_id=existing.id,
                    transitioned=bool(existing.new_status and existing.previous_status != existing.new_status),
                    reason="idempotent replay of persisted safety assessment",
                )
        # 写路径不能使用 Redis DTO；每次都从 MySQL 权威状态开始计算。
        previous_row = self.db.query(SafetyState).filter(SafetyState.user_id == user_id, SafetyState.session_id == session_id).first()
        previous_dto = self._to_dto(previous_row) if previous_row else None
        previous_status = SafetyLifecycleStatus(previous_dto.status) if previous_dto else SafetyLifecycleStatus.RESOLVED
        carryover = carryover_risk(previous_dto, self.settings)
        effective = max_risk(current_risk, carryover)

        settings = self.settings
        now = self._replay_event.created_at if self._replay_event else _now()
        new_status = previous_status
        reason = ""
        transitioned = False
        phase_changed = False
        downgraded_risk: RiskLevel | None = None

        if current_risk == RiskLevel.HIGH:
            new_status = SafetyLifecycleStatus.ACTIVE
            transitioned = new_status != previous_status
            reason = "本轮检测到新的 HIGH 风险，立即回到 ACTIVE"
        elif current_risk == RiskLevel.MEDIUM and previous_status in {
            SafetyLifecycleStatus.RESOLVED,
            SafetyLifecycleStatus.FALSE_POSITIVE,
        }:
            new_status = SafetyLifecycleStatus.ACTIVE
            transitioned = new_status != previous_status
            reason = "本轮 MEDIUM 信号将状态提升为 ACTIVE"

        valid_reassessment = (
            safety_reassessment
            and current_risk == RiskLevel.LOW
            and bool((protective_factors or "").strip())
            and not has_plan_or_intent
        )

        if not transitioned and current_risk < RiskLevel.HIGH and previous_dto is not None:
            counters = self._advance_counters(previous_dto, valid_reassessment, current_risk)
            new_status, downgraded_risk, phase_changed, downgrade_reason = self._maybe_downgrade(
                previous_status, previous_dto, counters, valid_reassessment, protective_factors
            )
            transitioned = new_status != previous_status
            reason = downgrade_reason or reason
        elif previous_dto is None:
            counters = {"safe": 1 if valid_reassessment else 0, "low": 1 if current_risk == RiskLevel.LOW and valid_reassessment else 0}
        else:
            counters = (
                {"safe": 0, "low": 0}
                if current_risk >= RiskLevel.MEDIUM
                else {
                    "safe": previous_dto.safe_reassessment_count + (1 if valid_reassessment else 0),
                    "low": previous_dto.consecutive_low_assessments + (1 if valid_reassessment else 0),
                }
            )
        latest_event_id = self._append_event(
            user_id=user_id,
            session_id=session_id,
            message_id=message_id,
            report_id=report_id,
            kind=RiskEventKind.RISK_DETECTED if current_risk >= RiskLevel.MEDIUM else RiskEventKind.REASSESSMENT,
            risk_type=risk_type or self._classify_type(evidence_summary or summary),
            risk_level=current_risk,
            confidence=confidence,
            evidence=self.privacy.sanitize(evidence_summary or summary)[:300],
            source=source,
            previous_status=previous_status if transitioned else None,
            new_status=new_status if transitioned else None,
            payload={
                "scope": "session",
                "effectiveRisk": effective.value,
                "validSafetyReassessment": valid_reassessment,
                "protectiveFactors": protective_factors[:120],
                "hasPlanOrIntent": has_plan_or_intent,
            },
            dedupe_key=f"message:{message_id}:assessment" if message_id is not None else None,
        )

        if transitioned or phase_changed:
            transition_event_id = self._append_event(
                user_id=user_id,
                session_id=session_id,
                message_id=message_id,
                report_id=report_id,
                kind=RiskEventKind.STATE_TRANSITION,
                risk_type=risk_type or self._classify_type(evidence_summary or summary),
                risk_level=downgraded_risk or effective,
                confidence=confidence,
                evidence=reason or "lifecycle transition",
                source=source,
                previous_status=previous_status,
                new_status=new_status,
                payload={
                    "currentRisk": current_risk.value,
                    "effectiveRisk": (downgraded_risk or effective).value,
                    "safeReassessmentCount": counters["safe"],
                    "consecutiveLowAssessments": counters["low"],
                    "protectiveFactors": protective_factors[:120],
                    "humanReviewed": False,
                    "reason": reason,
                },
            )
            latest_event_id = transition_event_id or latest_event_id

        # SafetyState.effective_risk 表示"当前历史风险下限"：转换后按新生命周期重算，
        # 避免把转出前的旧值带入下一轮（MONITORING 下限 MEDIUM；解除后 LOW）。
        stored_effective = effective
        if downgraded_risk is not None:
            stored_effective = downgraded_risk
        elif new_status == SafetyLifecycleStatus.MONITORING:
            stored_effective = max_risk(RiskLevel.MEDIUM, parse_level(self.settings.safety_monitoring_risk_floor))
        elif new_status in {SafetyLifecycleStatus.RESOLVED, SafetyLifecycleStatus.FALSE_POSITIVE}:
            stored_effective = RiskLevel.LOW

        try:
            self._upsert_state(
                user_id=user_id,
                session_id=session_id,
                status=new_status,
                effective_risk=stored_effective,
                latest_event_id=latest_event_id,
                now=now,
                transitioned=transitioned,
                phase_changed=phase_changed,
                valid_reassessment=valid_reassessment,
                current_risk=current_risk,
                summary=evidence_summary or summary,
                counters=counters,
                previous_dto=previous_dto,
                high_now=current_risk == RiskLevel.HIGH,
                has_plan_or_intent=has_plan_or_intent,
                expected_version=previous_dto.version if previous_dto else None,
            )
        except Exception:
            self.db.rollback()
            raise
        return SafetyTurnDecision(
            current_risk=current_risk,
            carryover_risk=carryover,
            effective_risk=effective,
            previous_status=previous_status,
            new_status=new_status,
            latest_event_id=latest_event_id,
            transitioned=transitioned,
            reason=reason,
        )

    # ------------------------------------------------------------- manual ops
    def manual_override(self, user_id: int, actor: str, action: str, reason: str, session_id: int) -> SafetyStateDTO:
        """管理端人工确认进入监测、解除或误判；始终追加 MANUAL_OVERRIDE 事件。"""
        self._require_session(user_id, session_id)
        row = self.db.query(SafetyState).filter(SafetyState.user_id == user_id, SafetyState.session_id == session_id).first()
        previous_status = SafetyLifecycleStatus(row.status) if row else SafetyLifecycleStatus.RESOLVED
        new_status = SafetyLifecycleStatus(action)
        if new_status not in {SafetyLifecycleStatus.MONITORING, SafetyLifecycleStatus.RESOLVED, SafetyLifecycleStatus.FALSE_POSITIVE}:
            raise ValueError("manual override only supports MONITORING, RESOLVED or FALSE_POSITIVE")
        target_risk = (
            max_risk(RiskLevel.MEDIUM, parse_level(self.settings.safety_monitoring_risk_floor))
            if new_status == SafetyLifecycleStatus.MONITORING
            else RiskLevel.LOW
        )
        event_id = self._append_event(
            user_id=user_id,
            session_id=session_id or (row.latest_session_id if row else 0) or 0,
            message_id=None,
            report_id=None,
            kind=RiskEventKind.MANUAL_OVERRIDE,
            risk_type=RiskEventType.OTHER,
            risk_level=target_risk,
            confidence=1.0,
            evidence=f"by {actor}: {reason}"[:300],
            source="HUMAN",
            previous_status=previous_status,
            new_status=new_status,
            payload={"humanReviewed": True, "reason": reason, "effectiveRisk": target_risk.value},
        )
        try:
            state = self._upsert_state(
                user_id=user_id,
                session_id=session_id or (row.latest_session_id if row else 0) or 0,
                status=new_status,
                effective_risk=target_risk,
                latest_event_id=event_id,
                now=self._replay_event.created_at if self._replay_event else _now(),
                transitioned=True,
                phase_changed=False,
                valid_reassessment=False,
                current_risk=RiskLevel.LOW,
                summary=reason,
                counters={"safe": 0, "low": 0},
                previous_dto=self._to_dto(row) if row else None,
                high_now=False,
                expected_version=(row.version or 0) if row else None,
                human_override=True,
            )
        except Exception:
            self.db.rollback()
            raise
        return self._to_dto(state)

    # ---------------------------------------------------------------- helpers
    def _advance_counters(
        self,
        dto: SafetyStateDTO,
        valid_reassessment: bool,
        current_risk: RiskLevel,
    ) -> dict[str, int]:
        if current_risk >= RiskLevel.MEDIUM:
            return {"safe": 0, "low": 0}
        safe = dto.safe_reassessment_count + (1 if valid_reassessment else 0)
        low = dto.consecutive_low_assessments + (1 if valid_reassessment else 0)
        return {"safe": safe, "low": low}

    def _maybe_downgrade(
        self,
        previous_status: SafetyLifecycleStatus,
        dto: SafetyStateDTO,
        counters: dict[str, int],
        valid_reassessment: bool,
        protective_factors: str,
    ) -> tuple[SafetyLifecycleStatus, RiskLevel | None, bool, str]:
        if not valid_reassessment:
            return previous_status, None, False, ""
        settings = self.settings
        if dto.human_review_required:
            return previous_status, None, False, ""

        if previous_status == SafetyLifecycleStatus.ACTIVE:
            if counters["safe"] >= settings.safety_high_to_monitoring_min_safe_assessments and protective_factors:
                floor = max_risk(RiskLevel.MEDIUM, parse_level(settings.safety_monitoring_risk_floor))
                return SafetyLifecycleStatus.MONITORING, floor, True, "完成有效安全复核且存在保护因素，ACTIVE 降为 MONITORING"
            return previous_status, None, False, ""
        if previous_status == SafetyLifecycleStatus.MONITORING:
            effective = parse_level(dto.effective_risk)
            if counters["low"] >= settings.safety_medium_to_low_min_consecutive_low:
                if effective == RiskLevel.MEDIUM:
                    return SafetyLifecycleStatus.MONITORING, RiskLevel.LOW, True, "连续 LOW 复核，MONITORING 风险下限降为 LOW"
            if effective == RiskLevel.LOW and counters["low"] >= settings.safety_low_to_resolved_min_additional_low:
                return SafetyLifecycleStatus.RESOLVED, RiskLevel.LOW, True, "追加 LOW 安全复核满足条件，MONITORING/LOW 转为 RESOLVED"
            return previous_status, None, False, ""
        return previous_status, None, False, ""

    def _classify_type(self, text: str) -> str:
        lowered = text or ""
        if any(word in lowered for word in ["自杀", "不想活", "轻生", "结束生命", "suicide"]):
            return RiskEventType.SUICIDE.value
        if any(word in lowered for word in ["自残", "伤害自己", "割", "self harm"]):
            return RiskEventType.SELF_HARM.value
        if any(word in lowered for word in ["伤人", "打人", "报复", "harm others"]):
            return RiskEventType.HARM_TO_OTHERS.value
        if any(word in lowered for word in ["危险", "紧急", "马上", "immediate"]):
            return RiskEventType.IMMEDIATE_DANGER.value
        return RiskEventType.OTHER.value

    def _append_event(
        self,
        user_id: int,
        session_id: int,
        message_id: int | None,
        report_id: int | None,
        kind: RiskEventKind,
        risk_type: RiskEventType,
        risk_level: RiskLevel,
        confidence: float,
        evidence: str,
        source: str,
        previous_status: SafetyLifecycleStatus | None,
        new_status: SafetyLifecycleStatus | None,
        payload: dict[str, Any],
        dedupe_key: str | None = None,
    ) -> int | None:
        if self._replay_event is not None:
            return self._replay_event.id
        row = RiskEvent(
            user_id=user_id,
            session_id=session_id or 0,
            message_id=message_id,
            report_id=report_id,
            event_kind=kind.value if isinstance(kind, RiskEventKind) else str(kind),
            risk_type=risk_type.value if isinstance(risk_type, RiskEventType) else str(risk_type),
            risk_level=risk_level.value if isinstance(risk_level, RiskLevel) else str(risk_level),
            confidence=confidence,
            evidence_summary=self.privacy.sanitize(evidence),
            source=source,
            previous_status=previous_status.value if previous_status else None,
            new_status=new_status.value if new_status else None,
            payload_json=json.dumps(payload, ensure_ascii=False),
            dedupe_key=dedupe_key,
            created_at=_now(),
        )
        self.db.add(row)
        self.db.flush()
        return row.id

    def _upsert_state(
        self,
        user_id: int,
        session_id: int,
        status: SafetyLifecycleStatus,
        effective_risk: RiskLevel,
        latest_event_id: int | None,
        now: datetime,
        transitioned: bool,
        phase_changed: bool,
        valid_reassessment: bool,
        current_risk: RiskLevel,
        summary: str,
        counters: dict[str, int],
        previous_dto: SafetyStateDTO | None,
        high_now: bool,
        has_plan_or_intent: bool = False,
        expected_version: int | None = None,
        human_override: bool = False,
    ) -> SafetyState:
        row = self.db.query(SafetyState).filter(SafetyState.user_id == user_id, SafetyState.session_id == session_id).first()
        is_new = row is None
        if row is None:
            row = SafetyState(user_id=user_id, session_id=session_id)
            self.db.add(row)
        elif expected_version is None or row.version != expected_version:
            self.db.rollback()
            raise RuntimeError("SafetyState optimistic lock conflict; assessment must be retried")

        old_next_review_at = row.next_review_at
        if human_override:
            row.normal_turn_count_since_review = 0
            row.safe_reassessment_count = 0
            row.consecutive_low_assessments = 0
            row.reassessment_required = False
            row.human_review_required = False
        else:
            if transitioned or phase_changed:
                # 阶段转换后重置连续复核计数，避免同批评估被重复用于多个降级阶段。
                row.safe_reassessment_count = 0
                row.consecutive_low_assessments = 0
                row.normal_turn_count_since_review = 0
            elif valid_reassessment:
                row.safe_reassessment_count = counters["safe"]
                row.consecutive_low_assessments = counters["low"]
                row.normal_turn_count_since_review = 0
            else:
                row.safe_reassessment_count = counters["safe"]
                row.consecutive_low_assessments = counters["low"]
                if current_risk == RiskLevel.LOW and status in {SafetyLifecycleStatus.ACTIVE, SafetyLifecycleStatus.MONITORING}:
                    row.normal_turn_count_since_review = (previous_dto.normal_turn_count_since_review if previous_dto else 0) + 1
                else:
                    row.normal_turn_count_since_review = 0
        row.history_rebuilt = self._replay_event is None
        row.latest_session_id = session_id or row.latest_session_id
        row.status = status.value
        row.effective_risk = effective_risk.value
        row.latest_event_id = latest_event_id if latest_event_id is not None else row.latest_event_id
        row.last_assessed_at = now
        if high_now:
            row.last_high_risk_at = now
            row.latest_event_severity = self.privacy.sanitize(summary)[:60]
        if current_risk == RiskLevel.LOW and valid_reassessment:
            row.last_safety_confirmation_at = now
        if has_plan_or_intent and status == SafetyLifecycleStatus.ACTIVE:
            row.human_review_required = True
        review_interval = self.settings.safety_review_interval_seconds
        review_turns = self.settings.safety_review_normal_turns
        turns = row.normal_turn_count_since_review
        due_by_time = bool(old_next_review_at and now >= old_next_review_at)
        due_by_turns = status in {SafetyLifecycleStatus.ACTIVE, SafetyLifecycleStatus.MONITORING} and turns >= review_turns
        if human_override or status in {SafetyLifecycleStatus.RESOLVED, SafetyLifecycleStatus.FALSE_POSITIVE}:
            row.reassessment_required = False
            row.next_review_at = None
        elif valid_reassessment or transitioned or phase_changed or high_now or old_next_review_at is None:
            row.reassessment_required = False
            row.next_review_at = now + timedelta(seconds=review_interval)
        else:
            row.reassessment_required = bool(due_by_time or due_by_turns)
            row.next_review_at = old_next_review_at

        next_version = (expected_version or 0) + 1
        row.version = next_version
        row.updated_at = now
        if is_new:
            self.db.flush()
        else:
            values = {
                column.name: getattr(row, column.name)
                for column in SafetyState.__table__.columns
                if column.name not in {"id", "user_id", "session_id", "created_at"}
            }
            # 避免 ORM dirty flush 绕过 CAS；先还原，再执行带 version 条件的 UPDATE。
            self.db.expire(row)
            self.db.flush()
            updated = self.db.execute(
                update(SafetyState)
                .where(SafetyState.id == row.id, SafetyState.version == expected_version)
                .values(**values)
            ).rowcount
            if updated != 1:
                self.db.rollback()
                raise RuntimeError("SafetyState optimistic lock conflict; assessment must be retried")
        self.db.commit()
        row = self.db.query(SafetyState).filter(SafetyState.user_id == user_id, SafetyState.session_id == session_id).first()
        self._write_cache(user_id, session_id, self._to_dto(row))
        return row

    def _to_dto(self, row: SafetyState) -> SafetyStateDTO:
        active = row.status in {SafetyLifecycleStatus.ACTIVE.value, SafetyLifecycleStatus.MONITORING.value}
        due_by_time = bool(active and row.next_review_at and _now() >= row.next_review_at)
        due_by_turns = bool(
            active and (row.normal_turn_count_since_review or 0) >= self.settings.safety_review_normal_turns
        )
        return SafetyStateDTO(
            status=row.status,
            effective_risk=row.effective_risk,
            version=row.version or 0,
            latest_event_id=row.latest_event_id,
            last_high_risk_at=row.last_high_risk_at.isoformat() if row.last_high_risk_at else None,
            last_assessed_at=row.last_assessed_at.isoformat() if row.last_assessed_at else None,
            next_review_at=row.next_review_at.isoformat() if row.next_review_at else None,
            reassessment_required=bool(row.reassessment_required or due_by_time or due_by_turns),
            normal_turn_count_since_review=row.normal_turn_count_since_review or 0,
            safe_reassessment_count=row.safe_reassessment_count or 0,
            consecutive_low_assessments=row.consecutive_low_assessments or 0,
            human_review_required=bool(row.human_review_required),
            latest_event_severity=row.latest_event_severity or "",
        )

    # ------------------------------------------------------------------ cache
    def _connect_redis(self):
        try:
            module = import_module("redis")
            client = module.Redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                socket_timeout=self.settings.redis_socket_timeout_seconds,
                socket_connect_timeout=self.settings.redis_socket_timeout_seconds,
            )
            client.ping()
            return client
        except Exception as exc:
            logger.warning("SafetyState cache disabled: %s", exc)
            return None

    def _read_cache(self, user_id: int, session_id: int) -> SafetyStateDTO | None:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(f"{SAFETY_STATE_CACHE_PREFIX}{user_id}:{session_id}")
            if not raw:
                return None
            data = json.loads(raw)
            if int(data.get("schema", 0)) != SAFETY_STATE_CACHE_SCHEMA:
                return None
            return SafetyStateDTO(
                status=str(data.get("status", "RESOLVED")),
                effective_risk=str(data.get("effectiveRisk", "LOW")),
                version=int(data.get("version", 0)),
                latest_event_id=data.get("latestEventId"),
                last_high_risk_at=data.get("lastHighRiskAt"),
                last_assessed_at=data.get("lastAssessedAt"),
                next_review_at=data.get("nextReviewAt"),
                reassessment_required=bool(data.get("reassessmentRequired", False)),
                normal_turn_count_since_review=int(data.get("normalTurnCountSinceReview", 0)),
                safe_reassessment_count=int(data.get("safeReassessmentCount", 0)),
                consecutive_low_assessments=int(data.get("consecutiveLowAssessments", 0)),
                human_review_required=bool(data.get("humanReviewRequired", False)),
                latest_event_severity=str(data.get("latestEventSeverity", "")),
            )
        except Exception as exc:
            logger.warning("SafetyState cache read failed, fallback to MySQL: %s", exc)
            return None

    def _write_cache(self, user_id: int, session_id: int, dto: SafetyStateDTO) -> None:
        if self._redis is None or self._replay_event is not None:
            return
        try:
            payload = {"schema": SAFETY_STATE_CACHE_SCHEMA, **dto.to_payload(), "updatedAt": _now().isoformat()}
            self._redis.set(
                f"{SAFETY_STATE_CACHE_PREFIX}{user_id}:{session_id}",
                json.dumps(payload, ensure_ascii=False),
                ex=self.settings.safety_state_cache_ttl_seconds,
            )
        except Exception as exc:
            logger.warning("SafetyState cache write failed (MySQL already committed): %s", exc)

    def invalidate_cache(self, user_id: int, session_id: int) -> None:
        if self._redis is None:
            return
        try:
            self._redis.delete(f"{SAFETY_STATE_CACHE_PREFIX}{user_id}:{session_id}")
        except Exception:
            pass
