"""普通对话滚动摘要 + MemorySnapshot 前置加载。

实现 docs/context-memory-safety-redesign-spec.md 第 6、8 节：
- chat_memory_summaries 是普通记忆的权威存储，Redis 只做可重建缓存；
- 8/12 批量窗口：未覆盖消息达到 12 条时，把最早 4 条完整轮次并入摘要，保留最近 8 条原文；
- token 高水位兜底可独立触发压缩；
- 摘要输入 = 旧累计摘要 + 本次移出窗口的消息，绝不重新摘要全量历史；
- covered_through_message_id 单调递增，摘要持久化成功后才推进；
- MemorySnapshot 为不可变 DTO，由 Runtime 在 Coordinator 之前统一加载。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Optional

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.enums import RiskLevel
from app.models.entities import ChatMemorySummary, ChatMessage, RiskEvent, SafetyState
from app.schemas.dtos import AiMessage
from app.services.privacy import PrivacySanitizer
from app.services.safety_memory import RiskEventDTO, SafetyMemoryService, SafetyStateDTO
from app.services.long_term_memory import MemoryRecall

logger = logging.getLogger(__name__)

SUMMARY_CACHE_PREFIX = "mindbridge:chat-summary:"
SUMMARY_KEYS = ["stableFacts", "preferences", "ongoingConcerns", "changes", "supportTried", "unresolvedItems", "safetyReferences"]
MAX_LIST_ITEMS = 12
MAX_ITEM_CHARS = 300


class SummaryError(Exception):
    """摘要生成或校验失败：不得推进覆盖游标。"""


@dataclass(frozen=True)
class MemorySnapshot:
    """本轮 Agent 协作开始时统一看到的状态（不可变）。"""

    history_summary: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    summary_text: str = ""
    covered_through_message_id: Optional[int] = None
    summary_version: int = 0
    recent_messages: tuple[AiMessage, ...] = field(default_factory=tuple)
    safety_state: SafetyStateDTO | None = None
    active_risk_events: tuple[RiskEventDTO, ...] = field(default_factory=tuple)
    rolling_cache_available: bool = True
    safety_cache_available: bool = True
    degraded: bool = False
    degradation_reasons: tuple[str, ...] = field(default_factory=tuple)
    long_term_memory: MemoryRecall = field(default_factory=MemoryRecall)

    def audit_metadata(self) -> dict[str, Any]:
        safety = self.safety_state.to_payload() if self.safety_state else None
        return {
            "longTermMemory": {"state": self.long_term_memory.state,
                "triggered": self.long_term_memory.triggered, "recalledCount": len(self.long_term_memory.items)},
            "summaryVersion": self.summary_version,
            "coveredThroughMessageId": self.covered_through_message_id,
            "recentMessageCount": len(self.recent_messages),
            "safetyState": {"status": safety["status"], "effectiveRisk": safety["effectiveRisk"], "version": safety["version"]} if safety else None,
            "activeRiskEventCount": len(self.active_risk_events),
            "rollingCacheAvailable": self.rolling_cache_available,
            "safetyCacheAvailable": self.safety_cache_available,
            "degraded": self.degraded,
            "degradationReasons": list(self.degradation_reasons),
        }


@dataclass(frozen=True)
class CompactionResult:
    compacted: bool
    reason: str = ""
    covered_through_message_id: Optional[int] = None
    remaining_count: int = 0


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：CJK 字符约 1 token/字，ASCII 约 0.25 token/字符。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if ord(ch) > 0x2E80)
    other = len(text) - cjk
    return max(1, int(cjk * 1.0 + other * 0.25))


def validate_summary_json(data: Any) -> dict[str, Any]:
    """校验并归一化结构化摘要，任何字段异常都会让压缩失败（不推进游标）。"""
    if not isinstance(data, dict):
        raise SummaryError("summary must be a JSON object")
    cleaned: dict[str, Any] = {}
    for key in SUMMARY_KEYS:
        value = data.get(key)
        if key == "supportTried":
            items = []
            if value:
                if not isinstance(value, list):
                    raise SummaryError("supportTried must be a list")
                for item in value[:MAX_LIST_ITEMS]:
                    if not isinstance(item, dict):
                        continue
                    items.append(
                        {
                            "action": str(item.get("action", ""))[:MAX_ITEM_CHARS],
                            "result": str(item.get("result", ""))[:MAX_ITEM_CHARS],
                        }
                    )
            cleaned[key] = items
        elif key == "safetyReferences":
            items = []
            if value:
                if not isinstance(value, list):
                    raise SummaryError("safetyReferences must be a list")
                for item in value[:MAX_LIST_ITEMS]:
                    if isinstance(item, dict) and item.get("eventId") is not None:
                        items.append({"eventId": int(item["eventId"]), "status": str(item.get("status", "ACTIVE"))[:32]})
            cleaned[key] = items
        else:
            if value is None:
                cleaned[key] = []
                continue
            if not isinstance(value, list):
                raise SummaryError(f"{key} must be a list")
            cleaned[key] = [str(item)[:MAX_ITEM_CHARS] for item in value[:MAX_LIST_ITEMS]]
    return cleaned


def render_summary_text(summary: dict[str, Any]) -> str:
    labels = {
        "stableFacts": "稳定背景",
        "preferences": "偏好",
        "ongoingConcerns": "持续关注",
        "changes": "变化",
        "supportTried": "已提供信息与支持",
        "unresolvedItems": "未解决事项",
        "safetyReferences": "安全事件引用",
    }
    parts = []
    for key in SUMMARY_KEYS:
        value = summary.get(key) or []
        if not value:
            continue
        if key == "supportTried":
            rendered = "；".join(f"{item.get('action', '')}({item.get('result', '')})" for item in value if item.get("action"))
        elif key == "safetyReferences":
            rendered = "；".join(f"#{item.get('eventId')}:{item.get('status')}" for item in value)
        else:
            rendered = "；".join(str(item) for item in value)
        if rendered:
            parts.append(f"{labels[key]}：{rendered}")
    return "\n".join(parts) if parts else "暂无摘要内容。"


def empty_summary() -> dict[str, Any]:
    return {key: [] for key in SUMMARY_KEYS}


def freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze_json(item) for item in value)
    return value


class RollingSummaryService:
    """12→8 批量压缩 + token 高水位兜底 + 乐观锁持久化。"""

    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.privacy = PrivacySanitizer()
        self._redis = self._connect_redis()

    # ------------------------------------------------------------------ read
    def load_state(self, session_id: int) -> ChatMemorySummary | None:
        return self.db.query(ChatMemorySummary).filter(ChatMemorySummary.session_id == session_id).first()

    def uncovered_messages(self, session_id: int, covered_through: int | None, exclude_message_id: int | None = None) -> list[ChatMessage]:
        query = self.db.query(ChatMessage).filter(ChatMessage.session_id == session_id)
        if covered_through is not None:
            query = query.filter(ChatMessage.id > covered_through)
        rows = query.order_by(ChatMessage.id.asc()).all()
        rows = [row for row in rows if not (row.role.lower() == "assistant" and row.completion_status == "INTERRUPTED")]
        if exclude_message_id is not None:
            rows = [row for row in rows if row.id != exclude_message_id]
        return rows

    # ------------------------------------------------------------- compaction
    def compact_if_needed(
        self,
        session_id: int,
        force_token_budget: bool = False,
        extra_tokens: int = 0,
        budget_limit_tokens: int | None = None,
    ) -> CompactionResult:
        if not self.settings.memory_rolling_summary_enabled:
            return CompactionResult(False, "rolling summary disabled")
        settings = self.settings
        state = self.load_state(session_id)
        covered = state.covered_through_message_id if state else None
        uncovered = self.uncovered_messages(session_id, covered)
        complete = self._exclude_incomplete_tail(uncovered)
        current_message = uncovered[-1] if uncovered and (complete and uncovered[-1].id != complete[-1].id) else None

        target = settings.memory_recent_target_messages
        trigger = settings.memory_compaction_trigger_messages
        pressure = False
        if force_token_budget:
            summary_tokens = estimate_tokens(state.summary_text) if state and state.summary_text else 0
            projected = summary_tokens + self._estimate_window_tokens(complete) + extra_tokens
            budget = budget_limit_tokens or self._high_watermark_tokens()
            pressure = projected >= budget

        if len(complete) < trigger and not pressure:
            return CompactionResult(False, "below trigger", covered, len(uncovered))

        evict_count = len(complete) - target
        if pressure and evict_count <= 0:
            evict_count = 2
        evict_count = min(evict_count, len(complete))
        evict_count = self._extend_to_complete_turn(complete, evict_count)
        if evict_count <= 0:
            return CompactionResult(False, "no complete turn to evict", covered, len(uncovered))
        evicted = complete[:evict_count]
        remaining = complete[evict_count:]

        previous_json = json.loads(state.summary_json) if state and state.summary_json else empty_summary()
        self.scope_safety_references(previous_json, session_id)
        current_safety_status = self._current_safety_status(session_id)
        if current_safety_status:
            previous_json["safetyReferences"] = [
                {"eventId": item.get("eventId"), "status": current_safety_status}
                for item in previous_json.get("safetyReferences", [])
                if isinstance(item, dict) and item.get("eventId") is not None
            ]
        previous_json["safetyReferences"] = self._merge_safety_references(
            previous_json.get("safetyReferences", []),
            self._active_safety_references(session_id),
        )
        try:
            new_summary = self._merge_summary(previous_json, evicted)
        except SummaryError as exc:
            logger.warning("summary merge failed; covered cursor NOT advanced: %s", exc)
            return CompactionResult(False, f"summary failed: {exc}", covered, len(uncovered))

        self._compare_and_swap(
            state=state,
            session_id=session_id,
            summary=new_summary,
            covered_through=evicted[-1].id,
        )
        return CompactionResult(True, "compacted", evicted[-1].id, len(remaining) + (1 if current_message else 0))

    def ensure_prompt_budget(
        self,
        session_id: int,
        fixed_tokens: int,
        rag_tokens: int,
        current_input_tokens: int,
    ) -> tuple[bool, list[str]]:
        """下一轮模型调用前兜底：常规压缩 + 高水位继续驱逐 + 返回降级说明。"""
        degradations: list[str] = []
        compacted = False
        # 先处理正常的 12→8 维护，再以真实 token 水位循环驱逐完整轮次。
        normal = self.compact_if_needed(session_id)
        compacted = normal.compacted
        if normal.compacted:
            degradations.append("pre-call batch compaction")

        extra = fixed_tokens + rag_tokens + current_input_tokens
        usable = self.settings.model_context_window_tokens - self.settings.reserved_output_tokens
        high = int(usable * self.settings.memory_prompt_high_watermark_ratio)
        target = int(usable * self.settings.memory_prompt_target_ratio)

        def projected_tokens() -> int:
            state = self.load_state(session_id)
            covered = state.covered_through_message_id if state else None
            summary_tokens = estimate_tokens(state.summary_text) if state and state.summary_text else 0
            complete = self._exclude_incomplete_tail(self.uncovered_messages(session_id, covered))
            return summary_tokens + self._estimate_window_tokens(complete) + extra

        projected = projected_tokens()
        if projected < high:
            return compacted, degradations

        for _ in range(64):
            result = self.compact_if_needed(
                session_id,
                force_token_budget=True,
                extra_tokens=extra,
                budget_limit_tokens=target,
            )
            if not result.compacted:
                break
            compacted = True
            projected = projected_tokens()
            degradations.append(f"token-pressure compaction; projected={projected}; target={target}")
            if projected <= target:
                break
        if projected > target:
            message = f"prompt memory remains over target after compaction: projected={projected}, target={target}"
            logger.error(message)
            degradations.append(message)
        return compacted, degradations

    # ------------------------------------------------------------ merge logic
    def _merge_summary(self, previous: dict[str, Any], evicted: list[ChatMessage]) -> dict[str, Any]:
        try:
            merged = self._llm_merge(previous, evicted)
            summary = validate_summary_json(merged)
        except SummaryError:
            raise
        except Exception as exc:
            # 任何摘要失败都不允许推进覆盖游标（规格 6.4/15.2）。
            raise SummaryError(f"summary merge failed: {type(exc).__name__}: {exc}") from exc
        # LLM 不得删除尚未解除的安全事件引用。
        summary["safetyReferences"] = self._merge_safety_references(
            previous.get("safetyReferences", []), summary.get("safetyReferences", [])
        )
        if estimate_tokens(render_summary_text(summary)) > self.settings.memory_summary_max_tokens:
            raise SummaryError("summary exceeds MEMORY_SUMMARY_MAX_TOKENS budget")
        return summary

    def _active_safety_references(self, session_id: int) -> list[dict[str, Any]]:
        first_message = self.db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.id.asc()).first()
        if first_message is None:
            return []
        state = self.db.query(SafetyState).filter(SafetyState.user_id == first_message.user_id, SafetyState.session_id == session_id).first()
        if state is None or state.status not in {"ACTIVE", "MONITORING"}:
            return []
        rows = (
            self.db.query(RiskEvent)
            .filter(RiskEvent.user_id == first_message.user_id, RiskEvent.session_id == session_id)
            .order_by(RiskEvent.id.desc())
            .limit(8)
            .all()
        )
        return [{"eventId": row.id, "status": state.status} for row in reversed(rows)]

    def scope_safety_references(self, summary: dict, session_id: int) -> None:
        references = summary.get("safetyReferences") or []
        ids = [r["eventId"] for r in references if isinstance(r, dict) and r.get("eventId") is not None]
        owned = {r.id for r in self.db.query(RiskEvent).filter(RiskEvent.session_id == session_id, RiskEvent.id.in_(ids)).all()} if ids else set()
        status = self._current_safety_status(session_id)
        summary["safetyReferences"] = [{**r, **({"status": status} if status else {})} for r in references
            if isinstance(r, dict) and r.get("eventId") in owned]

    def _current_safety_status(self, session_id: int) -> str:
        first_message = self.db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.id.asc()).first()
        if first_message is None:
            return ""
        state = self.db.query(SafetyState).filter(SafetyState.user_id == first_message.user_id, SafetyState.session_id == session_id).first()
        return state.status if state else ""

    def _merge_safety_references(self, left: Any, right: Any) -> list[dict[str, Any]]:
        merged: dict[int, dict[str, Any]] = {}
        # left 是服务端已有/权威引用，必须覆盖 LLM 返回的同 ID 状态。
        for item in [*(right or []), *(left or [])]:
            if isinstance(item, dict) and item.get("eventId") is not None:
                merged[int(item["eventId"])] = {
                    "eventId": int(item["eventId"]),
                    "status": str(item.get("status", "ACTIVE"))[:32],
                }
        return list(merged.values())[-MAX_LIST_ITEMS:]

    def _llm_merge(self, previous: dict[str, Any], evicted: list[ChatMessage]) -> Any:
        from app.services.ai import AiClient

        transcript = "\n".join(f"{row.role}: {self.privacy.sanitize(row.content)}" for row in evicted)
        prompt_messages = [
            AiMessage(role="system", content=(
                "你负责把“已有累计摘要”和“新移出近期窗口的消息”合并为新的累计摘要。"
                "只输出符合指定 schema 的 JSON："
                '{"stableFacts":[],"preferences":[],"ongoingConcerns":[],"changes":[],'
                '"supportTried":[{"action":"","result":""}],"unresolvedItems":[],"safetyReferences":[{"eventId":0,"status":""}]}。'
                "不得虚构；不得诊断；不得把推测写成事实。"
                "保留用户明确身份、政策主题、已确认适用条件、持续问题、变化、已提供信息与支持、未解决事项。当前用户明确更正优先，旧条件应更新；不要把助手给出的政策金额存为用户背景或现行政策事实。"
                "风险信息只保留引用和必要概述，不得删除已有未解除引用。删除闲聊噪声。"
            )),
            AiMessage(role="user", content=(
                f"已有累计摘要：\n{json.dumps(previous, ensure_ascii=False)}\n\n"
                f"新移出窗口的消息：\n{transcript}\n\n请输出合并后的新累计摘要 JSON。"
            )),
        ]
        raw = AiClient(self.settings).complete(prompt_messages)
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise SummaryError("summary model returned non-JSON output")
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise SummaryError(f"summary model returned invalid JSON: {exc}") from exc

    def _compare_and_swap(
        self,
        state: ChatMemorySummary | None,
        session_id: int,
        summary: dict[str, Any],
        covered_through: int,
    ) -> None:
        if state is None:
            state = ChatMemorySummary(
                session_id=session_id,
                user_id=self.db.query(ChatMessage).filter(ChatMessage.session_id == session_id).first().user_id,
            )
            self.db.add(state)
            self.db.flush()
        expected_version = state.version or 0
        current_covered = state.covered_through_message_id or 0
        if covered_through <= current_covered:
            raise SummaryError("covered cursor must move forward")
        rows_updated = self.db.query(ChatMemorySummary).filter(
            ChatMemorySummary.id == state.id,
            ChatMemorySummary.version == expected_version,
        ).update(
            {
                "summary_json": json.dumps(summary, ensure_ascii=False),
                "summary_text": render_summary_text(summary),
                "covered_through_message_id": covered_through,
                "summary_schema_version": self.settings.memory_summary_schema_version,
                "version": expected_version + 1,
                "updated_at": datetime.utcnow(),
            }
        )
        if rows_updated == 0:
            raise SummaryError("optimistic lock conflict on summary update")
        self.db.commit()
        self.db.refresh(state)
        self._write_cache(state)

    def _exclude_incomplete_tail(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """未得到回复的尾部 user 消息不参与常规压缩。"""
        complete = list(messages)
        while complete and complete[-1].role.lower() == "user":
            complete.pop()
        return complete

    def _extend_to_complete_turn(self, messages: list[ChatMessage], evict_count: int) -> int:
        """驱逐边界向后延伸到 assistant 结束，保证以完整轮次为单位。"""
        if evict_count <= 0 or evict_count >= len(messages):
            return 0 if evict_count >= len(messages) else evict_count
        while evict_count < len(messages) and messages[evict_count - 1].role.lower() != "assistant":
            evict_count += 1
        return evict_count

    def _estimate_window_tokens(self, messages: list[ChatMessage]) -> int:
        return sum(estimate_tokens(row.content) for row in messages)

    def _high_watermark_tokens(self) -> int:
        s = self.settings
        usable = s.model_context_window_tokens - s.reserved_output_tokens
        return int(usable * s.memory_prompt_high_watermark_ratio)

    # ------------------------------------------------------------------ cache
    def _connect_redis(self):
        try:
            from importlib import import_module

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
            logger.warning("Rolling summary cache disabled: %s", exc)
            return None

    def _write_cache(self, state: ChatMemorySummary) -> None:
        if self._redis is None:
            return
        try:
            payload = {
                "schema": self.settings.memory_summary_schema_version,
                "summaryJson": state.summary_json,
                "summaryText": state.summary_text,
                "coveredThroughMessageId": state.covered_through_message_id,
                "version": state.version,
                "updatedAt": datetime.utcnow().isoformat(),
            }
            self._redis.set(
                f"{SUMMARY_CACHE_PREFIX}{state.session_id}",
                json.dumps(payload, ensure_ascii=False),
                ex=self.settings.redis_memory_ttl_seconds,
            )
        except Exception as exc:
            logger.warning("Rolling summary cache write failed (MySQL already committed): %s", exc)


class MemoryContextService:
    """Runtime 前置加载 MemorySnapshot；assistant 保存后执行常规压缩。"""

    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.rolling = RollingSummaryService(db, settings)
        self.safety = SafetyMemoryService(db, settings)

    def load_snapshot(
        self,
        user_id: int,
        session_id: int,
        current_message_id: int | None = None,
    ) -> MemorySnapshot:
        reasons: list[str] = []
        try:
            state = self.rolling.load_state(session_id)
            summary_json = json.loads(state.summary_json) if state and state.summary_json else empty_summary()
            self.rolling.scope_safety_references(summary_json, session_id)
            covered = state.covered_through_message_id if state else None
            summary_version = state.version if state else 0
        except Exception as exc:
            logger.warning("load summary state failed: %s", exc)
            summary_json, covered, summary_version = empty_summary(), None, 0
            reasons.append(f"summary load failed: {type(exc).__name__}")

        try:
            uncovered = self.rolling.uncovered_messages(session_id, covered, exclude_message_id=current_message_id)
            recent = tuple(
                AiMessage(role=row.role.lower(), content=self.rolling.privacy.sanitize(row.content))
                for row in uncovered[-self.settings.redis_memory_max_messages :]
            )
        except Exception as exc:
            logger.warning("load recent messages failed: %s", exc)
            recent, reasons = tuple(), reasons + [f"recent load failed: {type(exc).__name__}"]

        try:
            safety_dto = self.safety.get_state(user_id, session_id) if self.settings.safety_memory_enabled else None
            events = tuple(self.safety.active_events(user_id, session_id)) if safety_dto and safety_dto.status in {"ACTIVE", "MONITORING"} else tuple()
            if safety_dto and safety_dto.degraded:
                reasons.append("safety state loaded from degraded cache fallback")
        except Exception as exc:
            logger.warning("load safety state failed: %s", exc)
            # 权威安全状态不可读时不能把“无数据”解释为 LOW。
            safety_dto = SafetyStateDTO(
                status="ACTIVE",
                effective_risk=RiskLevel.HIGH.value,
                reassessment_required=True,
                human_review_required=True,
                degraded=True,
            )
            events = tuple()
            reasons.append(f"safety state load failed: {type(exc).__name__}")

        return MemorySnapshot(
            history_summary=freeze_json(summary_json),
            summary_text=render_summary_text(summary_json),
            covered_through_message_id=covered,
            summary_version=summary_version,
            recent_messages=recent,
            safety_state=safety_dto,
            active_risk_events=events,
            rolling_cache_available=self.rolling._redis is not None,
            safety_cache_available=self.safety._redis is not None,
            degraded=bool(reasons),
            degradation_reasons=tuple(reasons),
        )

    def compact_after_turn(self, session_id: int) -> CompactionResult:
        return self.rolling.compact_if_needed(session_id)
