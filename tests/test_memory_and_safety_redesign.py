import json
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.core.database import Base
from app.core.enums import (
    RiskLevel,
    SafetyLifecycleStatus,
)
from app.models.entities import (
    ChatMessage,
    ChatSession,
    MemoryMaintenanceJob,
    RiskEvent,
    SafetyState,
    UserAccount,
)


def make_settings(**overrides):
    base = {
        "memory_rolling_summary_enabled": True,
        "memory_recent_target_messages": 8,
        "memory_compaction_trigger_messages": 12,
        "memory_summary_max_tokens": 600,
        "memory_summary_schema_version": 1,
        "memory_prompt_high_watermark_ratio": 0.65,
        "memory_prompt_target_ratio": 0.40,
        "model_context_window_tokens": 32768,
        "reserved_output_tokens": 1024,
        "rag_skill_budget_tokens": 1200,
        "redis_memory_ttl_seconds": 86400,
        "redis_memory_max_messages": 40,
        "safety_memory_enabled": True,
        "safety_monitoring_risk_floor": "MEDIUM",
        "safety_review_interval_seconds": 86400,
        "safety_review_normal_turns": 10,
        "safety_high_to_monitoring_min_safe_assessments": 2,
        "safety_medium_to_low_min_consecutive_low": 2,
        "safety_low_to_resolved_min_additional_low": 1,
        "ai_provider": "mock",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def isolated_db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = factory()
    return db, engine


def make_user_session_messages(db, message_count, safe_tail_user=False):
    user = UserAccount(username="u1", display_name="学生", password_hash="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    session = ChatSession(public_id="s-public-1", user_id=user.id, title="测试会话")
    db.add(session)
    db.commit()
    db.refresh(session)
    for index in range(message_count):
        role = "USER" if index % 2 == 0 else "ASSISTANT"
        db.add(ChatMessage(user_id=user.id, session_id=session.id, role=role, content=f"第{index}条消息内容"))
    if safe_tail_user:
        db.add(ChatMessage(user_id=user.id, session_id=session.id, role="USER", content="刚发出的新消息"))
    db.commit()
    return user, session


class RollingSummaryTests(unittest.TestCase):
    def setUp(self):
        self.db, self.engine = isolated_db()
        settings = make_settings()
        from app.services.memory_context import RollingSummaryService

        self.service = RollingSummaryService(self.db, settings)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_12_uncovered_messages_compact_oldest_turns_and_keep_8(self):
        user, session = make_user_session_messages(self.db, 12)
        result = self.service.compact_if_needed(session.id)
        self.assertTrue(result.compacted)
        state = self.service.load_state(session.id)
        self.assertIsNotNone(state)
        self.assertGreater(state.version, 0)
        summary = json.loads(state.summary_json)
        self.assertIn("ongoingConcerns", summary)
        remaining = self.service.uncovered_messages(session.id, state.covered_through_message_id)
        self.assertLessEqual(len(remaining), 8)
        # 以完整轮次为单位：游标应停在 ASSISTANT 消息上
        cursor_row = self.db.query(ChatMessage).filter(ChatMessage.id == state.covered_through_message_id).first()
        self.assertEqual(cursor_row.role, "ASSISTANT")

    def test_below_trigger_does_not_compact(self):
        _, session = make_user_session_messages(self.db, 11)
        result = self.service.compact_if_needed(session.id)
        self.assertFalse(result.compacted)
        self.assertIsNone(self.service.load_state(session.id))

    def test_unanswered_user_tail_is_not_compacted(self):
        user, session = make_user_session_messages(self.db, 12, safe_tail_user=True)
        result = self.service.compact_if_needed(session.id)
        self.assertTrue(result.compacted)
        state = self.service.load_state(session.id)
        tail = self.db.query(ChatMessage).filter(ChatMessage.content == "刚发出的新消息").first()
        self.assertLess(tail.id, state.covered_through_message_id + 10**9)
        self.assertGreater(tail.id, state.covered_through_message_id)

    def test_covered_cursor_moves_forward_only(self):
        user, session = make_user_session_messages(self.db, 12)
        first = self.service.compact_if_needed(session.id)
        covered_first = first.covered_through_message_id
        # 重复压缩：没有新的未覆盖消息时不再变化
        second = self.service.compact_if_needed(session.id)
        self.assertFalse(second.compacted)
        self.assertEqual(second.covered_through_message_id, covered_first)

    def test_invalid_llm_output_does_not_advance_cursor(self):
        user, session = make_user_session_messages(self.db, 12)

        def broken_merge(previous, evicted):
            raise ValueError("bad json")

        original = self.service._llm_merge
        self.service._llm_merge = broken_merge
        try:
            result = self.service.compact_if_needed(session.id)
        finally:
            self.service._llm_merge = original
        self.assertFalse(result.compacted)
        self.assertIsNone(self.service.load_state(session.id))

    def test_second_compaction_merges_into_previous_summary(self):
        user, session = make_user_session_messages(self.db, 12)
        self.service.compact_if_needed(session.id)
        version_after_first = self.service.load_state(session.id).version
        covered_after_first = self.service.load_state(session.id).covered_through_message_id
        # 追加 12 条新的完整轮次 → 再次达到触发阈值
        for offset in range(12):
            role = "USER" if offset % 2 == 0 else "ASSISTANT"
            self.db.add(ChatMessage(user_id=user.id, session_id=session.id, role=role, content=f"新增第{offset}条"))
        self.db.commit()
        result = self.service.compact_if_needed(session.id)
        self.assertTrue(result.compacted)
        state2 = self.service.load_state(session.id)
        self.assertGreater(state2.version, version_after_first)
        self.assertGreater(state2.covered_through_message_id, covered_after_first)

    def test_token_budget_repeats_compaction_until_target(self):
        self.service.settings.memory_compaction_trigger_messages = 100
        self.service.settings.model_context_window_tokens = 10000
        self.service.settings.reserved_output_tokens = 1000
        user, session = make_user_session_messages(self.db, 0)
        for index in range(6):
            self.db.add(ChatMessage(
                user_id=user.id,
                session_id=session.id,
                role="USER" if index % 2 == 0 else "ASSISTANT",
                content="长" * 1000,
            ))
        self.db.commit()
        compacted, reasons = self.service.ensure_prompt_budget(session.id, 0, 0, 0)
        self.assertTrue(compacted)
        self.assertTrue(any("token-pressure" in reason for reason in reasons))
        state = self.service.load_state(session.id)
        remaining = self.service.uncovered_messages(session.id, state.covered_through_message_id)
        from app.services.memory_context import estimate_tokens
        projected = estimate_tokens(state.summary_text) + sum(estimate_tokens(row.content) for row in remaining)
        self.assertLessEqual(projected, 3600)

    def test_safety_references_are_preserved_and_status_is_refreshed(self):
        from app.services.safety_memory import SafetyMemoryService

        user, session = make_user_session_messages(self.db, 12)
        safety = SafetyMemoryService(self.db, self.service.settings)
        safety.record_turn_assessment(
            user_id=user.id,
            session_id=session.id,
            message_id=None,
            report_id=None,
            current_risk=RiskLevel.HIGH,
            confidence=0.95,
            summary="高风险",
            risk_type="SUICIDE",
            evidence_summary="不想活了",
        )
        self.service.compact_if_needed(session.id)
        first = json.loads(self.service.load_state(session.id).summary_json)
        self.assertTrue(first["safetyReferences"])
        self.assertTrue(all(item["status"] == "ACTIVE" for item in first["safetyReferences"]))

        safety.manual_override(user.id, "admin", "RESOLVED", "人工确认解除", session.id)
        for index in range(12):
            self.db.add(ChatMessage(
                user_id=user.id,
                session_id=session.id,
                role="USER" if index % 2 == 0 else "ASSISTANT",
                content=f"后续消息{index}",
            ))
        self.db.commit()
        self.service.compact_if_needed(session.id)
        second = json.loads(self.service.load_state(session.id).summary_json)
        self.assertTrue(all(item["status"] == "RESOLVED" for item in second["safetyReferences"]))


class SafetyMemoryTests(unittest.TestCase):
    def setUp(self):
        self.db, self.engine = isolated_db()
        self.settings = make_settings()
        from app.services.safety_memory import SafetyMemoryService

        self.service = SafetyMemoryService(self.db, self.settings)
        self.user = UserAccount(username="s1", display_name="学生", password_hash="x")
        self.db.add(self.user)
        self.db.commit()
        self.db.refresh(self.user)
        self.session = ChatSession(public_id="sp-1", user_id=self.user.id, title="t")
        self.db.add(self.session)
        self.db.commit()
        self.db.refresh(self.session)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def assess(self, risk: RiskLevel, summary: str = "", safety_reassessment: bool = False, protective: str = ""):
        return self.service.record_turn_assessment(
            user_id=self.user.id,
            session_id=self.session.id,
            message_id=None,
            report_id=None,
            current_risk=risk,
            confidence=0.9,
            summary=summary,
            safety_reassessment=safety_reassessment,
            protective_factors=protective,
        )

    def test_high_creates_risk_event_and_active_state(self):
        decision = self.assess(RiskLevel.HIGH, summary="想自杀")
        self.assertEqual(decision.effective_risk, RiskLevel.HIGH)
        self.assertEqual(decision.new_status, SafetyLifecycleStatus.ACTIVE)
        state = self.service.get_state(self.user.id, self.session.id)
        self.assertEqual(state.status, "ACTIVE")
        self.assertEqual(state.effective_risk, "HIGH")
        events = self.db.query(RiskEvent).filter(RiskEvent.user_id == self.user.id).all()
        self.assertTrue(any(e.event_kind == "RISK_DETECTED" for e in events))

    def test_high_then_low_chat_inherits_carryover(self):
        self.assess(RiskLevel.HIGH, summary="不想活了")
        decision = self.assess(RiskLevel.LOW, summary="帮我看看 Python 报错")
        self.assertEqual(decision.current_risk, RiskLevel.LOW)
        self.assertEqual(decision.carryover_risk, RiskLevel.HIGH)
        self.assertEqual(decision.effective_risk, RiskLevel.HIGH)

    def test_active_high_downgrades_to_monitoring_with_valid_reassessment(self):
        self.assess(RiskLevel.HIGH, summary="想自杀")
        first = self.assess(
            RiskLevel.LOW,
            summary="我现在安全，有人陪着，已经联系了心理中心",
            safety_reassessment=True,
            protective="有人陪伴，已联系心理中心",
        )
        self.assertEqual(first.new_status, SafetyLifecycleStatus.ACTIVE)
        decision = self.assess(
            RiskLevel.LOW,
            summary="我现在仍然安全，有人陪着",
            safety_reassessment=True,
            protective="有人陪伴",
        )
        self.assertEqual(decision.new_status, SafetyLifecycleStatus.MONITORING)
        # MONITORING 下限 MEDIUM：即使 current LOW，effective 至少 MEDIUM
        self.assertGreaterEqual(
            {"LOW": 1, "MEDIUM": 2, "HIGH": 3}[decision.effective_risk.value],
            {"LOW": 1, "MEDIUM": 2, "HIGH": 3}[RiskLevel.MEDIUM],
        )

    def test_monitoring_requires_consecutive_low_before_floor_drop(self):
        self.assess(RiskLevel.HIGH, summary="想自杀")
        self.assess(RiskLevel.LOW, summary="我现在安全了，有人陪", safety_reassessment=True, protective="有人陪伴")
        self.assess(RiskLevel.LOW, summary="我现在仍然安全，有人陪", safety_reassessment=True, protective="有人陪伴")
        # 只有 1 次连续 LOW：不能降到 LOW 下限
        decision = self.assess(RiskLevel.LOW, summary="普通问题")
        # 第二次 LOW 复核未带安全确认 → 不计为有效复核
        self.assertEqual(decision.effective_risk, RiskLevel.MEDIUM)

    def test_resolved_does_not_inherit_risk(self):
        self.assess(RiskLevel.HIGH, summary="想自杀")
        self.service.manual_override(self.user.id, actor="admin", action="RESOLVED", reason="人工确认处理完成", session_id=self.session.id)
        decision = self.assess(RiskLevel.LOW, summary="普通聊天")
        self.assertEqual(decision.carryover_risk, RiskLevel.LOW)
        self.assertEqual(decision.effective_risk, RiskLevel.LOW)

    def test_false_positive_does_not_inherit_risk(self):
        self.assess(RiskLevel.HIGH, summary="想自杀")
        self.service.manual_override(self.user.id, actor="admin", action="FALSE_POSITIVE", reason="确认为误判", session_id=self.session.id)
        decision = self.assess(RiskLevel.LOW, summary="普通聊天")
        self.assertEqual(decision.effective_risk, RiskLevel.LOW)

    def test_manual_override_appends_event_and_resets_counters(self):
        self.assess(RiskLevel.HIGH, summary="想自杀")
        dto = self.service.manual_override(self.user.id, actor="admin", action="RESOLVED", reason="处理完成", session_id=self.session.id)
        self.assertEqual(dto.status, "RESOLVED")
        self.assertEqual(dto.safe_reassessment_count, 0)
        events = self.db.query(RiskEvent).filter(RiskEvent.event_kind == "MANUAL_OVERRIDE").all()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source, "HUMAN")

    def test_state_version_increments_each_turn(self):
        self.assess(RiskLevel.HIGH, summary="想自杀")
        first = self.service.get_state(self.user.id, self.session.id)
        self.assess(RiskLevel.LOW, summary="普通问题")
        second = self.service.get_state(self.user.id, self.session.id)
        self.assertGreater(second.version, first.version)

    def test_monitoring_new_high_returns_to_active(self):
        self.assess(RiskLevel.HIGH, summary="想自杀")
        self.assess(RiskLevel.LOW, summary="我现在安全，有人陪", safety_reassessment=True, protective="有人陪伴")
        self.assess(RiskLevel.LOW, summary="我现在仍安全，有人陪", safety_reassessment=True, protective="有人陪伴")
        self.assertEqual(self.service.get_state(self.user.id, self.session.id).status, "MONITORING")
        decision = self.assess(RiskLevel.HIGH, summary="又出现强烈的自杀念头")
        self.assertEqual(decision.new_status, SafetyLifecycleStatus.ACTIVE)
        self.assertEqual(self.service.get_state(self.user.id, self.session.id).status, "ACTIVE")

    def test_full_staged_downgrade_reaches_resolved(self):
        self.assess(RiskLevel.HIGH, summary="想自杀")
        for _ in range(2):
            self.assess(RiskLevel.LOW, summary="当前安全且有人陪", safety_reassessment=True, protective="有人陪伴")
        self.assertEqual(self.service.get_state(self.user.id, self.session.id).status, "MONITORING")
        self.assertEqual(self.service.get_state(self.user.id, self.session.id).effective_risk, "MEDIUM")
        for _ in range(2):
            self.assess(RiskLevel.LOW, summary="当前安全且有人陪", safety_reassessment=True, protective="有人陪伴")
        self.assertEqual(self.service.get_state(self.user.id, self.session.id).status, "MONITORING")
        self.assertEqual(self.service.get_state(self.user.id, self.session.id).effective_risk, "LOW")
        self.assess(RiskLevel.LOW, summary="当前仍然安全", safety_reassessment=True, protective="有人陪伴")
        self.assertEqual(self.service.get_state(self.user.id, self.session.id).status, "RESOLVED")

    def test_plan_or_intent_requires_human_review(self):
        self.service.record_turn_assessment(
            user_id=self.user.id,
            session_id=self.session.id,
            message_id=None,
            report_id=None,
            current_risk=RiskLevel.HIGH,
            confidence=0.99,
            summary="检测到明确高风险表达",
            has_plan_or_intent=True,
            risk_type="SUICIDE",
            evidence_summary="已经准备了实施手段",
        )
        self.assertTrue(self.service.get_state(self.user.id, self.session.id).human_review_required)
        for _ in range(4):
            self.assess(RiskLevel.LOW, summary="当前安全且有人陪", safety_reassessment=True, protective="有人陪伴")
        self.assertEqual(self.service.get_state(self.user.id, self.session.id).status, "ACTIVE")
        dto = self.service.manual_override(self.user.id, "admin", "MONITORING", "专业人员确认可进入监测", self.session.id)
        self.assertEqual(dto.status, "MONITORING")
        self.assertFalse(dto.human_review_required)

    def test_negated_plan_phrase_is_not_treated_as_plan(self):
        from app.services.assessment import detect_plan_or_intent

        self.assertFalse(detect_plan_or_intent("我现在没有计划，也不会伤害自己"))
        self.assertTrue(detect_plan_or_intent("我已经准备好了，今晚就实施具体计划"))

    def test_review_deadline_is_not_postponed_by_normal_chat(self):
        self.assess(RiskLevel.HIGH, summary="想自杀")
        row = self.db.query(SafetyState).filter(SafetyState.user_id == self.user.id).first()
        original_deadline = datetime.utcnow() + timedelta(hours=1)
        row.next_review_at = original_deadline
        self.db.commit()
        self.assess(RiskLevel.LOW, summary="普通聊天")
        row = self.db.query(SafetyState).filter(SafetyState.user_id == self.user.id).first()
        self.assertEqual(row.next_review_at, original_deadline)
        self.assertEqual(row.normal_turn_count_since_review, 1)

    def test_elapsed_review_deadline_is_visible_in_next_snapshot(self):
        self.assess(RiskLevel.HIGH, summary="想自杀")
        row = self.db.query(SafetyState).filter(SafetyState.user_id == self.user.id).first()
        row.next_review_at = datetime.utcnow() - timedelta(seconds=1)
        row.reassessment_required = False
        self.db.commit()
        self.assertTrue(self.service.get_state(self.user.id, self.session.id).reassessment_required)

    def test_same_message_safety_assessment_is_idempotent(self):
        message = ChatMessage(
            user_id=self.user.id,
            session_id=self.session.id,
            role="USER",
            content="我不想活了",
            client_message_id="safety-retry-1",
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        kwargs = dict(
            user_id=self.user.id,
            session_id=self.session.id,
            message_id=message.id,
            report_id=None,
            current_risk=RiskLevel.HIGH,
            confidence=0.95,
            summary="检测到高风险",
            risk_type="SUICIDE",
            evidence_summary="不想活了",
        )
        self.service.record_turn_assessment(**kwargs)
        version = self.service.get_state(self.user.id, self.session.id).version
        self.service.record_turn_assessment(**kwargs)
        self.assertEqual(self.service.get_state(self.user.id, self.session.id).version, version)
        primary = self.db.query(RiskEvent).filter(RiskEvent.dedupe_key == f"message:{message.id}:assessment").all()
        self.assertEqual(len(primary), 1)


class MemorySnapshotIntegrationTests(unittest.TestCase):
    def setUp(self):
        from app.core.config import Settings

        self.db, self.engine = isolated_db()
        # 集成链路用完整 Settings（pydantic），仅覆盖外部依赖和 provider。
        self.settings = Settings(
            ai_provider="mock",
            knowledge_vector_enabled=False,
            redis_url="redis://127.0.0.1:1/15",
            redis_socket_timeout_seconds=0.05,
        )
        self.user = UserAccount(username="i1", display_name="学生", password_hash="x")
        self.db.add(self.user)
        self.db.commit()
        self.db.refresh(self.user)
        self.session = ChatSession(public_id="ip-1", user_id=self.user.id, title="t")
        self.db.add(self.session)
        self.db.commit()
        self.db.refresh(self.session)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_load_snapshot_returns_recent_and_safety(self):
        from app.services.memory_context import MemoryContextService

        for index in range(6):
            role = "USER" if index % 2 == 0 else "ASSISTANT"
            self.db.add(ChatMessage(user_id=self.user.id, session_id=self.session.id, role=role, content=f"消息{index}"))
        self.db.commit()
        service = MemoryContextService(self.db, self.settings)
        snapshot = service.load_snapshot(self.user.id, self.session.id, current_message_id=None)
        self.assertEqual(len(snapshot.recent_messages), 6)
        self.assertFalse(snapshot.degraded)
        self.assertIsNone(snapshot.safety_state)

        SafetyMemorySvc = __import__("app.services.safety_memory", fromlist=["SafetyMemoryService"]).SafetyMemoryService
        SafetyMemorySvc(self.db, self.settings).record_turn_assessment(
            user_id=self.user.id,
            session_id=self.session.id,
            message_id=None,
            report_id=None,
            current_risk=RiskLevel.HIGH,
            confidence=0.9,
            summary="想自杀",
        )
        snapshot2 = service.load_snapshot(self.user.id, self.session.id, current_message_id=None)
        self.assertIsNotNone(snapshot2.safety_state)
        self.assertEqual(snapshot2.safety_state.status, "ACTIVE")
        self.assertGreaterEqual(len(snapshot2.active_risk_events), 1)

    def test_snapshot_payload_is_deeply_read_only(self):
        from app.services.memory_context import MemoryContextService

        snapshot = MemoryContextService(self.db, self.settings).load_snapshot(self.user.id, self.session.id)
        with self.assertRaises(TypeError):
            snapshot.history_summary["stableFacts"] = ("changed",)

    def test_runtime_prefers_effective_risk_for_chat_after_high(self):
        """场景 B：上一轮 HIGH，下一轮普通 CHAT → effective HIGH，回复模式 safety_support。"""
        from app.services.memory_context import MemoryContextService
        from app.services.safety_memory import SafetyMemoryService

        SafetyMemoryService(self.db, self.settings).record_turn_assessment(
            user_id=self.user.id,
            session_id=self.session.id,
            message_id=None,
            report_id=None,
            current_risk=RiskLevel.HIGH,
            confidence=0.95,
            summary="明确的自杀意图",
        )
        from app.agents.event_driven_runtime import EventDrivenAgentRuntimeService

        runtime = EventDrivenAgentRuntimeService(self.db, self.settings)
        result = runtime.run(self.user, self.session, "帮我解释一下 Python 列表", "帮我解释一下 Python 列表")
        self.assertEqual(result.current_risk, RiskLevel.LOW)
        self.assertEqual(result.carryover_risk, RiskLevel.HIGH)
        self.assertEqual(result.effective_risk, RiskLevel.HIGH)
        self.assertEqual(result.response_mode.value, "safety_support")
        joined = "\n".join(m.content for m in result.response_messages)
        self.assertIn("可信任的人", result.response_text)

    def test_carryover_high_does_not_create_a_new_alert_report_for_plain_chat(self):
        from app.agents.harness import MindBridgeAgentHarness
        from app.schemas.dtos import ChatRequest

        harness = MindBridgeAgentHarness(self.db, self.settings)
        first = harness.run(self.user, ChatRequest(
            message="我不想活了",
            sessionId=self.session.public_id,
            clientMessageId="carryover-high-1",
        ))
        self.assertIsNotNone(first.report_id)
        second = harness.run(self.user, ChatRequest(
            message="帮我解释 Python 列表",
            sessionId=self.session.public_id,
            clientMessageId="carryover-chat-2",
        ))
        self.assertEqual(second.effective_risk, "HIGH")
        self.assertIsNone(second.report_id)

    def test_client_message_id_idempotent(self):
        """同 clientMessageId 重试不写重复 user 消息（规格第 9 节）。"""
        from app.agents.harness import MindBridgeAgentHarness
        from app.schemas.dtos import ChatRequest

        harness = MindBridgeAgentHarness(self.db, self.settings)
        request = ChatRequest(message="记录一条内容", clientMessageId="retry-id-1")
        first = harness.run(self.user, request)
        harness.save_assistant_message(
            self.user,
            first.session,
            first.response_text,
            reply_to_message_id=first.current_message_id,
        )
        second = harness.run(self.user, request)
        self.assertTrue(second.cached)
        self.assertEqual(second.response_text, first.response_text)
        count = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.user_id == self.user.id, ChatMessage.client_message_id == "retry-id-1")
            .count()
        )
        self.assertEqual(count, 1)
        reply_count = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.reply_to_message_id == first.current_message_id, ChatMessage.completion_status == "COMPLETE")
            .count()
        )
        self.assertEqual(reply_count, 1)
        self.assertEqual(
            self.db.query(MemoryMaintenanceJob).filter(MemoryMaintenanceJob.session_id == first.session.id).count(),
            1,
        )

    def test_interrupted_assistant_is_excluded_from_snapshot(self):
        from app.services.memory_context import MemoryContextService

        self.db.add(ChatMessage(
            user_id=self.user.id,
            session_id=self.session.id,
            role="ASSISTANT",
            content="这是一段被中断的半截回",
            completion_status="INTERRUPTED",
        ))
        self.db.commit()
        snapshot = MemoryContextService(self.db, self.settings).load_snapshot(self.user.id, self.session.id)
        self.assertFalse(any("半截" in item.content for item in snapshot.recent_messages))

    def test_blackboard_snapshot_loaded_event_precedes_artifacts(self):
        from app.agents.event_driven_runtime import EventDrivenAgentRuntimeService

        runtime = EventDrivenAgentRuntimeService(self.db, self.settings)
        result = runtime.run(self.user, self.session, "你好", "你好")
        kinds = [event.type.value for event in result.collaboration_events]
        self.assertIn("MEMORY_SNAPSHOT_LOADED", kinds)
        self.assertLess(kinds.index("MEMORY_SNAPSHOT_LOADED"), kinds.index("ARTIFACT_PUBLISHED"))


if __name__ == "__main__":
    unittest.main()
