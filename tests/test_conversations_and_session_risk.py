import json
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from test_campus_youying import CampusBase, POLICY
from app.agents.harness import MindBridgeAgentHarness
from app.core.config import get_settings
from app.core.database import get_db
from app.core.enums import RiskLevel
from app.core.security import current_user
from app.models.entities import ChatSession, ChatMessage, UserAccount, RiskEvent, SafetyState, ChatMemorySummary
from app.schemas.dtos import ChatRequest
from app.services.safety_memory import SafetyMemoryService, SAFETY_STATE_CACHE_PREFIX
from app.services.memory_context import MemoryContextService


class SessionRiskTests(CampusBase):
    def setUp(self):
        super().setUp()
        self.user = UserAccount(username="isolation", display_name="学生", password_hash="x")
        self.db.add(self.user); self.db.commit()
        self.a, self.b = [ChatSession(public_id=key, user_id=self.user.id, title=key) for key in ("window-a", "window-b")]
        self.db.add_all([self.a, self.b]); self.db.commit()
        self.safety = SafetyMemoryService(self.db, self.settings)

    def assess(self, session, risk, safe=False):
        return self.safety.record_turn_assessment(user_id=self.user.id, session_id=session.id, message_id=None,
            report_id=None, current_risk=risk, confidence=.9, summary="会话测试",
            safety_reassessment=safe, protective_factors="有人陪伴" if safe else "")

    def test_high_risk_and_events_stay_in_their_original_window(self):
        self.assess(self.a, RiskLevel.HIGH)
        normal = self.assess(self.b, RiskLevel.LOW)
        self.assertEqual(normal.effective_risk, RiskLevel.LOW)
        snapshot = MemoryContextService(self.db, self.settings).load_snapshot(self.user.id, self.b.id)
        self.assertEqual(snapshot.safety_state.effective_risk, "LOW")
        self.assertEqual(snapshot.active_risk_events, ())
        continued = self.assess(self.a, RiskLevel.LOW)
        self.assertEqual(continued.effective_risk, RiskLevel.HIGH)
        self.assertTrue(all(event.risk_level != "HIGH" for event in self.safety.active_events(self.user.id, self.b.id)))

    def test_reassessment_counters_do_not_advance_in_other_windows(self):
        self.assess(self.a, RiskLevel.HIGH)
        self.assess(self.a, RiskLevel.LOW, safe=True)
        for _ in range(3):
            self.assess(self.b, RiskLevel.LOW, safe=True)
        a = self.safety.get_state(self.user.id, self.a.id)
        self.assertEqual(a.status, "ACTIVE")
        self.assertEqual(a.safe_reassessment_count, 1)
        self.assess(self.a, RiskLevel.LOW, safe=True)
        self.assertEqual(self.safety.get_state(self.user.id, self.a.id).status, "MONITORING")

    def test_manual_override_only_changes_the_chosen_session(self):
        self.assess(self.a, RiskLevel.HIGH); self.assess(self.b, RiskLevel.HIGH)
        self.safety.manual_override(self.user.id, "admin", "RESOLVED", "测试解除", self.a.id)
        self.assertEqual(self.safety.get_state(self.user.id, self.a.id).effective_risk, "LOW")
        self.assertEqual(self.safety.get_state(self.user.id, self.b.id).effective_risk, "HIGH")

    def test_redis_cache_keys_are_separate_and_old_user_cache_is_ignored(self):
        values = {f"mindbridge:safety-state:user:{self.user.id}": json.dumps({"schema": 1, "status": "ACTIVE", "effectiveRisk": "HIGH"})}
        self.safety._redis = SimpleNamespace(get=values.get, set=lambda key, value, ex: values.__setitem__(key, value))
        self.assess(self.a, RiskLevel.HIGH); self.assess(self.b, RiskLevel.LOW)
        self.assertEqual(self.safety._read_cache(self.user.id, self.a.id).effective_risk, "HIGH")
        self.assertEqual(self.safety._read_cache(self.user.id, self.b.id).effective_risk, "LOW")
        self.assertIn(f"{SAFETY_STATE_CACHE_PREFIX}{self.user.id}:{self.b.id}", values)

    def test_legacy_projection_rebuild_uses_current_risk_not_old_cross_session_carryover(self):
        self.db.execute(text("CREATE TABLE safety_states (user_id INTEGER PRIMARY KEY, effective_risk TEXT)"))
        self.db.execute(text("INSERT INTO safety_states VALUES (:user_id, 'HIGH')"), {"user_id": self.user.id})
        self.db.add_all([RiskEvent(user_id=self.user.id, session_id=self.a.id, event_kind="RISK_DETECTED", risk_level="HIGH",
            payload_json=json.dumps({"effectiveRisk": "HIGH"})),
            RiskEvent(user_id=self.user.id, session_id=self.b.id, event_kind="REASSESSMENT", risk_level="LOW",
            payload_json=json.dumps({"effectiveRisk": "HIGH"}))])
        self.db.commit()
        count = self.db.query(RiskEvent).count()
        self.assertEqual(self.safety.restore_legacy_sessions(), 2)
        self.assertEqual(self.safety.get_state(self.user.id, self.a.id).effective_risk, "HIGH")
        self.assertEqual(self.safety.get_state(self.user.id, self.b.id).effective_risk, "LOW")
        self.assertEqual(self.safety.restore_legacy_sessions(), 0)
        self.assertEqual(self.db.query(RiskEvent).count(), count)
        self.assertEqual(self.db.execute(text("SELECT effective_risk FROM safety_states")).scalar(), "HIGH")

    def test_legacy_replay_restores_within_session_downgrades(self):
        for index, risk in enumerate(["HIGH", "LOW", "LOW", "LOW", "LOW", "LOW"]):
            self.db.add(RiskEvent(user_id=self.user.id, session_id=self.a.id,
                event_kind="RISK_DETECTED" if index == 0 else "REASSESSMENT", risk_level=risk,
                payload_json=json.dumps({"validSafetyReassessment": index > 0, "protectiveFactors": "有人陪伴"})))
        self.db.commit()
        self.safety.restore_legacy_sessions()
        self.assertEqual(self.safety.get_state(self.user.id, self.a.id).status, "RESOLVED")
        self.assertEqual(self.db.query(RiskEvent).count(), 6)

    def test_interrupted_history_rebuild_restarts_without_duplicate_events(self):
        self.assess(self.a, RiskLevel.HIGH)
        self.assess(self.a, RiskLevel.LOW, safe=True)
        row = self.db.query(SafetyState).filter_by(session_id=self.a.id).one()
        row.history_rebuilt = False
        row.effective_risk = "LOW"
        self.db.commit()
        count = self.db.query(RiskEvent).count()
        self.safety.restore_legacy_sessions()
        restored = self.safety.get_state(self.user.id, self.a.id)
        self.assertEqual(restored.effective_risk, "HIGH")
        self.assertEqual(restored.safe_reassessment_count, 1)
        self.assertEqual(self.db.query(RiskEvent).count(), count)

    def test_foreign_safety_references_are_removed_from_old_summary(self):
        self.assess(self.a, RiskLevel.HIGH)
        foreign = self.safety.active_events(self.user.id, self.a.id)[0]
        self.db.add(ChatMessage(user_id=self.user.id, session_id=self.b.id, role="USER", content="普通会话"))
        self.db.add(ChatMemorySummary(user_id=self.user.id, session_id=self.b.id,
            summary_json=json.dumps({"safetyReferences": [{"eventId": foreign.id, "status": "ACTIVE"}]})))
        self.db.commit()
        snapshot = MemoryContextService(self.db, self.settings).load_snapshot(self.user.id, self.b.id)
        self.assertNotIn(str(foreign.id) + ":ACTIVE", snapshot.summary_text)
        self.assertIsNone(snapshot.safety_state)

    def test_runtime_new_chat_is_normal_and_reopening_old_chat_keeps_high(self):
        self.add_policy()
        harness = MindBridgeAgentHarness(self.db, self.settings)
        first = harness.run(self.user, ChatRequest(message="我想自杀，已经准备好了。", sessionId=self.a.public_id))
        harness.save_assistant_message(self.user, first.session, first.response_text, first.current_message_id)
        second = harness.run(self.user, ChatRequest(message="研究生差旅的伙食补助是什么？", sessionId=self.b.public_id))
        self.assertEqual(second.risk_level, "LOW")
        self.assertEqual(second.response_mode, "policy_answer")
        self.assertNotIn("准备好了", "".join(message.content for message in second.response_messages))
        third = harness.run(self.user, ChatRequest(message="帮我解释一段代码", sessionId=self.a.public_id))
        self.assertEqual(third.risk_level, "HIGH")
        self.assertEqual(third.response_mode, "safety_support")


class ConversationApiTests(CampusBase):
    def setUp(self):
        super().setUp()
        from app.api.routes import router
        self.user = UserAccount(username="history", display_name="学生", password_hash="x")
        self.other = UserAccount(username="other", display_name="其他学生", password_hash="x")
        self.db.add_all([self.user, self.other]); self.db.commit()
        self.own = ChatSession(public_id="mine", user_id=self.user.id, title="我的报销问题")
        self.foreign = ChatSession(public_id="foreign", user_id=self.other.id, title="他人的对话")
        self.db.add_all([self.own, self.foreign]); self.db.commit()
        self.db.add_all([ChatMessage(user_id=self.user.id, session_id=self.own.id, role="USER", content="我的问题"),
            ChatMessage(user_id=self.other.id, session_id=self.foreign.id, role="USER", content="其他人的秘密")]); self.db.commit()
        self.app = FastAPI(); self.app.include_router(router)
        self.app.dependency_overrides[get_db] = lambda: self.db
        self.app.dependency_overrides[current_user] = lambda: self.user
        self.patch = patch("app.api.routes.get_settings", return_value=self.settings)
        self.patch.start(); self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close(); self.patch.stop(); super().tearDown()

    def test_sidebar_lists_only_current_users_sessions_and_supports_paging(self):
        self.db.add(ChatSession(public_id="newer", user_id=self.user.id, title="最新会话")); self.db.commit()
        first = self.client.get("/api/chat/sessions?limit=1").json()
        second = self.client.get("/api/chat/sessions?offset=1&limit=1").json()
        self.assertEqual(first["total"], 2)
        self.assertEqual(first["nextOffset"], 1)
        self.assertEqual([first["items"][0]["sessionId"], second["items"][0]["sessionId"]], ["newer", "mine"])
        self.assertEqual(second["items"][0]["messageCount"], 1)
        self.assertEqual(self.client.get("/api/chat/sessions?offset=-1").status_code, 422)

    def test_history_does_not_expose_another_users_messages(self):
        self.assertEqual(self.client.get("/api/chat/sessions/foreign").status_code, 404)
        own = self.client.get("/api/chat/sessions/mine").json()
        self.assertEqual(own["messages"][0]["content"], "我的问题")

    def test_history_restores_original_sources_after_a_policy_version_update(self):
        self.add_policy(version="v1")
        harness = MindBridgeAgentHarness(self.db, self.settings)
        out = harness.run(self.user, ChatRequest(message="研究生差旅的伙食补助标准是什么？", sessionId=self.own.public_id))
        harness.save_assistant_message(self.user, out.session, out.response_text, out.current_message_id)
        self.service.ingest("travel.md", POLICY.replace("80 元", "90 元"), version="v2")
        history = self.client.get("/api/chat/sessions/mine").json()
        self.assertEqual(history["messages"][-1]["content"], out.response_text)
        self.assertNotIn("sources", history["messages"][-1])
        self.user.roles_csv = "ROLE_ADMIN,ROLE_USER"
        history = self.client.get("/api/admin/conversations/mine").json()
        self.assertTrue(history["messages"][-1]["sources"])
        self.assertTrue(all(s["version"] == "v1" for s in history["messages"][-1]["sources"]))

    def test_admin_override_requires_a_session_owned_by_the_selected_user(self):
        self.user.roles_csv = "ROLE_ADMIN,ROLE_USER"
        path = f"/api/admin/safety-states/{self.user.id}/override"
        body = {"action": "RESOLVED", "reason": "测试"}
        self.assertEqual(self.client.post(path, json=body).status_code, 422)
        self.assertEqual(self.client.post(path, json={**body, "sessionId": "foreign"}).status_code, 404)
        self.assertEqual(self.client.post(path, json={**body, "sessionId": "mine"}).status_code, 200)
