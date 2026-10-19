"""Campus routing, parent/child retrieval and API regressions (no external services)."""
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.core.database import Base, get_db
from app.core.enums import IntentType, RiskLevel
from app.core.security import current_user
from app.models.entities import KnowledgeChunk, KnowledgeDocument, PsychologicalReport, ChatMessage, UserAccount
from app.services.knowledge import KnowledgeService, SearchResult, parent_sections, child_sections
from app.services.ai import has_high_risk_signal
from app.agents.harness import MindBridgeAgentHarness
from app.schemas.dtos import ChatRequest

POLICY = """# 测试学校差旅办法（仅测试，不是实际政策）

## 研究生伙食补助

研究生境内出差的伙食补助按测试标准每天 80 元执行。

### 例外条件

统一安排伙食时，不重复领取伙食补助。此金额仅用于自动测试。

## 住宿费

研究生住宿费按测试标准每日上限 300 元凭票报销。

### 办理材料

需要住宿发票和出差审批单，测试标准不适用于实际办理。
"""


def settings(**overrides):
    values = dict(_env_file=None, ai_provider="mock", openai_api_key="", knowledge_vector_enabled=False,
        redis_url="redis://127.0.0.1:1/0", redis_socket_timeout_seconds=.01,
        tool_queue_enabled=False, memory_maintenance_queue_enabled=False, campus_name="测试学校")
    values.update(overrides)
    return Settings(**values)


class CampusBase(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine, autoflush=False)()
        self.settings = settings()
        self.service = KnowledgeService(self.db, self.settings)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def add_policy(self, **kwargs):
        return self.service.ingest("travel.md", POLICY, title="测试差旅规则", is_demo=True, **kwargs)


class ParentChildTests(CampusBase):
    def test_structure_preserves_exception_and_parent_mapping(self):
        self.add_policy()
        parents = self.db.query(KnowledgeChunk).filter_by(chunk_type="parent").all()
        children = self.db.query(KnowledgeChunk).filter_by(chunk_type="child").all()
        self.assertEqual(len(parents), 2)
        self.assertTrue(all(c.parent_id in {p.id for p in parents} for c in children))
        results = self.service.retrieve("研究生伙食补助标准", domain="policy", school="测试学校")
        self.assertIn("统一安排伙食", results[0].content)
        self.assertEqual(len({r.chunk_id for r in results}), len(results))
        self.assertEqual(self.service.last_retrieval["path"], "bm25_parent_child")
        self.assertTrue(results[0].citation()["isDemo"])

    def test_seed_school_update_preserves_content_and_status(self):
        self.service.ensure_source("seed.md", POLICY, domain="support", school="旧校名")
        self.service.ensure_source("seed.md", POLICY, domain="support", school="新校名")
        self.assertEqual(self.db.query(KnowledgeDocument).count(), 1)
        self.assertTrue(self.service.retrieve("伙食补助", domain="support", school="新校名"))
        self.assertFalse(self.service.retrieve("伙食补助", domain="support", school="旧校名"))

    def test_same_content_idempotent_and_named_version_immutable(self):
        self.add_policy(version="v1")
        count = self.service.count()
        self.add_policy(version="v1")
        self.assertEqual(count, self.service.count())
        with self.assertRaisesRegex(ValueError, "版本"):
            self.service.ingest("travel.md", POLICY + "不同内容", version="v1")
        self.assertEqual(self.db.query(KnowledgeDocument).count(), 1)

    def test_version_domain_school_and_date_filters(self):
        self.add_policy(version="v1")
        self.service.ingest("travel.md", POLICY.replace("80 元", "90 元"), version="v2")
        self.service.ingest("support.md", "# 情绪支持\n\n伙食和睡眠困扰可以找人倾诉。", domain="support")
        self.service.ingest("other.md", POLICY.replace("80 元", "999 元"), school="另一学校")
        results = self.service.retrieve("伙食补助", domain="policy", school="测试学校")
        self.assertTrue(results)
        self.assertTrue(all(r.version == "v2" for r in results))
        self.assertIn("90 元", results[0].content)
        self.assertNotIn("999 元", "".join(r.content for r in results))
        self.assertEqual(self.db.query(KnowledgeDocument).filter_by(source="travel.md", version="v1").one().status, "ARCHIVED")
        self.service.ingest("future.md", POLICY, effective_on=date.today()+timedelta(days=3))
        future = self.db.query(KnowledgeDocument).filter_by(source="future.md").one()
        self.assertEqual(future.status, "DRAFT")
        with self.assertRaises(ValueError):
            self.service.set_document_status(future.id, "ACTIVE")
        self.assertNotIn("future.md", [r.source for r in self.service.retrieve("伙食补助")])

    def test_restore_old_version_archives_new_version(self):
        self.add_policy(version="v1")
        self.service.ingest("travel.md", POLICY.replace("80 元", "90 元"), version="v2")
        old = self.db.query(KnowledgeDocument).filter_by(version="v1").one()
        self.service.set_document_status(old.id, "ACTIVE")
        self.assertEqual(self.service.retrieve("伙食补助")[0].version, "v1")

    def test_table_and_long_sentence_are_not_lost(self):
        table = "| 对象 | 标准 |\n|---|---|\n| 研究生 | 80元 |\n\n统一安排伙食不得重复领取。"
        parents = parent_sections("## 测试表格\n\n" + table, 100)
        self.assertIn("| 研究生 | 80元 |", "".join(p[1] for p in parents))
        chunks = child_sections(table, 50, 10)
        self.assertTrue(any("| 对象 | 标准 |" in c and "80元" in c for c in chunks))
        source = "甲" * 700
        children = child_sections(source, 300, 40)
        self.assertEqual(children[0][-40:], children[1][:40])
        self.assertTrue(all(len(c) <= 300 for c in children))

    def test_budget_skips_oversized_first_parent_without_losing_next(self):
        from app.agents.autonomous import ContextAgent
        agent = ContextAgent(SimpleNamespace(settings=SimpleNamespace(rag_skill_budget_tokens=100)))
        large = SearchResult(1, "a", "长"*500, 1)
        small = SearchResult(2, "b", "完整短条款。", .8)
        self.assertEqual(agent._bounded_knowledge([large, small]), [small])
        self.assertEqual(agent.budget_omitted[0]["chunkId"], 1)

    def test_failed_required_embedding_does_not_replace_active_policy(self):
        self.add_policy(version="v1")
        self.settings.knowledge_vector_required = True
        self.service.vector_store = SimpleNamespace(can_embed=True, upsert_chunks=lambda *args: None, embed_texts=lambda _: (_ for _ in ()).throw(RuntimeError("embedding down")))
        with self.assertRaisesRegex(RuntimeError, "embedding down"):
            self.service.ingest("travel.md", POLICY + "新版本", version="v2")
        self.assertEqual(self.db.query(KnowledgeDocument).filter_by(version="v1").one().status, "ACTIVE")
        self.assertIsNone(self.db.query(KnowledgeDocument).filter_by(version="v2").first())

    def test_unreadable_input_and_unsafe_source_url_are_rejected(self):
        with self.assertRaises(ValueError):
            self.service.ingest("x", "content", source_url="javascript:alert(1)")
        with self.assertRaises(ValueError):
            self.service.ingest_file("x.exe", b"x")
        with self.assertRaises(ValueError):
            self.service.ingest_file("x.txt", b"")
        with self.assertRaises(ValueError):
            self.service.ingest_file("broken.pdf", b"not a PDF")

    def test_parent_and_child_both_embedded_and_have_separate_recall(self):
        embedded = []
        queries = []
        service = self.service
        class FakeVector:
            can_embed = True
            error = ""
            def embed_texts(self, texts):
                embedded.extend(texts)
                return [[1., 0.] for _ in texts]
            def upsert_chunks(self, rows, vectors):
                self.rows = list(rows)
            def has_exact_chunk_ids(self, rows):
                return True
            def query(self, vector, k, *, chunk_type, allowed_ids):
                queries.append(chunk_type)
                self.assertions = allowed_ids
                return [SimpleNamespace(chunk_id=i, score=1.) for i in allowed_ids[:k]]
        service.vector_store = FakeVector()
        self.add_policy()
        self.assertEqual(len(embedded), service.count())
        self.assertTrue(any("例外条件" in text and "80 元" in text for text in embedded))
        results = service.retrieve("伙食补助", domain="policy", school="测试学校")
        self.assertEqual(queries, ["parent", "child"])
        self.assertEqual(len(results), len({r.chunk_id for r in results}))
        self.assertIn("parent_vector", results[0].matched_by)
        self.assertIn("child_vector", results[0].matched_by)
        self.assertIn("child_bm25", results[0].matched_by)


class RoutingTests(CampusBase):
    def setUp(self):
        super().setUp()
        self.user = UserAccount(username="student-test", display_name="测试学生", password_hash="x")
        self.db.add(self.user); self.db.commit()
        self.add_policy()

    def turn(self, text, session=None):
        harness = MindBridgeAgentHarness(self.db, self.settings)
        out = harness.run(self.user, ChatRequest(message=text, sessionId=session))
        harness.save_assistant_message(self.user, out.session, out.response_text, out.current_message_id)
        return out

    def test_policy_consult_has_sources_and_no_psychology_report(self):
        out = self.turn("研究生差旅伙食补助标准是什么？")
        self.assertEqual(out.intent, IntentType.CONSULT)
        self.assertEqual(out.response_mode, "policy_answer")
        self.assertIsNone(out.report_id)
        self.assertEqual(self.db.query(PsychologicalReport).count(), 0)
        self.assertTrue(out.retrieved_knowledge)
        self.assertIn("演示", out.response_text)
        self.assertNotIn("你是 ResponseAgent", out.response_text)
        self.assertNotIn("supportive_response_baseline", "".join(m.content for m in out.response_messages))

    def test_followup_uses_recent_context(self):
        first = self.turn("我是研究生，差旅报销的伙食补助标准是什么？")
        second = self.turn("那住宿呢？", first.session.public_id)
        self.assertEqual(second.intent, IntentType.CONSULT)
        self.assertTrue(any("住宿" in r.heading for r in second.retrieved_knowledge))

    def test_earlier_policy_context_survives_summary_compaction(self):
        first = self.turn("我是研究生，差旅报销的伙食补助标准是什么？")
        sid = first.session.public_id
        for i in range(5):
            self.turn(f"今天完成了第{i}项任务，想分享一下。", sid)
        from app.services.memory_context import RollingSummaryService
        state = RollingSummaryService(self.db, self.settings).load_state(first.session.id)
        self.assertIsNotNone(state)
        out = self.turn("那住宿呢？", sid)
        self.assertEqual(out.intent, IntentType.CONSULT)
        self.assertTrue(any("住宿" in r.heading for r in out.retrieved_knowledge))
        self.assertIn("研究生", "".join(m.content for m in out.response_messages))

    def test_risk_overrides_policy_without_duplicate_report(self):
        first = self.turn("我想自杀，已经准备好了。")
        self.assertEqual(first.risk_level, "HIGH")
        self.assertIsNotNone(first.report_id)
        count = self.db.query(PsychologicalReport).count()
        second = self.turn("研究生差旅报销需要什么材料？", first.session.public_id)
        self.assertEqual(second.risk_level, "HIGH")
        self.assertIn("可信任", second.response_text)
        self.assertEqual(self.db.query(PsychologicalReport).count(), count)
        third = self.turn("奖学金政策是什么？")
        self.assertEqual(third.risk_level, "LOW")

    def test_policy_discussion_and_explicit_negation_are_not_hard_high(self):
        self.assertFalse(has_high_risk_signal("学校如何预防自杀风险，有什么规定？"))
        self.assertFalse(has_high_risk_signal("我没有自杀的想法，现在有人陪着。"))
        self.assertTrue(has_high_risk_signal("虽然学校有预防自杀的规定，但我想自杀。"))
        out = self.turn("学校如何预防自杀风险，有什么规定？")
        self.assertEqual(out.intent, IntentType.CONSULT)
        self.assertEqual(out.risk_level, "LOW")

    def test_sharing_about_finished_reimbursement_is_chat(self):
        out = self.turn("今天终于交完报销材料了，想分享一下，松口气！")
        self.assertEqual(out.intent, IntentType.CHAT)
        self.assertFalse(out.retrieved_knowledge)

    def test_light_emotional_support_remains_available(self):
        out = self.turn("最近压力很大，睡不好，想找人聊聊。")
        self.assertEqual(out.intent, IntentType.RISK)
        self.assertEqual(out.risk_level, "LOW")
        self.assertEqual(out.response_mode, "support")
        self.assertIsNone(out.report_id)

    def test_response_model_failure_has_explicit_safe_fallback(self):
        with patch("app.services.ai.AiClient._mock", side_effect=RuntimeError("model unavailable")):
            out = self.turn("研究生差旅报销政策是什么？")
        self.assertEqual(out.intent, IntentType.CONSULT)
        self.assertIn("暂时无法", out.response_text)
        self.assertIsNone(out.report_id)

    def test_chat_receives_recent_history(self):
        first = self.turn("今天完成了毕业作品，很开心。")
        second = self.turn("你还记得我刚才说的吗？", first.session.public_id)
        self.assertIn("毕业作品", "".join(m.content for m in second.response_messages))


class CampusApiTests(CampusBase):
    def setUp(self):
        super().setUp()
        from app.api.routes import router
        from app.core.config import get_settings
        self.app = FastAPI()
        self.app.include_router(router)
        self.user = UserAccount(username="api-admin", display_name="管理员", password_hash="x", roles_csv="ROLE_ADMIN,ROLE_USER")
        self.db.add(self.user); self.db.commit()
        self.app.dependency_overrides[get_db] = lambda: self.db
        self.app.dependency_overrides[current_user] = lambda: self.user
        self.app.dependency_overrides[get_settings] = lambda: self.settings
        # 路由中有直接 get_settings() 调用，显式替换为隔离配置。
        self.patch = patch("app.api.routes.get_settings", return_value=self.settings)
        self.patch.start()
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close(); self.patch.stop()
        super().tearDown()

    def test_metadata_upload_preview_and_status(self):
        response = self.client.post("/api/admin/knowledge/file", files={"file": ("travel.md", POLICY.encode())},
            data={"metadata": json.dumps({"domain": "policy", "version": "v1", "is_demo": True})})
        self.assertEqual(response.status_code, 200, response.text)
        docs = self.client.get("/api/admin/knowledge/documents").json()
        self.assertEqual(docs[0]["version"], "v1")
        detail = self.client.get(f"/api/admin/knowledge/documents/{docs[0]['id']}").json()
        self.assertEqual({c["type"] for c in detail["chunks"]}, {"parent", "child"})
        response = self.client.patch(f"/api/admin/knowledge/documents/{docs[0]['id']}/status", json={"status": "ARCHIVED"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.service.retrieve("伙食补助", domain="policy"), [])
        self.assertEqual(self.client.post("/api/admin/knowledge/file", files={"file": ("x.md", b"x")}, data={"metadata": "bad"}).status_code, 422)

    def test_real_upload_defaults_to_non_demo_and_updates_policy_readiness(self):
        self.assertEqual(self.client.get("/api/admin/knowledge/status").json()["activePolicyDocuments"], 0)
        response = self.client.post("/api/admin/knowledge/file", files={"file": ("official.md", POLICY.encode())},
            data={"metadata": json.dumps({"domain": "policy", "version": "official-v1"})})
        self.assertEqual(response.status_code, 200, response.text)
        documents = self.client.get("/api/admin/knowledge/documents").json()
        self.assertFalse(documents[0]["isDemo"])
        self.assertEqual(self.client.get("/api/admin/knowledge/status").json()["activePolicyDocuments"], 1)

    def test_retry_reuses_reply_without_exposing_sources(self):
        self.add_policy()
        self.user.roles_csv = "ROLE_USER"
        body = {"message": "研究生伙食补助标准是什么？", "clientMessageId": "retry-sources"}
        first = self.client.post("/api/chat/stream", json=body)
        second = self.client.post("/api/chat/stream", json=body)
        def tokens(response):
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            self.assertFalse(any(e["type"] == "sources" for e in events))
            return "".join(e["content"] for e in events if e["type"] == "token")
        self.assertEqual(tokens(first), tokens(second))
        self.assertEqual(self.db.query(ChatMessage).count(), 2)

    def test_sources_only_available_to_admin(self):
        self.add_policy()
        self.user.roles_csv = "ROLE_USER"
        self.assertEqual(self.client.get("/api/admin/knowledge/documents").status_code, 403)
        response = self.client.post("/api/chat/stream", json={"message": "研究生伙食补助标准是什么？"})
        self.assertEqual(response.status_code, 200)
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertFalse(any(e["type"] == "sources" for e in events))
        session_id = next(e["sessionId"] for e in events if e["type"] == "meta")
        self.assertNotIn("sources", self.client.get(f"/api/chat/sessions/{session_id}").json()["messages"][-1])
        self.assertEqual(self.client.get(f"/api/admin/conversations/{session_id}").status_code, 403)
        self.user.roles_csv = "ROLE_ADMIN,ROLE_USER"
        reply = self.client.get(f"/api/admin/conversations/{session_id}").json()["messages"][-1]
        self.assertTrue(reply["traceId"])
        self.assertTrue(reply["sources"][0]["isDemo"])
        self.assertIn("content", reply["sources"][0])
        self.assertTrue(any(e["type"] == "done" for e in events))
        self.assertEqual(self.db.query(PsychologicalReport).count(), 0)



class VectorStoreIntegrationTests(CampusBase):
    def test_chroma_dual_index_filters_and_atomic_rebuild(self):
        try:
            import chromadb
        except ImportError:
            self.skipTest("Chroma not installed")
        from app.services.vector_store import ChromaKnowledgeStore
        with tempfile.TemporaryDirectory() as directory:
            cfg = settings(knowledge_vector_enabled=True, openai_api_key="test-not-a-real-key",
                chroma_persist_dir=directory, chroma_snapshot_dir=str(Path(directory)/"snapshots"))
            def embed(store, texts):
                return [[float(1 + text.count("伙食")), float(1 + text.count("住宿")), 1.] for text in texts]
            with patch.object(ChromaKnowledgeStore, "_embed", embed):
                service = KnowledgeService(self.db, cfg)
                service.ingest("dual.md", POLICY, version="v1", is_demo=True)
                results = service.retrieve("研究生伙食补助", domain="policy", school=cfg.campus_name)
                self.assertTrue(results)
                self.assertEqual(service.last_retrieval["path"], "hybrid_parent_child")
                count = service.rebuild_vector_index()
                manifest = service.vector_store.manifest.read_text()
                self.assertEqual(count, service.count())
                same_config = KnowledgeService(self.db, cfg)
                self.assertEqual(same_config.vector_store.collection_name, service.vector_store.collection_name)
                self.assertTrue(same_config.vector_store.has_exact_chunk_ids(service._all_index_rows()))
                with patch.object(service.vector_store, "_upsert", side_effect=RuntimeError("staging failed")):
                    with self.assertRaisesRegex(RuntimeError, "staging failed"):
                        service.rebuild_vector_index()
                self.assertEqual(service.vector_store.manifest.read_text(), manifest)
                doc = self.db.query(KnowledgeDocument).filter_by(source="dual.md").one()
                service.set_document_status(doc.id, "ARCHIVED")
                self.assertEqual(service.retrieve("伙食补助", domain="policy", school=cfg.campus_name), [])

    def test_old_sqlite_schema_is_additively_migrated(self):
        from app.core import bootstrap
        old = create_engine("sqlite://")
        with old.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE knowledge_chunks (id INTEGER PRIMARY KEY, source VARCHAR(256), source_index INTEGER, content TEXT, embedding_json TEXT, created_at DATETIME)")
            connection.exec_driver_sql("INSERT INTO knowledge_chunks VALUES (1, 'old.md', 0, '历史支持知识。', NULL, '2026-10-18 00:00:00')")
        with patch.object(bootstrap, "engine", old):
            bootstrap.create_schema()
            bootstrap.create_schema()
        db = sessionmaker(bind=old)()
        try:
            service = KnowledgeService(db, settings())
            self.assertEqual(service.migrate_legacy_chunks(), 1)
            self.assertEqual(db.get(KnowledgeChunk, 1).content, "历史支持知识。")
            self.assertTrue(db.query(KnowledgeDocument).first())
            self.assertTrue(service.retrieve("历史支持", domain="support"))
        finally:
            db.close(); old.dispose()


class SafetyReviewTests(unittest.TestCase):
    def test_unverified_phone_is_rejected_in_existing_review(self):
        from app.agents.autonomous import SafetyAgent
        from app.agents.events import AgentArtifact, AgentTask, CollaborationBlackboard
        services = SimpleNamespace(private_memory=SimpleNamespace(_key=lambda *args: "test"), session=SimpleNamespace(public_id="s"))
        agent = SafetyAgent(services)
        agent.remember = lambda text: None
        risk = AgentArtifact(id="risk", owner="SafetyAgent", kind="risk", payload={"risk": "HIGH", "effectiveRisk": "HIGH"})
        response = AgentArtifact(id="response", owner="ResponseAgent", kind="response_proposal", payload={"mode": "safety_support", "candidateText": "请关注当前安全，联系身边可信任的人，拨打虚构测试电话400-000-0000。"})
        board = CollaborationBlackboard(turn_id="test").add_artifact(risk).add_artifact(response)
        result = agent._review_response(AgentTask(id="review", title="review"), board, response)
        self.assertFalse(result.artifacts[0].payload["approved"])
        self.assertIn("联系电话", result.artifacts[0].payload["reason"])

    def test_fictional_quote_does_not_become_personal_hard_signal(self):
        self.assertFalse(has_high_risk_signal('校规的预防条款引用“我想自杀”这句话作为识别示例。'))
        self.assertTrue(has_high_risk_signal('小说引用“我想自杀”，但是现在我自己也不想活了。'))


class SelfReviewKnowledgeTests(CampusBase):
    def test_flattened_legacy_markdown_retains_body_during_startup_migration(self):
        self.db.add(KnowledgeChunk(source="legacy.md", source_index=0,
            content="# 旧版支持指南 ## 支持建议 可以先联系身边可信任的人。", chunk_type="legacy", domain="support"))
        self.db.commit()
        self.service.migrate_legacy_chunks()
        self.service.migrate_legacy_chunks()
        results = self.service.retrieve("支持建议", domain="support")
        self.assertIn("身边可信任的人", results[0].content)
        self.assertEqual(self.db.query(KnowledgeDocument).count(), 1)

    def test_expired_upload_does_not_archive_current_version(self):
        self.add_policy(version="current")
        self.service.ingest("travel.md", POLICY + "历史版本", version="expired",
            expires_on=date.today() - timedelta(days=1))
        self.assertEqual(self.db.query(KnowledgeDocument).filter_by(version="current").one().status, "ACTIVE")
        self.assertEqual(self.service.retrieve("伙食补助")[0].version, "current")

    def test_heading_only_upload_cannot_replace_current_policy(self):
        self.add_policy(version="current")
        with self.assertRaisesRegex(ValueError, "正文"):
            self.service.ingest("travel.md", "# 只有标题\n\n## 没有条款", version="empty")
        self.assertEqual(self.db.query(KnowledgeDocument).count(), 1)
        self.assertEqual(self.service.retrieve("伙食补助")[0].version, "current")

    def test_blank_pdf_is_rejected_instead_of_success_with_zero_chunks(self):
        from io import BytesIO
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        output = BytesIO()
        writer.write(output)
        with self.assertRaisesRegex(ValueError, "OCR"):
            self.service.ingest_file("scan.pdf", output.getvalue())
        self.assertEqual(self.db.query(KnowledgeDocument).count(), 0)

    def test_empty_historical_document_cannot_be_activated(self):
        self.add_policy(version="current")
        empty = KnowledgeDocument(source="travel.md", version="empty", title="空历史文件", domain="policy",
            school=self.settings.campus_name, content="# 标题", status="DRAFT")
        self.db.add(empty); self.db.commit()
        with self.assertRaisesRegex(ValueError, "正文"):
            self.service.set_document_status(empty.id, "ACTIVE")
        self.assertEqual(self.service.retrieve("伙食补助")[0].version, "current")

    def test_policy_dates_use_china_timezone(self):
        from datetime import datetime, timezone
        from app.services.knowledge import campus_today
        instant = datetime(2026, 10, 18, 17, 0, tzinfo=timezone.utc)
        with patch("app.services.knowledge.datetime") as clock:
            clock.now.side_effect = lambda zone: instant.astimezone(zone)
            self.assertEqual(campus_today(), date(2026, 10, 19))

    def test_same_version_with_different_scope_is_not_silently_accepted(self):
        self.add_policy(version="v1")
        for metadata in ({"school": "另一学校"}, {"domain": "support"}):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, "学校|领域"):
                self.service.ingest("travel.md", POLICY, version="v1", **metadata)

    def test_disabled_bm25_channel_does_not_return_zero_score_evidence(self):
        self.add_policy()
        self.settings.knowledge_hybrid_bm25_weight = 0
        self.assertEqual(self.service.retrieve("伙食补助"), [])

    def test_partial_vector_failure_falls_back_completely_to_bm25(self):
        self.add_policy()
        parents = self.db.query(KnowledgeChunk).filter_by(chunk_type="parent").all()
        def query(vector, k, *, chunk_type, allowed_ids):
            if chunk_type == "child":
                raise RuntimeError("child query failed")
            return [SimpleNamespace(chunk_id=p.id, score=1.) for p in parents]
        self.service.vector_store = SimpleNamespace(can_embed=True, error="",
            has_exact_chunk_ids=lambda rows: True, embed_texts=lambda texts: [[1.]], query=query)
        self.service.retrieve("unmatched_xyz")
        self.assertEqual(self.service.last_retrieval["path"], "bm25_parent_child")
        self.assertEqual(self.service.last_retrieval["selectedParents"], [])


class SelfReviewVectorTests(unittest.TestCase):
    def test_corrupt_manifest_respects_optional_and_required_vector_modes(self):
        import hashlib
        from app.services.vector_store import ChromaKnowledgeStore, VectorStoreUnavailable
        with tempfile.TemporaryDirectory() as directory:
            cfg = settings(knowledge_vector_enabled=True, openai_api_key="test", chroma_persist_dir=directory)
            fingerprint = hashlib.sha256(f"pc-v1:{cfg.openai_base_url}:{cfg.openai_embedding_model}".encode()).hexdigest()[:12]
            Path(directory, f"{cfg.chroma_collection_name}_{fingerprint}.json").write_text("broken json")
            store = ChromaKnowledgeStore(cfg)
            self.assertFalse(store.can_embed)
            self.assertTrue(store.error)
            cfg.knowledge_vector_required = True
            with self.assertRaises(VectorStoreUnavailable):
                ChromaKnowledgeStore(cfg)

    def test_stale_store_upserts_into_current_manifest_collection(self):
        from app.services.vector_store import ChromaKnowledgeStore
        with tempfile.TemporaryDirectory() as directory:
            cfg = settings(knowledge_vector_enabled=True, openai_api_key="test", chroma_persist_dir=directory)
            first, second = ChromaKnowledgeStore(cfg), ChromaKnowledgeStore(cfg)
            first.sync_chunks([], [])
            chunk = KnowledgeChunk(id=1, source="x", source_index=0, content="规则", chunk_type="parent",
                document_id=1, domain="policy", school="测试学校", version="v1", heading="标题")
            second.upsert_chunks([chunk], [[1., 0.]])
            self.assertEqual(first.count(), 1)
            self.assertTrue(first.has_exact_chunk_ids([chunk]))


class SelfReviewStateTests(CampusBase):
    def test_retry_restores_risk_mode_and_trace_without_duplicate_actions(self):
        user = UserAccount(username="retry-risk", display_name="测试", password_hash="x")
        self.db.add(user); self.db.commit()
        harness = MindBridgeAgentHarness(self.db, self.settings)
        request = ChatRequest(message="我想自杀，已经准备好了。", clientMessageId="retry-high")
        first = harness.run(user, request)
        harness.save_assistant_message(user, first.session, first.response_text, first.current_message_id)
        second = harness.run(user, request)
        self.assertTrue(second.cached)
        self.assertEqual(second.risk_level, first.risk_level)
        self.assertEqual(second.effective_risk, first.effective_risk)
        self.assertEqual(second.response_mode, first.response_mode)
        self.assertEqual(second.trace_id, first.trace_id)
        self.assertFalse(second.tool_plan.requires_tools)
        self.assertEqual(self.db.query(PsychologicalReport).count(), 1)

    def test_response_mode_uses_this_turn_safety_transition(self):
        from app.agents.autonomous import _safety_status
        from app.agents.events import AgentArtifact, CollaborationBlackboard
        for previous, current in (("ACTIVE", "MONITORING"), ("MONITORING", "RESOLVED"), ("ACTIVE", "FALSE_POSITIVE")):
            with self.subTest(previous=previous, current=current):
                board = CollaborationBlackboard(turn_id="transition", memory_snapshot=SimpleNamespace(
                    safety_state=SimpleNamespace(status=previous)))
                board = board.add_artifact(AgentArtifact(id="risk", owner="SafetyAgent", kind="risk",
                    payload={"newStatus": current}))
                self.assertEqual(_safety_status(board), current)

    def test_response_without_consumed_context_does_not_get_late_citations(self):
        from app.agents.event_driven_runtime import EventDrivenAgentRuntimeService
        from app.agents.events import AgentArtifact, CollaborationBlackboard
        from app.schemas.dtos import AiMessage
        board = CollaborationBlackboard(turn_id="late-context")
        board = board.add_artifact(AgentArtifact(id="response", owner="ResponseAgent", kind="response_proposal",
            payload={"candidateText": "请联系身边可信任的人，先关注安全。", "mode": "safety_support",
                "messages": [AiMessage(role="user", content="test")]}))
        board = board.accept_final("response", "CoordinatorAgent", "reviewed")
        board = board.add_artifact(AgentArtifact(id="late", owner="ContextAgent", kind="context",
            payload={"retrievedKnowledge": [SearchResult(1, "support.md", "稍后才获取的证据", 1.)]}))
        result = EventDrivenAgentRuntimeService(self.db, self.settings)._to_result(board, SimpleNamespace(display_name="测试"))
        self.assertEqual(result.retrieved_knowledge, [])

    def test_citations_keep_consumed_context_when_later_context_replaces_it(self):
        from app.agents.event_driven_runtime import EventDrivenAgentRuntimeService
        from app.agents.events import AgentArtifact, CollaborationBlackboard
        from app.schemas.dtos import AiMessage
        evidence = SearchResult(1, "old.md", "实际使用的条款", 1.)
        board = CollaborationBlackboard(turn_id="frozen-context")
        board = board.add_artifact(AgentArtifact(id="used", owner="ContextAgent", kind="context",
            payload={"retrievedKnowledge": [evidence]}))
        board = board.add_artifact(AgentArtifact(id="response", owner="ResponseAgent", kind="response_proposal",
            payload={"candidateText": "实际使用的条款", "mode": "policy_answer", "knowledgeContextId": "used",
                "messages": [AiMessage(role="user", content="test")]}))
        board = board.accept_final("response", "CoordinatorAgent", "reviewed")
        board = board.add_artifact(AgentArtifact(id="late", owner="ContextAgent", kind="context",
            payload={"retrievedKnowledge": [SearchResult(2, "new.md", "稍后条款", 1.)]}))
        result = EventDrivenAgentRuntimeService(self.db, self.settings)._to_result(board, SimpleNamespace(display_name="测试"))
        self.assertEqual(result.retrieved_knowledge, [evidence])


if __name__ == "__main__":
    unittest.main()
