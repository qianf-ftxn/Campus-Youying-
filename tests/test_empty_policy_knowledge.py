from unittest.mock import patch, Mock

from test_campus_youying import CampusBase, POLICY
from app.models.entities import KnowledgeDocument, KnowledgeChunk, UserAccount
from app.agents.harness import MindBridgeAgentHarness
from app.schemas.dtos import ChatRequest


class EmptyPolicyTests(CampusBase):
    def test_demo_cleanup_preserves_real_documents_and_removes_child_and_parent_vectors(self):
        self.service.ingest("demo.md", POLICY, is_demo=True)
        self.service.ingest("official.md", POLICY + "正式文件测试", is_demo=False)
        demo = self.db.query(KnowledgeDocument).filter_by(is_demo=True).one()
        ids = {c.id for c in self.db.query(KnowledgeChunk).filter_by(document_id=demo.id)}
        vector = Mock(can_embed=True)
        self.service.vector_store = vector
        self.assertEqual(self.service.remove_demo_documents(), 1)
        self.assertEqual(set(vector.delete_chunk_ids.call_args.args[0]), ids)
        self.assertEqual(self.db.query(KnowledgeDocument).count(), 1)
        self.assertEqual(self.db.query(KnowledgeDocument).one().source, "official.md")
        self.assertEqual(self.db.query(KnowledgeChunk).filter(KnowledgeChunk.id.in_(ids)).count(), 0)
        self.assertEqual(self.service.remove_demo_documents(), 0)

    def test_restart_does_not_recreate_policy_demo_or_remove_real_uploads(self):
        from app.core.bootstrap import seed_data
        self.service.ingest("old-demo.md", POLICY, is_demo=True)
        self.service.ingest("official.md", POLICY, is_demo=False)
        with patch("app.core.bootstrap.get_settings", return_value=self.settings):
            seed_data(self.db)
            seed_data(self.db)
        self.assertEqual(self.db.query(KnowledgeDocument).filter_by(is_demo=True).count(), 0)
        policies = self.db.query(KnowledgeDocument).filter_by(domain="policy").all()
        self.assertEqual([d.source for d in policies], ["official.md"])

    def test_empty_policy_answer_is_explicit_without_using_model_guesses(self):
        user = UserAccount(username="no-files", display_name="学生", password_hash="x")
        self.db.add(user); self.db.commit()
        out = MindBridgeAgentHarness(self.db, self.settings).run(user,
            ChatRequest(message="研究生出差每天的伙食补助是多少？"))
        self.assertIn("对应的有效文件", out.response_text)
        self.assertIn("请先添加对应资料", out.response_text)
        self.assertFalse(out.retrieved_knowledge)
        self.assertIsNone(out.report_id)
        calls = (out.memory_snapshot_metadata or {}).get("modelCalls", [])
        self.assertFalse(any(c["agent"] == "ResponseAgent" for c in calls))

    def test_real_upload_enables_retrieval_after_empty_state(self):
        self.assertEqual(self.service.status()["activePolicyDocuments"], 0)
        self.service.ingest_file("official.md", POLICY.encode(), is_demo=False)
        self.assertEqual(self.service.status()["activePolicyDocuments"], 1)
        self.assertTrue(self.service.retrieve("伙食补助", domain="policy", school=self.settings.campus_name))

    def test_vector_cleanup_failure_cannot_leave_demo_evidence_retrievable(self):
        self.service.ingest("demo.md", POLICY, is_demo=True)
        self.service.vector_store = Mock(can_embed=True)
        self.service.vector_store.delete_chunk_ids.side_effect = RuntimeError("index unavailable")
        self.service.remove_demo_documents()
        self.assertEqual(self.service.retrieve("伙食补助", domain="policy"), [])
