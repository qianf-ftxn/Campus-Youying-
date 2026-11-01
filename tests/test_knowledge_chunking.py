"""Hierarchy, source integrity, and migration regressions for policy chunking."""
from collections import Counter
import unittest
from unittest.mock import patch

from test_campus_youying import CampusBase
from app.models.entities import KnowledgeChunk, KnowledgeDocument
from app.services.knowledge import parent_sections, child_sections
from app.services.knowledge_chunking import child_heading, knowledge_heading


DOCUMENT = """# 研究生办理规则

第一章 申请与审批

第一条 在籍研究生可以提出申请。

第二条 须提交本人身份证明，并经导师审核。

第三条 紧急情况下可以先报备，返校后补齐材料。

第二章 地点变更

第四条 地点变化应重新申请。
"""


class HierarchicalChunkingTests(unittest.TestCase):
    def test_articles_remain_under_chapter_and_children_are_specific(self):
        parents = parent_sections(DOCUMENT, 1000)
        self.assertEqual(len(parents), 2)
        heading, body, _ = parents[0]
        self.assertIn("第一章 申请与审批", heading)
        self.assertIn("第一条—第三条", heading)
        self.assertIn("导师审核", body)
        self.assertIn("紧急情况", body)
        self.assertNotIn("地点变化", body)
        children = child_sections(body, 300, 40)
        self.assertTrue(any("第二条" in text and "导师审核" in text for text in children))
        self.assertTrue(any("第三条" in text and "补齐材料" in text for text in children))
        self.assertTrue(all(text != body for text in children))

    def test_numbered_topics_are_separate_but_arabic_steps_stay_in_topic(self):
        source = "# 指南\n\n一、住宿手续\n\n1.适用情况\n\n地址变更须重新申请。\n\n2.办理流程\n\n登录住宿系统办理。\n\n二、请假手续\n\n到期前申请续假。"
        parents = parent_sections(source, 1000)
        self.assertEqual(len(parents), 2)
        self.assertIn("一、住宿手续", parents[0][0])
        self.assertIn("地址变更", parents[0][1])
        self.assertIn("登录住宿系统", parents[0][1])
        self.assertNotIn("续假", parents[0][1])
        self.assertIn("二、请假手续", parents[1][0])

    def test_parenthesized_short_headings_keep_ancestor_path(self):
        source = "# 指南\n\n一、申请\n\n（一）提交材料\n\n准备身份证明。\n\n（二）审批流程\n\n交导师审批。"
        parents = parent_sections(source, 1000)
        self.assertEqual(len(parents), 2)
        self.assertIn("一、申请 / （一）提交材料", parents[0][0])
        self.assertIn("一、申请 / （二）审批流程", parents[1][0])

    def test_enumerated_conditions_inside_an_article_stay_with_the_rule(self):
        source = "第一章 申请\n\n第一条 申请须同时满足以下条件：\n\n（一）全日制在籍研究生\n\n（二）未超过修业年限\n\n第二条 例外由学院审批。"
        parents = parent_sections(source, 1000)
        self.assertEqual(len(parents), 1)
        self.assertIn("须同时满足", parents[0][1])
        self.assertIn("未超过修业年限", parents[0][1])

    def test_short_independent_document_is_not_artificially_split(self):
        parents = parent_sections("# 独立规定\n\n办理时需携带身份证。", 1000)
        self.assertEqual(len(parents), 1)
        self.assertEqual(child_sections(parents[0][1], 300, 40), ["办理时需携带身份证。"])

    def test_article_sentence_does_not_become_repeated_heading(self):
        heading, body, _ = parent_sections(DOCUMENT, 1000)[0]
        heading = knowledge_heading("研究生办理规则", heading)
        self.assertEqual(heading.count("研究生办理规则"), 1)
        self.assertNotIn("可以提出申请", heading)
        self.assertIn("第二条", child_heading(heading, "第二条 须提交本人身份证明，并经导师审核。"))

    def test_oversized_article_splits_without_losing_content(self):
        source = "第一章 办理\n\n第一条 " + "提交本人材料并经过审批。" * 100
        parents = parent_sections(source, 200)
        self.assertGreater(len(parents), 1)
        self.assertTrue(all(len(body) <= 200 for _, body, _ in parents))
        compact = lambda text: "".join(text.split())
        body = compact("".join(part[1] for part in parents))
        self.assertIn(compact("第一条 " + "提交本人材料并经过审批。" * 100), body)

    def test_adjacent_table_and_prose_are_retained_and_table_not_torn(self):
        table = "| 对象 | 标准 |\n|---|---|\n| 研究生 | 80元 |"
        text = "第一条 标准如下。\n" + table + "\n特殊情况不重复领取。"
        children = child_sections(text, 40, 10)
        self.assertTrue(any(table == item for item in children))
        self.assertIn("特殊情况不重复领取。", "".join(children))
        self.assertIn("第一条 标准如下。", "".join(children))

    def test_page_marker_does_not_sever_a_topic_and_first_page_is_recorded(self):
        source = "# 指南\n\n<!-- page:2 -->\n第一章 办理\n第一条 提交材料。\n<!-- page:3 -->\n第二条 例外需审批。"
        parents = parent_sections(source, 1000)
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0][2], 2)
        self.assertIn("例外需审批", parents[0][1])
        self.assertNotIn("<!--", parents[0][1])

    def test_heading_only_input_has_no_chunks(self):
        self.assertEqual(parent_sections("# 规则\n\n第一章 总则\n\n第一节 范围", 1000), [])

    def test_committed_corpus_has_real_hierarchy_and_source_alignment(self):
        from app.policy_eval.runner import build_chunks, compact, load_inputs, without_page_markers
        from app.policy_eval.pilot import DATASET
        _, documents = load_inputs(DATASET)
        chunks = build_chunks(documents)
        parents = [row for row in chunks if row.kind == "parent"]
        children = [row for row in chunks if row.kind == "child"]
        counts = Counter(row.parent_id for row in children)
        self.assertLess(sum(counts[row.id] == 1 for row in parents) / len(parents), .2)
        for row in chunks:
            normalize = compact if row.kind == "flat" else without_page_markers
            self.assertIn(normalize(row.content), normalize(documents[row.document_id][1]), row.id)


class ChunkGenerationMigrationTests(CampusBase):
    def test_existing_sqlite_documents_table_gets_idempotent_version_column(self):
        from app.core import bootstrap
        from sqlalchemy import inspect
        with self.engine.begin() as connection:
            connection.exec_driver_sql("ALTER TABLE knowledge_documents DROP COLUMN chunking_version")
        with patch.object(bootstrap, "engine", self.engine):
            bootstrap.create_schema()
            bootstrap.create_schema()
        columns = {column["name"] for column in inspect(self.engine).get_columns("knowledge_documents")}
        self.assertIn("chunking_version", columns)
        self.service.ingest("rules.md", DOCUMENT)
        self.assertEqual(self.db.query(KnowledgeDocument).one().chunking_version, self.service.chunking_version)

    def test_existing_document_is_rechunked_once_old_ids_and_status_retained(self):
        # Reproduce the old article-as-parent layout, not just a version label.
        doc = KnowledgeDocument(source="rules.md", content=DOCUMENT, title="研究生办理规则", version="official-v1",
            domain="policy", school=self.settings.campus_name, status="ARCHIVED", chunking_version="")
        self.db.add(doc)
        self.db.flush()
        for index, line in enumerate(line for line in DOCUMENT.splitlines() if line.startswith(("第一条", "第二条", "第三条", "第四条"))):
            parent = KnowledgeChunk(source=doc.source, source_index=index, document_id=doc.id, content=line,
                chunk_type="parent", heading=doc.title + " / " + line, version=doc.version, domain=doc.domain, school=doc.school)
            self.db.add(parent)
            self.db.flush()
            self.db.add(KnowledgeChunk(source=doc.source, source_index=0, document_id=doc.id, parent_id=parent.id,
                content=line, chunk_type="child", heading=parent.heading, version=doc.version, domain=doc.domain, school=doc.school))
        self.db.flush()
        old = self.db.query(KnowledgeChunk).all()
        old_ids = [row.id for row in old]
        self.db.commit()
        result = self.service.rechunk_documents()
        self.assertEqual(result["documents"], 1)
        self.assertEqual(doc.status, "ARCHIVED")
        self.assertEqual(doc.version, "official-v1")
        self.assertEqual(self.db.query(KnowledgeChunk).filter_by(document_id=doc.id, chunk_type="parent").count(), 2)
        self.assertTrue(all(self.db.get(KnowledgeChunk, key).chunk_type.startswith("retired_") for key in old_ids))
        self.assertTrue(set(old_ids).isdisjoint(row.id for row in self.service._all_index_rows()))
        self.assertTrue(all(row["type"] in {"parent", "child"} for row in self.service.document_detail(doc.id)["chunks"]))
        count = self.db.query(KnowledgeChunk).count()
        self.assertEqual(self.service.rechunk_documents()["documents"], 0)
        self.assertEqual(self.db.query(KnowledgeChunk).count(), count)

    def test_failed_rechunk_restores_generation_and_old_retrieval(self):
        self.service.ingest("rules.md", DOCUMENT)
        doc = self.db.query(KnowledgeDocument).one()
        doc.chunking_version = "old-v1"
        self.db.commit()
        original = {row.id: row.chunk_type for row in self.db.query(KnowledgeChunk).all()}
        create = self.service._create_document_chunks

        def fail(document):
            create(document)
            raise RuntimeError("injected migration failure")

        with patch.object(self.service, "_create_document_chunks", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "migration failure"):
                self.service.rechunk_documents()
        self.assertEqual({row.id: row.chunk_type for row in self.db.query(KnowledgeChunk).all()}, original)
        self.assertEqual(self.db.query(KnowledgeDocument).one().chunking_version, "old-v1")
        self.assertTrue(self.service.retrieve("导师审核", domain="policy"))

    def test_configuration_change_marks_document_for_rechunk(self):
        self.service.ingest("rules.md", DOCUMENT)
        self.settings.knowledge_chunk_size = 180
        self.assertEqual(self.service.status()["outdatedChunkDocuments"], 1)
        self.assertEqual(self.service.rechunk_documents()["documents"], 1)
        self.assertEqual(self.service.status()["outdatedChunkDocuments"], 0)

    def test_failed_vector_sync_uses_new_bm25_chunks_without_old_vector_ids(self):
        self.service.ingest("rules.md", DOCUMENT)
        old_ids = {row.id for row in self.service._all_index_rows()}
        self.db.query(KnowledgeDocument).one().chunking_version = "old-v1"
        self.db.commit()
        self.service.rechunk_documents()

        class StaleVector:
            can_embed = True
            error = ""

            def has_exact_chunk_ids(self, rows):
                return False

            def embed_texts(self, texts):
                return [[1., 0.] for _ in texts]

            def sync_chunks(self, rows, vectors):
                raise RuntimeError("injected vector sync failure")

            def query(self, *args, **kwargs):
                raise AssertionError("Stale vector query must not run")

        self.service.vector_store = StaleVector()
        result = self.service.retrieve("导师审核", domain="policy")
        self.assertTrue(result)
        self.assertTrue(all(row.chunk_id not in old_ids for row in result))
        self.assertEqual(self.service.last_retrieval["path"], "bm25_parent_child")

    def test_rechunked_ids_replace_real_chroma_index_atomically(self):
        import tempfile
        from app.services.knowledge import KnowledgeService
        from app.services.vector_store import ChromaKnowledgeStore
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.settings.model_copy(update={"knowledge_vector_enabled": True, "openai_api_key": "test-only",
                                                    "chroma_persist_dir": directory})
            with patch.object(ChromaKnowledgeStore, "_embed", lambda store, texts: [[float(1 + text.count("材料")), 1., 1.] for text in texts]):
                service = KnowledgeService(self.db, cfg)
                service.ingest("rules.md", DOCUMENT)
                old_ids = {row.id for row in service._all_index_rows()}
                old_collection = service.vector_store.collection_name
                self.db.query(KnowledgeDocument).one().chunking_version = "old-v1"
                self.db.commit()
                service.rechunk_documents()
                indexed = service.rebuild_vector_index()
                self.assertNotEqual(service.vector_store.collection_name, old_collection)
                current_ids = {row.id for row in service._all_index_rows()}
                self.assertTrue(current_ids.isdisjoint(old_ids))
                stored_ids = {item["db_id"] for item in service.vector_store.collection.get(include=["metadatas"])["metadatas"]}
                self.assertEqual(stored_ids, current_ids)
                self.assertEqual(indexed, len(current_ids))
                results = service.retrieve("导师审核材料", domain="policy", school=cfg.campus_name)
                self.assertTrue(results)
                self.assertTrue(all(row.chunk_id in current_ids for row in results))
