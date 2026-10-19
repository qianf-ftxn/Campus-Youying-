"""Portable official corpus, offline bootstrap, integrity and school isolation."""
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from test_campus_youying import CampusBase
from app.models.entities import KnowledgeDocument, KnowledgeChunk
from app.services.knowledge import KnowledgeService
from app.services.knowledge_bundle import load_bundle, import_bundle, bundle_status, PROJECT_ROOT


class OfficialBundleTests(CampusBase):
    def setUp(self):
        super().setUp()
        self.settings.campus_name = "西安电子科技大学"
        self.service = KnowledgeService(self.db, self.settings)

    def fixture(self, root):
        content = '# 官方测试文件\n\n请假需要办理审批。\n'
        raw = b'<html>official fixture</html>'
        (root/'text.md').write_text(content)
        (root/'raw.html').write_bytes(raw)
        entry = {'id':'fixture', 'school':'西安电子科技大学', 'domain':'policy',
            'title':'隔离测试文件', 'department':'测试部门', 'version':'v1', 'status':'ACTIVE',
            'source_url':'https://ygb.xidian.edu.cn/info/1061/13093.htm',
            'content_path':'text.md','content_sha256':hashlib.sha256(content.encode()).hexdigest(),
            'original_path':'raw.html','original_sha256':hashlib.sha256(raw).hexdigest()}
        manifest = {'schema_version':1,'school':'西安电子科技大学','checked_on':'2026-10-19','documents':[entry]}
        (root/'manifest.json').write_text(json.dumps(manifest))
        return manifest

    def test_committed_corpus_imports_into_fresh_db_and_is_idempotent(self):
        manifest, rows = load_bundle(PROJECT_ROOT/'knowledge/xidian')
        self.assertGreaterEqual(len(rows), 50)
        first = import_bundle(self.service,self.settings)
        docs = self.db.query(KnowledgeDocument).count()
        chunks = self.db.query(KnowledgeChunk).count()
        self.assertEqual(docs,len(rows))
        self.assertEqual(first['imported'],len(rows))
        self.assertGreater(chunks,docs)
        second = import_bundle(self.service,self.settings)
        self.assertEqual(second['imported'],0)
        self.assertEqual(self.db.query(KnowledgeDocument).count(),docs)
        self.assertEqual(self.db.query(KnowledgeChunk).count(),chunks)
        self.assertFalse(self.db.query(KnowledgeDocument).filter_by(is_demo=True).count())
        self.assertEqual(self.db.query(KnowledgeDocument).filter_by(status='DRAFT').count(),
                         sum(d['status']=='DRAFT' for d in manifest['documents']))
        for question,document,phrase in [
            ('研究生单次请假时长原则上不超过多久','ygb-1061-13093','2周'),
            ('研究生校外住宿连续居住超过几天需要申请','ygb-1061-13103','14天'),
            ('全日制硕士研究生最长学习年限','gr-1045-16474','4年'),
            ('研究生转导师管理办法','gr-1039-10195','导师'),
        ]:
            with self.subTest(question=question):
                results = self.service.retrieve(question,domain='policy',school=self.settings.campus_name)
                self.assertTrue(any(document in r.source and phrase in r.content for r in results),
                                [(r.source,r.heading) for r in results])

    def test_administrator_archive_is_not_reactivated_on_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); self.fixture(root)
            import_bundle(self.service,self.settings,root)
            doc=self.db.query(KnowledgeDocument).one()
            self.service.set_document_status(doc.id,'ARCHIVED')
            import_bundle(self.service,self.settings,root)
            self.assertEqual(self.db.query(KnowledgeDocument).one().status,'ARCHIVED')
            self.assertEqual(self.service.retrieve('请假审批',domain='policy'),[])

    def test_another_school_and_disabled_bundle_do_not_import_xidian(self):
        self.settings.campus_name='另一所学校'
        result=import_bundle(self.service,self.settings)
        self.assertEqual(result['imported'],0)
        self.settings.bundled_knowledge_enabled=False
        result=import_bundle(self.service,self.settings)
        self.assertFalse(result['enabled'])
        self.assertEqual(self.db.query(KnowledgeDocument).count(),0)

    def test_missing_and_modified_files_fail_before_writing_any_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); manifest=self.fixture(root)
            (root/'raw.html').unlink()
            with self.assertRaisesRegex(ValueError,'缺失'):
                import_bundle(self.service,self.settings,root)
            self.assertEqual(self.db.query(KnowledgeDocument).count(),0)
            self.fixture(root)
            (root/'text.md').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'校验失败'):
                import_bundle(self.service,self.settings,root)
            self.assertEqual(self.db.query(KnowledgeDocument).count(),0)

    def test_path_traversal_and_nonofficial_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); manifest=self.fixture(root)
            manifest['documents'][0]['content_path']='../outside.md'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'路径不合法'):
                load_bundle(root)
            manifest=self.fixture(root)
            manifest['documents'][0]['source_url']='https://example.com/copied-rule'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'官方来源'):
                load_bundle(root)

    def test_bundle_status_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); self.fixture(root)
            self.settings.bundled_knowledge_dir=str(root)
            (root/'raw.html').unlink()
            state=bundle_status(self.settings)
            self.assertFalse(state['available'])
            self.assertIn('raw.html',state['missingFiles'])

    def test_merged_award_table_keeps_student_type_on_each_row(self):
        _, rows = load_bundle(PROJECT_ROOT/'knowledge/xidian')
        text = next(text for entry,text in rows if entry['id']=='ygb-1059-1593')
        self.assertIn('| 硕士研究生 | 三年级 | 600元/月 | 10个月/年 |',text)
        self.assertIn('| 博士研究生 | 中期考核后 | 2250元/月 | 12个月/年 |',text)
