import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from app.services.knowledge import SearchResult
from app.services.knowledge_reranking import parse_order, rerank_request, semantic_rerank
from tests.test_campus_youying import CampusBase


class SemanticRankingUnitTests(unittest.TestCase):
    def test_ids_are_unique_known_and_bounded(self):
        for raw in ['{"ordered_ids":[99]}', '{"ordered_ids":[1,1]}', '{"ordered_ids":[1,2,3]}']:
            with self.assertRaises(ValueError):
                parse_order(raw, {1, 2, 3}, 2)

    def test_no_suitable_evidence_can_return_empty(self):
        self.assertEqual(parse_order('{"ordered_ids":[]}', {1, 2}, 5), [])

    def test_request_contains_only_query_and_source_candidates(self):
        request = rerank_request('课程审批', [{'id': 1, 'content': '先申请再审批。'}], 5, 'model')
        supplied = json.loads(request['messages'][1]['content'])
        self.assertEqual(set(supplied), {'query', 'top_k', 'candidates'})
        self.assertNotIn('reference', json.dumps(request))

    def test_selecting_fewer_than_k_does_not_fill_with_rejected_candidates(self):
        rows = [SearchResult(i, 'doc', f'原文{i}', .9) for i in [1, 2, 3]]
        settings = SimpleNamespace(ai_provider='openai', openai_api_key='test', openai_model='model',
            openai_base_url='https://example.invalid', openai_enable_thinking=False,
            knowledge_semantic_rerank_model='', knowledge_semantic_rerank_candidate_limit=32,
            knowledge_semantic_rerank_timeout_seconds=30)
        response = httpx.Response(200, json={'choices': [{'message': {'content': '{"ordered_ids":[3,1]}'}}]})
        with patch('app.services.knowledge_reranking.httpx.post', return_value=response):
            selected = semantic_rerank('问题', rows, settings, 5)
        self.assertEqual([r.chunk_id for r in selected], [3, 1])
        self.assertEqual([r.content for r in selected], ['原文3', '原文1'])


class SemanticRankingIntegrationTests(CampusBase):
    def test_rerank_failure_preserves_existing_retrieval(self):
        self.add_policy()
        baseline = self.service.retrieve('研究生住宿费上限', domain='policy', school='测试学校')
        self.settings.knowledge_semantic_rerank_enabled = True
        with patch('app.services.knowledge_reranking.semantic_rerank', side_effect=RuntimeError('test outage')):
            result = self.service.retrieve('研究生住宿费上限', domain='policy', school='测试学校')
        self.assertEqual([r.chunk_id for r in baseline], [r.chunk_id for r in result])
        self.assertEqual(self.service.last_retrieval['semanticRerankError'], 'RuntimeError')

    def test_service_honors_semantic_order(self):
        self.add_policy()
        baseline = self.service.retrieve('研究生伙食补助住宿费', domain='policy', school='测试学校')
        self.assertGreaterEqual(len(baseline), 2)
        self.settings.knowledge_semantic_rerank_enabled = True
        with patch('app.services.knowledge_reranking.semantic_rerank', side_effect=lambda q, rows, s, k: list(reversed(rows))):
            result = self.service.retrieve('研究生伙食补助住宿费', domain='policy', school='测试学校')
        self.assertEqual(result[0].chunk_id, baseline[-1].chunk_id)
