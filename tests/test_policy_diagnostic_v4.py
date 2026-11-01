import copy
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch
from app.core.config import Settings
from app.policy_eval.diagnostic_v4 import ablation_rows, question_variants, diagnostic_messages, evaluate_quality
from app.policy_eval.challenge_v4 import PATH
from app.policy_eval.rigorous import HOLDOUT
from app.policy_eval.runner import load_inputs, digest, build_chunks
from app.services.memory_context import estimate_tokens
from app.policy_eval.answer_compare import run_stage
from app.policy_eval.repair_citations_v4 import citations_for, map_citation


class DiagnosticV4Tests(unittest.TestCase):
    def test_source_paragraph_mapping_preserves_rejection_and_rejects_unknown_ids(self):
        contexts = [{'rank': 1, 'title': '助研', 'scope': '研究生',
                     'content': '助研津贴按月发放。\n\n每年发放10个月。'}]
        task = {'claims': [{'answer_quote': '每年10个月', 'statement': '助研津贴每年发10个月'}],
                'contexts': contexts}
        options = citations_for(contexts)
        result = map_citation(json.dumps({'verdicts': [{'index': 1, 'relation': 'entailed',
                                                       'citation_id': '1.2', 'reason': '原文明确规定'}]}), task, options)
        self.assertEqual(result['verdicts'][0]['support_quote'], '每年发放10个月。')
        rejected = map_citation(json.dumps({'verdicts': [{'index': 1, 'relation': 'insufficient',
                                                         'citation_id': '', 'reason': '部分条件未证实'}]}), task, options)
        self.assertEqual(rejected['verdicts'][0]['relation'], 'insufficient')
        self.assertEqual(rejected['verdicts'][0]['context_rank'], 0)
        with self.assertRaises(ValueError):
            map_citation(json.dumps({'verdicts': [{'index': 1, 'relation': 'entailed',
                                                   'citation_id': '1.99', 'reason': '伪造编号'}]}), task, options)

    def test_challenge_really_requires_multiple_documents_and_history(self):
        dataset, _ = load_inputs(PATH)
        self.assertEqual(len(dataset['cases']), 32)
        multi = [c for c in dataset['cases'] if c['category'] == 'cross_document']
        self.assertEqual(len(multi), 12)
        self.assertTrue(all(len({g['document_id'] for g in c['evidence']}) >= 2 for c in multi))
        histories = [c for c in dataset['cases'] if c['category'] == 'long_history']
        self.assertEqual(len(histories), 8)
        self.assertTrue(all(len(c['history']) >= 4 for c in histories))
        self.assertEqual(sum(not c['answerable'] for c in dataset['cases']), 6)
        self.assertTrue(all(c['review_status'] == 'pending_human' for c in dataset['cases']))
        self.assertEqual(dataset['parent_dataset_sha256'], digest(HOLDOUT.read_bytes()))

    def test_perturbations_do_not_retain_original_context_metrics_or_gold_docs(self):
        dataset, docs = load_inputs(HOLDOUT)
        parents = [p for p in build_chunks(docs) if p.kind == 'parent'][:3]
        contexts = [{'chunk_id': p.id, 'document_id': p.document_id, 'version': p.version,
                     'heading': p.heading, 'content': p.content, 'score': 1.} for p in parents]
        previous = {'results': [{'case_id': c['id'], 'method': 'parent_child_hybrid',
                                 'retrieval': {'case_id': c['id'], 'contexts': contexts,
                                               'estimated_tokens': sum(estimate_tokens(p.content) for p in parents),
                                               'metrics': {'sentinel': 'must not be reused'}}}
                                for c in dataset['cases']]}
        rows = ablation_rows(dataset, docs, previous)
        self.assertEqual(len(rows), 216)
        cases = {c['id']: c for c in dataset['cases']}
        for row in rows:
            case = cases[row['case_id']]
            self.assertIsNone(row['metrics'])
            if row['method'] in {'no_context_strict', 'parametric_probe'}:
                self.assertEqual(row['contexts'], [])
            if row['method'] == 'random_context':
                blocked = {g['document_id'] for g in case['evidence']} | set(case['checked_source_ids'])
                self.assertFalse(blocked & {c['document_id'] for c in row['contexts']})
            if row['method'] == 'oracle_evidence':
                self.assertEqual([c['content'] for c in row['contexts']], [g['quote'] for g in case['evidence']])
            self.assertLessEqual(row['estimated_tokens'], 4000)
        self.assertEqual(rows, ablation_rows(dataset, docs, previous))

    def test_wording_variants_preserve_gold_and_never_contaminate_messages(self):
        dataset, docs = load_inputs(HOLDOUT)
        unchanged = copy.deepcopy(dataset)
        variants = question_variants(dataset)
        self.assertEqual(dataset, unchanged)
        self.assertEqual(len(variants['cases']), 8)
        originals = {c['id']: c for c in dataset['cases']}
        for case in variants['cases']:
            old = originals[case['id']]
            self.assertEqual(case['evidence'], old['evidence'])
            self.assertEqual(case['answer_points'], old['answer_points'])
            self.assertEqual(case['history'], old['history'])
            self.assertNotEqual(case['question'], old['question'])
            # Reference labels are never serialized to the generation prompt.
            row = {'method': 'parametric_probe', 'contexts': []}
            msg = diagnostic_messages(case, row, docs)
            self.assertEqual(msg[-1]['content'], case['question'])
            self.assertNotIn('reference_points', json.dumps(msg, ensure_ascii=False))


class DiagnosticOutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_broken_constrained_json_retries_transport_without_changing_verdict(self):
        calls = []
        async def complete(client, payload, endpoint, key):
            calls.append(copy.deepcopy(payload))
            return {'text': '{"' if len(calls) == 1 else '{"covered":false}',
                    'finish_reason': 'stop', 'usage': {}, 'http_attempts': 1}
        original = {'model': 'test', 'temperature': 0, 'response_format': {'type': 'json_object'},
                    'messages': [{'role': 'system', 'content': 'Keep the rubric fixed.'}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'quality.jsonl'
            with patch('app.policy_eval.answer_compare.complete', new=AsyncMock(side_effect=complete)):
                results = await run_stage('quality', [{'case_id': 'test', 'method': 'test', 'payload': original}],
                                          path, 'https://example.invalid', '', 1,
                                          lambda text, task: json.loads(text))
        self.assertEqual(len(calls), 2)
        self.assertNotIn('response_format', calls[1])
        self.assertEqual(calls[1]['messages'][0], original['messages'][0])
        self.assertIn('response_format', original)
        self.assertFalse(results['test', 'test']['parsed']['covered'])
        self.assertTrue(results['test', 'test']['json_mode_transport_fallback'])

    async def test_nested_output_exists_before_first_api_stage(self):
        dataset, _ = load_inputs(HOLDOUT)
        case = dataset['cases'][0]
        method = 'no_context_strict'
        row = {'case_id': case['id'], 'method': method, 'contexts': [], 'estimated_tokens': 0, 'metrics': None}
        key = (case['id'], method)
        generation = {key: {'text': '原文测试回答', 'finish_reason': 'stop', 'usage': {}}}
        judgment = {key: {'parsed': {'point_verdicts': [{'index': i, 'covered': True} for i in range(1, len(case['answer_points']) + 1)],
                                       'contradictions': [], 'fabricated_specifics': False, 'behavior': 'answer', 'context_relevance': []}}}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'new' / 'nested'
            async def fake_stage(stage, tasks, path, *args):
                self.assertTrue(path.parent.is_dir())
                return generation if stage == 'generation' else judgment
            with patch('app.policy_eval.diagnostic_v4.run_stage', new=AsyncMock(side_effect=fake_stage)):
                result = await evaluate_quality(HOLDOUT, output, [row], Settings())
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0]['quality']['answer_correct'])


if __name__ == '__main__':
    unittest.main()
