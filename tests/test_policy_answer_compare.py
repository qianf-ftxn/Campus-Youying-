import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.policy_eval.answer_compare import parse_faithfulness, parse_judgment, quality_metrics, quote_text, repair_items, run_stage


class AnswerComparisonTests(unittest.TestCase):
    def setUp(self):
        self.case = {"answerable": True, "answer_points": ["超过14天"]}
        self.judge = {"point_verdicts": [{"index": 1, "covered": True, "reason": "原文支持"}],
                      "contradictions": [], "behavior": "answer", "fabricated_specifics": False,
                      "context_relevance": [{"rank": 1, "relevant": False, "quote": ""},
                                            {"rank": 2, "relevant": True, "quote": "超过14天"}],
                      "claims": [{"answer_quote": "超过14天", "context_rank": 2,
                                  "support_quote": "超过14天", "supported": True}]}
        self.task = {"case": self.case, "contexts": [{"content": "其他规定"}, {"content": "连续居住超过14天。"}],
                     "answer": "连续居住超过**14天**。"}

    def test_single_result_mrr_uses_its_rank_not_cumulative_gold_coverage(self):
        result = quality_metrics(self.case, self.judge)
        self.assertEqual(result["mrr_at_5"], .5)
        self.assertTrue(result["hit_at_5"])

    def test_unsupported_claims_in_gold_cannot_be_marked_as_retrieved_support(self):
        bad = {'verdicts': [{'index': 1, 'relation': 'entailed', 'context_rank': 2, 'support_quote': '应提前申请'}]}
        task = {**self.task, 'claims': self.judge['claims']}
        with self.assertRaisesRegex(ValueError, "not present"):
            parse_faithfulness(json.dumps(bad), task)

    def test_markdown_is_ignored_but_policy_numbers_are_preserved(self):
        parse_judgment(json.dumps(self.judge), self.task)
        self.assertEqual(quote_text("1. 超过**14天**\n2. 期限2周"), "超过14天期限2周")
        self.assertNotEqual(quote_text("超过14天"), quote_text("超过15天"))
        self.assertEqual(quote_text("课程 <!-- page:4 --> - 4 - 不列入"), "课程不列入")

    def test_claim_free_abstention_is_not_perfect_faithfulness(self):
        negative = {"answerable": False, "expected_behavior": "abstain", "answer_points": ["资料不足"]}
        judge = copy.deepcopy(self.judge)
        judge.update(behavior="abstain", claims=[])
        result = quality_metrics(negative, judge)
        self.assertIsNone(result["faithfulness"])
        self.assertIsNone(result["mrr_at_5"])
        self.assertTrue(result["abstention_correct"])

    def test_entailment_label_cannot_disagree_with_its_own_reason(self):
        raw = {'verdicts': [{'index': 1, 'relation': 'entailed', 'context_rank': 2,
                             'support_quote': '超过14天', 'reason': '具体月份信息不足'}]}
        with self.assertRaisesRegex(ValueError, 'contradicts its reason'):
            parse_faithfulness(json.dumps(raw), {**self.task, 'claims': self.judge['claims']})

    def test_fact_citation_can_be_located_in_another_returned_block_of_same_source(self):
        raw = {'verdicts': [{'index': 1, 'relation': 'entailed', 'context_rank': 1,
                             'support_quote': '超过14天', 'reason': '原文支持'}]}
        contexts = [{**c, 'title': '住宿管理办法', 'scope': '研究生'} for c in self.task['contexts']]
        parsed = parse_faithfulness(json.dumps(raw), {'claims': self.judge['claims'], 'contexts': contexts})
        self.assertEqual(parsed['verdicts'][0]['context_rank'], 2)
        self.assertEqual(parsed['verdicts'][0]['reported_context_rank'], 1)

    def test_clarification_is_distinct_from_abstention(self):
        case = {"answerable": False, "expected_behavior": "clarify", "answer_points": ["先询问身份"]}
        judge = copy.deepcopy(self.judge)
        judge["behavior"] = "abstain"
        self.assertFalse(quality_metrics(case, judge)["clarification_correct"])
        judge["behavior"] = "clarify"
        self.assertTrue(quality_metrics(case, judge)["clarification_correct"])

    def test_generic_advice_is_not_a_policy_fact_but_raw_score_is_preserved(self):
        judge = copy.deepcopy(self.judge)
        judge['claims'] = [{'answer_quote': '建议向学院咨询', 'statement': '建议向学院咨询', 'supported': False}]
        result = quality_metrics(self.case, judge)
        self.assertIsNone(result['faithfulness'])
        self.assertEqual(result['faithfulness_raw_extractor'], 0)

    def test_specific_phone_in_advice_still_requires_evidence(self):
        judge = copy.deepcopy(self.judge)
        judge['claims'] = [{'answer_quote': '建议拨打12345678901咨询', 'supported': False}]
        self.assertEqual(quality_metrics(self.case, judge)['faithfulness'], 0)

    def test_cached_answers_never_collide_between_arms_with_identical_prompts(self):
        response = {"text": "回答", "usage": {}, "finish_reason": "stop", "http_attempts": 1}
        tasks = [{"case_id": "D01", "method": method, "payload": {"messages": []}} for method in ["a", "b"]]
        with tempfile.TemporaryDirectory() as directory, patch("app.policy_eval.answer_compare.complete", new_callable=AsyncMock, return_value=response) as mocked:
            path = Path(directory) / "answers.jsonl"
            first = asyncio.run(run_stage("generation", tasks, path, "https://example.invalid", "test", 2))
            self.assertEqual(mocked.call_count, 2)
            second = asyncio.run(run_stage("generation", tasks, path, "https://example.invalid", "test", 2))
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(first.keys(), second.keys())
            self.assertEqual(len(second), 2)

    def test_item_repair_can_reject_a_claim_instead_of_forcing_support(self):
        original = {'verdicts': [{'index': 1, 'relation': 'entailed', 'context_rank': 2,
                                  'support_quote': '不存在的条款', 'reason': '支持'}]}
        corrected = {'verdicts': [{'index': 1, 'relation': 'insufficient', 'context_rank': 0,
                                   'support_quote': '', 'reason': '资料没有支持此结论'}]}
        task = {**self.task, 'claims': self.judge['claims'], 'case_id': 'D01', 'method': 'test',
                'payload': {'messages': [{}, {'content': json.dumps({'contexts': self.task['contexts']})}]}}
        reply = {'text': json.dumps(corrected), 'usage': {}, 'http_attempts': 1}
        with tempfile.TemporaryDirectory() as directory, patch('app.policy_eval.answer_compare.complete', new_callable=AsyncMock, return_value=reply):
            result = asyncio.run(repair_items('faithfulness', json.dumps(original), task, None, 'unused', 'unused', Path(directory)))
        self.assertEqual(result['verdicts'][0]['relation'], 'insufficient')


if __name__ == "__main__":
    unittest.main()
