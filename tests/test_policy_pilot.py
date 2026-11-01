"""Regression checks for evidence scoring and the evaluation isolation contract."""
import copy
import unittest

from app.policy_eval.runner import (
    Chunk, evidence_coverage, load_inputs, retrieval_query, score_case,
    select_context, summarize, validate_dataset,
)
from app.policy_eval.pilot import DATASET


class PolicyPilotScoringTests(unittest.TestCase):
    def setUp(self):
        self.documents = {"doc": ({"version": "v1"}, "甲条件乙条件丙例外丁")}
        self.gold = {"document_id": "doc", "version": "v1", "start_char": 0,
                     "end_char": 10, "quote": "甲条件乙条件丙例外丁"}

    def chunk(self, content, *, version="v1", document_id="doc"):
        return Chunk(1, document_id, version, "flat", "", content)

    def test_partial_evidence_is_not_a_complete_clause(self):
        selected = [(self.chunk("甲条件乙条件"), 1.)]
        case = {"answerable": True, "evidence": [self.gold]}
        metrics = score_case(case, selected, self.documents)
        self.assertFalse(metrics["evidence_hit_at_k"])
        self.assertEqual(metrics["clause_recall_at_k"], 0)
        self.assertFalse(metrics["evidence_complete"])

    def test_overlapping_windows_do_not_double_count(self):
        selected = [(self.chunk("甲条件乙条件"), 1.), (self.chunk("乙条件"), .9)]
        self.assertAlmostEqual(evidence_coverage(self.gold, selected, self.documents), .6)

    def test_complementary_windows_can_cover_a_whole_clause(self):
        selected = [(self.chunk("甲条件乙条件"), 1.), (self.chunk("乙条件丙例外丁"), .9)]
        self.assertEqual(evidence_coverage(self.gold, selected, self.documents), 1.)

    def test_same_quote_from_other_document_or_version_does_not_count(self):
        for row in [self.chunk(self.gold["quote"], version="v0"),
                    self.chunk(self.gold["quote"], document_id="other")]:
            self.assertEqual(evidence_coverage(self.gold, [(row, 1.)], self.documents), 0.)

    def test_parent_page_marker_cleanup_does_not_lose_real_evidence(self):
        source = "前言\n<!-- page:2 -->\n甲条件乙条件丙例外丁"
        docs = {"doc": ({"version": "v1"}, source)}
        gold = dict(self.gold, start_char=source.index("甲"), end_char=len(source))
        row = Chunk(1, "doc", "v1", "parent", "", "前言\n甲条件乙条件丙例外丁")
        self.assertEqual(evidence_coverage(gold, [(row, 1.)], docs), 1.)

    def test_multiple_required_clauses_are_scored_separately(self):
        second = dict(self.gold, start_char=6, end_char=10, quote="丙例外丁")
        case = {"answerable": True, "evidence": [dict(self.gold, end_char=6, quote="甲条件乙条件"), second]}
        result = score_case(case, [(self.chunk("甲条件乙条件"), 1.)], self.documents)
        self.assertEqual(result["clause_recall_at_k"], .5)
        self.assertFalse(result["evidence_complete"])

    def test_negative_questions_are_not_treated_as_successful_abstentions(self):
        metrics = score_case({"answerable": False}, [], self.documents)
        self.assertIsNone(metrics["abstention_correct"])
        summary = summarize([{"metrics": metrics, "estimated_tokens": 0, "retrieval_ms": 1}])
        self.assertEqual(summary["answerable_cases"], 0)
        self.assertIsNone(summary["evidence_complete_rate"])
        self.assertIsNone(summary["unanswerable_abstention_accuracy"])

    def test_unmeasured_latency_is_not_reported_as_zero(self):
        metrics = score_case({"answerable": False}, [], self.documents)
        summary = summarize([{"metrics": metrics, "estimated_tokens": 0, "retrieval_ms": None}])
        self.assertIsNone(summary["mean_local_retrieval_ms"])

    def test_top_k_is_applied_before_budget_filtering(self):
        ranked = [(self.chunk("甲" * 100), 1.), (self.chunk("乙"), .9), (self.chunk("丙"), .8)]
        selected, used = select_context(ranked, 2, 10)
        self.assertEqual([row.content for row, _ in selected], ["乙"])
        self.assertEqual(used, 1)

    def test_query_never_uses_gold_answer_or_evidence(self):
        case = {"question": "更正，我是自发修课。", "history": [{"role": "user", "content": "境外交流"},
                {"role": "assistant", "content": "答案泄漏"}], "answer_points": ["秘密"], "evidence": [{"quote": "秘密"}]}
        query = retrieval_query(case)
        self.assertEqual(query, "境外交流\n更正，我是自发修课。")


class PolicyPilotDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset, cls.documents = load_inputs(DATASET)

    def test_all_cases_are_grounded_and_have_review_fields(self):
        self.assertEqual(len(self.dataset["cases"]), 25)
        self.assertEqual(sum(case["answerable"] for case in self.dataset["cases"]), 20)
        self.assertTrue(all(case["review_status"] in {"pending_human", "approved", "needs_revision"}
                            for case in self.dataset["cases"]))

    def test_changed_snapshot_and_wrong_offsets_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_dataset(self.dataset, self.documents, "wrong hash")
        broken = copy.deepcopy(self.dataset)
        broken["cases"][0]["evidence"][0]["start_char"] += 1
        with self.assertRaises(ValueError):
            validate_dataset(broken, self.documents, self.dataset["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
