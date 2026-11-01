import copy
import json
import unittest

from app.task_eval.judging import judge_input, parse_judgment, semantic_checks
from app.task_eval.prepare import build_dataset
from app.task_eval.runner import summarize
from app.task_eval.review import inconsistent_labels


class TaskJudgingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = build_dataset()["cases"][0]

    def judgment(self):
        a = self.case["acceptance"]
        return {"point_verdicts": [{"index": i + 1, "covered": True, "reason": "fixture"} for i in range(len(a["required_answer_points"]))],
                "forbidden_verdicts": [{"index": i + 1, "violated": False, "reason": "fixture"} for i in range(len(a["forbidden_behaviors"]))],
                "contradictions": [], "fabricated_specifics": False, "internal_metadata_leak": False,
                "behavior": "answer", "reason": "fixture"}

    def test_judge_does_not_receive_architecture_or_tool_outcome(self):
        payload = judge_input(self.case, {"answer": "回答", "arm": "multi", "effects": {"secret": True}}, [])
        self.assertNotIn("arm", payload)
        self.assertNotIn("effects", payload)
        self.assertNotIn("intent", payload)

    def test_missing_duplicate_or_string_boolean_verdict_is_rejected(self):
        for mode in ("missing", "duplicate", "string"):
            value = self.judgment()
            if mode == "missing": value["point_verdicts"].pop()
            if mode == "duplicate": value["point_verdicts"][1]["index"] = 1
            if mode == "string": value["point_verdicts"][0]["covered"] = "true"
            with self.assertRaises(ValueError): parse_judgment(json.dumps(value), self.case)

    def test_any_substantive_error_fails_even_with_full_point_coverage(self):
        value = self.judgment()
        value["contradictions"] = ["wrong applicability"]
        parsed = parse_judgment(json.dumps(value), self.case)
        checks = semantic_checks(self.case, parsed)
        self.assertTrue(checks["required_answer_points"])
        self.assertFalse(all(checks.values()))

    def test_explicit_label_reason_conflict_is_flagged_without_relabeling(self):
        value = self.judgment()
        value["point_verdicts"][0]["reason"] = "遗漏必要材料，因此应判定为false。"
        problems = inconsistent_labels(value)
        self.assertEqual(len(problems), 1)
        self.assertTrue(value["point_verdicts"][0]["covered"])

    def test_ordinary_negation_is_not_an_explicit_label_conflict(self):
        value = self.judgment()
        value["point_verdicts"][0]["reason"] = "没有编造个人办理结果，符合否定约束。"
        self.assertEqual(inconsistent_labels(value), [])

    def test_incomplete_judging_cannot_produce_final_success_rate(self):
        observation = {"model_calls": 1, "input_tokens": 10, "output_tokens": 3,
                       "response_completed": True, "effect_execution_completed": True}
        result = {"case_id": "a", "group": "policy", "observations": {"single": observation, "multi": observation},
                  "task_success": {"single": None, "multi": True}, "checks": {"single": {}, "multi": {}}}
        summary = summarize([result], 1)
        self.assertIsNone(summary["arms"]["single"]["task_success_rate"])
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["multi_only_success"], [])

    def test_failures_stay_in_the_denominator_and_regressions_are_reported(self):
        observation = {"model_calls": 1, "input_tokens": 10, "output_tokens": 3,
                       "response_completed": False, "effect_execution_completed": False}
        rows = []
        for case_id, flags in (("a", {"single": False, "multi": True}), ("b", {"single": True, "multi": False})):
            rows.append({"case_id": case_id, "group": "policy", "observations": {"single": observation, "multi": observation},
                         "task_success": flags, "checks": {"single": {}, "multi": {}}})
        summary = summarize(rows, 2)
        self.assertEqual(summary["arms"]["single"]["task_success_rate"], .5)
        self.assertEqual(summary["multi_only_success"], ["a"])
        self.assertEqual(summary["single_only_success"], ["b"])


if __name__ == "__main__": unittest.main()
