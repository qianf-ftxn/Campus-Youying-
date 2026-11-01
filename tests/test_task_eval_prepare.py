"""Contract regressions for candidate tasks; no model/service calls."""
import copy
import unittest

from app.task_eval.prepare import build_dataset, candidate_input, validate_dataset


class TaskDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = build_dataset()

    def test_candidate_covers_negative_behavior_and_real_tool_contracts(self):
        report = validate_dataset(self.dataset)
        self.assertEqual(report["total_cases"], 40)
        self.assertEqual(report["families"], {"policy": 32, "risk_support": 8})
        self.assertEqual(report["behaviors"]["clarify"], 3)
        self.assertEqual(report["behaviors"]["abstain"], 3)
        self.assertEqual(report["policy_evidence_anchors_checked"], 59)
        self.assertEqual(report["cases_requiring_report"], 6)
        self.assertEqual(report["expected_tool_jobs_per_arm"], 16)

    def test_low_support_does_not_expect_report_or_alert(self):
        row = next(c for c in self.dataset["cases"] if c["id"] == "TS1-S001")
        self.assertEqual(row["acceptance"]["intent"], "RISK")
        self.assertEqual(row["acceptance"]["effects"]["report_count"], 0)
        self.assertEqual(row["acceptance"]["effects"]["tool_jobs"], [])

    def test_low_risk_report_and_medium_alert_are_invalid(self):
        for case_id in ("TS1-S001", "TS1-S004"):
            data = copy.deepcopy(self.dataset)
            row = next(c for c in data["cases"] if c["id"] == case_id)
            row["acceptance"]["effects"]["tool_jobs"].append("ALERT_SEND")
            with self.assertRaisesRegex(ValueError, "Risk/effect"):
                validate_dataset(data)

    def test_gold_and_expected_tools_never_enter_candidate_input(self):
        for case in self.dataset["cases"]:
            inputs = candidate_input(case)
            self.assertEqual(set(inputs), {"question", "history", "initial_safety_state"})
            inputs["history"].append({"role": "user", "content": "local modification"})
            self.assertNotEqual(inputs["history"], case["input"]["history"])

    def test_duplicate_task_and_missing_success_check_are_rejected(self):
        data = copy.deepcopy(self.dataset)
        data["cases"].append(copy.deepcopy(data["cases"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_dataset(data)
        data = copy.deepcopy(self.dataset)
        data["cases"][0]["acceptance"]["required_checks"].remove("no_substantive_errors")
        with self.assertRaisesRegex(ValueError, "required checks"):
            validate_dataset(data)

    def test_stale_source_and_invalid_quote_are_rejected(self):
        data = copy.deepcopy(self.dataset)
        data["cases"][0]["provenance"]["file_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Source changed"):
            validate_dataset(data)
        data = copy.deepcopy(self.dataset)
        data["cases"][0]["acceptance"]["reference"]["policy_evidence"][0]["quote"] = "nonexistent evidence"
        with self.assertRaisesRegex(ValueError, "quote missing"):
            validate_dataset(data)

    def test_reused_question_and_case_fingerprint_must_match_source(self):
        data = copy.deepcopy(self.dataset)
        data["cases"][0]["input"]["question"] = "silently changed question"
        with self.assertRaisesRegex(ValueError, "Input differs"):
            validate_dataset(data)
        data = copy.deepcopy(self.dataset)
        data["cases"][0]["provenance"]["case_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "case fingerprint"):
            validate_dataset(data)


if __name__ == "__main__":
    unittest.main()
