"""Integrity checks for the expanded, still unapproved development annotations."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from app.policy_eval.expand import EXPANDED_DATASET, build_expanded_dataset
from app.policy_eval.pilot import DATASET
from app.policy_eval.runner import digest, load_inputs, validate_dataset, write_review


class PolicyExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset, cls.documents = load_inputs(EXPANDED_DATASET)

    def test_original_annotations_and_provenance_are_preserved(self):
        original = json.loads(DATASET.read_text())
        self.assertEqual(self.dataset["base_dataset_sha256"], digest(DATASET.read_bytes()))
        base_ids = set(self.dataset["base_case_ids"])
        self.assertEqual([c for c in self.dataset["cases"] if c["id"] in base_ids], original["cases"])
        added = [c for c in self.dataset["cases"] if c["id"] in self.dataset["added_case_ids"]]
        self.assertEqual(len(added), 50)
        self.assertEqual(len({c["id"] for c in self.dataset["cases"]}), 75)
        self.assertTrue(all(c["review_status"] in {"pending_human", "approved", "needs_revision"} for c in added))
        # All source offsets, hashes and versions were checked by load_inputs.
        self.assertEqual(len({g["document_id"] for c in added for g in c["evidence"]}), 15)

    def test_authoring_spec_stays_grounded_without_overwriting_human_reviews(self):
        initial = build_expanded_dataset()
        # The authoring spec must remain valid, but humans can revise JSON labels.
        # Do not require their verdicts or revised answer points to equal AI drafts.
        self.assertEqual(initial["added_case_ids"], self.dataset["added_case_ids"])
        added_ids = set(initial["added_case_ids"])
        self.assertTrue(all(c["review_status"] == "pending_human" for c in initial["cases"] if c["id"] in added_ids))

    def test_answer_and_clarification_labels_cannot_contradict_the_evidence(self):
        for behavior, answerable in [("clarify", True), ("answer", False), ("unknown", True)]:
            broken = copy.deepcopy(self.dataset)
            case = next(c for c in broken["cases"] if c["answerable"] == answerable)
            case["expected_behavior"] = behavior
            with self.assertRaises(ValueError):
                validate_dataset(broken, self.documents, broken["manifest_sha256"])

    def test_negative_reference_to_an_ineligible_source_is_rejected(self):
        broken = copy.deepcopy(self.dataset)
        next(c for c in broken["cases"] if not c["answerable"])["checked_source_ids"] = ["missing"]
        with self.assertRaises(ValueError):
            validate_dataset(broken, self.documents, broken["manifest_sha256"])

    def test_review_sheet_points_to_the_dataset_actually_being_reviewed(self):
        with tempfile.TemporaryDirectory() as directory:
            write_review(self.dataset, Path(directory), EXPANDED_DATASET)
            sheet = (Path(directory) / "annotation-review.md").read_text()
            self.assertIn(str(EXPANDED_DATASET), sheet)
            self.assertIn("共 75 题", sheet)
            self.assertIn("预期行为：clarify", sheet)
            self.assertIn("U15", sheet)


if __name__ == "__main__":
    unittest.main()
