import json
import unittest
from pathlib import Path

from app.core.enums import IntentType, RiskLevel
from app.policy_eval.expand import EXPANDED_DATASET
from app.policy_eval.rigorous import FROZEN_PROTOCOL, HOLDOUT, REVIEWED_DEV
from app.policy_eval.runner import compact, digest, load_inputs
from app.services.ai import PromptTemplates


class ProspectiveDatasetTests(unittest.TestCase):
    def test_gold_and_negative_reference_sources_do_not_overlap_development_gold(self):
        old, _ = load_inputs(EXPANDED_DATASET)
        new, _ = load_inputs(HOLDOUT)
        old_ids = {g['document_id'] for c in old['cases'] for g in c['evidence']}
        new_ids = {g['document_id'] for c in new['cases'] for g in c['evidence']}
        new_ids |= {i for c in new['cases'] for i in c['checked_source_ids']}
        self.assertFalse(old_ids & new_ids)
        self.assertEqual(new['frozen_protocol_sha256'], digest(FROZEN_PROTOCOL.read_bytes()))
        self.assertEqual(len(new['cases']), 60)
        self.assertEqual(sum(c['answerable'] for c in new['cases']), 48)
        self.assertEqual(len({compact(c['question']) for c in new['cases']}), 60)

    def test_ai_source_audit_never_implies_human_approval(self):
        reviewed, _ = load_inputs(REVIEWED_DEV)
        self.assertTrue(all(c['review_status'] in {'pending_human', 'approved', 'needs_revision'} for c in reviewed['cases']))
        d13 = next(c for c in reviewed['cases'] if c['id'] == 'D13')
        self.assertEqual(d13['required_point_indices'], [1, 2])
        self.assertEqual(d13['optional_point_indices'], [3])
        original = json.loads(EXPANDED_DATASET.read_text())
        self.assertEqual(len(next(c for c in original['cases'] if c['id'] == 'D13')['answer_points']), 3)

    def test_candidate_prompt_is_opt_in_and_baseline_prompt_is_preserved(self):
        baseline = PromptTemplates.answer_system_prompt(IntentType.CONSULT, RiskLevel.LOW, '__CONTEXT__', '评测学生').content
        precise = PromptTemplates.answer_system_prompt(IntentType.CONSULT, RiskLevel.LOW, '__CONTEXT__', '评测学生', policy_evidence_precision=True).content
        saved = Path('app/policy_eval/legacy_consult_prompt.txt').read_text()
        self.assertEqual(baseline, saved)
        self.assertNotEqual(precise, saved)
        frozen = json.loads(FROZEN_PROTOCOL.read_text())['protocol']
        self.assertEqual(frozen['improved_prompt_sha256'], digest(precise.encode()))


if __name__ == '__main__':
    unittest.main()
