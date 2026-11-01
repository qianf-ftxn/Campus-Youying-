"""Deterministic integration checks; scripted text is never an effect score."""
import copy
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from app.core.config import Settings
from app.schemas.dtos import AiMessage
from app.services.knowledge import SearchResult
from app.services.ai import AiClient
from app.task_eval.execution import (
    BudgetExceeded, Completion, Limits, ModelSession, context_text,
    machine_checks, parse_single, prepare_shared_input, retrieve_once, run_arm, run_pair,
)
from app.task_eval.prepare import build_dataset


class FixedRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, top_k, **kwargs):
        self.queries.append((query, top_k, kwargs))
        return [SearchResult(17, "test-only.md", "离校须请假；本段仅用于链路测试。", 1.,
                             title="测试资料", version="test-v1", domain="policy", heading="测试条款")]


class ScriptedTransport:
    def __init__(self, risk="LOW", answer=None):
        self.risk = risk
        self.answer = answer or ("我很在意你的安全，请联系身边可信任的人陪伴。" if risk == "HIGH" else "请先核对适用条件并办理请假。")
        self.calls = []
        self.single_count = 0

    def __call__(self, messages, settings, stage, timeout):
        self.calls.append((stage, copy.deepcopy(messages), settings.openai_model, settings.ai_temperature, settings.ai_max_tokens))
        if stage == "SingleAgent":
            self.single_count += 1
            text = json.dumps({"intent": "RISK" if self.risk != "LOW" else "CONSULT", "risk": self.risk,
                               "answer": self.answer, "final": self.single_count > 1}, ensure_ascii=False)
        elif stage == "SafetyAgent":
            text = json.dumps({"emotion": {"LOW": "NORMAL", "MEDIUM": "DEPRESSED", "HIGH": "HIGH_RISK"}[self.risk],
                               "emotionScore": {"LOW": 0., "MEDIUM": 3., "HIGH": 4.}[self.risk],
                               "risk": self.risk, "confidence": .95, "summary": "scripted test assessment"})
        elif stage == "UnderstandingAgent":
            text = "RISK" if self.risk != "LOW" else "CONSULT"
        elif stage == "ResponseAgent":
            text = self.answer
        else:
            raise AssertionError("Unexpected model stage " + stage)
        return Completion(text)


class TaskExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = build_dataset()

    def setUp(self):
        self.settings = Settings(_env_file=None, ai_provider="mock", openai_model="pinned-fixture",
                                 knowledge_vector_enabled=False, openai_api_key="")
        self.network = patch("httpx.post", side_effect=AssertionError("No live API in integration tests"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def case(self, original):
        return copy.deepcopy(next(c for c in self.dataset["cases"] if c["id"] == "TS1-" + original))

    def test_retrieve_once_pair_uses_same_input_evidence_skills_and_budget(self):
        retriever, transport = FixedRetriever(), ScriptedTransport()
        result = run_pair(self.case("C4-X01"), retriever, self.settings, transport=transport)
        self.assertEqual(len(retriever.queries), 1)
        a, b = result["observations"]["single"], result["observations"]["multi"]
        for key in ("input_sha256", "evidence_sha256", "skills_sha256", "limits"):
            self.assertEqual(a[key], b[key])
        self.assertTrue(a["response_completed"], a)
        self.assertTrue(b["response_completed"], b)
        self.assertEqual(a["effects"]["report_count"], 0)
        self.assertEqual(b["effects"]["jobs"], [])
        self.assertIsNone(result["task_success"])
        self.assertEqual(result["semantic_checks"], "pending")

    def test_multi_uses_accepted_text_without_a_second_generation(self):
        transport = ScriptedTransport(answer="只允许生成一次的审核候选。")
        rows = FixedRetriever().retrieve("", 5)
        result = run_arm("multi", prepare_shared_input(self.case("C4-X01")), rows,
                         self.settings, transport=transport, skills="SHARED-SKILL-MARKER")
        self.assertEqual(result.get("answer"), transport.answer)
        response_calls = [c for c in transport.calls if c[0] == "ResponseAgent"]
        self.assertEqual(len(response_calls), 1)
        self.assertTrue(result["details"]["final_accepted"])
        prompt = "\n".join(m.content for m in response_calls[0][1])
        self.assertIn(context_text(rows), prompt)
        self.assertIn("SHARED-SKILL-MARKER", prompt)

    def test_single_has_self_review_and_retains_raw_decisions(self):
        transport = ScriptedTransport()
        result = run_arm("single", prepare_shared_input(self.case("C4-X01")), [],
                         self.settings, transport=transport, skills="")
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(len(result["details"]["raw_decisions"]), 2)
        self.assertIn("自检", transport.calls[1][1][-1].content)

    def test_medium_high_real_tools_run_in_sandbox_for_both_arms(self):
        for source, risk, jobs in (("S004", "MEDIUM", ["EXCEL_REPORT", "CASE_CREATE"]),
                                   ("C001", "HIGH", ["EXCEL_REPORT", "CASE_CREATE", "ALERT_SEND"])):
            for arm in ("single", "multi"):
                with self.subTest(arm=arm, risk=risk):
                    result = run_arm(arm, prepare_shared_input(self.case(source)), [], self.settings,
                                     transport=ScriptedTransport(risk), skills="")
                    self.assertTrue(result["response_completed"], result)
                    self.assertEqual(result["effects"]["report_count"], 1)
                    self.assertEqual([j["kind"] for j in result["effects"]["jobs"]], jobs)
                    self.assertTrue(all(j["status"] == "SUCCESS" for j in result["effects"]["jobs"]))
                    self.assertEqual(result["effects"]["artifact_counts"],
                                     {"excel": 1, "case": 1, "alert": int(risk == "HIGH")})
                    self.assertTrue(all(machine_checks(self.case(source), result).values()), result)

    def test_candidate_never_receives_gold_points_or_expected_tools(self):
        case = self.case("C4-X01")
        case["acceptance"]["required_answer_points"] = ["SECRET-GOLD-POINT"]
        transport = ScriptedTransport()
        run_pair(case, FixedRetriever(), self.settings, transport=transport)
        for _, messages, *_ in transport.calls:
            self.assertNotIn("SECRET-GOLD-POINT", "\n".join(m.content for m in messages))

    def test_single_format_error_can_be_repaired_without_gold_feedback(self):
        transport = ScriptedTransport()
        original = transport.__call__
        calls = []

        def malformed_once(messages, settings, stage, timeout):
            calls.append(stage)
            if len(calls) == 1:
                return Completion("bad JSON")
            return original(messages, settings, stage, timeout)

        result = run_arm("single", prepare_shared_input(self.case("C4-X01")), [], self.settings,
                         transport=malformed_once, skills="")
        self.assertTrue(result["response_completed"], result)
        self.assertEqual(result["model_calls"], 3)

    def test_timeout_or_truncated_output_is_not_a_successful_fallback(self):
        for arm in ("single", "multi"):
            for fail in ("timeout", "length"):
                def broken(*args):
                    if fail == "timeout":
                        raise TimeoutError("fixture")
                    return Completion("partial text", finish_reason="length")
                result = run_arm(arm, prepare_shared_input(self.case("C4-X01")), [], self.settings,
                                 transport=broken, skills="")
                self.assertFalse(result["response_completed"], result)
                self.assertTrue(result["model_errors"])

    def test_shared_call_cap_applies_to_single_and_multi(self):
        for arm in ("single", "multi"):
            result = run_arm(arm, prepare_shared_input(self.case("C4-X01")), FixedRetriever().retrieve("", 5),
                             self.settings, transport=ScriptedTransport(), limits=Limits(max_calls=1), skills="")
            self.assertLessEqual(result["model_calls"], 1)
            self.assertFalse(result["response_completed"], result)

    def test_provider_usage_is_recorded_and_exact_budget_enforced(self):
        def transport(*args):
            return Completion("answer", {"prompt_tokens": 9999, "completion_tokens": 2})
        session = ModelSession(self.settings, Limits(max_input_tokens=1000), transport)
        with self.assertRaises(BudgetExceeded):
            session.complete([AiMessage(role="user", content="test")], "test")
        self.assertEqual(session.calls[0]["usage_kind"], "provider")
        self.assertEqual(session.calls[0]["input_tokens"], 9999)

    def test_tool_failure_is_observed_without_counting_a_queued_job_as_success(self):
        with patch("app.services.tools.ToolOrchestrationService.write_excel", side_effect=RuntimeError("fixture failure")):
            result = run_arm("single", prepare_shared_input(self.case("S004")), [], self.settings,
                             transport=ScriptedTransport("MEDIUM"), skills="")
        self.assertTrue(result["response_completed"])
        self.assertFalse(result["effect_execution_completed"])
        self.assertFalse(machine_checks(self.case("S004"), result)["tool_jobs_success"])
        self.assertEqual(result["effects"]["jobs"][0]["status"], "PENDING")

    def test_legacy_entry_does_not_make_an_extra_unreviewed_answer_call(self):
        from app.agent_eval.runner import _evaluate_case, _isolated_runtime
        from app.core.enums import IntentType, RiskLevel
        from app.evaluation.dataset import ResponseEvalCase

        self.settings.long_term_memory_enabled = False
        self.settings.redis_socket_timeout_seconds = .001
        original, calls = AiClient.complete, []

        def tracked(client, messages):
            calls.append(client.agent_name)
            return original(client, messages)

        score = SimpleNamespace(as_dict=lambda: {})
        judge = SimpleNamespace(compare=lambda *args: SimpleNamespace(a=score, b=score, winner="TIE", confidence=1., reason="fixture"))
        case = ResponseEvalCase("legacy-fix", "general", "general", IntentType.CHAT, RiskLevel.LOW,
                                "解释 Python 列表推导式。")
        with patch("app.services.memory.RedisShortTermMemoryStore._connect", return_value=None), \
             patch("app.services.memory_context.RollingSummaryService._connect_redis", return_value=None), \
             patch("app.services.safety_memory.SafetyMemoryService._connect_redis", return_value=None):
            db, user, knowledge, close = _isolated_runtime(self.settings)
            try:
                with patch.object(AiClient, "complete", tracked):
                    result = _evaluate_case(case, db, user, knowledge, self.settings, judge)
            finally:
                close()
        self.assertTrue(result["multiAgent"]["answer"])
        self.assertEqual(calls.count(""), 1)  # only the actual single-agent generation

    def test_custom_role_models_cannot_change_the_shared_candidate(self):
        self.settings.agent_model_safety_model = "different-role-model"
        transport = ScriptedTransport()
        result = run_arm("multi", prepare_shared_input(self.case("C4-X01")), FixedRetriever().retrieve("", 5),
                         self.settings, transport=transport, skills="")
        self.assertTrue(result["response_completed"], result)
        self.assertTrue(transport.calls)
        self.assertTrue(all(c[2] == "pinned-fixture" and c[3] == 0 and c[4] <= 1024 for c in transport.calls))

    def test_missing_effects_does_not_pass_as_no_tools_needed(self):
        checks = machine_checks(self.case("C4-X01"), {"response_completed": True, "answer": "text"})
        self.assertFalse(checks["tool_jobs_exact"])
        self.assertFalse(checks["tool_jobs_success"])

    def test_oversized_parent_is_skipped_and_gold_domain_not_used(self):
        retriever = FixedRetriever()
        self.settings.rag_skill_budget_tokens = 30
        small = retriever.retrieve("", 5)[0]
        large = replace(small, content="长" * 10000)
        with patch.object(retriever, "retrieve", return_value=[large, small]) as retrieve:
            rows = retrieve_once(retriever, prepare_shared_input(self.case("C4-X01")), self.settings)
        self.assertEqual(rows, [small])
        self.assertNotIn("domain", retrieve.call_args.kwargs)

    def test_parser_rejects_missing_fields_and_non_boolean_final(self):
        for raw in ('{"answer":"hi"}', '{"intent":"CONSULT","risk":"LOW","answer":"hi","final":"true"}'):
            with self.assertRaises(ValueError):
                parse_single(raw)

    def test_fenced_json_is_accepted_without_changing_the_decision(self):
        raw = {"intent": "RISK", "risk": "HIGH", "answer": "请联系身边可信任的人。", "final": True}
        self.assertEqual(parse_single("```json\n" + json.dumps(raw) + "\n```"), raw)


if __name__ == "__main__":
    unittest.main()
