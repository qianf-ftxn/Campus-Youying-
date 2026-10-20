"""意图规则只提前处理明确请求；语义兜底须保留三个合法标签。"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.agents.autonomous import UnderstandingAgent
from app.agents.events import AgentTask, CollaborationBlackboard
from app.core.enums import IntentType
from app.schemas.dtos import AiMessage
from app.services.ai import has_consult_signal, has_high_risk_signal
from app.services.intent_rules import classify_intent_rules, is_policy_followup
from app.services.memory_context import MemorySnapshot


class IntentRuleTests(unittest.TestCase):
    def test_clear_boundaries(self):
        cases = {
            "什么是奖学金": IntentType.CHAT,
            "西电奖学金申请条件是什么": IntentType.CONSULT,
            "帮我写一封请假邮件": IntentType.CHAT,
            "按学校请假规定帮我写邮件": IntentType.CONSULT,
            "Java课程补考怎么申请": IntentType.CONSULT,
            "Java课程补考如何办理": IntentType.CONSULT,
            "解释Java课程补考的办理流程": IntentType.CONSULT,
            "帮我写邮件，学校餐补多少": IntentType.CONSULT,
            "开发报销系统，学校报销的标准是什么": IntentType.CONSULT,
            "报销系统的代码怎么写": IntentType.CHAT,
            "报销完了好开心": IntentType.CHAT,
            "报销到账了，下次报销需要哪些材料？": IntentType.CONSULT,
            "压力测试怎么做": IntentType.CHAT,
            "程序崩溃怎么修复": IntentType.CHAT,
            "什么是抑郁": IntentType.CHAT,
            "帮我写一篇关于焦虑的论文": IntentType.CHAT,
            "我想了解抑郁，帮我写一篇科普文章": IntentType.CHAT,
            "最近压力很大，想找人聊聊": IntentType.RISK,
            "我最近过的不好": IntentType.RISK,
            "我很焦虑，帮我写一封信": IntentType.RISK,
            "你好": IntentType.CHAT,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(classify_intent_rules(text, []).intent, expected)

    def test_uncertain_is_not_a_fourth_business_intent(self):
        for text in ["电吹风能带回寝室不", "门禁到啥时候才开", "我想问点事", "论文", "压力", "如何",
                     "帮我写一封邮件，寝室能带电吹风不"]:
            with self.subTest(text=text):
                self.assertIsNone(classify_intent_rules(text, []).intent)
        self.assertEqual({v.value for v in IntentType}, {"CHAT", "CONSULT", "RISK"})

    def test_followups_require_continuous_policy_context(self):
        policy = "差旅报销需要哪些材料？"
        self.assertTrue(is_policy_followup("我是研究生", [policy]))
        self.assertTrue(is_policy_followup("那住宿呢？", [policy, "我是研究生"]))
        self.assertFalse(is_policy_followup("那住宿呢？", [policy, "给我讲个笑话"]))
        self.assertFalse(is_policy_followup("那住宿呢？", [policy, "今天好开心，分享一下"]))
        self.assertFalse(is_policy_followup("那怎么办？", [policy, "最近很难过"]))
        self.assertFalse(is_policy_followup("那住宿呢？", [], summary=policy))
        self.assertFalse(is_policy_followup("那天气呢？", [policy]))

    def test_emotion_terms_quotes_and_negations(self):
        for text in ["压力测试怎么做", "程序崩溃", "什么是焦虑", "我没有焦虑", "我想了解抑郁",
                     '帮我翻译“我很焦虑”', "我在写关于焦虑的论文"]:
            with self.subTest(text=text):
                self.assertFalse(has_consult_signal(text))
        self.assertTrue(has_consult_signal('帮我翻译“我很焦虑”，但我最近也很难过'))
        self.assertFalse(has_high_risk_signal("什么是自杀"))
        self.assertFalse(has_high_risk_signal('小说引用“我想自杀”'))
        self.assertFalse(has_high_risk_signal("我没有自杀的想法"))
        self.assertTrue(has_high_risk_signal('小说引用“我想自杀”，但是我现在也不想活了'))


class ModelFallbackTests(unittest.TestCase):
    def make_agent(self, result="CONSULT"):
        client = Mock()
        client.complete.return_value = result
        agent = UnderstandingAgent(SimpleNamespace())
        agent.client = lambda: client
        agent.private_memory = lambda: []
        return agent, client

    def test_all_three_semantic_labels_are_accepted(self):
        for label in IntentType:
            for text in ["电吹风能带回寝室不", "门禁到啥时候才开", "我想问点事"]:
                with self.subTest(label=label, text=text):
                    agent, client = self.make_agent(" " + label.value.lower() + "\n")
                    board = CollaborationBlackboard("test", user_input=text)
                    self.assertEqual(agent._classify(text, board), label)
                    client.complete.assert_called_once()
                    self.assertEqual(agent.routing_source, "model")
                    self.assertEqual(agent.routing_model_label, label.value)

    def test_uncertain_followup_with_old_summary_is_left_to_model(self):
        agent, client = self.make_agent("CHAT")
        board = CollaborationBlackboard("test", user_input="那住宿呢？", memory_snapshot=MemorySnapshot(
            summary_text="之前咨询过报销政策", recent_messages=(AiMessage(role="user", content="讲个笑话"),)))
        self.assertEqual(agent._classify(board.user_input, board), IntentType.CHAT)
        client.complete.assert_called_once()
        prompt = "\n".join(m.content for m in client.complete.call_args.args[0])
        self.assertIn("讲个笑话", prompt)
        self.assertIn("之前咨询过报销政策", prompt)

    def test_invalid_or_failed_model_is_a_distinct_fallback(self):
        for invalid in ["", "UNKNOWN", "应当是 CHAT 而不是 CONSULT", "```CONSULT```", "CHAT CONSULT",
                        '{"intent":"UNKNOWN"}', '{"topic":"CONSULT"}', '["CONSULT"]', '{"intent":["CONSULT"]}',
                        '说明：{"intent":"CONSULT"}']:
            with self.subTest(invalid=invalid):
                agent, client = self.make_agent(invalid)
                self.assertEqual(agent._classify("电吹风能带回寝室不", CollaborationBlackboard("test")), IntentType.CHAT)
                client.complete.assert_called_once()
                self.assertEqual(agent.routing_source, "fallback")
                self.assertEqual(agent.routing_fallback_reason, "invalid_model_label")
        agent, client = self.make_agent()
        client.complete.side_effect = RuntimeError("unavailable")
        self.assertEqual(agent._classify("门禁到啥时候才开", CollaborationBlackboard("test")), IntentType.CHAT)
        self.assertEqual(agent.routing_fallback_reason, "model_error:RuntimeError")

    def test_structured_model_intent_is_not_discarded(self):
        for label in IntentType:
            with self.subTest(label=label):
                agent, client = self.make_agent('{"intent":"' + label.value + '","reason":"语义识别结果"}')
                self.assertEqual(agent._classify("门禁到啥时候才开", CollaborationBlackboard("test")), label)
                client.complete.assert_called_once()
                self.assertEqual(agent.routing_model_label, label.value)

    def test_clear_rules_bypass_model(self):
        for text, expected in [("你好", IntentType.CHAT), ("Java课程补考怎么申请", IntentType.CONSULT),
                               ("我最近过得不好，帮我写一封邮件", IntentType.RISK), ("我想自杀", IntentType.RISK)]:
            with self.subTest(text=text):
                agent, client = self.make_agent("CONSULT")
                self.assertEqual(agent._classify(text, CollaborationBlackboard("test")), expected)
                client.complete.assert_not_called()
                self.assertEqual(agent.routing_source, "rule")

    def test_trace_records_actual_source_and_label(self):
        agent, _ = self.make_agent("CONSULT")
        agent.services = SimpleNamespace(private_memory=SimpleNamespace(_key=lambda *args: "test"),
                                         session=SimpleNamespace(public_id="isolated"))
        agent.remember = lambda content: None
        result = agent.act(AgentTask("task", "understand"), CollaborationBlackboard("test", user_input="门禁到啥时候才开"))
        payload = result.artifacts[0].payload
        self.assertEqual(payload["intent"], "CONSULT")
        self.assertEqual(payload["routingSource"], "model")
        self.assertEqual(payload["modelLabel"], "CONSULT")
        self.assertTrue(payload["requiresPolicyRetrieval"])
