import asyncio
import json
import threading
from unittest.mock import patch, AsyncMock

from test_campus_youying import CampusBase
from app.agents.harness import MindBridgeAgentHarness
from app.core.enums import IntentType, RiskLevel, EmotionLabel
from app.models.entities import UserAccount, ChatMessage, ChatSession, AgentRunTrace, PsychologicalReport
from app.schemas.dtos import ChatRequest
from app.services.ai import AiClient
from app.services.chat import ChatService
from app.services.intent_rules import is_policy_request
from app.services.streaming import collect_guarded_stream


def event_data(chunk):
    for line in chunk.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    return {}


class RoutingBoundaryTests(CampusBase):
    def setUp(self):
        super().setUp()
        self.user = UserAccount(username="routing", display_name="学生", password_hash="x")
        self.db.add(self.user); self.db.commit()

    def turn(self, question, session=None):
        harness = MindBridgeAgentHarness(self.db, self.settings)
        out = harness.run(self.user, ChatRequest(message=question, sessionId=session))
        harness.save_assistant_message(self.user, out.session, out.response_text, out.current_message_id)
        return out

    def test_generic_questions_do_not_call_rag_even_if_classifier_says_consult(self):
        questions = ["你好", "谢谢", "你是谁", "我想咨询一下", "报销系统的代码怎么写", "帮我写一封请假邮件",
            "今天终于交完报销材料了，想分享一下", "预约理发有什么建议", "什么是机器学习", "怎么准备奖学金面试", "奖学金是什么"]
        for question in questions:
            with self.subTest(question=question), patch("app.services.knowledge.KnowledgeService.retrieve") as retrieve:
                out = self.turn(question)
                self.assertEqual(out.intent, IntentType.CHAT)
                retrieve.assert_not_called()

    def test_concrete_campus_facts_require_documents(self):
        for question in ["研究生差旅报销标准是什么", "学校心理中心怎么预约", "奖学金申请需要哪些材料",
                         "本校休学的办理流程", "宿舍的收费标准是多少", "研究生毕业的学分要求是什么",
                         "食堂几点关门", "校园卡挂失的流程是什么", "校医院的预约方式是什么",
                         "研究生转导师需要什么条件", "学位授予复核的申请期限是什么", "西电办理复核需要哪些材料",
                         "研究生在校外连续居住超过多少天需要办理校外住宿手续？"]:
            with self.subTest(question=question):
                self.assertTrue(is_policy_request(question))

    def test_topic_switch_after_policy_does_not_stick_to_consult(self):
        first = self.turn("研究生报销的伙食补助标准是什么？")
        for question in ["谢谢你", "那给我讲个笑话吧", "这个代码怎么写", "今天天气怎么样"]:
            with self.subTest(question=question), patch("app.services.knowledge.KnowledgeService.retrieve") as retrieve:
                out = self.turn(question, first.session.public_id)
                self.assertEqual(out.intent, IntentType.CHAT)
                retrieve.assert_not_called()

    def test_real_followup_and_identity_stay_with_the_policy_topic(self):
        first = self.turn("差旅报销需要哪些材料？")
        for question in ["我是研究生", "那住宿呢？"]:
            out = self.turn(question, first.session.public_id)
            self.assertEqual(out.intent, IntentType.CONSULT)

    def test_unstructured_classifier_output_cannot_route_to_consult(self):
        original = AiClient._mock
        def mock(client, messages):
            if any("意图分类器" in m.content for m in messages):
                return "应当是 CHAT 而不是 CONSULT"
            return original(client, messages)
        with patch.object(AiClient, "_mock", mock), patch("app.services.knowledge.KnowledgeService.retrieve") as retrieve:
            self.assertEqual(self.turn("你好啊").intent, IntentType.CHAT)
            retrieve.assert_not_called()


class ActualStreamingTests(CampusBase):
    def setUp(self):
        super().setUp()
        self.user = UserAccount(username="stream", display_name="学生", password_hash="x")
        self.db.add(self.user); self.db.commit()

    def test_first_delta_arrives_before_model_finishes_and_saved_text_matches(self):
        gate = threading.Event()
        finished = threading.Event()
        calls = []
        async def stream(client, messages):
            calls.append(1)
            yield "这是模型尚未生成完毕时发出的第一段普通回答，内容足够长。"
            while not gate.is_set():
                await asyncio.sleep(.01)
            yield "接着是第二段。"
            finished.set()
        service = ChatService(self.db, self.settings)
        async def check():
            events = []
            try:
                async for raw in service.stream_chat(self.user, ChatRequest(message="你好，聊聊今天")):
                    event = event_data(raw); events.append(event)
                    if event.get("type") == "token" and not gate.is_set():
                        self.assertFalse(finished.is_set())
                        gate.set()
            finally:
                gate.set()
            return events
        with patch.object(AiClient, "stream", stream):
            events = asyncio.run(asyncio.wait_for(check(), timeout=10))
        self.assertEqual(calls, [1])
        self.assertEqual(events[0]["type"], "status")
        self.assertTrue(any(e.get("type") == "done" for e in events))
        text = "".join(e["content"] for e in events if e.get("type") == "token")
        reply = self.db.query(ChatMessage).filter_by(role="ASSISTANT").one()
        self.assertEqual(reply.content, text)
        self.assertEqual(reply.completion_status, "COMPLETE")
        self.assertIsNotNone(reply.agent_trace_id)
        self.assertFalse(any(e.get("type") == "sources" for e in events))

    def test_internal_label_split_across_chunks_never_reaches_client(self):
        async def chunks():
            for token in ["普通回答 " * 10, "effective", "Risk", " 后续内容"]:
                yield token
        sent = []
        result = asyncio.run(collect_guarded_stream(type("Client", (), {"stream": lambda self, messages: chunks()})(), [], sent.append, "失败"))
        self.assertNotIn("effective", "".join(sent))
        self.assertEqual("".join(sent), result.text)
        self.assertTrue(result.error)

    def test_midstream_failure_keeps_delivered_prefix_without_second_generation(self):
        async def chunks():
            yield "已经开始输出的普通回答内容。" * 4
            raise RuntimeError("provider disconnected")
        sent = []
        result = asyncio.run(collect_guarded_stream(type("Client", (), {"stream": lambda self, messages: chunks()})(), [], sent.append, "失败"))
        self.assertTrue(result.text.startswith("已经开始输出"))
        self.assertIn("回答暂时中断", result.text)
        self.assertEqual("".join(sent), result.text)

    def test_disconnect_saves_only_delivered_prefix_and_admin_trace(self):
        closed = threading.Event()
        async def stream(client, messages):
            try:
                yield "客户端断开前已经看到的回答内容。" * 4
                await asyncio.Event().wait()  # 模型暂时不再产出 token，也必须能主动取消。
            finally:
                closed.set()
        service = ChatService(self.db, self.settings)
        async def check():
            generator = service.stream_chat(self.user, ChatRequest(message="你好，聊一聊"))
            prefix = ""
            try:
                async for raw in generator:
                    event = event_data(raw)
                    if event.get("type") == "token":
                        prefix += event["content"]
                        break
            finally:
                await generator.aclose()
                await asyncio.to_thread(service._worker_thread.join, 5)
            return prefix
        with patch.object(AiClient, "stream", stream):
            text = asyncio.run(check())
        self.db.expire_all()
        self.assertTrue(closed.is_set())
        reply = self.db.query(ChatMessage).filter_by(role="ASSISTANT").one()
        self.assertEqual(reply.completion_status, "INTERRUPTED")
        self.assertEqual(reply.content, text)
        trace = self.db.get(AgentRunTrace, reply.agent_trace_id)
        self.assertIn("interrupted_candidate", trace.response_messages_json)

    def assert_model_streams_before_completion(self, question, expected_mode, session_id=None):
        gate, finished = threading.Event(), threading.Event()
        async def stream(client, messages):
            yield "模型正文开始，这是在模型尚未完成时返回的支持回答。" * 3
            while not gate.is_set():
                await asyncio.sleep(.01)
            yield "请继续告诉我你的近况。"
            finished.set()
        async def collect():
            events = []
            try:
                async for raw in ChatService(self.db, self.settings).stream_chat(
                        self.user, ChatRequest(message=question, sessionId=session_id)):
                    event = event_data(raw); events.append(event)
                    if event.get("type") == "token" and "模型正文开始" in event.get("content", ""):
                        self.assertFalse(finished.is_set())
                        gate.set()
            finally:
                gate.set()
            return events
        with patch.object(AiClient, "stream", stream), patch.object(MindBridgeAgentHarness, "dispatch_tools", new=AsyncMock(return_value=[])):
            events = asyncio.run(asyncio.wait_for(collect(), 10))
        self.assertTrue(gate.is_set())
        self.assertTrue(any(e.get("type") == "done" for e in events))
        self.db.expire_all()
        trace = self.db.query(AgentRunTrace).order_by(AgentRunTrace.id.desc()).first()
        self.assertEqual(trace.response_mode, expected_mode)
        text = "".join(e.get("content", "") for e in events if e.get("type") == "token")
        reply = self.db.query(ChatMessage).filter_by(role="ASSISTANT").order_by(ChatMessage.id.desc()).first()
        self.assertEqual(reply.content, text)
        self.assertEqual(reply.completion_status, "COMPLETE")
        return text

    def test_low_emotional_support_streams_before_model_finishes(self):
        self.assert_model_streams_before_completion("最近心情不好，有点难过", "support")

    def test_medium_support_streams_before_model_finishes(self):
        from app.services.assessment import PsychologyAssessment, PsychologicalAssessmentService
        assessment = PsychologyAssessment(EmotionLabel.ANXIETY, 3, RiskLevel.MEDIUM, .9, "测试中度风险")
        with patch.object(PsychologicalAssessmentService, "assess", return_value=assessment):
            self.assert_model_streams_before_completion("最近心情不好，很难受", "support")

    def test_high_risk_streams_guidance_then_model_before_completion(self):
        text = self.assert_model_streams_before_completion("我想自杀，已经准备好了。", "safety_support")
        self.assertTrue(text.startswith("我很在意你现在的安全"))
        self.assertIn("可信任", text)
        self.assertEqual(self.db.query(PsychologicalReport).count(), 1)

    def test_monitoring_streams_before_model_finishes(self):
        from app.services.safety_memory import SafetyMemoryService
        session = ChatSession(public_id="monitor-stream", user_id=self.user.id, title="监测测试")
        self.db.add(session); self.db.commit()
        SafetyMemoryService(self.db, self.settings).manual_override(
            self.user.id, "admin", "MONITORING", "隔离测试", session.id)
        text = self.assert_model_streams_before_completion("你好，聊一聊", "monitored_chat", session.public_id)
        self.assertTrue(text.startswith("先关心一下，你最近还好吗"))

    def test_consult_streams_model_before_completion_with_evidence(self):
        self.add_policy()
        self.assert_model_streams_before_completion("研究生出差伙食补助标准是什么", "policy_answer")

    def test_high_risk_stream_redacts_unverified_phone_before_release(self):
        async def chunks():
            for token in ["请联系可信任的人。" * 4, "测试电话400-", "000-", "0000。", "已核实电话010-", "12345678。结束。"]:
                yield token
        sent = []
        result = asyncio.run(collect_guarded_stream(
            type("Client", (), {"stream": lambda self, messages: chunks()})(), [], sent.append, "失败",
            verified_phones={"01012345678"}))
        self.assertNotIn("400", "".join(sent))
        self.assertIn("[未核实号码已省略]", result.text)
        self.assertIn("010-12345678", result.text)
        self.assertEqual("".join(sent), result.text)

    def test_high_risk_disconnect_preserves_risk_report_and_tool_plan(self):
        closed = threading.Event()
        async def stream(client, messages):
            try:
                yield "已开始生成的支持内容。" * 6
                await asyncio.Event().wait()
            finally:
                closed.set()
        service = ChatService(self.db, self.settings)
        async def check():
            generator = service.stream_chat(self.user, ChatRequest(message="我想自杀，已经准备好了。"))
            text = ""
            try:
                async for raw in generator:
                    event = event_data(raw)
                    if event.get("type") == "token":
                        text += event["content"]
                        if "已开始生成" in text:
                            break
            finally:
                await generator.aclose()
                await asyncio.to_thread(service._worker_thread.join, 5)
            return text
        dispatch = AsyncMock(return_value=[])
        with patch.object(AiClient, "stream", stream), patch.object(MindBridgeAgentHarness, "dispatch_tools", new=dispatch):
            text = asyncio.run(asyncio.wait_for(check(), 10))
        self.assertTrue(closed.is_set())
        self.db.expire_all()
        reply = self.db.query(ChatMessage).filter_by(role="ASSISTANT").one()
        trace = self.db.get(AgentRunTrace, reply.agent_trace_id)
        report = self.db.query(PsychologicalReport).one()
        self.assertEqual(reply.content, text)
        self.assertEqual(reply.completion_status, "INTERRUPTED")
        self.assertEqual(trace.effective_risk, "HIGH")
        self.assertEqual(trace.risk_level, "HIGH")
        self.assertEqual(report.risk_level, "HIGH")
        dispatch.assert_awaited_once()
        self.assertEqual(dispatch.call_args.args[0].report_id, report.id)
        self.assertEqual(dispatch.call_args.args[0].risk_level, "HIGH")
