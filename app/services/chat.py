from __future__ import annotations

import asyncio
import json
import logging
import threading

from sqlalchemy.orm import Session, sessionmaker

from app.agents.harness import MindBridgeAgentHarness, AgentToolPlan
from app.agents.result import AgentRunResult
from app.core.config import Settings
from app.core.enums import IntentType, ResponseMode, RiskLevel
from app.models.entities import UserAccount, ChatSession
from app.schemas.dtos import ChatRequest
from app.services.ai import split_text
from app.services.streaming import StreamCancelled
from app.services.trace import AgentTraceService
from app.services.privacy import PrivacySanitizer

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings

    async def stream_chat(self, user: UserAccount, request: ChatRequest):
        # 模型与 ORM 都留在同一工作线程/独立 Session，事件循环随到随发 SSE。
        loop = asyncio.get_running_loop()
        events = asyncio.Queue()
        cancelled, delivered_end = threading.Event(), threading.Event()
        delivery = {"text": "", "complete": False}
        session_factory = sessionmaker(bind=self.db.get_bind(), autoflush=False)
        user_id = user.id

        def enqueue(event):
            if not loop.is_closed():
                loop.call_soon_threadsafe(events.put_nowait, event)

        def worker():
            db = session_factory()
            harness, outcome = None, None
            active, plan = {}, {}
            streamed = []
            success = False

            def publish(event):
                if event["type"] == "_response_plan":
                    plan.update(event)
                    if cancelled.is_set():
                        raise StreamCancelled()
                    enqueue({"type": "status", "message": "正在生成回复"})
                    return
                if cancelled.is_set():
                    raise StreamCancelled()
                if event["type"] == "meta":
                    active.update(sessionId=event["sessionId"], messageId=event.get("_messageId"))
                if event["type"] == "token":
                    streamed.append(event["content"])
                enqueue({k: v for k, v in event.items() if not k.startswith("_")})

            publish.is_cancelled = cancelled.is_set
            try:
                account = db.get(UserAccount, user_id)
                if account is None:
                    raise RuntimeError("Account no longer exists")
                harness = MindBridgeAgentHarness(db, self.settings)
                outcome = harness.run(account, request, stream_event=publish)
                if not active:
                    publish({"type": "meta", "sessionId": outcome.session.public_id,
                             "_messageId": outcome.current_message_id})
                if streamed:
                    # 不二次生成、不拼接修订版：学生所见必须是本轮最终采纳正文。
                    if "".join(streamed) != outcome.response_text:
                        raise RuntimeError("Streamed candidate was not the final accepted response")
                else:
                    # 缓存、缺文件固定提示和无模型的兜底文案也使用相同 SSE 通道。
                    for token in split_text(outcome.response_text, 12):
                        publish({"type": "token", "sessionId": outcome.session.public_id, "content": token})
                success = True
            except StreamCancelled:
                pass
            except Exception:
                logger.exception("Chat generation failed user=%s session=%s", user_id, active.get("sessionId"))
                db.rollback()
                enqueue({"type": "error", "message": "本次回复暂时无法完成，请稍后重试。"})
            finally:
                enqueue({"type": "_delivery_end", "success": success})
                # 等待传输确认后决定 COMPLETE / INTERRUPTED，不能把已生成等同于已送达。
                delivered_end.wait(timeout=120)
                try:
                    text = delivery["text"]
                    complete = delivery["complete"] and not cancelled.is_set() and success
                    tool_plan = outcome.tool_plan if outcome else None
                    if harness and not (outcome and outcome.cached):
                        session = outcome.session if outcome else db.query(ChatSession).filter_by(
                            public_id=active.get("sessionId"), user_id=user_id).first()
                        message_id = outcome.current_message_id if outcome else active.get("messageId")
                        if session is not None:
                            if plan and (outcome is None or not success):
                                tool_plan = self._save_interrupted_trace(db, account, session, request, message_id, text, plan, harness)
                            if text:
                                harness.save_assistant_message(account, session, text, message_id,
                                    completion_status="COMPLETE" if complete else "INTERRUPTED")
                    if harness and tool_plan:
                        try:
                            asyncio.run(harness.dispatch_tools(tool_plan))
                        except Exception:
                            logger.exception("Post-response tools failed session=%s", active.get("sessionId"))
                    if complete:
                        enqueue({"type": "done", "sessionId": outcome.session.public_id})
                except Exception:
                    logger.exception("Chat persistence failed user=%s", user_id)
                    enqueue({"type": "error", "message": "回复保存暂时失败，请稍后重试。"})
                finally:
                    db.close()
                    enqueue({"type": "_stop"})

        thread = threading.Thread(target=worker, name="campus-chat-stream", daemon=True)
        self._worker_thread = thread
        thread.start()
        try:
            yield sse("status", {"type": "status", "message": "正在理解问题"})
            while True:
                try:
                    event = await asyncio.wait_for(events.get(), timeout=10)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                kind = event["type"]
                if kind == "_stop":
                    break
                if kind == "_delivery_end":
                    delivery["complete"] = event["success"]
                    delivered_end.set()
                    continue
                if kind == "token":
                    delivery["text"] += event.get("content", "")
                yield sse(kind, event)
        finally:
            cancelled.set()
            delivered_end.set()

    def _save_interrupted_trace(self, db, user, session, request, message_id, text, plan, harness):
        effective = RiskLevel(plan["effectiveRisk"])
        run = AgentRunResult(intent=IntentType(plan["intent"]), risk_level=effective,
            assessment=plan.get("assessment"), retrieved_knowledge=plan.get("knowledge", []),
            response_messages=plan["messages"], response_text=text, steps=[], memory_brief=plan.get("memoryBrief", ""),
            current_risk=RiskLevel(plan["currentRisk"]), carryover_risk=RiskLevel(plan["carryoverRisk"]),
            effective_risk=effective, previous_safety_status=plan["previousStatus"], new_safety_status=plan["newStatus"],
            response_mode=ResponseMode(plan["mode"]), memory_snapshot_metadata={
                **plan.get("snapshotMetadata", {}), "completionStatus": "INTERRUPTED", "streaming": True})
        # 风险评估已先落库；断线不能把 HIGH 记成 LOW，或跳过应有报告和工具任务。
        report = harness._create_report(user, session, request.message, run, message_id=message_id)
        AgentTraceService(db).save_run(user, session, request.message, PrivacySanitizer().sanitize(request.message),
            run.memory_brief, run, report.id if report else None, message_id=message_id)
        return AgentToolPlan(report_id=report.id if report else None, risk_level=effective.value)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
