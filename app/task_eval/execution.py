"""Controlled single/multi-agent adapters, isolated from live services.

Only candidate input and shared retrieval enter the adapters. Gold criteria are
consumed separately by the evaluator. Patches are scoped and execution is serial.
"""
from __future__ import annotations

import copy
import json
import tempfile
import time
import uuid
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from unittest.mock import patch

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.core.enums import EmotionLabel, IntentType, ResponseMode, RiskLevel
from app.evaluation.judge import extract_json_object
from app.schemas.dtos import AiMessage
from app.services.ai import AiClient, has_high_risk_signal
from app.services.assessment import PsychologyAssessment
from app.services.intent_rules import classify_intent_rules
from app.services.knowledge import SearchResult
from app.services.memory_context import estimate_tokens
from app.services.privacy import PrivacySanitizer
from app.services.skills import MindBridgeSkillLibrary
from app.task_eval.prepare import candidate_input, encode, sha256

BUSINESS_RULES = """共同业务规则：
CHAT用于普通交流，CONSULT用于校园制度咨询，RISK用于情绪支持或危险表达。
风险LOW表示无明确危险及严重功能受损，MEDIUM表示持续困扰并明显影响生活学习，HIGH表示明确或紧迫的个人/他人危险。
参考资料不能代替适用条件判断；以用户最新身份和更正为准，区分在读、休学、结业、毕业等状态。
资料没有支持的金额、期限、录取、办理编号和个人状态不能编造；缺条件要追问，缺资料要说明。
高风险先关注当前安全和可信任的人或当地紧急支持，不提供危险细节或未经核实的联系方式。
用户回答中不得输出风险标签、评分、报告ID、任务字段等内部信息；工具成功也不代表已人工接手。
后置任务由共同业务代码映射：LOW无新报告或任务；MEDIUM新建报告并写记录、建个案；HIGH另记录预警。
"""


@dataclass(frozen=True)
class Limits:
    max_calls: int = 16
    max_input_tokens: int = 64000
    max_output_tokens: int = 8192
    output_per_call: int = 1024
    wall_seconds: float = 180.0


@dataclass(frozen=True)
class Completion:
    text: str
    usage: dict | None = None
    finish_reason: str = "stop"
    cache_hit: bool = False


class BudgetExceeded(RuntimeError):
    pass


def openai_transport(messages, settings, stage, timeout) -> Completion:
    """No hidden retries. Every submitted request is recorded by ModelSession."""
    if settings.ai_provider != "openai":
        raise ValueError("Real task comparison currently requires the OpenAI-compatible provider")
    payload = {
        "model": settings.openai_model,
        "messages": [m.model_dump() for m in messages],
        "temperature": settings.ai_temperature,
        "max_tokens": settings.ai_max_tokens,
        "stream": False,
    }
    if settings.openai_enable_thinking is not None:
        payload["enable_thinking"] = settings.openai_enable_thinking
    response = httpx.post(
        settings.openai_base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer " + settings.openai_api_key},
        json=payload, timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    return Completion(choice["message"].get("content") or "", body.get("usage"), choice.get("finish_reason", ""))


class ModelSession:
    def __init__(self, settings, limits, transport, skills=""):
        self.settings, self.limits, self.transport = settings, limits, transport
        self.skills = skills
        self.started = time.perf_counter()
        self.calls, self.errors = [], []

    def complete(self, messages, stage):
        # Always pin all roles to the exact same settings as the single agent.
        messages = [AiMessage(role="system", content=BUSINESS_RULES), *messages]
        if stage in {"SingleAgent", "ResponseAgent"} and self.skills:
            messages.insert(1, AiMessage(role="system", content="共享业务Skill规则（按场景适用）：\n" + self.skills))
        estimated = sum(estimate_tokens(m.content) + 8 for m in messages)
        used_input = sum(c.get("input_tokens", 0) for c in self.calls)
        used_output = sum(c.get("output_tokens", 0) for c in self.calls)
        remaining_seconds = self.limits.wall_seconds - (time.perf_counter() - self.started)
        output_cap = min(self.limits.output_per_call, self.limits.max_output_tokens - used_output)
        if (len(self.calls) >= self.limits.max_calls or used_input + estimated > self.limits.max_input_tokens
                or output_cap <= 0 or remaining_seconds <= 0):
            self.errors.append({"stage": stage, "type": "BudgetExceeded"})
            raise BudgetExceeded("Shared inference budget exhausted")
        config = copy.copy(self.settings)
        config.ai_max_tokens = output_cap
        entry = {
            "stage": stage, "messages": [m.model_dump() for m in messages],
            "request_sha256": sha256(encode({"provider": config.ai_provider,
                                             "endpoint_sha256": sha256(config.openai_base_url.rstrip("/").encode()),
                                             "enable_thinking": config.openai_enable_thinking,
                                             "model": config.openai_model, "temperature": config.ai_temperature,
                                             "max_tokens": output_cap, "messages": [m.model_dump() for m in messages]})),
            "input_tokens": estimated, "output_tokens": 0, "usage_kind": "estimate",
        }
        self.calls.append(entry)
        started = time.perf_counter()
        try:
            result = self.transport(messages, config, stage, min(60.0, remaining_seconds))
            if not isinstance(result, Completion):
                raise TypeError("Transport must return Completion")
            entry.update({"text": result.text, "finish_reason": result.finish_reason, "cache_hit": result.cache_hit})
            if result.usage and {"prompt_tokens", "completion_tokens"} <= result.usage.keys():
                entry.update({"input_tokens": result.usage["prompt_tokens"], "output_tokens": result.usage["completion_tokens"],
                              "usage_kind": "provider", "usage": result.usage})
            else:
                entry["output_tokens"] = estimate_tokens(result.text)
            if result.finish_reason not in {"stop", "end_turn"}:
                raise RuntimeError("Incomplete model response: " + result.finish_reason)
            if (sum(c["input_tokens"] for c in self.calls) > self.limits.max_input_tokens
                    or sum(c["output_tokens"] for c in self.calls) > self.limits.max_output_tokens
                    or time.perf_counter() - self.started > self.limits.wall_seconds):
                raise BudgetExceeded("Observed usage exceeds shared budget")
            if not result.text.strip():
                raise ValueError("Empty model output")
            return result.text
        except Exception as exc:
            entry["error"] = type(exc).__name__
            self.errors.append({"stage": stage, "type": type(exc).__name__})
            raise
        finally:
            entry["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)


def prepare_shared_input(case: dict) -> dict:
    inputs = candidate_input(case)
    privacy = PrivacySanitizer()
    inputs["question"] = privacy.sanitize(inputs["question"])
    for message in inputs["history"]:
        message["content"] = privacy.sanitize(message["content"])
    return inputs


def retrieve_once(knowledge, inputs, settings) -> list[SearchResult]:
    """Neutral query, no gold intent/domain or answer evidence used."""
    query = "\n".join([m["content"] for m in inputs["history"] if m["role"] == "user"] + [inputs["question"]])
    rows = knowledge.retrieve(query, settings.knowledge_top_k, school=settings.campus_name)
    selected, used = [], 0
    for row in rows[:settings.knowledge_top_k]:
        tokens = estimate_tokens(row.content)
        if used + tokens <= settings.rag_skill_budget_tokens:
            selected.append(copy.deepcopy(row))
            used += tokens
    return selected


def shared_skills() -> str:
    return "\n\n".join(skill.prompt_context() for skill in MindBridgeSkillLibrary.list_skills()
                         if skill.name != "counselor_handoff_summary")


def context_text(rows) -> str:
    return "\n\n".join(
        f"[资料{i+1}] {row.title or row.source} | 版本 {row.version} | {row.heading}"
        + (f" | 第{row.page_start}页" if row.page_start else "")
        + (" | 演示资料，非正式校规" if row.is_demo else "")
        + f"\n{row.content}" for i, row in enumerate(rows))


def rule_signals(inputs):
    previous = [m["content"] for m in inputs["history"] if m["role"] == "user"]
    routing = classify_intent_rules(inputs["question"], previous)
    return {"hard_high_risk": has_high_risk_signal(inputs["question"]),
            "intent_rule": routing.intent.value if routing.intent else None}


def parse_single(raw: str) -> dict:
    # Match the production assessment parser's tolerance for JSON wrappers.
    # Validate the fields strictly after extraction; do not invent missing labels.
    value = extract_json_object(raw)
    if not isinstance(value, dict) or set(value) != {"intent", "risk", "answer", "final"}:
        raise ValueError("Return exactly intent, risk, answer, final")
    IntentType(value["intent"])
    RiskLevel(value["risk"])
    if not isinstance(value["answer"], str) or not value["answer"].strip() or type(value["final"]) is not bool:
        raise ValueError("answer must be nonempty and final must be a boolean")
    return value


def run_single(inputs, rows, model):
    from app.agents.result import AgentRunResult
    signals = rule_signals(inputs)
    messages = [
        AiMessage(role="system", content=(
            "你是单Agent校园助手，自行完成意图理解、风险判断、证据取舍、回答与自检。"
            "只输出JSON对象，字段严格为intent（CHAT/CONSULT/RISK）、risk（LOW/MEDIUM/HIGH）、answer（给用户的正文）、final（布尔）。"
            "首轮final必须为false；收到自检指令后可以修订并提交final=true。可多轮自检，资源预算与多Agent一致。"
            "工具由共享执行器按最终风险映射，不要在正文输出内部字段。"
            f"\n共享确定性规则信号：{json.dumps(signals, ensure_ascii=False)}"
            f"\n可用知识：\n{context_text(rows) or '无'}"
        )),
        *[AiMessage(**m) for m in inputs["history"]],
        AiMessage(role="user", content=inputs["question"]),
    ]
    raw_decisions, decision = [], None
    for attempt in range(model.limits.max_calls):
        raw = model.complete(messages, "SingleAgent")
        messages.append(AiMessage(role="assistant", content=raw))
        try:
            decision = parse_single(raw)
            raw_decisions.append(copy.deepcopy(decision))
            if attempt > 0 and decision["final"]:
                break
            feedback = "请自检并可修订：适用身份、关键事实、资料不足、当前安全和风险、内部字段泄露。确认后提交final=true。不要重复回答历史旧问题。"
        except (ValueError, TypeError) as exc:
            decision = None
            feedback = f"输出格式不合约：{exc}。仅修复或修订你的JSON；没有提供标准答案。"
        messages.append(AiMessage(role="system", content=feedback))
    else:
        raise BudgetExceeded("Single agent did not finalize within shared call limit")
    guarded = copy.deepcopy(decision)
    # Same deterministic guardrails used by UnderstandingAgent/SafetyAgent;
    # these are input-derived rules, not reference labels.
    if signals["hard_high_risk"]:
        guarded["risk"] = "HIGH"
    if signals["intent_rule"]:
        guarded["intent"] = signals["intent_rule"]
    if guarded["risk"] == "HIGH":
        guarded["intent"] = "RISK"
    risk, intent = RiskLevel(guarded["risk"]), IntentType(guarded["intent"])
    emotion = {RiskLevel.LOW: EmotionLabel.NORMAL, RiskLevel.MEDIUM: EmotionLabel.DEPRESSED,
               RiskLevel.HIGH: EmotionLabel.HIGH_RISK}[risk]
    assessment = PsychologyAssessment(emotion, {RiskLevel.LOW: 0., RiskLevel.MEDIUM: 3., RiskLevel.HIGH: 4.}[risk],
                                      risk, 1.0, guarded["answer"][:500])
    run = AgentRunResult(intent, risk, assessment, list(rows), [], guarded["answer"], [], "",
                         current_risk=risk, effective_risk=risk,
                         response_mode=ResponseMode.SAFETY_SUPPORT if risk == RiskLevel.HIGH else ResponseMode.SUPPORT)
    return run, {"raw_decisions": raw_decisions, "guarded_decision": guarded, "rule_signals": signals}


def _shared_runtime_patches(stack, inputs, rows):
    from app.agents.autonomous import ContextAgent, ResponseAgent
    from app.agents.registry import AgentDecision

    class SharedContext(ContextAgent):
        def _rewrite_query(self, *args, **kwargs):
            return inputs["question"]

        def act(self, task, board):
            result = super().act(task, board)
            artifacts = tuple(replace(a, payload={**a.payload, "retrievedKnowledge": copy.deepcopy(rows),
                              "skillContext": "", "retrievalTrace": {"path": "shared-frozen-context"}})
                              if a.kind == "context" else a for a in result.artifacts)
            return replace(result, artifacts=artifacts)

    class SharedResponse(ResponseAgent):
        def decide(self, task, board):
            if not board.latest_artifact("context"):
                return AgentDecision(False, reason="controlled comparison waits for shared context")
            return super().decide(task, board)

    class FrozenKnowledge:
        last_retrieval = {"path": "shared-frozen-context"}

        def __init__(self, *args, **kwargs):
            pass

        def retrieve(self, *args, **kwargs):
            return copy.deepcopy(rows)

    stack.enter_context(patch("app.agents.event_driven_runtime.ContextAgent", SharedContext))
    stack.enter_context(patch("app.agents.event_driven_runtime.ResponseAgent", SharedResponse))
    stack.enter_context(patch("app.agents.event_driven_runtime.KnowledgeService", FrozenKnowledge))


def execute_effects(db, settings, user, session, message, run):
    """Shared real report/queue/tool code, synchronously drained in a sandbox."""
    from app.agents.harness import MindBridgeAgentHarness
    from app.models.entities import AlertRecord, ExcelRecord, PsychologicalReport, RiskCase, ToolJob
    from app.services.tool_governance import ToolGovernanceService
    from app.services.tool_queue import ToolQueueService, ToolQueueWorker

    report = MindBridgeAgentHarness(db, settings)._create_report(user, session, message.content, run, message.id)
    if report:
        ToolQueueService(db, settings).enqueue_report(report.id, run.effective_risk.value)
    jobs = db.query(ToolJob).order_by(ToolJob.id).all()
    worker = ToolQueueWorker(settings)
    try:
        for job in jobs:
            if not worker._dependency_ready(db, job):
                continue
            job.status, job.attempts = "RUNNING", 1
            db.commit()
            governance = ToolGovernanceService(db)
            audit = governance.start_job(job, report)
            try:
                governance.require_allowed(job, report)
                worker._execute(db, job)
                job.status = "SUCCESS"
                governance.finish(audit, "SUCCESS")
            except Exception as exc:
                db.rollback()
                worker._fail_or_dead_letter(db, job.id, exc)
                governance.finish(audit, "FAILED", type(exc).__name__)
            db.commit()
    finally:
        worker.stop()
    # Each arm has its own database, so counts cannot leak across arms/cases.
    return {
        "report_count": db.query(PsychologicalReport).count(),
        "jobs": [{"kind": j.kind, "status": j.status, "attempts": j.attempts} for j in jobs],
        "artifact_counts": {"excel": db.query(ExcelRecord).count(), "case": db.query(RiskCase).count(),
                            "alert": db.query(AlertRecord).count()},
        "notification_adapter": "log", "execution_adapter": "synchronous_sandbox_queue_drain",
    }


def run_arm(arm, inputs, rows, settings, *, transport=openai_transport, limits=Limits(), skills=None):
    from app.agents.event_driven_runtime import EventDrivenAgentRuntimeService
    from app.core.database import Base
    from app.models.entities import ChatMessage, ChatSession, UserAccount

    if arm not in {"single", "multi"}:
        raise ValueError("Unknown comparison arm")
    if set(inputs) != {"question", "history", "initial_safety_state"} or inputs["initial_safety_state"] != "RESOLVED":
        raise ValueError("Only candidate input in an initially resolved session is allowed")
    if len(rows) > settings.knowledge_top_k or sum(estimate_tokens(r.content) for r in rows) > settings.rag_skill_budget_tokens:
        raise ValueError("Shared evidence exceeds the common retrieval budget")
    config = copy.copy(settings)
    config.ai_temperature = 0
    config.ai_max_tokens = limits.output_per_call
    config.policy_evidence_precision_enabled = False
    config.long_term_memory_enabled = config.long_term_memory_vector_enabled = False
    config.memory_maintenance_queue_enabled = config.memory_rolling_summary_enabled = False
    config.bundled_knowledge_enabled = config.knowledge_vector_enabled = False
    config.alert_email_delivery_mode = "log"
    skills = shared_skills() if skills is None else skills
    model = ModelSession(config, limits, transport, skills)
    result = {"arm": arm, "input_sha256": sha256(encode(inputs)), "evidence_sha256": sha256(encode([asdict(r) for r in rows])),
              "skills_sha256": sha256(skills.encode()), "limits": asdict(limits), "calls": model.calls,
              "execution_sha256": sha256(Path(__file__).read_bytes()),
              "candidate_profile": {"provider": config.ai_provider, "model": config.openai_model,
                                    "temperature": config.ai_temperature,
                                    "endpoint_sha256": sha256(config.openai_base_url.rstrip("/").encode()),
                                    "knowledge_top_k": config.knowledge_top_k,
                                    "knowledge_budget": config.rag_skill_budget_tokens,
                                    "policy_evidence_precision_enabled": False},
              "response_completed": False, "effect_execution_completed": False}
    with tempfile.TemporaryDirectory(prefix=f"task-{arm}-") as temporary, ExitStack() as stack:
        # Patch constructors before instantiating runtime or services. No live Redis.
        stack.enter_context(patch("app.services.memory.RedisShortTermMemoryStore._connect", return_value=None))
        stack.enter_context(patch("app.services.memory_context.RollingSummaryService._connect_redis", return_value=None))
        stack.enter_context(patch("app.services.safety_memory.SafetyMemoryService._connect_redis", return_value=None))
        config.database_url = "sqlite://"
        config.excel_path = str(Path(temporary) / "ledger.xlsx")
        config.chroma_persist_dir = str(Path(temporary) / "chroma")
        config.chroma_snapshot_dir = str(Path(temporary) / "snapshots")
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine, autoflush=False)()
        try:
            user = UserAccount(username="isolated-eval", display_name="评测学生", password_hash="test-only")
            db.add(user); db.flush()
            session = ChatSession(public_id=uuid.uuid4().hex, user_id=user.id, title="isolated task")
            db.add(session); db.flush()
            for message in inputs["history"]:
                db.add(ChatMessage(user_id=user.id, session_id=session.id, role=message["role"].upper(), content=message["content"]))
            current = ChatMessage(user_id=user.id, session_id=session.id, role="USER", content=inputs["question"])
            db.add(current); db.commit()
            if arm == "single":
                run, details = run_single(inputs, rows, model)
            else:
                _shared_runtime_patches(stack, inputs, rows)

                def complete(client, messages):
                    return model.complete(messages, client.agent_name or "Runtime")

                stack.enter_context(patch.object(AiClient, "complete", complete))
                runtime = EventDrivenAgentRuntimeService(db, config)
                run = runtime.run(user, session, inputs["question"], inputs["question"], current_message_id=current.id)
                # Critically: use the accepted text; never generate a new answer here.
                details = {"steps": [asdict(s) for s in run.steps],
                           "final_accepted": any(s.action == "FINAL_ACCEPTED" for s in run.steps)}
            result.update({"answer": run.response_text, "intent": run.intent.value,
                           "current_risk": run.current_risk.value, "effective_risk": run.effective_risk.value,
                           "response_completed": bool(run.response_text.strip()) and not model.errors,
                           "details": details})
            result["effects"] = execute_effects(db, config, user, session, current, run)
            result["effect_execution_completed"] = all(j["status"] == "SUCCESS" for j in result["effects"]["jobs"])
        except Exception as exc:
            db.rollback()
            result["error"] = type(exc).__name__
        finally:
            result["model_errors"] = model.errors
            result["model_calls"] = len(model.calls)
            result["input_tokens"] = sum(c["input_tokens"] for c in model.calls)
            result["output_tokens"] = sum(c["output_tokens"] for c in model.calls)
            result["elapsed_ms"] = round((time.perf_counter() - model.started) * 1000, 3)
            db.close(); engine.dispose()
    return result


def machine_checks(case, observation):
    """No semantic score is guessed here; a judge must still check the answer."""
    expected = case["acceptance"]
    effects = observation.get("effects", {})
    jobs = effects.get("jobs", [])
    wanted_jobs = expected["effects"]["tool_jobs"]
    artifacts = effects.get("artifact_counts", {})
    expected_artifacts = {"excel": int("EXCEL_REPORT" in wanted_jobs), "case": int("CASE_CREATE" in wanted_jobs),
                          "alert": int("ALERT_SEND" in wanted_jobs)}
    return {
        "response_completed": observation.get("response_completed") is True,
        "nonempty_answer": bool(observation.get("answer", "").strip()),
        "intent_correct": observation.get("intent") == expected["intent"],
        "risk_correct": observation.get("current_risk") == expected["current_risk"]
                        and observation.get("effective_risk") == expected["effective_risk"],
        "report_count_exact": effects.get("report_count") == expected["effects"]["report_count"],
        "tool_jobs_exact": "effects" in observation and Counter(j["kind"] for j in jobs) == Counter(wanted_jobs),
        "tool_jobs_success": observation.get("effect_execution_completed") is True
                             and all(j["status"] == "SUCCESS" for j in jobs),
        "no_duplicate_artifacts": artifacts == expected_artifacts,
    }


def run_pair(case, knowledge, settings, *, transport=openai_transport, limits=Limits()):
    inputs = prepare_shared_input(case)
    rows = retrieve_once(knowledge, inputs, settings)
    skills = shared_skills()
    order = ("single", "multi") if int(sha256(case["id"].encode()), 16) % 2 == 0 else ("multi", "single")
    observations = {arm: run_arm(arm, inputs, rows, settings, transport=transport, limits=limits, skills=skills)
                    for arm in order}
    for key in ("input_sha256", "evidence_sha256", "skills_sha256", "limits", "candidate_profile", "execution_sha256"):
        if observations["single"][key] != observations["multi"][key]:
            raise RuntimeError("Unfair comparison: " + key)
    return {"case_id": case["id"], "execution_order": list(order), "observations": observations,
            "machine_checks": {arm: machine_checks(case, value) for arm, value in observations.items()},
            "semantic_checks": "pending", "task_success": None}
