from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session

from app.agents.factory import agent_framework_status
from app.agents.event_driven_runtime import EventDrivenAgentRuntimeService
from app.core.config import get_settings
from app.core.database import get_db
from app.core.security import current_user, require_admin
from app.models.entities import UserAccount
from app.schemas.dtos import KnowledgeIngestRequest, KnowledgeIngestResponse, KnowledgeMetadata, KnowledgeDocumentStatusRequest, ChatRequest, SafetyOverrideRequest, authority
from app.services.chat import ChatService
from app.services.knowledge import KnowledgeService
from app.services.model_assets import finetuned_model_status
from app.services.report import ReportService
from app.services.safety_memory import SafetyMemoryService
from app.services.skills import MindBridgeSkillLibrary
from app.services.readiness import readiness
from app.services.conversations import ConversationService

router = APIRouter()


@router.get("/api/memory/status")
def personal_memory_status(user: Annotated[UserAccount, Depends(current_user)], db: Annotated[Session, Depends(get_db)]):
    from app.services.long_term_memory import LongTermMemoryService
    # No user selector or content endpoint; caller can only observe their own update state.
    status = LongTermMemoryService(db, get_settings()).status(user.id)
    return {"state": status["state"]}


@router.get("/api/admin/long-term-memory/status")
def personal_memory_admin_status(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    from app.models.entities import LongTermMemory, LongTermMemoryJob
    from app.services.long_term_memory import LongTermMemoryService
    jobs = db.query(LongTermMemoryJob).order_by(LongTermMemoryJob.updated_at.desc()).limit(100).all()
    service = LongTermMemoryService(db, get_settings())
    return {"enabled": get_settings().long_term_memory_enabled, "items": [{
        "userId": j.user_id, "jobState": j.status, "attempts": j.attempts,
        "processedMessageId": j.processed_message_id, "targetMessageId": j.target_message_id,
        "memoryCount": db.query(LongTermMemory).filter_by(user_id=j.user_id).count(),
        "recallCount": j.recall_count, "lastRecalledCount": j.last_recall_count,
        **service.status(j.user_id)} for j in jobs]}


@router.get("/actuator/health")
def health():
    return {"status": "UP"}


@router.get("/actuator/ready")
def ready(request: Request, db: Annotated[Session, Depends(get_db)]):
    result = readiness(db, get_settings(), request.app.state)
    return JSONResponse(result, status_code=200 if result["status"] == "UP" else 503)


@router.get("/api/profile")
def profile(user: Annotated[UserAccount, Depends(current_user)]):
    return {
        "id": user.id,
        "username": user.username,
        "displayName": user.display_name,
        "roles": [authority(role) for role in user.roles],
    }


@router.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user: Annotated[UserAccount, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    if "ROLE_ADMIN" in user.roles:
        raise HTTPException(403, "管理员账号只能查看后台记录，不能发起学生对话。")
    service = ChatService(db, get_settings())
    return StreamingResponse(service.stream_chat(user, request), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


@router.get("/api/agent/status")
def agent_status(user: Annotated[UserAccount, Depends(current_user)]):
    settings = get_settings()
    provider = settings.ai_provider.lower()
    model = settings.ollama_model if provider == "ollama" else settings.openai_model if provider == "openai" else "mock"
    framework = agent_framework_status(settings)
    return {
        "provider": provider,
        "model": model,
        "realModelEnabled": provider in {"ollama", "openai"},
        "agentFramework": framework,
        "finetunedModel": finetuned_model_status(settings),
        "agents": [
            {"name": "CoordinatorAgent", "status": "READY", "description": "维护任务板、预算、安全门槛、冲突仲裁和最终采纳"},
            {"name": "UnderstandingAgent", "status": "READY", "description": "独立理解用户输入，发布 intent artifact"},
            {"name": "SafetyAgent", "status": "READY", "description": "独立风险评估、SAFETY_OVERRIDE 和候选回复安全审查"},
            {"name": "ContextAgent", "status": "READY", "description": "独立记忆视图、RAG 检索和 skill 上下文聚合"},
            {"name": "ResponseAgent", "status": "READY", "description": "根据黑板 artifact 发布候选回复方案"},
        ],
        "skills": MindBridgeSkillLibrary.status_items(),
        "runtimeHarness": {
            "name": "MindBridgeAgentHarness",
            "status": "READY",
            "description": "统一管理单轮 Agent run 的输入脱敏、上下文注入、风险报告、工具计划和 trace 输出",
        },
        "loop": {
            "type": "event-driven-multi-agent",
            "maxSteps": EventDrivenAgentRuntimeService.max_steps,
            "scheduler": "claim-based-actor-runtime",
        },
        "collaboration": {
            "scheduler": "claim-based",
            "state": "append-only-blackboard",
            "messageBus": "per-agent inbox over shared mailbox",
            "fixedWorkflow": False,
            "agentIsolation": {
                "prompt": "per-agent system prompt",
                "memory": "per-agent private Redis key",
                "model": "per-agent model profile",
                "tools": "per-agent tool permissions",
            },
        },
    }


@router.get("/api/reports/me")
def my_reports(user: Annotated[UserAccount, Depends(current_user)], db: Annotated[Session, Depends(get_db)]):
    return ReportService(db).latest_reports(user.id)


@router.get("/api/chat/sessions")
def chat_sessions(user: Annotated[UserAccount, Depends(current_user)], db: Annotated[Session, Depends(get_db)],
                  offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)):
    return ConversationService(db).sessions(user.id, offset, limit)


@router.get("/api/chat/sessions/{session_id}")
def chat_session(session_id: str, user: Annotated[UserAccount, Depends(current_user)], db: Annotated[Session, Depends(get_db)]):
    try:
        return ConversationService(db).detail(user.id, session_id)
    except LookupError:
        raise HTTPException(404, "会话不存在")


@router.get("/api/admin/reports")
def admin_reports(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    return ReportService(db).latest_reports()


@router.get("/api/admin/excel-records")
def admin_excel(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    return ReportService(db).excel_records()


@router.get("/api/admin/alerts")
def admin_alerts(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    return ReportService(db).alert_records()


@router.get("/api/admin/cases")
def admin_cases(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    return ReportService(db).risk_cases()


@router.get("/api/admin/cases/{case_id}/notes")
def admin_case_notes(case_id: int, _: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    return ReportService(db).case_notes(case_id)


@router.get("/api/admin/tool-jobs")
def admin_tool_jobs(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    return ReportService(db).tool_jobs()


@router.get("/api/admin/dead-letters")
def admin_dead_letters(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    return ReportService(db).dead_letters()


@router.get("/api/admin/agent-traces")
def admin_agent_traces(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    return ReportService(db).agent_run_traces()


@router.get("/api/admin/tool-audits")
def admin_tool_audits(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    return ReportService(db).tool_audits()


@router.get("/api/admin/safety-states")
def admin_safety_states(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    return ReportService(db).safety_states()


@router.get("/api/admin/memory-maintenance-jobs")
def admin_memory_maintenance_jobs(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    return ReportService(db).memory_maintenance_jobs()


@router.get("/api/admin/risk-events")
def admin_risk_events(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[UserAccount, Depends(require_admin)],
    user_id: int | None = None,
):
    return ReportService(db).risk_events(user_id)


@router.post("/api/admin/safety-states/{user_id}/override")
def admin_safety_override(
    user_id: int,
    request: SafetyOverrideRequest,
    actor: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    if request.action not in {"MONITORING", "RESOLVED", "FALSE_POSITIVE"}:
        raise HTTPException(400, "action 仅支持 MONITORING、RESOLVED 或 FALSE_POSITIVE")
    try:
        session = ConversationService(db).owned_session(user_id, request.sessionId)
    except LookupError:
        raise HTTPException(404, "会话不存在或不属于指定用户")
    dto = SafetyMemoryService(db, get_settings()).manual_override(
        user_id=user_id,
        actor=actor.username,
        action=request.action,
        reason=request.reason,
        session_id=session.id,
    )
    return {"userId": user_id, "sessionId": request.sessionId, "status": dto.status, "effectiveRisk": dto.effective_risk, "version": dto.version}


@router.get("/api/admin/conversations/{session_id}")
def admin_conversation(session_id: str, _: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    try:
        return ReportService(db).conversation(session_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/api/admin/knowledge")
def ingest_knowledge(
    request: KnowledgeIngestRequest,
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    try:
        chunks = KnowledgeService(db, get_settings()).ingest(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return KnowledgeIngestResponse(source=request.source, chunks=chunks)


@router.get("/api/admin/knowledge/status")
def knowledge_status(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    return KnowledgeService(db, get_settings()).status()


@router.post("/api/admin/knowledge/rebuild-vector")
def rebuild_knowledge_vector(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    try:
        indexed = KnowledgeService(db, get_settings()).rebuild_vector_index()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"indexedChunks": indexed}


@router.post("/api/admin/knowledge/backup")
def backup_knowledge_vector(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    try:
        snapshot = KnowledgeService(db, get_settings()).backup_vector_index()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"snapshot": snapshot}


@router.post("/api/admin/knowledge/file")
async def ingest_file(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
):
    try:
        options = KnowledgeMetadata.model_validate_json(metadata)
        chunks = KnowledgeService(db, get_settings()).ingest_file(file.filename or "uploaded-file", await file.read(10 * 1024 * 1024 + 1), **options.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return KnowledgeIngestResponse(source=file.filename or "uploaded-file", chunks=chunks)


@router.get("/api/admin/knowledge/documents")
def knowledge_documents(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    return KnowledgeService(db, get_settings()).documents()


@router.get("/api/admin/knowledge/documents/{document_id}")
def knowledge_document(document_id: int, _: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    try:
        return KnowledgeService(db, get_settings()).document_detail(document_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.patch("/api/admin/knowledge/documents/{document_id}/status")
def knowledge_document_status(document_id: int, request: KnowledgeDocumentStatusRequest,
    _: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    try:
        return KnowledgeService(db, get_settings()).set_document_status(document_id, request.status)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
