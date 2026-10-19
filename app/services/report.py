from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.entities import AlertRecord, AgentRunTrace, CaseNote, ChatMessage, ChatSession, DeadLetterRecord, ExcelRecord, MemoryMaintenanceJob, PsychologicalReport, RiskCase, RiskEvent, SafetyState, ToolAuditRecord, ToolJob, UserAccount
from app.schemas.dtos import AgentRunTraceResponse, CaseNoteResponse, ConversationMessageResponse, ConversationResponse, DeadLetterResponse, ReportResponse, RiskCaseResponse, ToolAuditResponse, ToolJobResponse, ToolRecordResponse


class ReportService:
    def __init__(self, db: Session):
        self.db = db

    def safety_states(self) -> list[dict]:
        rows = self.db.query(SafetyState).order_by(SafetyState.updated_at.desc()).limit(100).all()
        users = {user.id: user for user in self.db.query(UserAccount).all()}
        sessions = {s.id: s for s in self.db.query(ChatSession).all()}
        return [
            {
                "userId": row.user_id,
                "sessionId": sessions[row.session_id].public_id if row.session_id in sessions else "",
                "sessionTitle": sessions[row.session_id].title if row.session_id in sessions else "",
                "username": users[row.user_id].username if row.user_id in users else str(row.user_id),
                "displayName": users[row.user_id].display_name if row.user_id in users else "",
                "status": row.status,
                "effectiveRisk": row.effective_risk,
                "reassessmentRequired": bool(row.reassessment_required),
                "humanReviewRequired": bool(row.human_review_required),
                "safeReassessmentCount": row.safe_reassessment_count or 0,
                "consecutiveLowAssessments": row.consecutive_low_assessments or 0,
                "normalTurnCountSinceReview": row.normal_turn_count_since_review or 0,
                "lastHighRiskAt": row.last_high_risk_at.isoformat() if row.last_high_risk_at else None,
                "lastAssessedAt": row.last_assessed_at.isoformat() if row.last_assessed_at else None,
                "version": row.version or 0,
            }
            for row in rows
        ]

    def memory_maintenance_jobs(self) -> list[dict]:
        rows = self.db.query(MemoryMaintenanceJob).order_by(MemoryMaintenanceJob.id.desc()).limit(100).all()
        return [
            {
                "id": row.id,
                "sessionId": row.session_id,
                "status": row.status,
                "attempts": row.attempts,
                "maxAttempts": row.max_attempts,
                "runAfter": row.run_after.isoformat() if row.run_after else None,
                "lastError": row.last_error,
                "createdAt": row.created_at.isoformat() if row.created_at else None,
                "updatedAt": row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in rows
        ]

    def risk_events(self, user_id: int | None = None) -> list[dict]:
        query = self.db.query(RiskEvent)
        if user_id is not None:
            query = query.filter(RiskEvent.user_id == user_id)
        rows = query.order_by(RiskEvent.id.desc()).limit(100).all()
        return [
            {
                "id": row.id,
                "userId": row.user_id,
                "sessionId": row.session_id,
                "messageId": row.message_id,
                "reportId": row.report_id,
                "eventKind": row.event_kind,
                "riskType": row.risk_type,
                "riskLevel": row.risk_level,
                "confidence": row.confidence,
                "evidenceSummary": row.evidence_summary,
                "source": row.source,
                "previousStatus": row.previous_status,
                "newStatus": row.new_status,
                "createdAt": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]

    def latest_reports(self, user_id: int | None = None) -> list[ReportResponse]:
        query = self.db.query(PsychologicalReport).order_by(PsychologicalReport.created_at.desc())
        if user_id is not None:
            query = query.filter(PsychologicalReport.user_id == user_id)
        return [self._report_response(item) for item in query.limit(100).all()]

    def excel_records(self) -> list[ToolRecordResponse]:
        rows = self.db.query(ExcelRecord).order_by(ExcelRecord.created_at.desc()).limit(100).all()
        return [
            ToolRecordResponse(id=row.id, reportId=row.report_id, status=row.status, message=row.message, createdAt=row.created_at, filePath=row.file_path)
            for row in rows
        ]

    def alert_records(self) -> list[ToolRecordResponse]:
        rows = self.db.query(AlertRecord).order_by(AlertRecord.created_at.desc()).limit(100).all()
        return [
            ToolRecordResponse(
                id=row.id,
                reportId=row.report_id,
                status=row.status,
                message=row.message,
                createdAt=row.created_at,
                channel=row.channel,
                recipient=row.recipient,
            )
            for row in rows
        ]

    def risk_cases(self) -> list[RiskCaseResponse]:
        rows = self.db.query(RiskCase).order_by(RiskCase.updated_at.desc()).limit(100).all()
        return [
            RiskCaseResponse(
                id=row.id,
                reportId=row.report_id,
                riskLevel=row.risk_level,
                status=row.status,
                owner=row.owner,
                summary=row.summary,
                handoffSummary=row.handoff_summary,
                acknowledgedBy=row.acknowledged_by,
                acknowledgedAt=row.acknowledged_at,
                createdAt=row.created_at,
                updatedAt=row.updated_at,
            )
            for row in rows
        ]

    def case_notes(self, case_id: int) -> list[CaseNoteResponse]:
        rows = self.db.query(CaseNote).filter(CaseNote.case_id == case_id).order_by(CaseNote.created_at.asc()).all()
        return [
            CaseNoteResponse(id=row.id, caseId=row.case_id, actor=row.actor, note=row.note, createdAt=row.created_at)
            for row in rows
        ]

    def tool_jobs(self) -> list[ToolJobResponse]:
        rows = self.db.query(ToolJob).order_by(ToolJob.created_at.desc()).limit(100).all()
        return [
            ToolJobResponse(
                id=row.id,
                reportId=row.report_id,
                kind=row.kind,
                status=row.status,
                attempts=row.attempts,
                maxAttempts=row.max_attempts,
                dependsOnJobId=row.depends_on_job_id,
                runAfter=row.run_after,
                lastError=row.last_error,
                createdAt=row.created_at,
                updatedAt=row.updated_at,
            )
            for row in rows
        ]

    def dead_letters(self) -> list[DeadLetterResponse]:
        rows = self.db.query(DeadLetterRecord).order_by(DeadLetterRecord.created_at.desc()).limit(100).all()
        return [
            DeadLetterResponse(
                id=row.id,
                jobId=row.job_id,
                reportId=row.report_id,
                kind=row.kind,
                reason=row.reason,
                payload=row.payload,
                createdAt=row.created_at,
            )
            for row in rows
        ]


    def agent_run_traces(self) -> list[AgentRunTraceResponse]:
        rows = self.db.query(AgentRunTrace).order_by(AgentRunTrace.created_at.desc()).limit(100).all()
        responses = []
        for row in rows:
            user = self.db.get(UserAccount, row.user_id)
            session = self.db.get(ChatSession, row.session_id)
            responses.append(
                AgentRunTraceResponse(
                    id=row.id,
                    sessionId=session.public_id if session else "",
                    reportId=row.report_id,
                    username=user.username if user else "",
                    intent=row.intent,
                    riskLevel=row.risk_level,
                    originalInput=row.original_input,
                    sanitizedInput=row.sanitized_input,
                    memoryBrief=row.memory_brief,
                    agentSteps=_loads(row.agent_steps_json, []),
                    retrievedKnowledge=_loads(row.retrieved_knowledge_json, []),
                    responseMessages=_loads(row.response_messages_json, []),
                    assessment=_loads(row.assessment_json, {}),
                    createdAt=row.created_at,
                )
            )
        return responses

    def tool_audits(self) -> list[ToolAuditResponse]:
        rows = self.db.query(ToolAuditRecord).order_by(ToolAuditRecord.created_at.desc()).limit(100).all()
        return [
            ToolAuditResponse(
                id=row.id,
                jobId=row.job_id,
                reportId=row.report_id,
                toolName=row.tool_name,
                policy=row.policy,
                allowed=row.allowed,
                status=row.status,
                reason=row.reason,
                payload=_loads(row.payload, {}),
                createdAt=row.created_at,
                updatedAt=row.updated_at,
            )
            for row in rows
        ]

    def conversation(self, public_id: str) -> ConversationResponse:
        from app.services.conversations import ConversationService
        session = self.db.query(ChatSession).filter(ChatSession.public_id == public_id).first()
        if session is None:
            raise ValueError("Session not found")
        return ConversationResponse(**ConversationService(self.db).detail(session.user_id, public_id, include_sources=True))

    def _report_response(self, report: PsychologicalReport) -> ReportResponse:
        user = self.db.get(UserAccount, report.user_id)
        session = self.db.get(ChatSession, report.session_id)
        return ReportResponse(
            id=report.id,
            sessionId=session.public_id if session else "",
            username=user.username if user else "",
            displayName=user.display_name if user else "",
            content=report.content,
            intent=report.intent,
            emotion=report.emotion,
            emotionScore=report.emotion_score,
            riskLevel=report.risk_level,
            confidence=report.confidence,
            summary=report.summary,
            createdAt=report.created_at,
        )


def _loads(raw: str, default):
    import json

    try:
        return json.loads(raw or "")
    except Exception:
        return default
