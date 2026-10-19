"""Account-owned saved conversations for the student sidebar."""
import json
from dataclasses import fields

from sqlalchemy import func

from app.models.entities import ChatSession, ChatMessage, AgentRunTrace
from app.services.knowledge import SearchResult


class ConversationService:
    def __init__(self, db):
        self.db = db

    def owned_session(self, user_id, public_id):
        session = self.db.query(ChatSession).filter_by(public_id=public_id, user_id=user_id).first()
        if session is None:
            raise LookupError("会话不存在")
        return session

    def sessions(self, user_id, offset=0, limit=50):
        query = self.db.query(ChatSession).filter_by(user_id=user_id)
        total = query.count()
        rows = query.order_by(ChatSession.updated_at.desc(), ChatSession.id.desc()).offset(offset).limit(limit).all()
        counts = dict(self.db.query(ChatMessage.session_id, func.count(ChatMessage.id)).filter(
            ChatMessage.session_id.in_([s.id for s in rows])).group_by(ChatMessage.session_id).all())
        return {"items": [{"sessionId": s.public_id, "title": s.title, "createdAt": s.created_at,
            "updatedAt": s.updated_at, "messageCount": counts.get(s.id, 0)} for s in rows],
            "total": total, "nextOffset": offset + len(rows) if offset + len(rows) < total else None}

    def detail(self, user_id, public_id, include_sources=False):
        session = self.owned_session(user_id, public_id)
        rows = self.db.query(ChatMessage).filter_by(session_id=session.id, user_id=user_id).order_by(ChatMessage.id).all()
        traces = self.db.query(AgentRunTrace).filter_by(session_id=session.id, user_id=user_id).order_by(AgentRunTrace.id.desc()).all() if include_sources else []
        traces_by_id = {trace.id: trace for trace in traces}
        messages_by_id = {m.id: m for m in rows}
        allowed = {f.name for f in fields(SearchResult)}
        messages = []
        previous_user = None
        for message in rows:
            sources = []
            trace_id = message.agent_trace_id if message.agent_trace_id in traces_by_id else None
            parent = messages_by_id.get(message.reply_to_message_id) or previous_user
            if message.role == "USER":
                previous_user = message
            if include_sources and message.role == "ASSISTANT":
                candidates = [traces_by_id[trace_id]] if trace_id in traces_by_id else traces
                for trace in candidates:
                    if trace_id is None and not (parent and trace.original_input == parent.content and parent.created_at <= trace.created_at <= message.created_at):
                        continue
                    try:
                        prompt = json.loads(trace.response_messages_json or "[]")
                        if trace_id is None and not any(item.get("kind") == "approved_candidate" and item.get("content") == message.content for item in prompt):
                            continue
                        evidence = json.loads(trace.retrieved_knowledge_json or "[]")
                        sources = [{**SearchResult(**{k: v for k, v in row.items() if k in allowed}).citation(),
                            "content": row.get("content", ""), "score": row.get("score"), "matchedBy": row.get("matched_by", [])} for row in evidence]
                        trace_id = trace.id
                    except (ValueError, TypeError, AttributeError):
                        sources = []
                    break
            item = {"id": message.id, "role": message.role.lower(), "content": message.content,
                "createdAt": message.created_at, "completionStatus": message.completion_status}
            if include_sources:
                item.update(sources=sources, traceId=trace_id)
            messages.append(item)
        return {"sessionId": session.public_id, "title": session.title, "messages": messages}
