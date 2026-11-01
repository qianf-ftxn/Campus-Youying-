"""Readiness checks for the desktop launcher; no credentials or business data returned."""
from redis import Redis
from sqlalchemy import text

from app.services.vector_store import ChromaKnowledgeStore


def readiness(db, settings, state):
    checks = {}
    try:
        checks["database"] = "UP" if db.execute(text("SELECT 1")).scalar() == 1 else "DOWN"
    except Exception:
        checks["database"] = "DOWN"
    try:
        cache = Redis.from_url(settings.redis_url, socket_connect_timeout=1, socket_timeout=1)
        try:
            checks["redis"] = "UP" if cache.ping() else "DOWN"
        finally:
            cache.close()
    except Exception:
        checks["redis"] = "DOWN"
    for key, enabled, worker_name, thread_name in (
        ("toolQueue", settings.tool_queue_enabled, "tool_queue_worker", "dispatcher"),
        ("memoryMaintenance", settings.memory_maintenance_queue_enabled, "memory_maintenance_worker", "thread"),
        ("longTermMemory", settings.long_term_memory_enabled, "long_term_memory_worker", "thread"),
    ):
        worker = getattr(state, worker_name, None)
        thread = getattr(worker, thread_name, None)
        checks[key] = "DISABLED" if not enabled else ("UP" if thread and thread.is_alive() else "DOWN")
    checks["vectorStore"] = "DISABLED"
    if settings.knowledge_vector_enabled:
        try:
            store = ChromaKnowledgeStore(settings)
            checks["vectorStore"] = "UP" if store.can_embed and store.count() >= 0 else "DOWN"
        except Exception:
            checks["vectorStore"] = "DOWN"
    return {"status": "UP" if all(value in {"UP", "DISABLED"} for value in checks.values()) else "DOWN",
            "checks": checks}
