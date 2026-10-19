from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.config import Settings
from app.core.database import SessionLocal
from app.models.entities import MemoryMaintenanceJob
from app.services.memory_context import MemoryContextService


logger = logging.getLogger(__name__)

PENDING = "PENDING"
RUNNING = "RUNNING"
SUCCESS = "SUCCESS"
DEAD = "DEAD"


class MemoryMaintenanceQueue:
    """MySQL-backed durable queue for non-critical rolling-summary maintenance."""

    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings

    def enqueue(self, session_id: int) -> MemoryMaintenanceJob:
        existing = (
            self.db.query(MemoryMaintenanceJob)
            .filter(MemoryMaintenanceJob.session_id == session_id)
            .first()
        )
        if existing is not None:
            if existing.status not in {PENDING, RUNNING}:
                existing.status = PENDING
                existing.attempts = 0
                existing.run_after = datetime.utcnow()
                existing.last_error = ""
                existing.updated_at = datetime.utcnow()
                self.db.commit()
            return existing
        job = MemoryMaintenanceJob(
            session_id=session_id,
            status=PENDING,
            max_attempts=self.settings.memory_maintenance_max_attempts,
            run_after=datetime.utcnow(),
        )
        self.db.add(job)
        try:
            self.db.commit()
        except IntegrityError:
            self.db.rollback()
            existing = self.db.query(MemoryMaintenanceJob).filter(MemoryMaintenanceJob.session_id == session_id).first()
            if existing is None:
                raise
            return existing
        self.db.refresh(job)
        return job


class MemoryMaintenanceWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.settings.memory_maintenance_queue_enabled or self.thread is not None:
            return
        self._recover_running_jobs()
        self.thread = threading.Thread(target=self._loop, name="mindbridge-memory-maintenance", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._dispatch_once()
            except Exception:
                logger.exception("Memory maintenance queue dispatch failed")
            self.stop_event.wait(self.settings.memory_maintenance_poll_interval_seconds)

    def _dispatch_once(self) -> None:
        db = SessionLocal()
        try:
            job = (
                db.query(MemoryMaintenanceJob)
                .filter(MemoryMaintenanceJob.status == PENDING, MemoryMaintenanceJob.run_after <= datetime.utcnow())
                .order_by(MemoryMaintenanceJob.created_at.asc())
                .first()
            )
            if job is None:
                return
            job.status = RUNNING
            job.attempts += 1
            job.updated_at = datetime.utcnow()
            db.commit()
            job_id = job.id
            try:
                result = MemoryContextService(db, self.settings).compact_after_turn(job.session_id)
                job.status = SUCCESS
                job.last_error = ""
                logger.debug(
                    "memory maintenance completed job=%s session=%s compacted=%s reason=%s",
                    job.id,
                    job.session_id,
                    result.compacted,
                    result.reason,
                )
            except Exception as exc:
                db.rollback()
                job = db.get(MemoryMaintenanceJob, job_id)
                job.last_error = f"{type(exc).__name__}: {exc}"
                if job.attempts >= job.max_attempts:
                    job.status = DEAD
                else:
                    job.status = PENDING
                    job.run_after = datetime.utcnow() + timedelta(seconds=5 * job.attempts)
            job.updated_at = datetime.utcnow()
            db.commit()
        finally:
            db.close()

    def _recover_running_jobs(self) -> None:
        db = SessionLocal()
        try:
            for job in db.query(MemoryMaintenanceJob).filter(MemoryMaintenanceJob.status == RUNNING).all():
                job.status = PENDING
                job.last_error = "服务重启后恢复未完成的记忆维护任务"
                job.run_after = datetime.utcnow()
                job.updated_at = datetime.utcnow()
            db.commit()
        finally:
            db.close()


_worker: MemoryMaintenanceWorker | None = None


def get_memory_maintenance_worker(settings: Settings) -> MemoryMaintenanceWorker:
    global _worker
    if _worker is None:
        _worker = MemoryMaintenanceWorker(settings)
    return _worker
