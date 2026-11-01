"""Durable per-user background extraction and retryable Chroma indexing."""
import logging
import threading
import time
from datetime import datetime, timedelta
from sqlalchemy import update, case

from app.core.database import SessionLocal
from app.models.entities import LongTermMemory, LongTermMemoryJob, LongTermMemoryExtraction
from app.services.long_term_memory import LongTermMemoryService

logger = logging.getLogger(__name__)


class LongTermMemoryWorker:
    def __init__(self, settings, session_factory=None):
        self.settings = settings
        self.factory = session_factory or SessionLocal
        self.stop_event = threading.Event()
        self.thread = None
        self.last_index_attempt = {}
        self.last_source_recovery = 0.

    def recover_sources(self):
        with self.factory() as db:
            return LongTermMemoryService(db, self.settings).recover_unqueued()

    def start(self):
        if not self.settings.long_term_memory_enabled or self.thread:
            return
        with self.factory() as db:
            db.query(LongTermMemoryJob).filter_by(status='RUNNING').update({'status': 'PENDING'})
            db.commit()
        self.thread = threading.Thread(target=self._loop, name='personal-memory-worker', daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                if time.monotonic() - self.last_source_recovery >= 60:
                    self.recover_sources()
                    self.last_source_recovery = time.monotonic()
                self.dispatch_once()
            except Exception as exc:
                logger.warning('Personal memory worker failure code=%s', type(exc).__name__)
            self.stop_event.wait(self.settings.long_term_memory_poll_seconds)

    def dispatch_once(self):
        with self.factory() as db:
            job = db.query(LongTermMemoryJob).filter(LongTermMemoryJob.status == 'PENDING',
                LongTermMemoryJob.run_after <= datetime.utcnow()).order_by(LongTermMemoryJob.updated_at).first()
            if not job:
                # Index failures never discard SQL memory; retry without extracting source turns again.
                users = db.query(LongTermMemory.user_id).distinct().all() if self.settings.long_term_memory_vector_enabled else []
                for (user_id,) in users:
                    if time.monotonic() - self.last_index_attempt.get(user_id, -60) >= 60:
                        self.last_index_attempt[user_id] = time.monotonic()
                        LongTermMemoryService(db, self.settings).sync_vectors(user_id)
                return False
            job_id, user_id = job.id, job.user_id
            claimed = db.execute(update(LongTermMemoryJob).where(LongTermMemoryJob.id == job_id,
                LongTermMemoryJob.status == 'PENDING').values(status='RUNNING', attempts=LongTermMemoryJob.attempts + 1)).rowcount
            db.commit()
            if claimed != 1:
                return False
            try:
                service = LongTermMemoryService(db, self.settings)
                db.expire_all()
                job = db.get(LongTermMemoryJob, job_id)
                target = job.target_message_id
                done = {key for (key,) in db.query(LongTermMemoryExtraction.message_id).filter_by(user_id=user_id).all()}
                pending = [m for m in service.completed_inputs(user_id) if m.id <= target and m.id not in done]
                for message in pending[:20]:
                    if self.stop_event.is_set():
                        break
                    changed = service.extract(message)
                    db.add(LongTermMemoryExtraction(message_id=message.id, user_id=user_id, changed_count=changed))
                    # Source marker and memory edits commit atomically; failed extraction does not advance.
                    db.flush()
                    db.execute(update(LongTermMemoryJob).where(LongTermMemoryJob.id == job_id).values(
                        processed_message_id=case((LongTermMemoryJob.processed_message_id < message.id, message.id),
                            else_=LongTermMemoryJob.processed_message_id), updated_at=datetime.utcnow()))
                    db.commit()
                service.sync_vectors(user_id)
                self.last_index_attempt[user_id] = time.monotonic()
                # Preserve new targets enqueued during LLM calls. A bounded batch may leave older inputs too.
                more = len(pending) > 20 or self.stop_event.is_set()
                db.execute(update(LongTermMemoryJob).where(LongTermMemoryJob.id == job_id).values(
                    status='PENDING' if more else 'SUCCESS', attempts=0, last_error='', updated_at=datetime.utcnow()))
                db.execute(update(LongTermMemoryJob).where(LongTermMemoryJob.id == job_id,
                    LongTermMemoryJob.target_message_id > target).values(status='PENDING'))
                db.commit()
            except Exception as exc:
                db.rollback()
                job = db.get(LongTermMemoryJob, job_id)
                job.last_error = type(exc).__name__  # No model output, prompt, memory body or provider URL in audit.
                job.status = 'DEAD' if job.attempts >= self.settings.long_term_memory_max_attempts else 'PENDING'
                job.run_after = datetime.utcnow() + timedelta(seconds=5 * job.attempts)
                job.updated_at = datetime.utcnow()
                db.commit()
                logger.warning('Personal memory job=%s failed code=%s', job_id, type(exc).__name__)
            return True
