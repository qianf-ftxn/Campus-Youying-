#!/usr/bin/env python3
"""Explicit, repeatable import of existing completed app conversations; prints metadata only."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import get_settings
from app.core.bootstrap import create_schema
from app.core.database import SessionLocal
from app.models.entities import LongTermMemory, LongTermMemoryExtraction, LongTermMemoryJob
from app.services.long_term_memory import LongTermMemoryService
from app.services.long_term_worker import LongTermMemoryWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drain', action='store_true', help='Also process pending jobs now; stop the app worker first.')
    parser.add_argument('--recheck', action='store_true', help='Re-evaluate previously checked source turns after an extraction-rule change; keep existing memories.')
    args = parser.parse_args()
    settings = get_settings()
    if not settings.long_term_memory_enabled:
        raise SystemExit('Long-term memory is disabled.')
    create_schema()
    with SessionLocal() as db:
        if args.recheck:
            db.query(LongTermMemoryExtraction).delete()
            db.commit()
        users = LongTermMemoryService(db, settings).enqueue_history()
        print(f'Enqueued users: {users}', flush=True)
    if args.drain:
        worker = LongTermMemoryWorker(settings)
        while True:
            dispatched = worker.dispatch_once()
            with SessionLocal() as db:
                print({'processedTurns': db.query(LongTermMemoryExtraction).count(),
                       'memoryCount': db.query(LongTermMemory).count()}, flush=True)
                waiting = db.query(LongTermMemoryJob).filter_by(status='PENDING').count()
            if not dispatched and not waiting:
                break
            if not dispatched:
                time.sleep(1)  # Includes MySQL DATETIME rounding and bounded retry backoff.
        with SessionLocal() as db:
            unfinished = db.query(LongTermMemoryJob).filter(LongTermMemoryJob.status.in_(['PENDING', 'RUNNING', 'DEAD'])).count()
            if unfinished:
                raise SystemExit(f'{unfinished} jobs need retry; progress is saved. See status endpoint.')


if __name__ == '__main__':
    main()
