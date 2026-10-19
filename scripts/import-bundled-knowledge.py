#!/usr/bin/env python3
"""Validate or import the committed official corpus without downloading sources."""
import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.services.knowledge_bundle import bundle_root, load_bundle, import_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只校验文件、来源及哈希；不连接数据库或模型")
    parser.add_argument("--no-vectors", action="store_true", help="仅导入 SQL/BM25，之后可在管理端重建向量")
    args = parser.parse_args()
    settings = get_settings().model_copy()
    if args.check:
        manifest, rows = load_bundle(bundle_root(settings))
        print(json.dumps({"school": manifest["school"], "documents": len(rows),
            "checkedOn": manifest["checked_on"], "status": "OK"}, ensure_ascii=False))
        return
    if args.no_vectors:
        settings.knowledge_vector_enabled = False
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from app.core.bootstrap import create_schema
    from app.core.database import SessionLocal
    from app.services.knowledge import KnowledgeService
    create_schema()
    with SessionLocal() as db:
        result = import_bundle(KnowledgeService(db, settings), settings)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
