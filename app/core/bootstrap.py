from pathlib import Path
import logging

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import Base, engine
from app.core.security import hash_password
from app.models.entities import UserAccount
from app.services.knowledge import KnowledgeService


# create_all() 只创建缺失的新表，不会给已有表补列；这里维护幂等的轻量列迁移。
# 列迁移项按 (表名, 列名, DDL) 登记，执行前先查 information_schema，可重复执行。
COLUMN_MIGRATIONS: list[tuple[str, str, str]] = [
    ("knowledge_documents", "chunking_version", "ALTER TABLE knowledge_documents ADD COLUMN chunking_version VARCHAR(96) NOT NULL DEFAULT ''"),
    ("knowledge_chunks", "document_id", "ALTER TABLE knowledge_chunks ADD COLUMN document_id INT NULL"),
    ("knowledge_chunks", "parent_id", "ALTER TABLE knowledge_chunks ADD COLUMN parent_id INT NULL"),
    ("knowledge_chunks", "chunk_type", "ALTER TABLE knowledge_chunks ADD COLUMN chunk_type VARCHAR(16) NOT NULL DEFAULT 'legacy'"),
    ("knowledge_chunks", "heading", "ALTER TABLE knowledge_chunks ADD COLUMN heading VARCHAR(512) NOT NULL DEFAULT ''"),
    ("knowledge_chunks", "page_start", "ALTER TABLE knowledge_chunks ADD COLUMN page_start INT NULL"),
    ("knowledge_chunks", "domain", "ALTER TABLE knowledge_chunks ADD COLUMN domain VARCHAR(32) NOT NULL DEFAULT 'support'"),
    ("knowledge_chunks", "school", "ALTER TABLE knowledge_chunks ADD COLUMN school VARCHAR(128) NOT NULL DEFAULT '本校'"),
    ("knowledge_chunks", "version", "ALTER TABLE knowledge_chunks ADD COLUMN version VARCHAR(64) NOT NULL DEFAULT '1'"),
    ("knowledge_chunks", "idx_knowledge_parent", "CREATE INDEX idx_knowledge_parent ON knowledge_chunks (parent_id)"),
    ("knowledge_chunks", "idx_knowledge_document", "CREATE INDEX idx_knowledge_document ON knowledge_chunks (document_id)"),
    ("psychological_reports", "message_id", "ALTER TABLE psychological_reports ADD COLUMN message_id INT NULL"),
    ("psychological_reports", "idx_reports_message_id", "CREATE INDEX idx_reports_message_id ON psychological_reports (message_id)"),
    ("chat_messages", "client_message_id", "ALTER TABLE chat_messages ADD COLUMN client_message_id VARCHAR(128) NULL"),
    ("chat_messages", "idx_chat_messages_client_message_id", "CREATE INDEX idx_chat_messages_client_message_id ON chat_messages (client_message_id)"),
    ("chat_messages", "reply_to_message_id", "ALTER TABLE chat_messages ADD COLUMN reply_to_message_id INT NULL"),
    ("chat_messages", "idx_chat_messages_reply_to_message_id", "CREATE INDEX idx_chat_messages_reply_to_message_id ON chat_messages (reply_to_message_id)"),
    ("chat_messages", "agent_trace_id", "ALTER TABLE chat_messages ADD COLUMN agent_trace_id INT NULL"),
    ("chat_messages", "idx_chat_messages_agent_trace_id", "CREATE INDEX idx_chat_messages_agent_trace_id ON chat_messages (agent_trace_id)"),
    ("chat_messages", "completion_status", "ALTER TABLE chat_messages ADD COLUMN completion_status VARCHAR(32) NOT NULL DEFAULT 'COMPLETE'"),
    ("chat_messages", "idx_chat_messages_completion_status", "CREATE INDEX idx_chat_messages_completion_status ON chat_messages (completion_status)"),
    ("chat_messages", "response_dedupe_key", "ALTER TABLE chat_messages ADD COLUMN response_dedupe_key VARCHAR(160) NULL"),
    ("chat_messages", "uq_chat_messages_response_dedupe_key", "CREATE UNIQUE INDEX uq_chat_messages_response_dedupe_key ON chat_messages (response_dedupe_key)"),
    ("chat_messages", "uq_chat_messages_user_client_message", "CREATE UNIQUE INDEX uq_chat_messages_user_client_message ON chat_messages (user_id, client_message_id)"),
    # trace 列 Text(64KB) 不够存完整协作事件/候选 prompt，升级为 MEDIUMTEXT(16MB)
    ("agent_run_traces", "agent_steps_json", "ALTER TABLE agent_run_traces MODIFY agent_steps_json MEDIUMTEXT NULL"),
    ("agent_run_traces", "retrieved_knowledge_json", "ALTER TABLE agent_run_traces MODIFY retrieved_knowledge_json MEDIUMTEXT NULL"),
    ("agent_run_traces", "response_messages_json", "ALTER TABLE agent_run_traces MODIFY response_messages_json MEDIUMTEXT NULL"),
    ("agent_run_traces", "assessment_json", "ALTER TABLE agent_run_traces MODIFY assessment_json MEDIUMTEXT NULL"),
    # 规格第 14.16/19 节：trace 记录记忆与安全上下文版本
    ("agent_run_traces", "current_risk", "ALTER TABLE agent_run_traces ADD COLUMN current_risk VARCHAR(32) NULL"),
    ("agent_run_traces", "carryover_risk", "ALTER TABLE agent_run_traces ADD COLUMN carryover_risk VARCHAR(32) NULL"),
    ("agent_run_traces", "effective_risk", "ALTER TABLE agent_run_traces ADD COLUMN effective_risk VARCHAR(32) NULL"),
    ("agent_run_traces", "response_mode", "ALTER TABLE agent_run_traces ADD COLUMN response_mode VARCHAR(32) NULL"),
    ("agent_run_traces", "summary_version", "ALTER TABLE agent_run_traces ADD COLUMN summary_version INT NULL"),
    ("agent_run_traces", "covered_through_message_id", "ALTER TABLE agent_run_traces ADD COLUMN covered_through_message_id INT NULL"),
    ("agent_run_traces", "safety_state_version", "ALTER TABLE agent_run_traces ADD COLUMN safety_state_version INT NULL"),
    ("agent_run_traces", "snapshot_degraded", "ALTER TABLE agent_run_traces ADD COLUMN snapshot_degraded TINYINT(1) NULL"),
    ("risk_events", "dedupe_key", "ALTER TABLE risk_events ADD COLUMN dedupe_key VARCHAR(160) NULL"),
    ("risk_events", "uq_risk_events_dedupe_key", "CREATE UNIQUE INDEX uq_risk_events_dedupe_key ON risk_events (dedupe_key)"),
]


def create_schema() -> None:
    Base.metadata.create_all(bind=engine)
    _apply_column_migrations()


def _apply_column_migrations() -> None:
    if engine.dialect.name != "mysql":
        # 新测试库由 create_all 建立；也允许已有 SQLite 开发库增补知识列。
        if engine.dialect.name == "sqlite":
            from sqlalchemy import inspect
            with engine.begin() as connection:
                for knowledge_table in ("knowledge_chunks", "knowledge_documents"):
                    columns = {c["name"] for c in inspect(engine).get_columns(knowledge_table)}
                    for table, column, ddl in COLUMN_MIGRATIONS:
                        if table == knowledge_table and "ADD COLUMN" in ddl and column not in columns:
                            connection.execute(text(ddl))
                message_columns = {c["name"] for c in inspect(engine).get_columns("chat_messages")}
                if "agent_trace_id" not in message_columns:
                    connection.execute(text("ALTER TABLE chat_messages ADD COLUMN agent_trace_id INT NULL"))
                    connection.execute(text("CREATE INDEX idx_chat_messages_agent_trace_id ON chat_messages (agent_trace_id)"))
        return
    with engine.connect() as connection:
        index_ddls: list[tuple[str, str, str]] = []
        # 第一遍：列的增补与类型迁移
        for table, column, ddl in COLUMN_MIGRATIONS:
            if ddl.upper().startswith("CREATE INDEX") or ddl.upper().startswith("CREATE UNIQUE INDEX"):
                index_ddls.append((table, column, ddl))
                continue
            if ddl.upper().startswith("ALTER TABLE") and "MODIFY" in ddl.upper():
                # 类型升级：只有当列已经是我们期望的类型时才跳过
                exists = connection.execute(
                    text(
                        "SELECT COUNT(*) FROM information_schema.columns "
                        "WHERE table_schema = DATABASE() AND table_name = :table AND column_name = :name "
                        "AND data_type = 'mediumtext'"
                    ),
                    {"table": table, "name": column},
                ).scalar()
            else:
                exists = connection.execute(
                    text(
                        "SELECT COUNT(*) FROM information_schema.columns "
                        "WHERE table_schema = DATABASE() AND table_name = :table AND column_name = :name"
                    ),
                    {"table": table, "name": column},
                ).scalar()
            if not exists:
                connection.execute(text(ddl))

        # 第二遍：唯一索引建立前先整理历史重复键（保留最早一条，其余置 NULL，
        # 消息内容不删除），否则旧部署环境会因 Duplicate entry 启动失败。
        _nullify_duplicate_keys(connection, "chat_messages", ["user_id", "client_message_id"], "client_message_id")
        _nullify_duplicate_keys(connection, "chat_messages", ["response_dedupe_key"], "response_dedupe_key")
        _nullify_duplicate_keys(connection, "risk_events", ["dedupe_key"], "dedupe_key")

        # 第三遍：索引
        for table, column, ddl in index_ddls:
            exists = connection.execute(
                text(
                    "SELECT COUNT(*) FROM information_schema.statistics "
                    "WHERE table_schema = DATABASE() AND table_name = :table AND index_name = :name"
                ),
                {"table": table, "name": column},
            ).scalar()
            if not exists:
                connection.execute(text(ddl))
        connection.commit()


def _nullify_duplicate_keys(connection, table: str, key_columns: list[str], target_column: str) -> int:
    """同组唯一键出现多次时，仅保留最早一条的键值，其余置 NULL（幂等）。"""
    partition = ", ".join(key_columns)
    result = connection.execute(
        text(
            f"UPDATE {table} AS t "
            f"JOIN ("
            f"  SELECT id FROM ("
            f"    SELECT id, ROW_NUMBER() OVER (PARTITION BY {partition} ORDER BY id ASC) AS rn "
            f"    FROM {table} WHERE {target_column} IS NOT NULL"
            f"  ) ranked WHERE ranked.rn > 1"
            f") duplicates ON t.id = duplicates.id "
            f"SET t.{target_column} = NULL"
        )
    )
    return result.rowcount


def seed_data(db: Session) -> None:
    from app.services.safety_memory import SafetyMemoryService
    SafetyMemoryService(db, get_settings()).restore_legacy_sessions()
    if db.query(UserAccount).count() == 0:
        admin = UserAccount(
            username="admin",
            display_name="Counselor Admin",
            password_hash=hash_password("admin123"),
        )
        admin.roles = {"ROLE_ADMIN", "ROLE_USER"}
        student = UserAccount(
            username="student",
            display_name="Demo Student",
            password_hash=hash_password("student123"),
        )
        student.roles = {"ROLE_USER"}
        db.add_all([admin, student])
        db.commit()

    service = KnowledgeService(db, get_settings())
    service.remove_demo_documents()
    root = Path(__file__).resolve().parents[1]
    service.migrate_legacy_chunks()
    for file in sorted((root / "knowledge").glob("*.md")):
        service.ensure_source(file.name, file.read_text(encoding="utf-8"), domain="support", school=get_settings().campus_name)
    from app.services.knowledge_bundle import import_bundle
    try:
        result = import_bundle(service, get_settings())
        logging.getLogger(__name__).info("Official knowledge bundle: %s", result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # 仍允许进入管理端补充文件；缺资料的咨询继续给出明确提示。
        logging.getLogger(__name__).error("官方资料包导入失败，请恢复 knowledge/xidian 或手动导入：%s", exc)
    migrated = service.rechunk_documents()
    if migrated["documents"]:
        logging.getLogger(__name__).info("Knowledge chunk migration: %s", migrated)
