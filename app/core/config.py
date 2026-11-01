from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    campus_name: str = "西安电子科技大学"
    bundled_knowledge_enabled: bool = True
    bundled_knowledge_dir: str = "knowledge/xidian"
    agent_framework: str = "event_driven_multi_agent"
    agent_max_rounds: int = 8
    agent_max_claims_per_round: int = 4
    agent_max_claims_per_agent: int = 3
    agent_final_acceptance_min_confidence: float = 0.6
    agent_model_default_provider: str = ""
    agent_model_default_model: str = ""
    agent_model_coordinator_provider: str = ""
    agent_model_coordinator_model: str = ""
    agent_model_understanding_provider: str = ""
    agent_model_understanding_model: str = ""
    agent_model_safety_provider: str = ""
    agent_model_safety_model: str = ""
    agent_model_context_provider: str = ""
    agent_model_context_model: str = ""
    agent_model_response_provider: str = ""
    agent_model_response_model: str = ""
    ai_provider: str = "ollama"
    ai_temperature: float = 0.35
    ai_max_tokens: int = 512
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mindbridge-qwen2.5-7b-ft:latest"
    finetuned_model_name: str = "mindbridge-qwen2.5-7b-ft:latest"
    finetuned_model_dir: str = "models/mindbridge-qwen2.5-7b-ft"
    finetuned_model_file: str = "mindbridge-qwen2.5-7b-ft-q4_k_m.gguf"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    # 兼容供应商可选参数。None 不向官方/其他兼容接口发送该字段。
    openai_enable_thinking: bool | None = None
    judge_base_url: str = "https://api.openai.com/v1"
    judge_api_key: str = ""
    judge_model: str = "gpt-4o-mini"
    judge_temperature: float = 0.0
    judge_timeout_seconds: float = 90.0
    database_url: str = "mysql+pymysql://mindbridge:mindbridge@127.0.0.1:3306/mindbridge?charset=utf8mb4"
    chat_history_limit: int = 10
    knowledge_top_k: int = 5
    knowledge_candidate_k: int = 16
    knowledge_chunk_size: int = 300
    knowledge_chunk_overlap: int = 40
    knowledge_parent_chunk_size: int = 1000
    knowledge_parent_vector_weight: float = 0.30
    knowledge_child_vector_weight: float = 0.35
    knowledge_hybrid_vector_weight: float = 0.65
    knowledge_hybrid_bm25_weight: float = 0.35
    knowledge_rerank_enabled: bool = False
    knowledge_semantic_rerank_enabled: bool = False
    knowledge_semantic_rerank_model: str = ""
    knowledge_semantic_rerank_candidate_limit: int = 32
    knowledge_semantic_rerank_timeout_seconds: float = 30.
    policy_evidence_precision_enabled: bool = False
    knowledge_vector_enabled: bool = True
    knowledge_vector_required: bool = False
    chroma_persist_dir: str = "data/chroma"
    chroma_collection_name: str = "mindbridge_knowledge"
    chroma_snapshot_dir: str = "data/chroma-snapshots"
    chroma_snapshot_keep: int = 5
    embedding_timeout_seconds: float = 30.0
    rag_eval_dataset: str = "app/rag_eval/mindbridge-rag-eval.json"
    rag_eval_output: str = "target/rag-eval-report.json"
    rag_eval_enabled: bool = False
    rag_eval_exit_after_run: bool = False
    answer_eval_dataset: str = "app/evaluation/mindbridge-response-eval.jsonl"
    answer_eval_output: str = "target/answer-quality-eval-report.json"
    agent_eval_output: str = "target/multi-agent-comparison-report.json"
    excel_path: str = "data/mindbridge-risk-ledger.xlsx"
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_memory_ttl_seconds: int = 86400
    redis_memory_max_messages: int = 40
    redis_socket_timeout_seconds: float = 2.0
    memory_compaction_enabled: bool = True
    memory_compaction_recent_messages: int = 8
    memory_summary_max_chars: int = 500
    # ---- 滚动摘要（docs/context-memory-safety-redesign-spec.md 第 11 节）----
    memory_rolling_summary_enabled: bool = True
    memory_recent_target_messages: int = 8
    memory_compaction_trigger_messages: int = 12
    memory_summary_max_tokens: int = 600
    memory_summary_schema_version: int = 1
    memory_prompt_high_watermark_ratio: float = 0.65
    memory_prompt_target_ratio: float = 0.40
    model_context_window_tokens: int = 32768
    reserved_output_tokens: int = 1024
    rag_skill_budget_tokens: int = 4000
    memory_maintenance_queue_enabled: bool = True
    memory_maintenance_poll_interval_seconds: float = 1.0
    memory_maintenance_max_attempts: int = 3
    # ---- 安全事件记忆 ----
    safety_memory_enabled: bool = True
    safety_state_scope: Literal["session"] = "session"
    safety_state_cache_ttl_seconds: int = 900
    safety_monitoring_risk_floor: str = "MEDIUM"
    safety_review_interval_seconds: int = 86400
    safety_review_normal_turns: int = 10
    safety_high_to_monitoring_min_safe_assessments: int = 2
    safety_medium_to_low_min_consecutive_low: int = 2
    safety_low_to_resolved_min_additional_low: int = 1
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    smtp_timeout_seconds: float = 10.0
    alert_email_delivery_mode: str = "log"
    alert_email_from: str = ""
    alert_email_to: str = ""
    alert_email_subject_prefix: str = "[MindBridge 高风险预警]"
    tool_queue_enabled: bool = True
    tool_queue_poll_interval_seconds: float = 1.0
    tool_queue_batch_size: int = 10
    tool_queue_max_attempts: int = 3
    tool_queue_retry_delay_seconds: float = 15.0
    tool_queue_excel_workers: int = 1
    tool_queue_email_workers: int = 2
    alert_email_rate_limit_per_minute: int = 30

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]


@lru_cache
def get_settings() -> Settings:
    return Settings()
