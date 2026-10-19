from __future__ import annotations

import shutil
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from app.core.config import Settings
from app.models.entities import KnowledgeChunk


PRIMARY_RETRIEVAL_LABEL = "parent vector + child vector + child BM25 → parent fusion"
FALLBACK_RETRIEVAL_LABEL = "child BM25 → parent retrieval"


class VectorStoreUnavailable(RuntimeError):
    pass


@dataclass
class VectorSearchHit:
    chunk_id: int | None
    source: str
    source_index: int
    content: str
    score: float


class ChromaKnowledgeStore:
    """Parent/child embeddings from the configured compatible API, stored in Chroma."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.can_embed = False
        self.error = ""
        fingerprint = hashlib.sha256(f"pc-v1:{settings.openai_base_url}:{settings.openai_embedding_model}".encode()).hexdigest()[:12]
        self.index_key = f"{settings.chroma_collection_name}_{fingerprint}"
        self.collection_name = self.index_key
        if not settings.knowledge_vector_enabled:
            self.error = "Chroma 向量库未启用"
            return
        if not settings.openai_api_key:
            if settings.knowledge_vector_required:
                raise VectorStoreUnavailable("缺少 OPENAI_API_KEY，无法启用 Chroma + configured embeddings 主检索方案")
            self.error = f"缺少 OPENAI_API_KEY，Chroma + configured embeddings 不可用，已回退到{FALLBACK_RETRIEVAL_LABEL}"
            return
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            if settings.knowledge_vector_required:
                raise VectorStoreUnavailable("缺少 chromadb 依赖，无法启用 Chroma + configured embeddings 主检索方案") from exc
            self.error = f"缺少 chromadb 依赖，Chroma + configured embeddings 不可用，已回退到{FALLBACK_RETRIEVAL_LABEL}"
            return

        try:
            persist_dir = self._resolve_path(settings.chroma_persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            self.persist_dir = persist_dir
            self.manifest = persist_dir / f"{self.index_key}.json"
            if self.manifest.exists():
                self.collection_name = json.loads(self.manifest.read_text())["collection"]
            self.client = chromadb.PersistentClient(path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine", "embedding_model": settings.openai_embedding_model},
            )
            self.can_embed = True
        except Exception as exc:
            self.error = f"向量库初始化失败（{type(exc).__name__}），已回退到{FALLBACK_RETRIEVAL_LABEL}"
            if settings.knowledge_vector_required:
                raise VectorStoreUnavailable("向量库初始化失败，required 模式不允许降级。") from exc

    def upsert_chunks(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> int:
        self._refresh_collection()
        rows = [chunk for chunk in chunks if chunk.id is not None and chunk.content.strip()]
        if not rows:
            return 0
        return self._upsert(self.collection, rows, embeddings)

    def _upsert(self, collection, rows, embeddings) -> int:
        ids = [self._id(chunk.id) for chunk in rows]
        documents = [chunk.content for chunk in rows]
        metadatas = [
            {"db_id": int(chunk.id), "source": chunk.source, "source_index": int(chunk.source_index),
             "chunk_type": chunk.chunk_type, "parent_id": chunk.parent_id or 0,
             "document_id": chunk.document_id or 0, "domain": chunk.domain,
             "school": chunk.school, "version": chunk.version,
             "content_hash": hashlib.sha256(chunk.embedding_text.encode()).hexdigest()}
            for chunk in rows
        ]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        return len(rows)

    def sync_chunks(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> int:
        # 完整构建并检查后，原子切换 manifest；失败时保留原索引和 pointer。
        name = f"{self.index_key}_{uuid.uuid4().hex[:8]}"
        staged = self.client.create_collection(name=name, embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_model": self.settings.openai_embedding_model})
        try:
            count = self._upsert(staged, chunks, embeddings) if chunks else 0
            if staged.count() != len(chunks):
                raise VectorStoreUnavailable("新索引块数校验失败，未切换索引。")
            temp = self.manifest.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temp.write_text(json.dumps({"collection": name, "model": self.settings.openai_embedding_model}))
            temp.replace(self.manifest)
            self.collection, self.collection_name = staged, name
            return count
        except Exception:
            self.client.delete_collection(name)
            raise

    def has_exact_chunk_ids(self, chunks: list[KnowledgeChunk]) -> bool:
        self._refresh_collection()
        expected = {self._id(int(c.id)): hashlib.sha256(c.embedding_text.encode()).hexdigest() for c in chunks if c.id is not None}
        current = self.collection.get(include=["metadatas"])
        actual = {key: metadata.get("content_hash") for key, metadata in zip(current.get("ids", []), current.get("metadatas", []))}
        return expected == actual

    def _refresh_collection(self) -> None:
        if self.manifest.exists():
            name = json.loads(self.manifest.read_text())["collection"]
            if name != self.collection_name:
                self.collection = self.client.get_collection(name, embedding_function=None)
                self.collection_name = name

    def delete_source(self, source: str) -> None:
        if not self.can_embed:
            return
        self._refresh_collection()
        self.collection.delete(where={"source": source})

    def delete_chunk_ids(self, chunk_ids: list[int]) -> None:
        if not self.can_embed or not chunk_ids:
            return
        self._refresh_collection()
        self.collection.delete(ids=[self._id(chunk_id) for chunk_id in chunk_ids])

    def query(self, query_embedding: list[float], top_k: int, *, chunk_type: str | None = None,
              allowed_ids: list[int] | None = None) -> list[VectorSearchHit]:
        self._refresh_collection()
        if allowed_ids == [] or not self.collection.count():
            return []
        filters = []
        if chunk_type:
            filters.append({"chunk_type": chunk_type})
        if allowed_ids is not None:
            filters.append({"db_id": {"$in": allowed_ids}})
        where = ({"$and": filters} if len(filters) > 1 else filters[0]) if filters else None
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            **({"where": where} if where else {}),
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            hits.append(
                VectorSearchHit(
                    chunk_id=int(metadata["db_id"]) if metadata.get("db_id") is not None else None,
                    source=str(metadata.get("source", "")),
                    source_index=int(metadata.get("source_index", 0)),
                    content=document or "",
                    score=1.0 / (1.0 + max(0.0, distance)),
                )
            )
        return hits

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.can_embed:
            raise VectorStoreUnavailable(self.error or "Chroma + configured embeddings 主检索方案不可用")
        return self._embed(texts)

    def snapshot(self) -> str | None:
        if not self.can_embed:
            return None
        if not self.persist_dir.exists():
            return None
        snapshot_root = self._resolve_path(self.settings.chroma_snapshot_dir)
        snapshot_root.mkdir(parents=True, exist_ok=True)
        destination = snapshot_root / datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        shutil.copytree(self.persist_dir, destination)
        self._prune_snapshots(snapshot_root)
        return str(destination)

    def count(self) -> int:
        if not self.can_embed:
            return 0
        self._refresh_collection()
        return int(self.collection.count())

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # 阿里云百炼 text-embedding-v4 单次请求最多接受 10 条 input，
        # 因此按批拆分；OpenAI 官方接口无此限制，分批同样兼容。
        batch_size = 10
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = [text if text.strip() else " " for text in texts[start : start + batch_size]]
            payload = {
                "model": self.settings.openai_embedding_model,
                "input": batch,
            }
            headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
            response = httpx.post(
                f"{self.settings.openai_base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=self.settings.embedding_timeout_seconds,
            )
            response.raise_for_status()
            rows = sorted(response.json().get("data", []), key=lambda item: item.get("index", 0))
            embeddings.extend(row.get("embedding") for row in rows)
        if len(embeddings) != len(texts) or any(not embedding for embedding in embeddings):
            raise VectorStoreUnavailable("OpenAI embeddings 接口返回向量数量不匹配")
        return [[float(value) for value in embedding] for embedding in embeddings]

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.settings.project_root / path

    def _prune_snapshots(self, snapshot_root: Path) -> None:
        keep = max(1, self.settings.chroma_snapshot_keep)
        snapshots = sorted([path for path in snapshot_root.iterdir() if path.is_dir()], reverse=True)
        for stale in snapshots[keep:]:
            shutil.rmtree(stale, ignore_errors=True)

    def _id(self, chunk_id: int) -> str:
        return f"knowledge-chunk-{chunk_id}"


ChromaKnowledgeVectorStore = ChromaKnowledgeStore
