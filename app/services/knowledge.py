from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from zoneinfo import ZoneInfo
from typing import Hashable

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.entities import KnowledgeChunk, KnowledgeDocument
from app.services.vector_store import FALLBACK_RETRIEVAL_LABEL, PRIMARY_RETRIEVAL_LABEL, ChromaKnowledgeStore

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: int | None
    source: str
    content: str
    score: float
    title: str = ""
    source_url: str = ""
    version: str = ""
    domain: str = ""
    heading: str = ""
    page_start: int | None = None
    is_demo: bool = False
    document_id: int | None = None
    matched_by: list[str] = field(default_factory=list)
    matched_chunks: list[int] = field(default_factory=list)

    def citation(self) -> dict:
        return {"chunkId": self.chunk_id, "documentId": self.document_id, "source": self.source,
                "title": self.title or self.source, "url": self.source_url, "version": self.version,
                "heading": self.heading, "page": self.page_start, "isDemo": self.is_demo}


class KnowledgeService:
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.vector_store = ChromaKnowledgeStore(settings)
        self.last_retrieval: dict = {}

    def count(self) -> int:
        return self.db.query(KnowledgeChunk).count()

    def remove_demo_documents(self) -> int:
        """移除旧演示文档及父子块；保留真实上传资料和对话审计。"""
        documents = self.db.query(KnowledgeDocument).filter_by(is_demo=True).all()
        ids = [d.id for d in documents]
        if not ids:
            return 0
        chunks = self.db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id.in_(ids))
        chunk_ids = [row.id for row in chunks.all()]
        try:
            # MySQL 自引用外键要求先删除子块，再删除父块。
            chunks.filter(KnowledgeChunk.parent_id.isnot(None)).delete(synchronize_session=False)
            chunks.delete(synchronize_session=False)
            self.db.query(KnowledgeDocument).filter(KnowledgeDocument.id.in_(ids)).delete(synchronize_session=False)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        if self.vector_store.can_embed:
            try:
                self.vector_store.delete_chunk_ids(chunk_ids)
            except Exception as exc:
                # SQL 允许列表已排除这些块；旧向量不会被引用，下一次同步会清理。
                logger.warning("Demo vector cleanup deferred: %s", type(exc).__name__)
        return len(ids)

    def ensure_source(self, source: str, content: str, **metadata) -> int:
        digest = hashlib.sha256(content.encode()).hexdigest()
        existing = self.db.query(KnowledgeDocument).filter_by(source=source, content_hash=digest).first()
        if existing:
            # 默认资料显式跟随校园配置；不重新启用已经归档的版本。
            school = metadata.get("school")
            if school and existing.school != school:
                existing.school = school
                self.db.query(KnowledgeChunk).filter_by(document_id=existing.id).update({"school": school})
                self.db.commit()
            return self.db.query(KnowledgeChunk).filter_by(document_id=existing.id).count()
        return self.ingest(source, content, **metadata)

    def ingest(self, source: str, content: str, *, domain: str = "policy", title: str = "",
               version: str | None = None, school: str | None = None, department: str = "",
               source_url: str = "", published_on: date | None = None, effective_on: date | None = None,
               expires_on: date | None = None, status: str = "ACTIVE", is_demo: bool = False) -> int:
        if not source.strip() or not content.strip():
            raise ValueError("文件来源和正文不能为空。")
        if domain not in {"policy", "support"} or status not in {"ACTIVE", "DRAFT", "ARCHIVED"}:
            raise ValueError("知识领域或文档状态无效。")
        if source_url and not source_url.startswith(("https://", "http://")):
            raise ValueError("原文链接只支持 http/https。")
        if expires_on and effective_on and expires_on < effective_on:
            raise ValueError("失效日期不能早于生效日期。")
        if len(content.encode("utf-8")) > 10 * 1024 * 1024:
            raise ValueError("单份文本最多 10 MB，请按独立政策文件拆分。")
        digest = hashlib.sha256(content.encode()).hexdigest()
        version = version or digest[:12]
        existing = self.db.query(KnowledgeDocument).filter_by(source=source, version=version).first()
        if existing:
            if existing.school != (school or self.settings.campus_name) or existing.domain != domain:
                raise ValueError("该来源和版本已属于其他学校或领域，请使用不同来源名或版本号。")
            if existing.content_hash != digest:
                raise ValueError("该版本已存在且内容不同，请使用新的版本号。")
            return self.db.query(KnowledgeChunk).filter_by(document_id=existing.id).count()
        # 待生效文件先保存为草稿，避免提前取代当前适用版本。
        today = campus_today()
        if effective_on and effective_on > today and status == "ACTIVE":
            status = "DRAFT"
        if expires_on and expires_on < today and status == "ACTIVE":
            status = "ARCHIVED"
        sections = parent_sections(content, self.settings.knowledge_parent_chunk_size)
        if not sections:
            raise ValueError("未提取到有效正文，请补充条款内容后再导入。")
        title = (title or first_title(content) or source)[:256]
        document = KnowledgeDocument(source=source, title=title, version=version, domain=domain,
            school=school or self.settings.campus_name, department=department, source_url=source_url,
            published_on=published_on, effective_on=effective_on, expires_on=expires_on,
            status=status, is_demo=is_demo, content=content, content_hash=digest)
        try:
            self.db.add(document)
            self.db.flush()
            rows = []
            for index, (heading, body, page) in enumerate(sections):
                parent = KnowledgeChunk(source=source, source_index=index, content=body, document_id=document.id,
                    chunk_type="parent", heading=f"{title} / {heading}"[:512], page_start=page,
                    domain=domain, school=document.school, version=version)
                self.db.add(parent)
                self.db.flush()
                rows.append(parent)
                for child_index, child in enumerate(child_sections(body, self.settings.knowledge_chunk_size,
                                                                         self.settings.knowledge_chunk_overlap)):
                    row = KnowledgeChunk(source=source, source_index=child_index, content=child,
                        document_id=document.id, parent_id=parent.id, chunk_type="child", heading=parent.heading,
                        page_start=page, domain=domain, school=document.school, version=version)
                    self.db.add(row)
                    rows.append(row)
            if status == "ACTIVE":
                self._archive_other_versions(document)
            self.db.flush()
            # SQL 的事务尚未切换版本；索引失败（required 模式）不能激活半成品。
            self._index_vector_chunks(rows)
            self.db.commit()
            return len(rows)
        except Exception:
            self.db.rollback()
            raise

    def _archive_other_versions(self, document: KnowledgeDocument) -> None:
        self.db.query(KnowledgeDocument).filter(KnowledgeDocument.source == document.source,
            KnowledgeDocument.id != document.id, KnowledgeDocument.school == document.school,
            KnowledgeDocument.domain == document.domain, KnowledgeDocument.status == "ACTIVE").update(
                {"status": "ARCHIVED"}, synchronize_session="fetch")

    def set_document_status(self, document_id: int, status: str) -> dict:
        document = self.db.get(KnowledgeDocument, document_id)
        if document is None:
            raise LookupError("文档不存在。")
        if status not in {"ACTIVE", "DRAFT", "ARCHIVED"}:
            raise ValueError("文档状态无效。")
        if status == "ACTIVE":
            if document.effective_on and document.effective_on > campus_today():
                raise ValueError("文件尚未生效，不能启用。")
            if document.expires_on and document.expires_on < campus_today():
                raise ValueError("文件已失效，不能启用。")
            if not self.db.query(KnowledgeChunk).filter_by(document_id=document.id, chunk_type="parent").first():
                raise ValueError("文档没有有效正文，不能启用。")
            self._archive_other_versions(document)
        document.status = status
        self.db.commit()
        return self.document_detail(document_id)

    def migrate_legacy_chunks(self) -> int:
        rows = self.db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id.is_(None)).order_by(
            KnowledgeChunk.source, KnowledgeChunk.source_index).all()
        grouped: dict[str, list[KnowledgeChunk]] = {}
        for row in rows:
            grouped.setdefault(row.source, []).append(row)
        for source, chunks in grouped.items():
            # 旧块有重叠，重建时去掉相邻块共有的后缀/前缀，原表记录暂不删除。
            text = chunks[0].content
            for item in chunks[1:]:
                shared = max((n for n in range(1, min(len(text), len(item.content), 256) + 1)
                              if text.endswith(item.content[:n])), default=0)
                text += item.content[shared:]
            # 旧版 chunk_text 会把换行压成空格。若整份 Markdown 只剩一行，
            # 开头的 # 会使所有正文被误认作标题；按恢复的纯文本保留全文。
            if "\n" not in text.strip():
                text = re.sub(r"^#{1,6}\s+", "", text.strip())
            self.ensure_source(source, text, domain=chunks[0].domain or "support")
        return len(grouped)

    def ingest_file(self, filename: str, data: bytes, **metadata) -> int:
        if len(data) > 10 * 1024 * 1024 or len(filename) > 256:
            raise ValueError("文件不能超过 10 MB，文件名不能超过 256 字符。")
        suffix = filename.lower().rsplit(".", 1)[-1]
        try:
            if suffix == "pdf":
                text = extract_pdf(data)
            elif suffix in {"txt", "md", "markdown"}:
                text = data.decode("utf-8-sig", errors="strict")
            else:
                raise ValueError("仅支持文字型 PDF、Markdown 和 UTF-8 文本文件。")
        except (PdfReadError, UnicodeError) as exc:
            raise ValueError("无法解析文件，请确认 PDF 完整可读或文本使用 UTF-8 编码。") from exc
        if not text.strip():
            raise ValueError("未提取到正文；扫描 PDF 需要先做 OCR。")
        return self.ingest(filename, text, **metadata)

    def _eligible_documents(self, domain: str | None, school: str | None) -> list[KnowledgeDocument]:
        today = campus_today()
        query = self.db.query(KnowledgeDocument).filter(KnowledgeDocument.status == "ACTIVE")
        if domain:
            query = query.filter(KnowledgeDocument.domain == domain)
        if school:
            query = query.filter(KnowledgeDocument.school == school)
        return [d for d in query.all() if (not d.effective_on or d.effective_on <= today)
                and (not d.expires_on or d.expires_on >= today)]

    def retrieve(self, query: str, top_k: int | None = None, *, domain: str | None = None,
                 school: str | None = None) -> list[SearchResult]:
        from time import perf_counter
        started = perf_counter()
        top_k = top_k or self.settings.knowledge_top_k
        documents = {d.id: d for d in self._eligible_documents(domain, school)}
        rows = self.db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id.in_(documents),
            KnowledgeChunk.chunk_type.in_(["parent", "child"])).all() if documents else []
        parents = {r.id: r for r in rows if r.chunk_type == "parent"}
        children = [r for r in rows if r.chunk_type == "child"]
        candidate_k = max(top_k, self.settings.knowledge_candidate_k)
        channels: dict[str, dict[int, float]] = {}
        matches: dict[int, set[int]] = {}
        weights = {"parent_vector": self.settings.knowledge_parent_vector_weight,
                   "child_vector": self.settings.knowledge_child_vector_weight,
                   "child_bm25": self.settings.knowledge_hybrid_bm25_weight}
        vector_error = ""
        if rows and self.vector_store.can_embed and any(weights[key] > 0 for key in ("parent_vector", "child_vector")):
            try:
                self._ensure_vector_index()
                embedding = self.vector_store.embed_texts([query])[0]
                by_id = {r.id: r for r in rows}
                for kind, allowed in (("parent", list(parents)), ("child", [c.id for c in children])):
                    if weights[f"{kind}_vector"] <= 0:
                        continue
                    raw: dict[int, float] = {}
                    if allowed:
                        for hit in self.vector_store.query(embedding, candidate_k, chunk_type=kind, allowed_ids=allowed):
                            row = by_id.get(hit.chunk_id)
                            if row is None:
                                continue
                            pid = row.id if kind == "parent" else row.parent_id
                            raw[pid] = max(raw.get(pid, 0), hit.score)
                            matches.setdefault(pid, set()).add(row.id)
                    channels[f"{kind}_vector"] = raw
            except Exception as exc:
                self._handle_vector_error("retrieve", exc)
                # 同一请求的任一路向量召回失败，完整回退，避免混入半次召回的证据。
                channels.clear()
                matches.clear()
                vector_error = f"{type(exc).__name__}: {exc}"
        scores = bm25_scores(query, children) if weights["child_bm25"] > 0 else {}
        bm: dict[int, float] = {}
        for row in sorted(children, key=lambda c: scores.get(c.id, 0), reverse=True)[:candidate_k]:
            score = scores.get(row.id, 0)
            if score > 0:
                bm[row.parent_id] = max(bm.get(row.parent_id, 0), score)
                matches.setdefault(row.parent_id, set()).add(row.id)
        channels["child_bm25"] = bm
        active = {key: normalize_scores(value) for key, value in channels.items() if value and weights[key] > 0}
        total_weight = sum(max(0, weights[key]) for key in active) or 1.0
        results = []
        for pid in {pid for value in active.values() for pid in value}:
            parent = parents.get(pid)
            if parent is None:
                continue
            score = sum(max(0, weights[key]) * value.get(pid, 0) for key, value in active.items()) / total_weight
            doc = documents[parent.document_id]
            if self.settings.knowledge_rerank_enabled:
                score = rerank_score(query, parent.embedding_text, score)
            results.append(SearchResult(parent.id, parent.source, parent.content, score,
                doc.title, doc.source_url, doc.version, doc.domain, parent.heading, parent.page_start,
                doc.is_demo, doc.id, [key for key, value in channels.items() if pid in value],
                sorted(matches.get(pid, set()))))
        results.sort(key=lambda item: (-item.score, item.chunk_id or 0))
        results = results[:top_k]
        self.last_retrieval = {"query": query, "domain": domain, "school": school,
            "path": "hybrid_parent_child" if any(k.endswith("vector") for k in active) else "bm25_parent_child",
            "vectorError": vector_error or getattr(self.vector_store, "error", ""),
            "channels": {key: {str(pid): score for pid, score in value.items()} for key, value in channels.items()},
            "selectedParents": [r.chunk_id for r in results], "rerankEnabled": self.settings.knowledge_rerank_enabled,
            "elapsedMs": round((perf_counter() - started) * 1000, 2)}
        return results

    def _all_index_rows(self) -> list[KnowledgeChunk]:
        return self.db.query(KnowledgeChunk).filter(KnowledgeChunk.chunk_type.in_(["parent", "child"])).all()

    def _ensure_vector_index(self) -> None:
        rows = self._all_index_rows()
        if not self.vector_store.has_exact_chunk_ids(rows):
            self._sync_vector_chunks(rows)

    def _index_vector_chunks(self, rows: list[KnowledgeChunk]) -> None:
        if not rows or not self.vector_store.can_embed:
            return
        try:
            self.vector_store.upsert_chunks(rows, self._embeddings_for_chunks(rows))
        except Exception as exc:
            self._handle_vector_error("index", exc)

    def _embeddings_for_chunks(self, rows: list[KnowledgeChunk]) -> list[list[float]]:
        # JSON 缓存只用于当前 embedding 模型；模型更换时重建会强制重新计算。
        embeddings = self.vector_store.embed_texts([r.embedding_text for r in rows])
        for row, embedding in zip(rows, embeddings):
            row.embedding_json = json.dumps(embedding, separators=(",", ":"))
        return embeddings

    def _sync_vector_chunks(self, rows: list[KnowledgeChunk]) -> None:
        self.vector_store.sync_chunks(rows, self._embeddings_for_chunks(rows) if rows else [])

    def rebuild_vector_index(self) -> int:
        if not self.vector_store.can_embed:
            raise RuntimeError(self.vector_store.error or "向量服务不可用。")
        rows = self._all_index_rows()
        self._sync_vector_chunks(rows)
        self.db.commit()
        return len(rows)

    def backup_vector_index(self) -> str:
        snapshot = self.vector_store.snapshot()
        if snapshot is None:
            raise RuntimeError(self.vector_store.error or "向量库不可用。")
        return snapshot

    def _handle_vector_error(self, action: str, exc: Exception) -> None:
        if self.settings.knowledge_vector_required:
            raise exc
        logger.warning("Vector %s failed; using BM25: %s", action, exc)

    def status(self) -> dict:
        from app.services.knowledge_bundle import bundle_status
        vector_count = None
        error = self.vector_store.error
        if self.vector_store.can_embed:
            try:
                vector_count = self.vector_store.count()
            except Exception as exc:
                error = str(exc)
        return {"primaryRetrieval": PRIMARY_RETRIEVAL_LABEL, "fallbackRetrieval": FALLBACK_RETRIEVAL_LABEL,
            "activePolicyDocuments": len(self._eligible_documents("policy", self.settings.campus_name)),
            "databaseChunks": self.count(), "parentChunks": self.db.query(KnowledgeChunk).filter_by(chunk_type="parent").count(),
            "childChunks": self.db.query(KnowledgeChunk).filter_by(chunk_type="child").count(),
            "documents": self.db.query(KnowledgeDocument).count(), "vectorChunks": vector_count,
            "vectorEnabled": self.settings.knowledge_vector_enabled, "vectorAvailable": self.vector_store.can_embed,
            "vectorRequired": self.settings.knowledge_vector_required, "embeddingModel": self.settings.openai_embedding_model,
            "candidateK": self.settings.knowledge_candidate_k, "rerankEnabled": self.settings.knowledge_rerank_enabled,
            "parentChunkSize": self.settings.knowledge_parent_chunk_size, "childChunkSize": self.settings.knowledge_chunk_size,
            "childOverlap": self.settings.knowledge_chunk_overlap, "knowledgeTokenBudget": self.settings.rag_skill_budget_tokens,
            "fusionWeights": {"parentVector": self.settings.knowledge_parent_vector_weight,
                "childVector": self.settings.knowledge_child_vector_weight, "bm25": self.settings.knowledge_hybrid_bm25_weight},
            "chromaPersistDir": self.settings.chroma_persist_dir, "chromaCollectionName": self.vector_store.collection_name,
            "vectorError": error, "school": self.settings.campus_name,
            "bundledKnowledge": bundle_status(self.settings)}

    def documents(self) -> list[dict]:
        return [self._document_payload(d) for d in self.db.query(KnowledgeDocument).order_by(KnowledgeDocument.id.desc()).all()]

    def _document_payload(self, d: KnowledgeDocument) -> dict:
        return {"id": d.id, "source": d.source, "title": d.title, "version": d.version, "domain": d.domain,
            "school": d.school, "department": d.department, "url": d.source_url, "status": d.status,
            "isDemo": d.is_demo, "publishedOn": d.published_on, "effectiveOn": d.effective_on, "expiresOn": d.expires_on}

    def document_detail(self, document_id: int) -> dict:
        d = self.db.get(KnowledgeDocument, document_id)
        if d is None:
            raise LookupError("文档不存在。")
        payload = self._document_payload(d)
        payload["chunks"] = [{"id": r.id, "type": r.chunk_type, "parentId": r.parent_id,
            "heading": r.heading, "page": r.page_start, "content": r.content} for r in
            self.db.query(KnowledgeChunk).filter_by(document_id=d.id).order_by(KnowledgeChunk.id).all()]
        return payload


def first_title(content: str) -> str:
    match = re.search(r"^#{1,6}\s+(.+)$", content, re.M)
    return match.group(1).strip() if match else ""


def parent_sections(content: str, target: int) -> list[tuple[str, str, int | None]]:
    """父块按条款/主标题组织，三级以下标题留在父块内；PDF 页码只定位不强制断条。"""
    sections = []
    heading, page, start_page, lines = "正文", None, None, []

    def flush_section():
        nonlocal lines
        body = "\n".join(lines).strip()
        if body:
            sections.append((heading, body, start_page))
        lines = []

    for line in content.replace("\r\n", "\n").splitlines():
        page_marker = re.match(r"<!-- page:(\d+) -->", line.strip())
        if page_marker:
            page = int(page_marker.group(1))
            continue
        major = re.match(r"^(?:#{1,2}\s+|第[一二三四五六七八九十百零\d]+[章节条])", line)
        if major:
            flush_section()
            heading = line.lstrip("# ")[:200]
            start_page = page
            # Markdown 标题正文在 heading 中保留；不产生只有标题的空父块。
            if not line.startswith("#"):
                lines.append(line)
        else:
            if not lines:
                start_page = page
            lines.append(line)
    flush_section()
    result = []
    for heading, body, page in sections:
        paragraphs = re.split(r"\n\s*\n", body)
        pending = []
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            units = [paragraph] if "|" in paragraph else re.split(r"(?<=[。！？；])(?=.)", paragraph)
            for unit in units:
                if pending and sum(map(len, pending)) + len(unit) > max(100, target):
                    result.append((heading, "\n\n".join(pending), page))
                    pending = []
                pending.append(unit)
        if pending:
            result.append((heading, "\n\n".join(pending), page))
    return result


def child_sections(content: str, size: int, overlap: int) -> list[str]:
    size = max(40, size)
    overlap = min(max(0, overlap), size // 3)
    units = re.split(r"\n\s*\n|(?<=[。！？；])", content)
    result, pending = [], ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        # 表格不按字数撕裂，正文长句才使用有重叠的滑窗。
        if "|" in unit:
            if pending:
                result.append(pending)
                pending = ""
            result.append(unit)
        elif len(unit) > size:
            if pending:
                result.append(pending)
                pending = ""
            result.extend(chunk_text(unit, size, overlap))
        elif pending and len(pending) + len(unit) > size:
            result.append(pending)
            pending = unit
        else:
            pending += ("\n" if pending else "") + unit
    if pending:
        result.append(pending)
    return result or [content]


def chunk_text(content: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", content or "").strip()
    step = max(1, size - overlap)
    return [text[start:start + size] for start in range(0, len(text), step)] if text else []


def hybrid_score(query: str, content: str) -> float:
    return token_cosine(query, content) * 0.75 + keyword_score(query, content) * 0.25


def bm25_scores(query: str, chunks: list[KnowledgeChunk]) -> dict[int, float]:
    query_terms = counts(tokenize(query))
    if not query_terms or not chunks:
        return {}
    documents, doc_freqs = [], {}
    for chunk in chunks:
        if chunk.id is None:
            continue
        token_counts = counts(tokenize(chunk.embedding_text))
        documents.append((chunk.id, token_counts, sum(token_counts.values())))
        for term in token_counts:
            doc_freqs[term] = doc_freqs.get(term, 0) + 1
    total_docs = len(documents)
    if not total_docs:
        return {}
    average_length = sum(length for _, _, length in documents) / total_docs or 1.0
    scores = {}
    for chunk_id, token_counts, doc_length in documents:
        score = 0.0
        length_norm = 1.5 * (1.0 - 0.75 + 0.75 * doc_length / average_length)
        for term, query_frequency in query_terms.items():
            tf = token_counts.get(term, 0)
            if tf:
                df = doc_freqs[term]
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                score += idf * (1.0 + math.log(query_frequency)) * (tf * 2.5 / (tf + length_norm))
        if score > 0:
            scores[chunk_id] = score
    return scores


def rerank_score(query: str, content: str, base_score: float) -> float:
    return base_score * .55 + hybrid_score(query, content) * .25 + query_token_coverage(query, content) * .15 + phrase_score(query, content) * .05


def query_token_coverage(query: str, content: str) -> float:
    tokens = set(tokenize(query))
    return len(tokens & set(tokenize(content))) / len(tokens) if tokens else 0.0


def phrase_score(query: str, content: str) -> float:
    return 1.0 if compact_text(query) and compact_text(query) in compact_text(content) else keyword_score(query, content)


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def normalize_scores(scores: dict[Hashable, float]) -> dict[Hashable, float]:
    positives = [value for value in scores.values() if value > 0]
    if not positives:
        return {}
    # 除以本路最高分，避免 min-max 将唯一命中或末位有效命中压成零。
    maximum = max(positives)
    return {key: max(0, value) / maximum for key, value in scores.items()}


def result_key(result: SearchResult) -> Hashable:
    return result.chunk_id if result.chunk_id is not None else (result.source, result.content)


def replace_score(result: SearchResult, score: float) -> SearchResult:
    from dataclasses import replace
    return replace(result, score=score)


def parse_embedding(raw: str | None) -> list[float] | None:
    try:
        value = json.loads(raw or "null")
        return [float(x) for x in value] if isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value) else None
    except (ValueError, TypeError):
        return None


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())
    compact = "".join(ch for ch in text.lower() if "\u4e00" <= ch <= "\u9fff")
    return words + [compact[i:i + 2] for i in range(max(0, len(compact) - 1))]


def token_cosine(left: str, right: str) -> float:
    a, b = counts(tokenize(left)), counts(tokenize(right))
    norm = math.sqrt(sum(x*x for x in a.values())) * math.sqrt(sum(x*x for x in b.values()))
    return sum(v * b.get(k, 0) for k, v in a.items()) / norm if norm else 0.0


def keyword_score(query: str, content: str) -> float:
    terms = [term for term in re.split(r"[\s，。！？、；：,.!?;:]+", query.lower()) if len(term) >= 2]
    return sum(term in content.lower() for term in terms) / len(terms) if terms else 0.0


def counts(values: list[str]) -> dict[str, int]:
    result = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def extract_pdf(data: bytes) -> str:
    from io import BytesIO
    reader = PdfReader(BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"<!-- page:{i+1} -->\n{text}")
    return "\n\n".join(pages)


def campus_today() -> date:
    # 数据库存 UTC 时间，政策生效日按中国校园所在时区解释。
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()
