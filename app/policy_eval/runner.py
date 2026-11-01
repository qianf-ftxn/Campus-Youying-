"""Three-arm retrieval pilot with exact evidence scoring and isolated embedding cache.

No business DB, Redis, Chroma index, chat generation, or notification tools are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sqlite3
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from app.core.config import Settings
from app.policy_eval.pilot import DATASET, ROOT
from app.services.knowledge import bm25_scores, child_sections, chunk_text, normalize_scores, parent_sections
from app.services.knowledge_chunking import CHUNKING_VERSION, child_heading, knowledge_heading
from app.services.knowledge_bundle import load_bundle
from app.services.memory_context import estimate_tokens

METHODS = ("flat_vector", "parent_child_vector", "parent_child_hybrid")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def without_page_markers(text: str) -> str:
    # Production parent_sections removes these markers; fixed windows retain them.
    return compact(re.sub(r"<!-- page:\d+ -->", "", text))


@dataclass
class Chunk:
    id: int
    document_id: str
    version: str
    kind: str
    heading: str
    content: str
    parent_id: int | None = None

    @property
    def embedding_text(self) -> str:
        return f"{self.heading}\n{self.content}".strip()


def validate_dataset(dataset: dict, documents: dict, manifest_hash: str) -> None:
    if dataset.get("schema_version") != 1 or dataset.get("split") not in {"pilot_dev", "frozen_candidate"}:
        raise ValueError("Only schema v1 pilot_dev or frozen_candidate datasets are supported.")
    if dataset["manifest_sha256"] != manifest_hash:
        raise ValueError("Corpus manifest changed; review the dataset before running again.")
    seen = set()
    for case in dataset["cases"]:
        if case["id"] in seen:
            raise ValueError("Duplicate case ID")
        seen.add(case["id"])
        if case["review_status"] not in {"pending_human", "approved", "needs_revision"}:
            raise ValueError("Invalid review status")
        if not case["question"].strip() or not case["answer_points"]:
            raise ValueError("Each case requires a question and answer points")
        if case["answerable"] != bool(case["evidence"]):
            raise ValueError("Answerable cases must have evidence; negative cases must not")
        behavior = case.get("expected_behavior")
        if behavior is not None:
            if behavior not in {"answer", "clarify", "abstain"}:
                raise ValueError("Invalid expected behavior")
            if (behavior == "answer") != case["answerable"]:
                raise ValueError("Expected behavior contradicts answerability")
        if any(doc_id not in documents for doc_id in case.get("checked_source_ids", [])):
            raise ValueError("Checked source is not eligible in the frozen corpus")
        for item in case["evidence"]:
            if item["document_id"] not in documents:
                raise ValueError("Gold document is not eligible in the frozen corpus")
            meta, text = documents[item["document_id"]]
            if meta["version"] != item["version"] or digest(text.encode()) != item["content_sha256"]:
                raise ValueError("Gold source version/hash mismatch")
            start, end = item["start_char"], item["end_char"]
            if start < 0 or end <= start or text[start:end] != item["quote"]:
                raise ValueError("Gold quote is not at the declared source offsets")
            if text.count("\n", 0, start) + 1 != item["line_start"] or text.count("\n", 0, end) + 1 != item["line_end"]:
                raise ValueError("Gold source line numbers changed")
            if not compact(item["quote"]):
                raise ValueError("Empty gold quote")
            if "<!-- page:" in item["quote"]:
                raise ValueError("Split gold evidence at PDF page markers")


def load_inputs(dataset_path: Path):
    dataset = json.loads(dataset_path.read_text())
    manifest_path = ROOT / "knowledge/xidian/manifest.json"
    manifest, rows = load_bundle(manifest_path.parent)
    frozen_date = date.fromisoformat(dataset["corpus_checked_on"])
    if dataset["corpus_checked_on"] != manifest["checked_on"]:
        raise ValueError("Dataset and corpus snapshot dates disagree")
    documents = {meta["id"]: (meta, text) for meta, text in rows
                 if meta["status"] == "ACTIVE"
                 and (not meta.get("effective_on") or date.fromisoformat(meta["effective_on"]) <= frozen_date)
                 and (not meta.get("expires_on") or date.fromisoformat(meta["expires_on"]) >= frozen_date)}
    validate_dataset(dataset, documents, digest(manifest_path.read_bytes()))
    return dataset, documents


def build_chunks(documents: dict) -> list[Chunk]:
    chunks = []

    def add(meta, kind, heading, content, parent_id=None):
        row = Chunk(len(chunks) + 1, meta["id"], meta["version"], kind, heading, content, parent_id)
        chunks.append(row)
        return row

    for meta, text in documents.values():
        # Independent ordinary fixed windows; no gold evidence is used for segmentation.
        for body in chunk_text(text, 300, 40):
            add(meta, "flat", meta["title"], body)
        for heading, body, _ in parent_sections(text, 1000):
            parent = add(meta, "parent", knowledge_heading(meta['title'], heading), body)
            for child in child_sections(body, 300, 40):
                add(meta, "child", child_heading(parent.heading, child), child, parent.id)
    return chunks


def retrieval_query(case: dict) -> str:
    # Contexts contain only supplied user messages, never gold answers or evidence.
    history = [m["content"] for m in case["history"] if m["role"] == "user"]
    return "\n".join([*history, case["question"]])


class EmbeddingCache:
    def __init__(self, settings: Settings, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS vectors (provider TEXT, text_hash TEXT, vector TEXT, PRIMARY KEY(provider,text_hash))")
        self.provider = digest(f"{settings.openai_base_url.rstrip('/')}:{settings.openai_embedding_model}".encode())
        self.settings = settings
        self.new_texts = self.api_calls = self.cached_texts = 0

    def embed(self, texts: list[str]) -> dict[str, np.ndarray]:
        unique = list(dict.fromkeys(texts))
        result, pending = {}, []
        for text in unique:
            row = self.connection.execute("SELECT vector FROM vectors WHERE provider=? AND text_hash=?",
                                          (self.provider, digest(text.encode()))).fetchone()
            if row:
                result[text] = json.loads(row[0])
                self.cached_texts += 1
            else:
                pending.append(text)
        print(f"Embedding: {len(unique)} unique texts, {self.cached_texts} cached, {len(pending)} new", flush=True)
        if pending and not self.settings.openai_api_key:
            raise ValueError("Missing embedding API key")
        with httpx.Client(timeout=self.settings.embedding_timeout_seconds) as client:
            for start in range(0, len(pending), 10):
                batch = pending[start:start + 10]
                for attempt in range(3):
                    response = client.post(f"{self.settings.openai_base_url.rstrip('/')}/embeddings",
                                           headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                                           json={"model": self.settings.openai_embedding_model, "input": batch})
                    self.api_calls += 1
                    if response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
                        break
                    time.sleep(attempt + 1)
                if response.is_error:
                    raise RuntimeError(f"Embedding HTTP {response.status_code}; response body withheld")
                data = response.json()["data"]
                if sorted(item["index"] for item in data) != list(range(len(batch))):
                    raise ValueError("Embedding response indices are incomplete or duplicated")
                vectors = [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
                for text, vector in zip(batch, vectors):
                    array = np.asarray(vector, dtype=np.float64)
                    if array.ndim != 1 or not array.size or not np.isfinite(array).all() or np.linalg.norm(array) == 0:
                        raise ValueError("Invalid embedding vector")
                    self.connection.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?)",
                                            (self.provider, digest(text.encode()), json.dumps(vector)))
                    result[text] = vector
                self.connection.commit()
                self.new_texts += len(batch)
                if start % 100 == 0 or start + 10 >= len(pending):
                    print(f"Embedded {min(start + 10, len(pending))}/{len(pending)} new texts", flush=True)
        dimensions = {len(vector) for vector in result.values()}
        if len(dimensions) != 1:
            raise ValueError("Embedding dimensions differ; use a fresh cache")
        normalized = {}
        for text, vector in result.items():
            array = np.asarray(vector, dtype=np.float64)
            norm = np.linalg.norm(array)
            if array.ndim != 1 or not np.isfinite(array).all() or not math.isfinite(float(norm)) or norm == 0:
                raise ValueError("Invalid cached vector")
            normalized[text] = array / norm
        return normalized

    def close(self):
        self.connection.close()


def vector_rank(rows, similarities, candidate_k):
    ordered = sorted(rows, key=lambda row: (-similarities[row.id], row.id))[:candidate_k]
    # Same monotonic score transformation as the production Chroma adapter.
    return {row.id: 1 / (1 + max(0., 1 - similarities[row.id])) for row in ordered}


def rank_chunks(method: str, query: str, chunks: list[Chunk], similarities: dict[int, float],
                candidate_k: int = 16) -> list[tuple[Chunk, float]]:
    by_id = {row.id: row for row in chunks}
    if method == "flat_vector":
        scores = vector_rank([r for r in chunks if r.kind == "flat"], similarities, len(chunks))
        return [(by_id[key], value) for key, value in scores.items()]
    children = [r for r in chunks if r.kind == "child"]
    channels, weights = {}, {}
    if method != "bm25_parent_smoke":
        for kind, weight in (("parent", .30), ("child", .35)):
            raw = vector_rank([r for r in chunks if r.kind == kind], similarities, candidate_k)
            mapped = {}
            for key, score in raw.items():
                parent_id = key if kind == "parent" else by_id[key].parent_id
                mapped[parent_id] = max(mapped.get(parent_id, 0.), score)
            channels[kind] = normalize_scores(mapped)
            weights[kind] = weight
    if method in {"parent_child_hybrid", "bm25_parent_smoke"}:
        raw = bm25_scores(query, children)
        mapped = {}
        for key in sorted(raw, key=lambda key: (-raw[key], key))[:candidate_k]:
            parent_id = by_id[key].parent_id
            mapped[parent_id] = max(mapped.get(parent_id, 0.), raw[key])
        channels["bm25"] = normalize_scores(mapped)
        weights["bm25"] = .35
    active = {name: scores for name, scores in channels.items() if scores}
    total = sum(weights[name] for name in active) or 1.
    scores = {key: sum(weights[name] * values.get(key, 0.) for name, values in active.items()) / total
              for key in {key for values in active.values() for key in values}}
    return [(by_id[key], score) for key, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def select_context(ranked, top_k, budget):
    selected, used = [], 0
    # Matches production: take top-K first, then skip blocks exceeding remaining budget.
    for row, score in ranked[:top_k]:
        cost = estimate_tokens(row.content)
        if used + cost <= budget:
            selected.append((row, score))
            used += cost
    return selected, used


def evidence_coverage(evidence: dict, selected: list[tuple[Chunk, float]], documents: dict) -> float:
    """Union of source-coordinate coverage; overlap never double-counts evidence."""
    meta, source = documents[evidence["document_id"]]
    if meta["version"] != evidence["version"]:
        return 0.
    length = len(compact(evidence["quote"]))
    intervals = []
    for row, _ in selected:
        if row.document_id != evidence["document_id"] or row.version != evidence["version"]:
            continue
        normalize = compact if row.kind == "flat" else without_page_markers
        normalized = normalize(source)
        begin = len(normalize(source[:evidence["start_char"]]))
        end = len(normalize(source[:evidence["end_char"]]))
        body = normalize(row.content)
        position = normalized.find(body)
        if position < 0:
            raise ValueError(f"Chunk {row.id} cannot be aligned to its exact source")
        cursor = 0
        while (position := normalized.find(body, cursor)) >= 0:
            left, right = max(begin, position), min(end, position + len(body))
            if left < right:
                intervals.append((left - begin, right - begin))
            cursor = position + max(1, len(body))
    covered, cursor = 0, 0
    for left, right in sorted(intervals):
        covered += max(0, right - max(cursor, left))
        cursor = max(cursor, right)
    return covered / length


def score_case(case, selected, documents):
    if not case["answerable"]:
        return {"clause_recall_at_k": None, "evidence_complete": None,
                "evidence_hit_at_k": None, "evidence_coverage": [], "reciprocal_rank": None,
                "abstention_correct": None}
    coverage = [evidence_coverage(gold, selected, documents) for gold in case["evidence"]]
    complete = [value >= 1 - 1e-9 for value in coverage]
    first_rank = next((rank for rank in range(1, len(selected) + 1)
                       if any(evidence_coverage(gold, selected[:rank], documents) >= 1 - 1e-9
                              for gold in case["evidence"])), 0)
    return {"clause_recall_at_k": statistics.mean(complete), "evidence_complete": all(complete),
            "evidence_hit_at_k": first_rank > 0, "evidence_coverage": coverage,
            "reciprocal_rank": 1 / first_rank if first_rank else 0., "abstention_correct": None}


def summarize(results):
    answerable = [row for row in results if row["metrics"]["clause_recall_at_k"] is not None]
    latencies = [row["retrieval_ms"] for row in results if row["retrieval_ms"] is not None]
    mean = lambda key: statistics.mean(row["metrics"][key] for row in answerable) if answerable else None
    return {"cases": len(results), "answerable_cases": len(answerable),
            "clause_recall_at_k": mean("clause_recall_at_k"), "evidence_complete_rate": mean("evidence_complete"),
            "evidence_hit_at_k": mean("evidence_hit_at_k"), "evidence_mrr": mean("reciprocal_rank"),
            "mean_context_estimated_tokens": statistics.mean(row["estimated_tokens"] for row in results) if results else None,
            "mean_local_retrieval_ms": statistics.mean(latencies) if latencies else None,
            "answer_accuracy": None, "unanswerable_abstention_accuracy": None}


def write_review(dataset, directory: Path, dataset_path: Path = DATASET):
    states = dict(Counter(case["review_status"] for case in dataset["cases"]))
    split_note = ("批准后仍属于开发集，不能改名作为独立保留集。" if dataset['split'] == 'pilot_dev' else
                  "这是冻结协议后的前瞻候选集；人工批准不自动构成外部认证的独立金标准，还需核对来源隔离和使用记录。")
    lines = ["# 评测集标注审核表", "", f"数据集：[{dataset['name']}]({dataset_path.resolve()})。共 {len(dataset['cases'])} 题，审核状态：{states}。",
             "", "请在上述 JSON 中填写 reviewer、review_status 和 review_note；审核表中的勾选不会自动更新 JSON。",
             "", "逐题核对：问题可回答性；制度适用关系及冲突；必要条款是否齐全；答案要点是否过度推断。" + split_note, ""]
    for case in dataset["cases"]:
        lines.extend([f"## {case['id']} · {case['category']} · {case['review_status']}", ""])
        for message in case["history"]:
            lines.append(f"历史（{message['role']}）：{message['content']}\n")
        lines.extend([case["question"], ""])
        if case.get("expected_behavior"):
            lines.extend([f"预期行为：{case['expected_behavior']}；场景标签：{'、'.join(case.get('tags', []))}", ""])
        if case.get("checked_source_ids"):
            lines.extend([f"相关资料检查范围：{'、'.join(case['checked_source_ids'])}。资料不足结论仍需结合全库人工核对。", ""])
        lines.extend(["答案要点：", ""])
        lines.extend(f"- {point}" for point in case["answer_points"])
        lines.append("")
        for gold in case["evidence"]:
            lines.extend([f"证据：{gold['document_id']} / {gold['version']} / {gold['section']} / 原文第 {gold['line_start']}–{gold['line_end']} 行", "",
                          f"[仓库原文]({ROOT / 'knowledge/xidian' / gold['content_path']}:{gold['line_start']}) · [快照记录的官方链接]({gold['source_url']})", "",
                          "> " + gold["quote"].replace("\n", "\n> "), ""])
        if case["review_note"]:
            lines.extend([f"待核对：{case['review_note']}", ""])
        lines.extend(["人工结论：□批准　□需修改；审核人：________；备注：________", ""])
    (directory / "annotation-review.md").write_text("\n".join(lines))


def write_report(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 校园政策检索试测", "", f"状态：{report['status']}。全部题目为开发试测；未经人工审核的数字不得作为简历效果结论。", "",
             "| 方法 | 可回答题数 | 必要证据单元召回率 | 证据完整率 | 证据命中率 | 平均上下文估算 tokens |", "|---|---:|---:|---:|---:|---:|"]
    for name, summary in report["summary"].items():
        percent = lambda value: "—" if value is None else f"{value:.1%}"
        lines.append(f"| {name} | {summary['answerable_cases']} | {percent(summary['clause_recall_at_k'])} | {percent(summary['evidence_complete_rate'])} | {percent(summary['evidence_hit_at_k'])} | {summary['mean_context_estimated_tokens']:.0f} |")
    counts = next(iter(report["summary"].values()), {})
    answerable = counts.get("answerable_cases", 0)
    negative = report["case_count"] - answerable
    lines.extend(["", f"指标按{answerable}道可回答题计算，{negative}道资料不足题只导出候选供审核，不进入检索召回分母。", "",
                  "必要证据单元召回率：逐题计算完整取回的标注证据单元数/该题全部必要证据单元数，再宏平均。证据完整率：全部必要证据均完整取回的题目占比。命中率：至少一个必要证据单元被完整取回；这不是回答正确率。JSON中的evidence_mrr是首次累计覆盖完整证据单元的倒数排名均值，不是传统的单块相关性MRR。", "",
                  f"共同 Top-K={report['configuration']['top_k']}、相同4000估算token预算上限；实际上下文长度不同，需同时报告。向量使用相同真实Embedding与精确余弦扫描，融合逻辑复用生产分块、BM25和归一化函数；未使用生产Chroma近似检索，不把本地扫描时间当线上响应时间。", "",
                  "多轮题统一拼接预设用户历史与当前问题，不调用生产模型改写。未生成答案、未评判拒答、未计算成本/首字延迟。全部结论受AI标注和开发集偏差限制。", "",
                  "## 优先查看的错例", ""])
    for row in report["results"]:
        if row["metrics"]["evidence_complete"] is False:
            lines.append(f"- {row['method']} / {row['case_id']}：覆盖比例 {', '.join(f'{v:.0%}' for v in row['metrics']['evidence_coverage'])}")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    # Blind review sheet: the reviewer can inspect anonymous method outputs before
    # looking at report.json, which retains the reproducible method mapping.
    order = {name: f"方案{index + 1}" for index, name in enumerate(sorted(report["summary"], key=lambda name: digest(("pilot-v1:" + name).encode())))}
    with (output / "retrieval-review.jsonl").open("w") as stream:
        for row in report["results"]:
            payload = {"case_id": row["case_id"], "method_alias": order[row["method"]],
                       "query": row["query"], "contexts": row["contexts"],
                       "human_evidence_complete": None, "human_notes": ""}
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=ROOT / "target/policy-pilot/offline")
    parser.add_argument("--cache", type=Path, default=ROOT / "target/policy-pilot/embeddings.sqlite3")
    parser.add_argument("--real", action="store_true", help="Call configured Embedding API; cache is isolated and resumable")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=None, help="Maximum returned chunks; defaults to configured KNOWLEDGE_TOP_K; explicit values do not change application settings")
    args = parser.parse_args(argv)
    if args.top_k is None:
        args.top_k = Settings().knowledge_top_k
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    dataset, documents = load_inputs(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    write_review(dataset, args.output, args.dataset)
    chunks = build_chunks(documents)
    print(json.dumps({"cases": len(dataset["cases"]), "categories": Counter(c["category"] for c in dataset["cases"]),
                      "documents": len(documents), "chunks": Counter(c.kind for c in chunks),
                      "review_status": Counter(c["review_status"] for c in dataset["cases"])}, ensure_ascii=False), flush=True)
    if args.validate_only:
        return 0
    methods = METHODS if args.real else ("bm25_parent_smoke",)
    vectors = None
    embedding_stats = {"api_calls": 0, "new_texts": 0, "model": None}
    if args.real:
        settings = Settings()
        cache = EmbeddingCache(settings, args.cache)
        try:
            vectors = cache.embed([row.embedding_text for row in chunks] + [retrieval_query(case) for case in dataset["cases"]])
            embedding_stats = {"api_calls": cache.api_calls, "new_texts": cache.new_texts,
                               "cached_texts": cache.cached_texts, "model": settings.openai_embedding_model,
                               "provider_fingerprint": cache.provider}
        finally:
            cache.close()
    results = []
    # Warm-up, then round-robin method order. Timing is diagnostic, not an online benchmark.
    if vectors:
        matrix = np.vstack([vectors[row.embedding_text] for row in chunks])
        _ = matrix @ vectors[retrieval_query(dataset["cases"][0])]
    for case_index, case in enumerate(dataset["cases"]):
        query = retrieval_query(case)
        similarities = {}
        if vectors:
            values = matrix @ vectors[query]
            similarities = {row.id: float(value) for row, value in zip(chunks, values)}
        rotated = methods[case_index % len(methods):] + methods[:case_index % len(methods)]
        for method in rotated:
            start = time.perf_counter()
            ranked = rank_chunks(method, query, chunks, similarities, max(16, args.top_k))
            selected, used = select_context(ranked, args.top_k, 4000)
            elapsed = (time.perf_counter() - start) * 1000
            results.append({"case_id": case["id"], "category": case["category"], "review_status": case["review_status"],
                            "method": method, "query": query, "estimated_tokens": used, "retrieval_ms": round(elapsed, 3),
                            "metrics": score_case(case, selected, documents),
                            "contexts": [{"chunk_id": row.id, "document_id": row.document_id, "version": row.version,
                                          "heading": row.heading, "content": row.content, "score": score} for row, score in selected]})
    summary = {name: summarize([row for row in results if row["method"] == name]) for name in methods}
    report = {"schema_version": 1, "status": "provisional_ai_annotations" if args.real else "offline_bm25_smoke_only",
              "created_at": datetime.now(timezone.utc).isoformat(), "dataset_sha256": digest(args.dataset.read_bytes()),
              "implementation_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in
                                         [Path(__file__), ROOT / "app/services/knowledge.py", ROOT / "app/services/knowledge_chunking.py", ROOT / "app/services/memory_context.py"]},
              "runtime": {"python": platform.python_version(), "numpy": np.__version__},
              "corpus_manifest_sha256": dataset["manifest_sha256"], "corpus_checked_on": dataset["corpus_checked_on"],
              "eligible_documents": len(documents), "case_count": len(dataset["cases"]),
              "review_status": dict(Counter(c["review_status"] for c in dataset["cases"])),
              "configuration": {"chunking_version": CHUNKING_VERSION, "flat_size": 300, "flat_overlap": 40, "parent_size": 1000, "child_size": 300,
                                "child_overlap": 40, "top_k": args.top_k, "candidate_k_per_channel": max(16, args.top_k),
                                "estimated_token_budget": 4000, "hybrid_weights": [.30, .35, .35],
                                "vector_search": "exact cosine (not production Chroma)",
                                "timing_excludes": ["embedding API", "cosine matrix multiplication", "generation", "HTTP"]},
              "embedding": embedding_stats, "summary": summary,
              "by_category": {name: {category: summarize([row for row in results if row["method"] == name and row["category"] == category])
                                     for category in sorted({case["category"] for case in dataset["cases"]})} for name in methods},
              "results": results}
    write_report(report, args.output)
    print(json.dumps({"status": report["status"], "summary": summary, "output": str(args.output)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
