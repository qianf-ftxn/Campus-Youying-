"""Compare K=4/5, including the archived pre-parent/child retrieval implementation."""
from __future__ import annotations

import json
import statistics
import sys
import tarfile
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.core.database import Base
from app.models.entities import KnowledgeChunk
from app.policy_eval.pilot import DATASET, ROOT
from app.policy_eval.runner import (
    Chunk, EmbeddingCache, digest, load_inputs, retrieval_query, score_case, select_context, summarize,
)


def archived_flat_reports(dataset, documents, output, *, dataset_path=DATASET, top_ks=(4, 5)):
    archive = ROOT / "target/backups/pre-campus-youying-source.tgz"
    with tarfile.open(archive) as bundle:
        name = next(name for name in bundle.getnames() if name.endswith("app/services/knowledge.py"))
        source = bundle.extractfile(name).read()
    # Load the already inspected project backup to call its original fusion,
    # deterministic rerank and neighboring-chunk expansion unchanged.
    legacy = types.ModuleType("archived_flat_knowledge_eval")
    legacy.__file__ = str(archive) + ":" + name
    sys.modules[legacy.__name__] = legacy
    exec(compile(source, legacy.__file__, "exec"), legacy.__dict__)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False)()
    cache = None
    try:
        for meta, text in documents.values():
            for index, content in enumerate(legacy.chunk_text(text, 512, 64)):
                db.add(KnowledgeChunk(source=meta["id"], source_index=index, content=content, chunk_type="legacy"))
        db.commit()
        rows = db.query(KnowledgeChunk).order_by(KnowledgeChunk.id).all()
        by_id = {row.id: row for row in rows}
        neighbors = {(row.source, row.source_index): row for row in rows}
        settings = Settings()
        cache = EmbeddingCache(settings, ROOT / "target/policy-pilot/embeddings.sqlite3")
        vectors = cache.embed([row.content for row in rows] + [retrieval_query(case) for case in dataset["cases"]])
        embedding = {"model": settings.openai_embedding_model, "provider_fingerprint": cache.provider,
                     "api_calls": cache.api_calls, "new_texts": cache.new_texts, "cached_texts": cache.cached_texts}
        matrix = np.vstack([vectors[row.content] for row in rows])
        service = object.__new__(legacy.KnowledgeService)
        service.db = db
        service.settings = SimpleNamespace(knowledge_top_k=4, knowledge_candidate_k=16,
                                          knowledge_hybrid_vector_weight=.65, knowledge_hybrid_bm25_weight=.35,
                                          knowledge_rerank_enabled=True)
        results = {k: [] for k in top_ks}
        for case in dataset["cases"]:
            query = retrieval_query(case)
            similarities = matrix @ vectors[query]
            vector_order = sorted(range(len(rows)), key=lambda index: (-float(similarities[index]), rows[index].id))
            vector_results = [legacy.SearchResult(rows[index].id, rows[index].source, rows[index].content,
                                                 1 / (1 + max(0., 1 - float(similarities[index])))) for index in vector_order]
            service._retrieve_vector = lambda text, k, candidates=vector_results: candidates[:k]
            for top_k in top_ks:
                retrieved = service.retrieve(query, top_k)
                ranked = [(Chunk(item.chunk_id, item.source, documents[item.source][0]["version"], "flat", "", item.content), item.score)
                          for item in retrieved]
                selected, used = select_context(ranked, top_k, 4000)
                groups = []
                for row, _ in selected:
                    original = by_id[row.id]
                    if row.content != original.content:
                        pieces = [neighbors[(original.source, index)] for index in range(max(0, original.source_index - 1), original.source_index + 2)
                                  if (original.source, index) in neighbors]
                    else:
                        pieces = [original]
                    groups.append([(Chunk(piece.id, piece.source, documents[piece.source][0]["version"], "flat", "", piece.content), 1.) for piece in pieces])
                # Expanded neighbors contain overlapping text; score the union of
                # the original source windows, rather than a noncontiguous join.
                flattened = [pair for group in groups for pair in group]
                metrics = score_case(case, flattened, documents)
                if case["answerable"]:
                    first = next((rank for rank in range(1, len(groups) + 1)
                                  if score_case(case, [pair for group in groups[:rank] for pair in group], documents)["evidence_hit_at_k"]), 0)
                    metrics["reciprocal_rank"] = 1 / first if first else 0.
                results[top_k].append({"case_id": case["id"], "category": case["category"], "review_status": case["review_status"],
                                       "method": "archived_flat_hybrid", "query": query, "metrics": metrics,
                                       "estimated_tokens": used, "retrieval_ms": None,
                                       "contexts": [{"chunk_id": row.id, "document_id": row.document_id, "version": row.version,
                                                     "content": row.content, "score": score,
                                                     "source_window_ids": [piece.id for piece, _ in group]}
                                                    for (row, score), group in zip(selected, groups)]})
        reports = {}
        for top_k, group in results.items():
            reports[top_k] = {"schema_version": 1, "status": "provisional_ai_annotations",
                              "dataset_sha256": digest(dataset_path.read_bytes()), "corpus_manifest_sha256": dataset["manifest_sha256"],
                              "configuration": {"top_k": top_k, "chunk_size": 512, "overlap": 64, "candidate_k_per_channel": 16,
                                                "weights": [.65, .35], "rule_rerank": True, "first_result_neighbors": True,
                                                "estimated_token_budget": 4000, "original_default_token_budget": 1200,
                                                "vector_search": "exact cosine; archived original ranking functions"},
                              "embedding": embedding, "chunks": len(rows), "source_archive_sha256": digest(archive.read_bytes()),
                              "archived_knowledge_source_sha256": digest(source), "summary": summarize(group), "results": group}
            (output / f"archived-k{top_k}.json").write_text(json.dumps(reports[top_k], ensure_ascii=False, indent=2) + "\n")
        return reports
    finally:
        if cache:
            cache.close()
        db.close()
        engine.dispose()


def main():
    output = ROOT / "target/policy-pilot/topk-4-vs-5"
    output.mkdir(parents=True, exist_ok=True)
    dataset, documents = load_inputs(DATASET)
    current = {k: json.loads((output / f"k{k}/report.json").read_text()) for k in (4, 5)}
    for k, report in current.items():
        if report["dataset_sha256"] != digest(DATASET.read_bytes()) or report["configuration"]["top_k"] != k:
            raise ValueError("Comparison input report/config differs")
    archived = archived_flat_reports(dataset, documents, output)
    if any(current[k]["embedding"]["provider_fingerprint"] != archived[k]["embedding"]["provider_fingerprint"] for k in (4, 5)):
        raise ValueError("Embedding provider/model differs between comparison arms")
    summaries, deltas = {}, {}
    for method in ["archived_flat_hybrid", "flat_vector", "parent_child_vector", "parent_child_hybrid"]:
        method_results = {k: archived[k]["results"] if method == "archived_flat_hybrid" else
                          [row for row in current[k]["results"] if row["method"] == method] for k in (4, 5)}
        summaries[method] = {}
        for k, group in method_results.items():
            summary = summarize(group)
            summary["complete_count"] = sum(row["metrics"]["evidence_complete"] is True for row in group)
            summary["mean_selected_chunks"] = statistics.mean(len(row["contexts"]) for row in group)
            summaries[method][k] = summary
        before = {row["case_id"]: row for row in method_results[4]}
        deltas[method] = []
        for row in method_results[5]:
            old = before[row["case_id"]]
            if row["metrics"]["clause_recall_at_k"] is not None:
                if row["metrics"]["clause_recall_at_k"] < old["metrics"]["clause_recall_at_k"]:
                    raise ValueError("Unexpected evidence regression when adding a fifth item")
                if row["metrics"]["evidence_coverage"] != old["metrics"]["evidence_coverage"]:
                    deltas[method].append({"case_id": row["case_id"], "question": row["query"],
                                           "coverage_k4": old["metrics"]["evidence_coverage"], "coverage_k5": row["metrics"]["evidence_coverage"],
                                           "complete_k4": old["metrics"]["evidence_complete"], "complete_k5": row["metrics"]["evidence_complete"],
                                           "extra_contexts": [context for context in row["contexts"]
                                                              if context["chunk_id"] not in {c["chunk_id"] for c in old["contexts"]}]})
    report = {"status": "provisional_ai_annotations", "dataset_sha256": digest(DATASET.read_bytes()),
              "eligible_documents": len(documents), "total_cases": len(dataset["cases"]), "scored_answerable_cases": 20,
              "comparison_code_sha256": digest(Path(__file__).read_bytes()), "token_budget": 4000,
              "summaries": summaries, "changes_from_fifth_result": deltas,
              "limitations": ["AI-drafted development set awaiting human review; retrieval evidence, not answer accuracy.",
                              "Original archived pipeline uses 512/64 windows and rules/neighbor expansion; earlier flat-vector control uses 300/40 windows.",
                              "All arms share 4000 estimated-token cap; original app default of 1200 is intentionally normalized.",
                              "Query history concatenation is fixed; production model query rewriting and approximate Chroma search are not exercised."]}
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 返回4条与5条的检索对照", "", "49份固定资料；25道开发题，20道可回答题参与指标；统一4000估算token预算。标注尚待人工审核。", "",
             "| 方法 | K | 必要证据召回率 | 证据齐全题数 | 平均上下文估算tokens（25题） |", "|---|---:|---:|---:|---:|"]
    labels = {"archived_flat_hybrid": "原始平面混合（512/64，含规则重排及邻块扩展）", "flat_vector": "普通切块纯向量对照（300/40）",
              "parent_child_vector": "当前父子块纯向量", "parent_child_hybrid": "当前父子块混合"}
    for method, summary in summaries.items():
        for k, item in summary.items():
            lines.append(f"| {labels[method]} | {k} | {item['clause_recall_at_k']:.2%} | {item['complete_count']}/20 | {item['mean_context_estimated_tokens']:.2f} |")
    lines.extend(["", "原始平面混合调用源码备份中的融合、规则重排及首条±1邻块扩展实现；普通切块纯向量是之前使用的简化基线，二者不同。各组实际上下文长度不同，不能将效果差异全部归因于是否采用父子索引。", "",
                  "必要证据召回率按题宏平均：完整取回的标注证据单元数/该题全部必要证据单元数。证据齐全题数要求全部单元都被取回。5道资料不足题不进入召回分母。", "",
                  "统一使用相同模型、查询和精确余弦检索，未调用生产查询改写、未生成答案、未评价拒答或线上延迟。归档原始实现原先1200的预算在这里统一到4000，以比较检索算法。", "",
                  "## 第五条改变了哪些题", ""])
    for method, changes in deltas.items():
        lines.extend([f"### {labels[method]}", ""])
        if not changes:
            lines.append("没有新增标注证据覆盖。\n")
        for row in changes:
            lines.append(f"- {row['case_id']}：覆盖比例 {row['coverage_k4']} → {row['coverage_k5']}；证据是否全部齐全 {row['complete_k4']} → {row['complete_k5']}")
        lines.append("")
    (output / "comparison.md").write_text("\n".join(lines))
    print(json.dumps({"summaries": summaries, "changes": {method: [row['case_id'] for row in rows] for method, rows in deltas.items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
