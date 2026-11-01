"""Read-only diagnosis of pilot retrieval failures using existing cached embeddings.

Interventions are diagnostic development-set experiments, never production changes.
Default mode makes no API calls. --probe-correction optionally embeds one new query.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from app.policy_eval.pilot import DATASET, ROOT
from app.core.config import Settings
from app.policy_eval.runner import (
    EmbeddingCache, build_chunks, compact, digest, evidence_coverage, load_inputs, rank_chunks,
    retrieval_query, score_case, select_context, vector_rank,
)
from app.services.knowledge import bm25_scores, normalize_scores
from app.services.knowledge_chunking import CHUNKING_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-correction", action="store_true", help="Compare M04 with/without old history, embedding only its current question")
    parser.add_argument("--baseline-report", type=Path, default=ROOT / "target/policy-pilot/baseline-k5/report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "target/policy-pilot/diagnosis-k5")
    args = parser.parse_args()
    dataset, documents = load_inputs(DATASET)
    baseline_path = args.baseline_report
    baseline = json.loads(baseline_path.read_text())
    baseline_k = baseline["configuration"]["top_k"]
    baseline_name = f"baseline_hybrid_k{baseline_k}"
    if baseline.get("configuration", {}).get("chunking_version") != CHUNKING_VERSION:
        raise ValueError("Baseline report uses another chunking algorithm; rerun the pilot into a separate directory first")
    if baseline["dataset_sha256"] != digest(DATASET.read_bytes()):
        raise ValueError("Pilot data differs from baseline report")
    provider = baseline["embedding"]["provider_fingerprint"]
    cache_path = ROOT / "target/policy-pilot/embeddings.sqlite3"
    chunks = build_chunks(documents)
    by_id = {row.id: row for row in chunks}
    parents = [row for row in chunks if row.kind == "parent"]
    children = [row for row in chunks if row.kind == "child"]
    connection = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
    vectors = {}
    try:
        texts = list(dict.fromkeys([r.embedding_text for r in chunks] + [retrieval_query(c) for c in dataset["cases"]]))
        for text in texts:
            record = connection.execute("SELECT vector FROM vectors WHERE provider=? AND text_hash=?",
                                        (provider, digest(text.encode()))).fetchone()
            if not record:
                raise ValueError("Required vector is missing; diagnosis never calls APIs")
            vector = np.asarray(json.loads(record[0]), dtype=np.float64)
            vectors[text] = vector / np.linalg.norm(vector)
    finally:
        connection.close()
    matrix = np.vstack([vectors[row.embedding_text] for row in chunks])
    parent_matrix = {row.id: vectors[row.embedding_text] for row in parents}
    child_counts = Counter(row.parent_id for row in children)
    singleton = [row for row in parents if child_counts[row.id] == 1]
    child_by_parent = {row.parent_id: row for row in children}
    corpus_stats = {"parents": len(parents), "children": len(children),
                    "median_parent_chars": statistics.median(len(row.content) for row in parents),
                    "single_child_parents": len(singleton),
                    "identical_parent_child_text": sum(compact(row.content) == compact(child_by_parent[row.id].content) for row in singleton),
                    "identical_parent_child_embedding_input": sum(row.embedding_text == child_by_parent[row.id].embedding_text for row in singleton),
                    "body_fully_repeated_in_heading": sum(compact(row.content) in compact(row.heading) for row in parents)}
    results, failures = [], []
    for case in dataset["cases"]:
        if not case["answerable"]:
            continue
        query = retrieval_query(case)
        similarities = {row.id: float(value) for row, value in zip(chunks, matrix @ vectors[query])}
        pv_all = vector_rank(parents, similarities, len(parents))
        cv_all = vector_rank(children, similarities, len(children))
        bm_all = bm25_scores(query, children)
        bm_all = dict(sorted(bm_all.items(), key=lambda pair: (-pair[1], pair[0])))

        def channels(candidate_k):
            mapped_channels = []
            for raw, child in ((pv_all, False), (cv_all, True), (bm_all, True)):
                mapped = {}
                for key, value in list(raw.items())[:candidate_k]:
                    pid = by_id[key].parent_id if child else key
                    mapped[pid] = max(mapped.get(pid, 0.), value)
                mapped_channels.append(mapped)
            return mapped_channels

        def rrf(candidate_k):
            scores = {}
            for channel, weight in zip(channels(candidate_k), (.30, .35, .35)):
                for rank, key in enumerate(channel, 1):
                    scores[key] = scores.get(key, 0.) + weight / (60 + rank)
            return [(by_id[key], score) for key, score in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]

        def raw_cosine_fusion():
            mapped_channels = []
            for kind in ("parent", "child"):
                mapped = {}
                eligible = sorted([r for r in chunks if r.kind == kind], key=lambda r: (-similarities[r.id], r.id))[:16]
                for row in eligible:
                    pid = row.id if kind == "parent" else row.parent_id
                    mapped[pid] = max(mapped.get(pid, 0.), max(0., similarities[row.id]))
                mapped_channels.append(normalize_scores(mapped))
            mapped_channels.append(normalize_scores(channels(16)[2]))
            scores = {key: sum(weight * channel.get(key, 0.) for channel, weight in zip(mapped_channels, (.30, .35, .35)))
                      for key in set().union(*(set(channel) for channel in mapped_channels))}
            return [(by_id[key], value) for key, value in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]

        ranked = rank_chunks("parent_child_hybrid", query, chunks, similarities, 16)

        def mmr_order():
            pool, selected = list(ranked), []
            while pool:
                def utility(pair):
                    redundancy = max((float(parent_matrix[pair[0].id] @ parent_matrix[row.id]) for row, _ in selected), default=0.)
                    return .7 * pair[1] - .3 * redundancy
                best = max(range(len(pool)), key=lambda index: (utility(pool[index]), -pool[index][0].id))
                selected.append(pool.pop(best))
            return selected

        experiments = {
            baseline_name: (ranked, baseline_k),
            "hybrid_k8": (ranked, 8),
            "hybrid_k16": (ranked, 16),
            f"candidate64_k{baseline_k}": (rank_chunks("parent_child_hybrid", query, chunks, similarities, 64), baseline_k),
            f"rrf16_k{baseline_k}": (rrf(16), baseline_k),
            f"raw_cosine16_k{baseline_k}": (raw_cosine_fusion(), baseline_k),
            f"mmr16_k{baseline_k}": (mmr_order(), baseline_k),
            f"parent_only_k{baseline_k}": ([(by_id[key], score) for key, score in pv_all.items()], baseline_k),
            f"bm25_only_k{baseline_k}": (rank_chunks("bm25_parent_smoke", query, chunks, similarities), baseline_k),
        }
        for name, (order, top_k) in experiments.items():
            selected, used = select_context(order, top_k, 4000)
            metrics = score_case(case, selected, documents)
            results.append({"case_id": case["id"], "experiment": name, "metrics": metrics,
                            "estimated_tokens": used, "selected": [row.id for row, _ in selected]})

        base = next(row for row in results if row["case_id"] == case["id"] and row["experiment"] == baseline_name)
        if base["metrics"]["evidence_complete"]:
            continue
        candidate_ids = set().union(*(set(channel) for channel in channels(16)))
        candidate_context = [(by_id[key], 1.) for key in sorted(candidate_ids)]
        all_context = [(row, 1.) for row in parents]
        normalized_channels = [normalize_scores(channel) for channel in channels(16)]

        def contributions(parent_id):
            return {name: weight * values.get(parent_id, 0.) for name, weight, values in
                    zip(("parent_vector", "child_vector", "child_bm25"), (.30, .35, .35), normalized_channels)}

        missing = []
        for index, gold in enumerate(case["evidence"]):
            if base["metrics"]["evidence_coverage"][index] >= 1 - 1e-9:
                continue
            supports = [row for row in parents if evidence_coverage(gold, [(row, 1.)], documents) >= 1 - 1e-9]
            support_ids = {row.id for row in supports}
            ranks = {}
            for name, raw, child in (("parent_vector", pv_all, False), ("child_vector", cv_all, True), ("child_bm25", bm_all, True)):
                ranks[name] = next((rank for rank, key in enumerate(raw, 1)
                                    if (by_id[key].parent_id if child else key) in support_ids), None)
            fused_rank = next((rank for rank, (row, _) in enumerate(ranked, 1) if row.id in support_ids), None)
            first_fused_id = next((row.id for row, _ in ranked if row.id in support_ids), None)
            missing.append({"evidence_index": index, "document_id": gold["document_id"], "section": gold["section"],
                            "source_available_coverage": evidence_coverage(gold, all_context, documents),
                            "candidate_coverage": evidence_coverage(gold, candidate_context, documents),
                            "single_support_parent_ids": sorted(support_ids), "first_support_channel_ranks": ranks,
                            "fused_rank": fused_rank,
                            "first_fused_support_contributions": contributions(first_fused_id) if first_fused_id else None})
        top16 = list(pv_all.items())[:16]
        top_scores = [score for _, score in top16]
        cosines = [similarities[key] for key, _ in top16]
        failures.append({"case_id": case["id"], "question": case["question"], "missing": missing,
                         "top_k": baseline_k, "budget_omitted_top_k": len(ranked[:baseline_k]) - len(base["selected"]),
                         "top_results": [{"id": row.id, "document_id": row.document_id, "heading": row.heading, "score": score,
                                    "contributions": contributions(row.id)} for row, score in ranked[:baseline_k]],
                         "parent_top16_cosine_range": [min(cosines), max(cosines)],
                         "parent_top16_normalized_score_range": [min(top_scores) / max(top_scores), 1.]})
    summary = {}
    for experiment in dict.fromkeys(row["experiment"] for row in results):
        group = [row for row in results if row["experiment"] == experiment]
        summary[experiment] = {"evidence_complete_count": sum(row["metrics"]["evidence_complete"] for row in group),
                               "answerable_cases": len(group),
                               "macro_clause_recall": statistics.mean(row["metrics"]["clause_recall_at_k"] for row in group),
                               "mean_estimated_tokens": statistics.mean(row["estimated_tokens"] for row in group),
                               "complete_case_ids": [row["case_id"] for row in group if row["metrics"]["evidence_complete"]]}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "diagnostic_only_unreviewed_pilot", "dataset_sha256": digest(DATASET.read_bytes()),
              "baseline_report_sha256": digest(baseline_path.read_bytes()), "diagnosis_code_sha256": digest(Path(__file__).read_bytes()),
              "baseline_top_k": baseline_k, "api_calls": 0, "corpus": corpus_stats, "summary": summary, "failures": failures, "results": results}
    (output / "diagnosis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.probe_correction:
        case = next(case for case in dataset["cases"] if case["id"] == "M04")
        current = case["question"]
        settings = Settings()
        cache = EmbeddingCache(settings, cache_path)
        try:
            if cache.provider != provider:
                raise ValueError("Provider/model changed since baseline")
            query_vector = cache.embed([current])[current]
            api_calls = cache.api_calls
        finally:
            cache.close()
        similarities = {row.id: float(value) for row, value in zip(chunks, matrix @ query_vector)}
        selected, used = select_context(rank_chunks("parent_child_hybrid", current, chunks, similarities), baseline_k, 4000)
        probe = {"case_id": "M04", "intervention": "remove old user history; keep current question verbatim",
                 "query": current, "api_calls": api_calls,
                 "baseline": next(row for row in results if row["case_id"] == "M04" and row["experiment"] == baseline_name),
                 "current_question_only": {"metrics": score_case(case, selected, documents), "estimated_tokens": used,
                                           "contexts": [{"id": row.id, "document_id": row.document_id, "heading": row.heading, "score": score} for row, score in selected]},
                 "limitation": "Single-case query probe; does not execute production ContextAgent query rewriting."}
        (output / "correction-query-probe.json").write_text(json.dumps(probe, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(probe, ensure_ascii=False, indent=2))
    print(json.dumps({"corpus": corpus_stats, "summary": summary, "failures": failures}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
