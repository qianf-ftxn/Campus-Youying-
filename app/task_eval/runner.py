"""Resumable real pilot/full development comparison. Uses only isolated tools."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.schemas.dtos import AiMessage
from app.services.knowledge import SearchResult
from app.task_eval.execution import (Completion, Limits, machine_checks, openai_transport,
                                    prepare_shared_input, run_arm, shared_skills)
from app.task_eval.judging import JUDGE_MODEL, PROMPT, judge_input, parse_judgment, semantic_checks
from app.task_eval.prepare import ROOT, COMMON_CHECKS, POLICY_SOURCE, build_dataset, encode, sha256

PILOT_IDS = ("C4-X01", "C4-B03", "C4-M03", "C4-U01", "C4-U06", "S004", "S014", "C005")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(encode(value)); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def source_fingerprint():
    paths = ["app/task_eval/execution.py", "app/task_eval/judging.py", "app/task_eval/runner.py",
             "app/agents/autonomous.py", "app/agents/coordinator.py", "app/agents/event_driven_runtime.py",
             "app/agents/harness.py", "app/services/ai.py", "app/services/assessment.py",
             "app/services/intent_rules.py", "app/services/knowledge_chunking.py", "app/services/tool_queue.py",
             "app/services/tools.py", "app/services/safety_memory.py", "app/services/memory_context.py"]
    return {p: sha256((ROOT / p).read_bytes()) for p in paths}


class CachedTransport:
    def __init__(self, root, namespace):
        self.root, self.namespace = root, namespace
        self.api_calls = self.cache_hits = 0

    def __call__(self, messages, settings, stage, timeout):
        key = sha256(encode({"namespace": self.namespace, "model": settings.openai_model,
                             "endpoint": sha256(settings.openai_base_url.rstrip("/").encode()),
                             "thinking": settings.openai_enable_thinking,
                             "temperature": settings.ai_temperature, "max_tokens": settings.ai_max_tokens,
                             "stage": stage, "messages": [m.model_dump() for m in messages]}))
        path = self.root / (key + ".json")
        if path.exists():
            data = json.loads(path.read_text())
            self.cache_hits += 1
            return Completion(data["text"], data["usage"], data["finish_reason"], True)
        self.api_calls += 1
        result = openai_transport(messages, settings, stage, timeout)
        save(path, {"request_sha256": key, "stage": stage, "text": result.text,
                    "usage": result.usage, "finish_reason": result.finish_reason,
                    "messages": [m.model_dump() for m in messages]})
        return result


def prepare_contexts(dataset, settings, out):
    """One joint policy/support corpus for every question, no gold domain filter."""
    from app.policy_eval.runner import EmbeddingCache, build_chunks, load_inputs, rank_chunks, select_context
    _, documents = load_inputs(ROOT / POLICY_SOURCE)
    documents = dict(documents)
    for path in sorted((ROOT / "app/knowledge").glob("*.md")):
        text = path.read_text()
        identity = "support/" + path.stem
        documents[identity] = ({"id": identity, "title": path.stem, "version": sha256(text.encode()),
                                "source_url": "", "domain": "support"}, text)
    signature = sha256(encode({"documents": {key: sha256(encode(value)) for key, value in documents.items()},
                               "inputs": [prepare_shared_input(c) for c in dataset["cases"]],
                               "embedding_model": settings.openai_embedding_model,
                               "endpoint": sha256(settings.openai_base_url.rstrip("/").encode()),
                               "ranking": "parent_child_hybrid", "top_k": 5, "budget": 4000}))
    path = out / "retrieval.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["signature"] != signature:
            raise ValueError("Retrieval inputs changed; use a new experiment output directory")
        return result
    chunks = [c for c in build_chunks(documents) if c.kind != "flat"]
    queries = {}
    for case in dataset["cases"]:
        inputs = prepare_shared_input(case)
        queries[case["id"]] = "\n".join([m["content"] for m in inputs["history"] if m["role"] == "user"] + [inputs["question"]])
    cache = EmbeddingCache(settings, ROOT / "target/policy-pilot/embeddings.sqlite3")
    try:
        vectors = cache.embed([c.embedding_text for c in chunks] + list(queries.values()))
        stats = {"new_embedding_texts": cache.new_texts, "embedding_api_calls": cache.api_calls,
                 "cached_embedding_texts": cache.cached_texts}
    finally:
        cache.close()
    matrix = np.vstack([vectors[c.embedding_text] for c in chunks])
    results = {}
    for case_id, query in queries.items():
        similarities = {c.id: float(score) for c, score in zip(chunks, matrix @ vectors[query])}
        ranked = rank_chunks("parent_child_hybrid", query, chunks, similarities, 16)
        selected, _ = select_context(ranked, 5, 4000)
        results[case_id] = [asdict(SearchResult(c.id, c.document_id, c.content, score,
                                  title=documents[c.document_id][0]["title"], version=c.version,
                                  source_url=documents[c.document_id][0].get("source_url", ""),
                                  domain="support" if c.document_id.startswith("support/") else "policy",
                                  heading=c.heading)) for c, score in selected]
    result = {"signature": signature, "corpus_documents": len(documents), "stats": stats, "results": results,
              "scope": "Joint 49-policy/11-support corpus; same K5 hybrid contexts supplied to both arms, no rerank"}
    save(path, result)
    return result


def grade(case, observation, contexts, settings, transport):
    config = copy.copy(settings)
    config.openai_model, config.ai_temperature, config.ai_max_tokens = JUDGE_MODEL, 0., 4096
    messages = [AiMessage(role="system", content=PROMPT),
                AiMessage(role="user", content=json.dumps(judge_input(case, observation, contexts), ensure_ascii=False))]
    errors = []
    for _ in range(2):
        response = transport(messages, config, "IndependentJudge", 90.)
        try:
            if response.finish_reason != "stop":
                raise ValueError("Judge response incomplete")
            parsed = parse_judgment(response.text, case)
            return {"judgment": parsed, "checks": semantic_checks(case, parsed), "format_errors": errors}
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(type(exc).__name__)
            messages.extend([AiMessage(role="assistant", content=response.text),
                             AiMessage(role="system", content="仅修复JSON结构并补齐全部要点和禁止行为的布尔判定。不要更改验收规则。")])
    raise ValueError("Judge did not return a valid complete criterion set")


def summarize(results, total):
    summary = {"planned_tasks": total, "completed_pairs": len(results), "arms": {}, "groups": {}}
    for arm in ("single", "multi"):
        observations = [r["observations"][arm] for r in results]
        flags = [r["task_success"][arm] for r in results]
        complete = len(flags) == total and all(v is not None for v in flags)
        summary["arms"][arm] = {"successful_tasks": sum(v is True for v in flags),
                                 "scored_tasks": sum(v is not None for v in flags),
                                 "task_success_rate": sum(v is True for v in flags) / total if complete else None,
                                 "model_calls": sum(o["model_calls"] for o in observations),
                                 "input_tokens": sum(o["input_tokens"] for o in observations),
                                 "output_tokens": sum(o["output_tokens"] for o in observations),
                                 "system_failures": sum(not o["response_completed"] or not o["effect_execution_completed"] for o in observations),
                                 "failed_checks": dict(Counter(k for r in results for k, v in r["checks"][arm].items() if v is False))}
    for group in sorted({r["group"] for r in results}):
        rows = [r for r in results if r["group"] == group]
        summary["groups"][group] = {"tasks": len(rows), **{arm: sum(r["task_success"][arm] is True for r in rows) for arm in ("single", "multi")}}
    summary["multi_only_success"] = [r["case_id"] for r in results if r["task_success"] == {"single": False, "multi": True}]
    summary["single_only_success"] = [r["case_id"] for r in results if r["task_success"] == {"single": True, "multi": False}]
    summary["status"] = "complete_development_automatic_scoring" if all(a["task_success_rate"] is not None for a in summary["arms"].values()) else "incomplete"
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "target/task-success-v1/real")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    settings = Settings()
    if settings.ai_provider != "openai" or not settings.openai_api_key:
        raise ValueError("Configured real OpenAI-compatible model/key required")
    settings.knowledge_top_k, settings.rag_skill_budget_tokens = 5, 4000
    dataset, limits, skills = build_dataset(), Limits(), shared_skills()
    protocol = {"dataset_sha256": sha256(encode(dataset)), "sources": source_fingerprint(),
                "skills_sha256": sha256(skills.encode()), "model": settings.openai_model,
                "endpoint_sha256": sha256(settings.openai_base_url.rstrip("/").encode()),
                "thinking": settings.openai_enable_thinking, "judge_model": JUDGE_MODEL,
                "limits": asdict(limits), "scope": "Controlled known development comparison; automatic labels pending human"}
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Protocol changed; use a new output directory, do not mix results")
    save(protocol_path, protocol); save(args.output / "dataset.json", dataset)
    contexts = prepare_contexts(dataset, settings, args.output)
    namespace = sha256(encode({"protocol": protocol, "retrieval": contexts["signature"]}))
    transport = CachedTransport(args.output / "responses", namespace)
    cases = dataset["cases"] if args.phase == "full" else [c for c in dataset["cases"] if c["provenance"]["case_id"] in PILOT_IDS]
    results = []
    for case in cases:
        if source_fingerprint() != protocol["sources"]:
            raise RuntimeError("Source files changed during experiment; stopped to avoid mixing code versions")
        case_id = case["id"]
        inputs = prepare_shared_input(case)
        blocks = contexts["results"][case_id]
        rows = [SearchResult(**b) for b in blocks]
        observations, checks, success, judgments = {}, {}, {}, {}
        order = ("single", "multi") if int(sha256(case_id.encode()), 16) % 2 == 0 else ("multi", "single")
        for arm in order:
            path = args.output / "arms" / (case_id + "-" + arm + ".json")
            if path.exists():
                observation = json.loads(path.read_text())
            else:
                print(json.dumps({"stage": "generate", "case": case_id, "arm": arm}), flush=True)
                observation = run_arm(arm, inputs, rows, settings, transport=transport, limits=limits, skills=skills)
                save(path, observation)
            observations[arm] = observation
            checks[arm] = machine_checks(case, observation)
            if observation["response_completed"] and observation["effect_execution_completed"]:
                path = args.output / "judgments" / (case_id + "-" + arm + ".json")
                try:
                    if path.exists():
                        judgment = json.loads(path.read_text())
                    else:
                        judgment = grade(case, observation, blocks, settings, transport)
                        save(path, judgment)
                    checks[arm].update(judgment["checks"])
                    judgments[arm] = judgment
                    if set(checks[arm]) != set(COMMON_CHECKS):
                        raise ValueError("Incomplete success criteria")
                    success[arm] = all(checks[arm].values())
                except Exception as exc:
                    judgments[arm] = {"error": type(exc).__name__}
                    success[arm] = None
            else:
                success[arm] = False
                judgments[arm] = {"status": "not_scored_due_to_system_failure"}
        for key in ("input_sha256", "evidence_sha256", "skills_sha256", "limits", "candidate_profile", "execution_sha256"):
            if observations["single"][key] != observations["multi"][key]:
                raise ValueError("Arm fairness mismatch: " + key)
        group = "risk_support" if case["family"] == "risk_support" else ("insufficient_information" if case["acceptance"]["behavior"] != "answer" else "policy_answerable")
        result = {"case_id": case_id, "group": group, "checks": checks, "task_success": success,
                  "observations": observations, "judgments": judgments}
        results.append(result)
        report = {"phase": args.phase, "protocol": protocol, "retrieval_signature": contexts["signature"],
                  "summary": summarize(results, len(cases)), "results": results,
                  "new_chat_requests_this_process": transport.api_calls, "response_cache_hits_this_process": transport.cache_hits}
        save(args.output / (args.phase + "-report.json"), report)
        print(json.dumps({"stage": "scored", "case": case_id, "success": success,
                          "calls": {a: observations[a]["model_calls"] for a in observations}}, ensure_ascii=False), flush=True)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
