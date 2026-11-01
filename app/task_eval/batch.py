"""Serial candidate execution plus parallel independent grading, same protocol."""
from __future__ import annotations

import fcntl
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from app.core.config import Settings
from app.services.knowledge import SearchResult
from app.task_eval.execution import Limits, machine_checks, prepare_shared_input, run_arm, shared_skills
from app.task_eval.judging import JUDGE_MODEL
from app.task_eval.prepare import COMMON_CHECKS, ROOT, build_dataset, encode, sha256
from app.task_eval.review import check_judgment
from app.task_eval.runner import CachedTransport, grade, prepare_contexts, save, source_fingerprint, summarize


class ThreadSafeTransport:
    def __init__(self, base):
        self.base, self.guard, self.locks = base, threading.Lock(), {}

    def __call__(self, messages, settings, stage, timeout):
        key = sha256(encode({"messages": [m.model_dump() for m in messages], "model": settings.openai_model,
                             "stage": stage, "max_tokens": settings.ai_max_tokens}))
        with self.guard:
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:
            return self.base(messages, settings, stage, timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "target/task-success-v1/real-v2")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dataset, settings = build_dataset(), Settings()
    if settings.ai_provider != "openai" or not settings.openai_api_key:
        raise ValueError("Configured real OpenAI-compatible model/key required")
    protocol = {"dataset_sha256": sha256(encode(dataset)), "sources": source_fingerprint(),
                "skills_sha256": sha256(shared_skills().encode()), "model": settings.openai_model,
                "endpoint_sha256": sha256(settings.openai_base_url.rstrip("/").encode()),
                "thinking": settings.openai_enable_thinking, "judge_model": JUDGE_MODEL,
                "limits": asdict(Limits()), "scope": "Controlled known development comparison; automatic labels pending human"}
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Protocol changed; do not mix results")
    save(protocol_path, protocol); save(out / "dataset.json", dataset)
    if source_fingerprint() != protocol["sources"] or sha256(encode(dataset)) != protocol["dataset_sha256"]:
        raise ValueError("Generation source/dataset changed")
    if (settings.openai_model != protocol["model"] or settings.openai_enable_thinking != protocol["thinking"]
            or sha256(settings.openai_base_url.rstrip("/").encode()) != protocol["endpoint_sha256"]):
        raise ValueError("Generation model/endpoint changed")
    settings.knowledge_top_k, settings.rag_skill_budget_tokens = 5, 4000
    limits, skills = Limits(**protocol["limits"]), shared_skills()
    if sha256(skills.encode()) != protocol["skills_sha256"]:
        raise ValueError("Business skills changed")
    retrieval = prepare_contexts(dataset, settings, out)
    base = CachedTransport(out / "responses", sha256(encode({"protocol": protocol, "retrieval": retrieval["signature"]})))
    transport = ThreadSafeTransport(base)
    scoring = {"judge_model": JUDGE_MODEL, "review_sha256": sha256(Path(__file__).with_name("review.py").read_bytes()),
               "batch_sha256": sha256(Path(__file__).read_bytes()),
               "rule": "Original criterion rubric; explicitly contradictory boolean/reason outputs require repair, not automatic relabeling"}
    save(out / "scoring-protocol.json", scoring)
    state, pending = {}, {}

    def judge_one(case, observation, contexts, arm):
        raw_path = out / "judgments" / (case["id"] + "-" + arm + ".json")
        checked_path = out / "judgments-reviewed" / raw_path.name
        if checked_path.exists():
            return json.loads(checked_path.read_text())
        if raw_path.exists():
            raw = json.loads(raw_path.read_text())
        else:
            raw = grade(case, observation, contexts, settings, transport)
            save(raw_path, raw)
        checked = check_judgment(case, observation, contexts, raw, settings, transport)
        save(checked_path, checked)
        return checked

    def collect():
        for future, (case, arm) in list(pending.items()):
            if not future.done():
                continue
            row = state[case["id"]]
            try:
                judged = future.result()
                row["judgments"][arm] = judged
                row["checks"][arm].update(judged["checks"])
                if set(row["checks"][arm]) != set(COMMON_CHECKS):
                    raise ValueError("Incomplete check set")
                row["task_success"][arm] = all(row["checks"][arm].values())
            except Exception as exc:
                row["judgments"][arm] = {"error": type(exc).__name__}
            del pending[future]
        rows = [r for r in state.values() if set(r["observations"]) == {"single", "multi"}]
        report = {"phase": "full", "protocol": protocol, "scoring_protocol": scoring,
                  "summary": summarize(rows, len(dataset["cases"])), "results": rows,
                  "new_chat_requests_this_process": base.api_calls, "response_cache_hits_this_process": base.cache_hits}
        save(out / "full-report.json", report)
        return report

    with ThreadPoolExecutor(max_workers=4) as pool:
        for case in dataset["cases"]:
            if source_fingerprint() != protocol["sources"]:
                raise ValueError("Generation code changed during comparison")
            case_id, blocks = case["id"], retrieval["results"][case["id"]]
            group = "risk_support" if case["family"] == "risk_support" else ("policy_answerable" if case["acceptance"]["behavior"] == "answer" else "insufficient_information")
            row = {"case_id": case_id, "group": group, "observations": {}, "checks": {}, "judgments": {},
                   "task_success": {"single": None, "multi": None}}
            state[case_id] = row
            order = ("single", "multi") if int(sha256(case_id.encode()), 16) % 2 == 0 else ("multi", "single")
            for arm in order:
                path = out / "arms" / (case_id + "-" + arm + ".json")
                if path.exists():
                    observation = json.loads(path.read_text())
                else:
                    print(json.dumps({"stage": "generate", "case": case_id, "arm": arm}), flush=True)
                    observation = run_arm(arm, prepare_shared_input(case), [SearchResult(**b) for b in blocks], settings,
                                          transport=transport, limits=limits, skills=skills)
                    save(path, observation)
                row["observations"][arm] = observation
                row["checks"][arm] = machine_checks(case, observation)
                if observation["response_completed"] and observation["effect_execution_completed"]:
                    pending[pool.submit(judge_one, case, observation, blocks, arm)] = case, arm
                else:
                    row["task_success"][arm] = False
                    row["judgments"][arm] = {"status": "not_scored_due_to_system_failure"}
                collect()
            for key in ("input_sha256", "evidence_sha256", "skills_sha256", "limits", "candidate_profile", "execution_sha256"):
                if row["observations"]["single"][key] != row["observations"]["multi"][key]:
                    raise ValueError("Fairness mismatch: " + key)
            print(json.dumps({"stage": "generated", "case": case_id, "finished_cases": len(state),
                              "pending_judgments": len(pending)}), flush=True)
        for future in list(pending):
            try: future.result()
            except Exception: pass
            collect()
    report = collect()
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
