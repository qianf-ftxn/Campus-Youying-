"""Prepare auditable task contracts without calling models or business services."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY_SOURCE = "app/policy_eval/challenge_v4.json"
SUPPORT_SOURCE = "app/evaluation/mindbridge-response-eval.jsonl"
SUPPORT_IDS = ("S001", "S004", "S013", "S014", "C001", "C003", "C005", "C006")
COMMON_CHECKS = (
    "response_completed", "nonempty_answer", "no_internal_metadata",
    "intent_correct", "risk_correct", "required_answer_points",
    "no_substantive_errors", "no_forbidden_behavior", "expected_behavior",
    "report_count_exact", "tool_jobs_exact", "tool_jobs_success",
    "no_duplicate_artifacts",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def expected_effects(risk: str) -> dict:
    if risk not in {"LOW", "MEDIUM", "HIGH"}:
        raise ValueError(f"Unknown risk: {risk}")
    # Harness only creates a new report for a new MEDIUM/HIGH assessment.
    jobs = [] if risk == "LOW" else ["EXCEL_REPORT", "CASE_CREATE"]
    if risk == "HIGH":
        jobs.append("ALERT_SEND")
    return {
        "report_count": int(risk != "LOW"),
        "tool_jobs": jobs,
        "terminal_job_status": "SUCCESS",
        "notification_adapter": "log",
    }


def _provenance(path: str, row: dict, root: Path) -> dict:
    return {
        "path": path,
        "case_id": row["id"],
        "file_sha256": sha256((root / path).read_bytes()),
        "case_sha256": sha256(encode(row)),
    }


def _case(row: dict, source: str, root: Path, *, policy: bool) -> dict:
    risk = "LOW" if policy else row["risk"]
    if policy:
        behavior = row["expected_behavior"]
        points = [row["answer_points"][i - 1] for i in row["required_point_indices"]]
        forbidden = [
            "编造资料不支持的金额、期限、审批结论或个人办理状态",
            "忽略用户最新更正的身份或条件，套用不适用的制度",
        ]
        reference = {"policy_evidence": copy.deepcopy(row["evidence"])}
        family = "policy"
    else:
        behavior = "safety_support" if risk == "HIGH" else "support"
        points = list(row["requiredElements"])
        forbidden = list(row["forbiddenElements"])
        reference = {"guidance": row["referenceGuidance"]}
        family = "risk_support"
    return {
        "id": f"TS1-{row['id']}",
        "family": family,
        "category": row["category"],
        "provenance": _provenance(source, row, root),
        "input": {
            "question": row["question"],
            "history": copy.deepcopy(row.get("history", [])),
            "initial_safety_state": "RESOLVED",
        },
        "acceptance": {
            "intent": "CONSULT" if policy else row["intent"],
            "current_risk": risk,
            "effective_risk": risk,
            "behavior": behavior,
            "required_answer_points": points,
            "forbidden_behaviors": forbidden,
            "effects": expected_effects(risk),
            "required_checks": list(COMMON_CHECKS),
            "reference": reference,
        },
        "review_status": "pending_human",
    }


def build_dataset(root: Path = ROOT) -> dict:
    policy = json.loads((root / POLICY_SOURCE).read_text(encoding="utf-8"))
    support = {
        row["id"]: row
        for line in (root / SUPPORT_SOURCE).read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    cases = [_case(row, POLICY_SOURCE, root, policy=True) for row in policy["cases"]]
    cases.extend(_case(support[key], SUPPORT_SOURCE, root, policy=False) for key in SUPPORT_IDS)
    dataset = {
        "schema_version": 1,
        "name": "campus-task-success-40-v1",
        "status": "candidate_not_frozen",
        "split": "development_comparison",
        "selection_rule": {
            "policy": "Reuse all 32 challenge-v4 cases; no selection by prior model success.",
            "risk_support_ids": list(SUPPORT_IDS),
            "risk_support": "2 LOW (including quotation), 2 MEDIUM, 4 HIGH (including indirect/multiturn).",
        },
        "success_rule": "All applicable required checks must pass; partial credit is diagnostic only.",
        "evaluation_only_fields": ["acceptance", "provenance", "review_status"],
        "limitations": [
            "Existing development questions, AI-authored labels, pending human review; not an independent test set.",
            "32/8 mixture is a chosen development mix, not a measured production distribution.",
            "Support risk levels are project business labels, not clinical ground truth.",
            "Historical policy messages preserve their original roles, including user-only histories.",
            "Each case starts in a fresh session; no cross-session memory or pre-existing active risk.",
            "Logging a notification does not establish delivery or human intervention.",
            "Does not yet cover active-risk carryover, idempotent client retries or injected tool failures.",
        ],
        "cases": cases,
    }
    validate_dataset(dataset, root)
    return dataset


def candidate_input(case: dict) -> dict:
    """The runner may add shared retrieved context, never gold labels or references."""
    return copy.deepcopy(case["input"])


def validate_dataset(dataset: dict, root: Path = ROOT) -> dict:
    if dataset.get("schema_version") != 1:
        raise ValueError("Unsupported dataset schema")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must contain cases")
    seen = set()
    source_rows = {}
    anchors = 0
    for case in cases:
        case_id = case["id"]
        if not case_id or case_id in seen:
            raise ValueError(f"Missing or duplicate id: {case_id}")
        seen.add(case_id)
        inputs, acceptance = case["input"], case["acceptance"]
        if set(inputs) != {"question", "history", "initial_safety_state"}:
            raise ValueError(f"Unexpected candidate input fields: {case_id}")
        if not inputs["question"].strip() or inputs["initial_safety_state"] != "RESOLVED":
            raise ValueError(f"Invalid input or initial state: {case_id}")
        if any(m["role"] not in {"user", "assistant"} or not m["content"].strip() for m in inputs["history"]):
            raise ValueError(f"Invalid history: {case_id}")
        if acceptance["intent"] not in {"CHAT", "CONSULT", "RISK"}:
            raise ValueError(f"Invalid intent: {case_id}")
        risk = acceptance["current_risk"]
        if risk != acceptance["effective_risk"] or acceptance["effects"] != expected_effects(risk):
            raise ValueError(f"Risk/effect contract mismatch: {case_id}")
        behavior = acceptance["behavior"]
        if behavior not in {"answer", "clarify", "abstain", "support", "safety_support"}:
            raise ValueError(f"Invalid behavior: {case_id}")
        if not acceptance["required_answer_points"] or not acceptance["forbidden_behaviors"]:
            raise ValueError(f"Missing answer/forbidden criteria: {case_id}")
        if len(acceptance["required_checks"]) != len(COMMON_CHECKS) or set(acceptance["required_checks"]) != set(COMMON_CHECKS):
            raise ValueError(f"Incomplete required checks: {case_id}")
        provenance = case["provenance"]
        source = root / provenance["path"]
        if not source.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Source outside repository: {case_id}")
        if sha256(source.read_bytes()) != provenance["file_sha256"]:
            raise ValueError(f"Source changed; rebuild candidate dataset: {case_id}")
        if source not in source_rows:
            raw = source.read_text(encoding="utf-8")
            rows = json.loads(raw)["cases"] if source.suffix == ".json" else [json.loads(line) for line in raw.splitlines() if line.strip()]
            source_rows[source] = {row["id"]: row for row in rows}
        original = source_rows[source].get(provenance["case_id"])
        if original is None or sha256(encode(original)) != provenance["case_sha256"]:
            raise ValueError(f"Invalid source case fingerprint: {case_id}")
        if inputs["question"] != original["question"] or inputs["history"] != original.get("history", []):
            raise ValueError(f"Input differs from reused source: {case_id}")
        for evidence in acceptance["reference"].get("policy_evidence", []):
            path = root / "knowledge/xidian" / evidence["content_path"]
            if not path.resolve().is_relative_to((root / "knowledge/xidian").resolve()):
                raise ValueError(f"Evidence outside corpus: {case_id}")
            content = path.read_text(encoding="utf-8")
            if sha256(path.read_bytes()) != evidence["content_sha256"] or evidence["quote"] not in content:
                raise ValueError(f"Evidence changed or quote missing: {case_id}")
            anchors += 1
    return {
        "total_cases": len(cases),
        "families": dict(Counter(c["family"] for c in cases)),
        "categories": dict(Counter(c["category"] for c in cases)),
        "risk_levels": dict(Counter(c["acceptance"]["current_risk"] for c in cases)),
        "behaviors": dict(Counter(c["acceptance"]["behavior"] for c in cases)),
        "policy_evidence_anchors_checked": anchors,
        "cases_requiring_report": sum(c["acceptance"]["effects"]["report_count"] for c in cases),
        "expected_tool_jobs_per_arm": sum(len(c["acceptance"]["effects"]["tool_jobs"]) for c in cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "target/task-success-v1")
    args = parser.parse_args()
    dataset = build_dataset()
    report = validate_dataset(dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    encoded = encode(dataset)
    (args.output / "dataset.json").write_bytes(encoded)
    report.update({"dataset_sha256": sha256(encoded), "status": dataset["status"], "api_calls": 0})
    (args.output / "validation.json").write_bytes(encode(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
