"""Reject explicitly self-contradictory judge labels; preserve all raw scores."""
import copy
import json
import re

from app.schemas.dtos import AiMessage
from app.task_eval.judging import JUDGE_MODEL, PROMPT, judge_input, parse_judgment, semantic_checks


def inconsistent_labels(judgment):
    problems = []
    for field, flag in (("point_verdicts", "covered"), ("forbidden_verdicts", "violated")):
        for row in judgment[field]:
            explicit = re.findall(r"(?:判定|应判|判为|应为|判断为)[^。；\n]{0,8}?(true|false)(?![a-z])", row["reason"], re.I)
            if explicit and (explicit[-1].lower() == "true") != row[flag]:
                problems.append({"field": field, "index": row["index"], "label": row[flag], "reason": row["reason"]})
    return problems


def check_judgment(case, observation, contexts, raw, settings, transport):
    original = raw
    current = raw["judgment"]
    problems = inconsistent_labels(current)
    history = []
    if not problems:
        return {**raw, "consistency_review": {"triggered": False, "repairs": []}}
    config = copy.copy(settings)
    config.openai_model, config.ai_temperature, config.ai_max_tokens = JUDGE_MODEL, 0., 4096
    messages = [AiMessage(role="system", content=PROMPT),
                AiMessage(role="user", content=json.dumps(judge_input(case, observation, contexts), ensure_ascii=False))]
    for _ in range(2):
        messages.extend([AiMessage(role="assistant", content=json.dumps(current, ensure_ascii=False)),
                         AiMessage(role="system", content=(
                             "输出存在结构语义不自洽：某个布尔标签与对应reason中的明确判定相反。"
                             "请重新核对全部原始数据与原验收规则，返回相同结构的完整JSON。"
                             "不要因为本提示偏向通过或不通过；每项布尔值必须与该项最终理由一致。"))])
        response = transport(messages, config, "IndependentJudgeConsistency", 90.)
        if response.finish_reason != "stop":
            raise ValueError("Consistency review output incomplete")
        revised = parse_judgment(response.text, case)
        history.append({"previous": current, "explicit_conflicts": problems, "revised": revised})
        current, problems = revised, inconsistent_labels(revised)
        if not problems:
            return {"judgment": revised, "checks": semantic_checks(case, revised),
                    "consistency_review": {"triggered": True, "repairs": history, "original": original}}
    raise ValueError("Explicit judge label/reason conflict remains unresolved")
