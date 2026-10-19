from __future__ import annotations

import json
from dataclasses import dataclass

from app.core.enums import EmotionLabel, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.ai import AiClient, PromptTemplates, has_consult_signal, has_high_risk_signal


@dataclass
class PsychologyAssessment:
    emotion: EmotionLabel
    emotion_score: float
    risk: RiskLevel
    confidence: float
    summary: str
    safety_reassessment: bool = False
    protective_factors: str = ""
    has_plan_or_intent: bool = False
    risk_type: str = "OTHER"
    evidence_summary: str = ""


class PsychologicalAssessmentService:
    def __init__(self, ai: AiClient):
        self.ai = ai

    def assess(self, text: str, history: list[AiMessage] | None = None) -> PsychologyAssessment:
        if has_high_risk_signal(text):
            return PsychologyAssessment(
                EmotionLabel.HIGH_RISK,
                4.0,
                RiskLevel.HIGH,
                0.95,
                "检测到明确高风险表达",
                has_plan_or_intent=detect_plan_or_intent(text),
                risk_type=classify_risk_type(text),
                evidence_summary=necessary_risk_evidence(text),
            )
        try:
            raw = self.ai.complete(PromptTemplates.psychology_prompt(history or [], text))
            start = raw.find("{")
            end = raw.rfind("}")
            data = json.loads(raw[start:end + 1] if start >= 0 and end > start else raw)
            emotion = EmotionLabel(data.get("emotion", "NORMAL").upper())
            score = float(data.get("emotionScore", score_for_emotion(emotion)))
            risk = RiskLevel(data.get("risk", risk_from_score(score).value).upper())
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.75))))
            score_risk = risk_from_score(score)
            if risk_order(score_risk) > risk_order(risk):
                risk = score_risk
            if emotion == EmotionLabel.HIGH_RISK:
                risk = RiskLevel.HIGH
            safety_reassessment = bool(data.get("safetyConfirmation") or data.get("safetyReassessment"))
            protective = str(data.get("protectiveFactors", "") or "")
            has_plan = bool(data.get("hasPlanOrIntent"))
            return PsychologyAssessment(emotion, score, risk, confidence, data.get("summary", "模型评估结果"),
                                        safety_reassessment=safety_reassessment, protective_factors=protective,
                                        has_plan_or_intent=has_plan or detect_plan_or_intent(text),
                                        risk_type=classify_risk_type(text),
                                        evidence_summary=necessary_risk_evidence(text))
        except Exception:
            return heuristic(text)


def heuristic(text: str) -> PsychologyAssessment:
    if has_consult_signal(text):
        if any(word in text.lower() for word in ["抑郁", "低落", "崩溃", "难过", "depress", "hopeless"]):
            return PsychologyAssessment(EmotionLabel.DEPRESSED, 3.1, RiskLevel.MEDIUM, 0.75, "检测到低落或抑郁相关表达")
        return PsychologyAssessment(EmotionLabel.ANXIETY, 2.2, RiskLevel.LOW, 0.72, "检测到焦虑或压力相关表达")
    return PsychologyAssessment(EmotionLabel.NORMAL, 0.0, RiskLevel.LOW, 0.66, "未检测到明显风险信号")


PLAN_OR_INTENT_SIGNALS = (
    "遗书", "已经准备", "准备好了", "准备去", "打算", "决定要", "马上去", "今晚就",
    "手里有", "已经买", "已经攒", "已经站在", "正在实施", "已经割", "已经吞",
    "具体计划", "有计划", "有办法", "有手段", "means available", "have a plan",
)

PLAN_NEGATIONS = (
    "没有计划", "没计划", "没有打算", "没打算", "不会伤害", "不想伤害",
    "没有准备", "没准备", "没有手段", "没有办法伤害", "no plan", "not going to",
)


def detect_plan_or_intent(text: str) -> bool:
    """保守识别明确计划/准备行为，同时避免“没有计划”的子串误判。"""
    normalized = " ".join((text or "").lower().split())
    without_negations = normalized
    for phrase in PLAN_NEGATIONS:
        without_negations = without_negations.replace(phrase, "")
    return any(signal in without_negations for signal in PLAN_OR_INTENT_SIGNALS)


def classify_risk_type(text: str) -> str:
    normalized = (text or "").lower()
    if any(word in normalized for word in ("自杀", "不想活", "轻生", "结束生命", "suicide", "kill myself")):
        return "SUICIDE"
    if any(word in normalized for word in ("自残", "伤害自己", "self harm")):
        return "SELF_HARM"
    if any(word in normalized for word in ("伤人", "伤害别人", "报复", "harm others")):
        return "HARM_TO_OTHERS"
    if any(word in normalized for word in ("马上实施", "正在实施", "眼前危险", "immediate danger")):
        return "IMMEDIATE_DANGER"
    return "OTHER"


def necessary_risk_evidence(text: str, limit: int = 180) -> str:
    """只保留审计所需的最短证据片段；入库前还会再次执行隐私脱敏。"""
    normalized = " ".join((text or "").split())
    return normalized[:limit]


def score_for_emotion(emotion: EmotionLabel) -> float:
    return {
        EmotionLabel.HIGH_RISK: 4.0,
        EmotionLabel.DEPRESSED: 3.0,
        EmotionLabel.ANXIETY: 2.0,
        EmotionLabel.NORMAL: 0.0,
    }[emotion]


def risk_from_score(score: float) -> RiskLevel:
    if score >= 4:
        return RiskLevel.HIGH
    if score >= 3:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def risk_order(risk: RiskLevel) -> int:
    return {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}[risk]
