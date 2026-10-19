from enum import Enum


class MessageRole(str, Enum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"


class IntentType(str, Enum):
    CHAT = "CHAT"
    CONSULT = "CONSULT"
    RISK = "RISK"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    @property
    def order(self) -> int:
        return {"LOW": 1, "MEDIUM": 2, "HIGH": 3}[self]

    # str 枚举的默认 < 比较是字典序（"LOW" > "HIGH"），必须显式按风险序比较。
    def __lt__(self, other):
        return self.order < RiskLevel(other).order

    def __le__(self, other):
        return self.order <= RiskLevel(other).order

    def __gt__(self, other):
        return self.order > RiskLevel(other).order

    def __ge__(self, other):
        return self.order >= RiskLevel(other).order


class EmotionLabel(str, Enum):
    NORMAL = "NORMAL"
    ANXIETY = "ANXIETY"
    DEPRESSED = "DEPRESSED"
    HIGH_RISK = "HIGH_RISK"


class ToolStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ToolJobKind(str, Enum):
    EXCEL_REPORT = "EXCEL_REPORT"
    CASE_CREATE = "CASE_CREATE"
    ALERT_SEND = "ALERT_SEND"
    RISK_ALERT = "RISK_ALERT"


class ToolJobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    DEAD = "DEAD"


class RiskCaseStatus(str, Enum):
    OPEN = "OPEN"
    ALERT_SENT = "ALERT_SENT"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


class SafetyLifecycleStatus(str, Enum):
    """风险生命周期，与 RiskCaseStatus（工单进度）严格分离。"""

    ACTIVE = "ACTIVE"
    MONITORING = "MONITORING"
    RESOLVED = "RESOLVED"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class RiskEventKind(str, Enum):
    RISK_DETECTED = "RISK_DETECTED"
    REASSESSMENT = "REASSESSMENT"
    STATE_TRANSITION = "STATE_TRANSITION"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"


class RiskEventType(str, Enum):
    SUICIDE = "SUICIDE"
    SELF_HARM = "SELF_HARM"
    HARM_TO_OTHERS = "HARM_TO_OTHERS"
    IMMEDIATE_DANGER = "IMMEDIATE_DANGER"
    OTHER = "OTHER"


class ResponseMode(str, Enum):
    POLICY_ANSWER = "policy_answer"
    NORMAL_CHAT = "normal_chat"
    SUPPORT = "support"
    MONITORED_CHAT = "monitored_chat"
    SAFETY_SUPPORT = "safety_support"


RISK_ORDER = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}


def max_risk(left: "RiskLevel", right: "RiskLevel") -> "RiskLevel":
    return left if RISK_ORDER[left] >= RISK_ORDER[right] else right
