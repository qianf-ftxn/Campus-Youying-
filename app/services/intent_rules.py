"""CONSULT means a request for institution-specific facts, not a generic question."""
import json
import re
from dataclasses import dataclass

from app.core.enums import IntentType

POLICY_TOPICS = (
    "校规", "规章", "学生手册", "学籍", "休学", "复学", "培养方案", "培养规定", "毕业条件", "学分要求",
    "奖学金", "助学金", "差旅", "出差", "报销", "餐补", "伙食补助", "住宿费", "补贴", "学费", "请假",
    "选课", "退课", "补考", "缓考", "重修", "心理中心", "心理咨询中心", "图书馆", "宿舍", "寝室", "学位申请",
    "食堂", "校医院", "校园卡", "一卡通", "学生证", "图书借阅", "校车", "校历", "转专业", "退学", "答辩", "送审",
    "转导师", "转学科", "学位授予", "学位复核", "双盲评审", "课程置换", "学分认定", "必修环节", "综合考试",
    "销假", "违纪处分", "电子注册", "助研津贴", "助研岗位", "助管岗位", "助教工作", "推免", "绩点",
    "校外住宿", "校外租房", "公寓管理", "退宿", "调宿", "校外学习实践",
)
CAMPUS_ANCHORS = ("学校", "本校", "校内", "校园", "学院", "教务", "财务处", "研究生院", "学生处", "辅导员", "西电", "西安电子科技大学")
DOCUMENT_CUES = ("校规", "规章", "政策", "规定", "办法", "指南", "制度", "标准", "条件", "要求", "截止", "流程", "申请材料")
FACT_CUES = DOCUMENT_CUES + ("多少", "哪些", "什么", "怎么", "如何", "是否", "能否", "可以", "需要", "查询", "请问", "想了解", "介绍", "解释", "电话", "地址", "在哪", "时间", "几点", "办理", "预约", "申请")


def _is_definition(text: str) -> bool:
    definition = re.fullmatch(r"(?:什么是(.{1,8})|(.{1,8})是什么)[？?。！!]*", text)
    if definition and not any(cue in text for cue in DOCUMENT_CUES) and not any(anchor in text for anchor in CAMPUS_ANCHORS):
        if not any(word in text for word in ("每天", "出差", "本次", "的", "预约", "办理", "方式", "时间", "地点", "电话", "地址", "金额")):
            return True
    return False


def _general_task(text: str) -> bool:
    # 动作与对象共同出现才是明确的通用任务；Java、论文、帮我等词本身不决定意图。
    return bool(re.search(
        r"(?:写|起草|生成|润色|改写|翻译|解释|介绍|设计|开发|调试|修复|总结|讲).{0,30}"
        r"(?:代码|程序|算法|脚本|系统|软件|接口|数据库|邮件|申请书|作文|故事|笑话|诗|论文|文章|字典|python|java|sql)"
        r"|(?:代码|程序|算法|脚本|系统|软件|数据库|接口|报错|bug|压力测试|压强|python|java|sql).{0,25}"
        r"(?:怎么|如何|是什么|为什么|解释|实现|设计|修复|写|做)"
        r"|(?:翻译|润色|改写|讲个笑话|讲笑话|讲个故事)|(?:今天|明天|现在|最近).{0,8}天气|^那(?:么)?天气呢[？?。]*$",
        text, re.I))


def has_emotional_support_signal(text: str) -> bool:
    """只匹配明确的现实情绪表达，模糊表达留给模型，不按单个情绪词判定。"""
    lower = text.lower()
    # 引文不代表说话者本人；引文以外的真实求助仍参与匹配。
    lower = re.sub(r'“[^”]*”|「[^」]*」|"[^"\n]*"', "", lower)
    lower = re.sub(r"压力测试|压力传感器|压力容器|程序崩溃|系统崩溃|服务崩溃|stress test", "", lower)
    lower = re.sub(r"(?:我)?(?:并不|没有|不再|不)(?:感到|觉得)?(?:焦虑|抑郁|难过|失眠|痛苦|无助)", "", lower)
    emotion = r"(?:过得不好|过的不好|心情不好|很难受|不开心|焦虑|抑郁|压力|失眠|难过|崩溃|痛苦|无助|情绪低落|睡不着|撑不住)"
    state = r"(?:最近|现在|一直|每天|总是|都|感到|觉得|感觉|有点|有些|有|很|特别|非常|好|太|挺|持续|越来越|有一点|有一些)*"
    return bool(
        re.search(rf"(?:我|最近|这几天|每天|这段时间){state}{emotion}", lower)
        or re.search(rf"{emotion}[^。！？；\n]{{0,20}}(?:让我|影响到我|影响学习|想找人聊聊|需要支持|需要心理咨询|怎么办)", lower)
        or re.search(r"(?:我)?(?:很想|想|需要)找人(?:聊聊|倾诉)|(?:我觉得|我感到|最近)?(?:好|很|特别|有点)(?:难过|无助|痛苦|焦虑|抑郁|崩溃)", lower)
        or re.search(r"\bi(?:'m| am| feel| have been)\s+(?:very\s+)?(?:anxious|depressed|stressed|hopeless)\b", lower)
    )


def has_policy_signal(text: str) -> bool:
    return any(topic in text for topic in POLICY_TOPICS) or (
        any(anchor in text for anchor in CAMPUS_ANCHORS) and any(cue in text for cue in FACT_CUES))


def is_policy_request(text: str) -> bool:
    if not has_policy_signal(text) or _is_definition(text):
        return False
    explicit_document = any(cue in text for cue in DOCUMENT_CUES)
    # 写作/编程可与制度查询并存。只有明确依赖权威事实时才直接 CONSULT。
    if _general_task(text):
        normative = bool(re.search(
            r"(?:按|依据|根据|符合|遵守).{0,25}(?:校规|规定|政策|办法|制度|标准|要求)"
            r"|(?:校规|规定|政策|制度|办法).{0,15}(?:写|起草|生成|翻译|总结|润色)"
            r"|(?:校规|规定|政策|制度|办法|标准|申请条件|办理流程|所需材料).{0,15}(?:是什么|哪些|多少|查询|怎么|如何)"
            r"|(?:补考|缓考|重修|报销|休学|复学|请假|奖学金).{0,12}(?:怎么申请|如何申请|怎么报销|如何报销|多少|需要哪些|需要什么)", text))
        normative = normative or bool(re.search(
            rf"(?:{'|'.join(POLICY_TOPICS)}).{{0,18}}"
            r"(?:规定|规章|政策|制度|标准|条件|截止|流程|申请材料|多少|几点|能否|能不能|可不可以|怎么申请|如何申请|怎么办理|如何办理|需要哪些|需要什么)", text))
        if not normative:
            return False
    if not explicit_document and any(word in text for word in ("建议", "技巧", "经验", "怎么准备", "如何准备", "规划", "学习方法", "怎么写", "如何写")):
        return False
    if _is_sharing(text):
        return False
    return explicit_document or any(cue in text for cue in FACT_CUES)


def _is_sharing(text: str) -> bool:
    return (any(word in text for word in ("办完", "交完", "完成了", "拿到", "到账", "开心", "分享", "松口气"))
            and not any(cue in text for cue in FACT_CUES + ("？", "?", "能不能", "能带", "吗", "几时")))


def is_general_request(text: str) -> bool:
    if has_emotional_support_signal(text) or is_policy_request(text):
        return False
    # 多个子请求中若还有未决业务对象，不能因其中一个写作/编程任务强行 CHAT。
    clauses = [part.strip() for part in re.split(r"[，,；;。！？!?\n]+", text) if part.strip()]
    if len(clauses) > 1 and any(has_policy_signal(part) and not _general_task(part) and not _is_sharing(part)
                                and not _is_definition(part) for part in clauses):
        return False
    return (_is_definition(text) or _general_task(text) or _is_sharing(text)
            or bool(re.fullmatch(r"(?:你好|您好|嗨|哈喽|早上好|晚上好|谢谢(?:你|啦|了)?|好的|好|再见|你是谁|我想咨询一下)[啊呀呢吧哦哈～~！!？?。，\s]*", text)))


def is_policy_followup(text: str, previous_users: list[str], summary: str = "") -> bool:
    if len(text) > 60 or is_general_request(text) or has_emotional_support_signal(text):
        return False
    elliptical = bool(re.match(r"^(?:那|那么|还有|如果|这个|那个|刚才|上次|更正|我其实)", text))
    identity = bool(re.fullmatch(r"我?是?(?:研究生|本科生|硕士生|博士生)[。！.]?", text))
    if not elliptical and not identity:
        return False
    concrete = any(word in text for word in POLICY_TOPICS + (
        "住宿", "伙食", "交通", "材料", "金额", "标准", "发票", "申请", "截止", "年级", "费用", "研究生", "本科生", "硕士生", "博士生"))
    generic = bool(re.fullmatch(r"(?:那|那么|这个|那个)(?:怎么办|需要多久|要多久|可以吗|行吗|每天多少|需要什么|还有什么)[？?。！!]*", text))
    if not identity and not concrete and not generic:
        return False
    # 仅沿连续、明确的业务追问回溯；历史摘要不能单独证明当前仍在该话题。
    for previous in reversed(previous_users[-6:]):
        if is_policy_request(previous):
            return True
        if is_general_request(previous):
            return False
        if re.fullmatch(r"我?是?(?:研究生|本科生|硕士生|博士生)[。！.]?", previous):
            continue
        if re.match(r"^(?:那|那么|还有|如果|更正)", previous) and len(previous) < 45 and has_policy_signal(previous):
            continue
        return False
    return False


@dataclass(frozen=True)
class IntentRuleDecision:
    # None 是规则内部 UNKNOWN，不新增对外业务意图。
    intent: IntentType | None
    rule_id: str


def parse_intent_label(raw: str) -> IntentType | None:
    """接纳完整枚举或结构化 intent；不从自然语言解释中搜索标签。"""
    label = raw.strip().upper()
    if label in {intent.value for intent in IntentType}:
        return IntentType(label)
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("intent"), str):
        return None
    label = data["intent"].strip().upper()
    return IntentType(label) if label in {intent.value for intent in IntentType} else None


def classify_intent_rules(text: str, previous_users: list[str]) -> IntentRuleDecision:
    text = text.strip().lower()
    if has_emotional_support_signal(text):
        return IntentRuleDecision(IntentType.RISK, "personal_emotional_support")
    if is_policy_request(text):
        return IntentRuleDecision(IntentType.CONSULT, "institutional_fact_request")
    if is_general_request(text):
        return IntentRuleDecision(IntentType.CHAT, "clear_general_request")
    if is_policy_followup(text, previous_users):
        return IntentRuleDecision(IntentType.CONSULT, "current_policy_followup")
    return IntentRuleDecision(None, "unknown")
