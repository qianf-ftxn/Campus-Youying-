"""CONSULT means a request for institution-specific facts, not a generic question."""
import re

POLICY_TOPICS = (
    "校规", "规章", "学生手册", "学籍", "休学", "复学", "培养方案", "培养规定", "毕业条件", "学分要求",
    "奖学金", "助学金", "差旅", "出差", "报销", "餐补", "伙食补助", "住宿费", "补贴", "学费", "请假",
    "选课", "退课", "补考", "缓考", "重修", "心理中心", "心理咨询中心", "图书馆", "宿舍", "学位申请",
    "食堂", "校医院", "校园卡", "一卡通", "学生证", "图书借阅", "校车", "校历", "转专业", "退学", "答辩", "送审",
    "转导师", "转学科", "学位授予", "学位复核", "双盲评审", "课程置换", "学分认定", "必修环节", "综合考试",
    "销假", "违纪处分", "电子注册", "助研津贴", "助研岗位", "助管岗位", "助教工作", "推免", "绩点",
    "校外住宿", "校外租房", "公寓管理", "退宿", "调宿", "校外学习实践",
)
CAMPUS_ANCHORS = ("学校", "本校", "校内", "校园", "学院", "教务", "财务处", "研究生院", "学生处", "辅导员", "西电", "西安电子科技大学")
DOCUMENT_CUES = ("校规", "规章", "政策", "规定", "办法", "指南", "制度", "标准", "条件", "要求", "截止", "流程", "申请材料")
FACT_CUES = DOCUMENT_CUES + ("多少", "哪些", "什么", "怎么", "如何", "是否", "能否", "可以", "需要", "查询", "请问", "想了解", "介绍", "解释", "电话", "地址", "在哪", "时间", "几点", "办理", "预约", "申请")


def is_general_request(text: str) -> bool:
    lower = text.lower()
    if any(word in lower for word in ("代码", "编程", "python", "javascript", "java", "算法", "脚本", "sql", "数据库", "前端", "后端", "报错", "bug", "翻译", "润色", "改写", "讲个笑话", "讲笑话", "讲个故事", "天气")):
        return True
    if re.search(r"(?:写|起草|生成).{0,15}(?:邮件|申请书|作文|故事|笑话|诗|代码)|(?:开发|设计).{0,12}(?:系统|软件|程序|接口)", text):
        return True
    if re.search(r"^(?:什么是|解释一下)(?:焦虑|抑郁|心理咨询|机器学习)|^(?:焦虑|抑郁|心理咨询)是什么", text):
        return True
    definition = re.fullmatch(r"(?:什么是(.{1,8})|(.{1,8})是什么)[？?。！!]*", text)
    if definition and not any(cue in text for cue in DOCUMENT_CUES) and not any(anchor in text for anchor in CAMPUS_ANCHORS):
        if not any(word in text for word in ("每天", "出差", "本次", "的", "预约", "办理", "方式", "时间", "地点", "电话", "地址", "金额")):
            return True
    return False


def has_policy_signal(text: str) -> bool:
    return any(topic in text for topic in POLICY_TOPICS) or (
        any(anchor in text for anchor in CAMPUS_ANCHORS) and any(cue in text for cue in FACT_CUES))


def is_policy_request(text: str) -> bool:
    if is_general_request(text) or not has_policy_signal(text):
        return False
    explicit_document = any(cue in text for cue in DOCUMENT_CUES)
    if not explicit_document and any(word in text for word in ("建议", "技巧", "经验", "怎么准备", "如何准备", "规划", "学习方法", "怎么写", "如何写")):
        return False
    if any(word in text for word in ("办完", "交完", "完成了", "拿到", "到账", "开心", "分享", "松口气")) and not explicit_document:
        return False
    return explicit_document or any(cue in text for cue in FACT_CUES)


def is_policy_followup(text: str, previous_users: list[str], summary: str = "") -> bool:
    if len(text) > 60 or is_general_request(text):
        return False
    elliptical = bool(re.match(r"^(?:那|那么|还有|如果|这个|那个|刚才|上次|更正|我其实)", text))
    identity = bool(re.fullmatch(r"我?是?(?:研究生|本科生|硕士生|博士生)[。！.]?", text))
    if not elliptical and not identity:
        return False
    # 有明确业务对象的追问可以回到较早的政策话题；泛指“那怎么办”只跟随最近话题。
    concrete = any(word in text for word in ("住宿", "伙食", "交通", "餐补", "补助", "报销", "材料", "金额", "标准", "发票", "申请", "截止", "年级"))
    context = "\n".join(previous_users) + "\n" + summary
    if concrete and has_policy_signal(context):
        return True
    for previous in reversed(previous_users[-6:]):
        if is_policy_request(previous):
            return True
        if is_general_request(previous):
            return False
        if re.match(r"^(?:那|那么|还有|如果|我是|更正)", previous) and len(previous) < 45:
            continue
        return False
    return not previous_users and has_policy_signal(summary)
