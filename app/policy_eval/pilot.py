"""Authoring specification for an AI-drafted development pilot (not a held-out set)."""
from __future__ import annotations

import json
from pathlib import Path

from app.services.knowledge_bundle import load_bundle

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(__file__).with_name("pilot.json")


def build_dataset() -> dict:
    manifest, rows = load_bundle(ROOT / "knowledge/xidian")
    docs = {d["id"]: (d, text) for d, text in rows}
    cases = []

    def add(case_id, category, question, points, evidence, *, history=None, note=""):
        gold = []
        for doc_id, section, quote in evidence:
            meta, text = docs[doc_id]
            # Quotes are literal source spans, not paraphrases or keyword labels.
            if text.count(quote) != 1:
                raise ValueError(f"{case_id}: quote must occur exactly once: {quote[:60]}")
            start = text.index(quote)
            gold.append({"document_id": doc_id, "version": meta["version"],
                         "content_sha256": meta["content_sha256"], "source_url": meta["source_url"],
                         "content_path": meta["content_path"], "section": section, "quote": quote,
                         "start_char": start, "end_char": start + len(quote),
                         "line_start": text.count("\n", 0, start) + 1,
                         "line_end": text.count("\n", 0, start + len(quote)) + 1})
        cases.append({"id": case_id, "split": "pilot_dev", "category": category,
                      "review_status": "pending_human", "reviewer": "", "review_note": note,
                      "question": question, "history": history or [], "answerable": bool(gold),
                      "answer_points": points, "evidence": gold})

    leave = "ygb-1061-13093"
    housing = "ygb-1061-13103"
    practice = "ygb-1061-4703"
    degree = "gr-1045-7343"
    stipend = "ygb-1059-1593"
    assistant = "ygb-1059-1592"
    exam = "gr-1039-10205"
    course = "gr-1039-10194"
    student = "ygb-1061-1706"
    duration = "第三条 研究生单次请假时长原则上不超过2周，报备时长原则上不超过假期收假时间。"
    housing_scope = "第三条 本办法规定的校外住宿，是指在学校或学校联系的校外学习实践单位安排的场所之外，连续居住超过14天的情况。"
    housing_valid = "第四条 研究生校外住宿申请可随时办理，有效期不超过本学年。有效期内若研究生校外住宿地点发生变更，须重新申请。"
    leave_kind = "1.法定节假日及寒暑假期间，离校须履行报备手续。\n\n2.除法定节假日及寒暑假外，离校须履行请假手续。"
    leave_flow = "研究生应在离校前登录“研究生请销假”系统办理请假/报备手续，如实、准确提交离校外出去向、时间、紧急联系人等信息。研究生请假离校须经导师、辅导员及所在管理单位领导依次审批，返校后应及时销假。研究生报备离校无需审批。研究生请假/报备手续办理完成后，方可离校。"
    practice_scope = "本办法适用于因培养需要，离开学校本部前往高校、企业、科研院所、依托二级学院建立的异地研究院等校外单位开展学习科研、交流交换、实习实践等活动，且连续超过15天的全日制非定向研究生。"
    practice_flow = "研究生参加校外学习实践须履行审批备案手续，由研究生本人提出申请，说明相关情况，确定紧急联系人、校外单位联系人等信息，签订安全承诺书，经校内培养单位导师、辅导员、分管研究生培养工作和学生工作领导审批通过后，报学校备案。研究生离校前还须按相关规定履行请假手续。"
    # PDF page furniture belongs to the snapshot but not to the semantic evidence.
    # Split the same rule at the page boundary so coverage can be evaluated honestly.
    course_before = "须在赴校外学习前进行审批和备案。学生填写《西安电子科\n技大学研究生申请选修校外课程审批表》，经导师同意报培养学\n院备案。"
    course_exception = "第十三条 如遇课程调整、临时修课等没有提前审批的情\n况， 可待学生返校后及时完成选修外校研究生课程的申请与备案\n工作。"
    stipend_scope = "第三条　研究生国家助学金用于资助我校纳入全国研究生招生计划的全日制研究生（有固定工资收入的除外）。获得资助的研究生须具有中华人民共和国国籍。"
    stipend_leave = "第七条研究生在学期间，因出国、入伍、休学等缘故办理保留学籍手续的，保留学籍期间停发国家助学金，待学籍恢复次月起重新发放，发放年限按照保留学籍时间顺延。"
    table = "| 博士研究生 | 中期考核后 | 2250元/月 | 12个月/年 |"
    degree_effect = "第一条  因学位证书遗失或损坏的学位授予人员，可以向学校申请办理学位证明书，学位证明书与学位证书具有同等效力。"
    degree_materials = "1.个人申请1份；\n\n2.申请人身份证复印件1份（因故需委托他人代办，需同时提供委托书、被委托人身份证复印件1份）；\n\n3.申请人近期免冠蓝底彩色证件照一张（两寸）。"

    add("D01", "direct", "研究生单次请假一般最长多久？报备能超过收假吗？",
        ["单次请假原则上不超过2周", "报备原则上不超过假期收假时间"], [(leave, "第三条", duration)])
    add("D02", "direct", "研究生在外租房连续住多少天，就属于校外住宿？",
        ["连续超过14天", "限学校或学校联系实践单位安排场所之外"], [(housing, "第三条", housing_scope)])
    add("D03", "direct", "学位证明书申请受理后，原则上几个工作日办好？",
        ["受理后原则上3个工作日内"], [(degree, "第四条", "第四条  受理学位证明书的申请后，原则上3个工作日内办结。")])
    add("D04", "direct", "研究生助管半天津贴参考多少？每月工作量和津贴有上限吗？",
        ["半天35元", "每月不超过12个工作日", "津贴上限840元", "按实际工作时间支付"],
        [(assistant, "第十八条", "第十八条研究生助管岗位津贴根据实际工作时间支付，参考标准为半天35元，由各单位据实统计，按月上报。为不影响助管学业，每人每月工作时间应不超过12个工作日，即津贴上限为840元。")])
    add("D05", "direct", "博士综合考试一般应何时进行？没通过可以开题吗？",
        ["一般第三学期末之前", "不通过不能开题"], [(exam, "三、考试时间", "综合考试一般在第三学期末之前进行。 综合考试是开题报告\n的必要准备， 如综合考试不通过， 不能进行学位论文的开题工作。")])

    add("C01", "conditions", "研究生寒假离校回家需要导师审批吗？和普通工作日离校有什么区别？",
        ["寒假报备无需审批", "非节假日寒暑假离校须请假", "请假依次经导师、辅导员及管理单位领导审批", "完成手续后离校"],
        [(leave, "第四条", leave_kind), (leave, "请假审批和报备", leave_flow)],
        note="新办法与旧管理规定可能并存，请审核适用关系；按2026年专门办法标注。")
    add("C02", "conditions", "已经办了校外住宿，这学年换了租房地址还需要重办吗？能自动沿用到下学年吗？",
        ["有效期内地点变更须重新申请", "有效期不超过本学年"], [(housing, "第四条", housing_valid)])
    add("C03", "conditions", "我是纳入全国招生计划、有固定工资的中国籍全日制研究生，能领国家助学金吗？",
        ["有固定工资收入属于排除条件", "不能仅凭全日制和国籍判断可领取"], [(stipend, "第三条", stipend_scope)])
    add("C04", "conditions", "自发去外校修课一般要提前审批吗？如果因为课程调整临时修课没来得及审批怎么办？",
        ["一般须提前审批备案", "导师同意、培养学院备案", "课程调整或临时修课例外可返校后及时补申请备案"],
        [(course, "第十二条", course_before), (course, "第十三条", course_exception)])
    add("C05", "conditions", "博士综合考试第一次没过是否立刻淘汰？还能重考几次、多久内申请？",
        ["可在导师指导下继续学习", "半年内申请重考一次", "重考仍未通过予以分流或淘汰"],
        [(exam, "五、考试结果", "综合考试通过者，继续攻读博士学位；不通过者，可在导师\n指导之下继续学习，并在半年内申请重考一次，重考仍未通过者\n予以分流或淘汰，详情请见《西安电子科技大学研究生培养过程\n分流选择管理办法》。")])

    add("X01", "cross_clause", "我是全日制非定向研究生，要自行联系企业实践两个月并在外租房，校外住宿申请能代替实践备案吗？",
        ["不能替代", "实践超过15天适用实践办法", "校外住宿研究生参加实践还需另办实践手续", "实践审批备案且离校前请假"],
        [(housing, "第十六条", "第十六条 校外学习实践的研究生还需按照《西安电子科技大学研究生校外学习实践管理办法》办理相关手续。"),
         (practice, "第一章适用范围", practice_scope), (practice, "第三章审批备案", practice_flow)])
    add("X02", "cross_clause", "我是在籍研究生，准备离校办理休学并保留学籍，国家助学金在休学期间和恢复学籍后怎么发？离校手续能不办吗？",
        ["保留学籍期间停发", "恢复学籍次月重新发", "发放年限顺延", "离校前完成请假或报备手续"],
        [(stipend, "第七条", stipend_leave), (leave, "请假审批和报备", leave_flow)])
    add("X03", "cross_clause", "我的硕士学位证遗失，能找人代办证明书吗？要什么材料、是否同等效力、归哪个部门办？",
        ["可委托代办", "个人申请、本人身份证复印件、委托书和被委托人身份证复印件、两寸蓝底照片", "同等效力", "研究生院办理硕博证明书"],
        [(degree, "第一条", degree_effect), (degree, "第三条", degree_materials),
         (degree, "第六条", "1.研究生院负责办理博士研究生和硕士研究生的学位证明书。")])
    add("X04", "cross_clause", "我是全日制非定向研究生，自发去境内另一高校学习两个月并想认定课程学分，除了修课审批还需校外学习实践备案吗？",
        ["校外高校学习连续超过15天属于实践办法范围", "修课须提前审批备案", "实践另需审批备案和离校请假"],
        [(course, "第十二条", course_before), (practice, "第一章适用范围", practice_scope), (practice, "第三章", practice_flow)],
        note="跨制度适用关系需业务人员确认，首轮仅测证据取回，不自动判定法律或政策解释正确。")
    add("X05", "cross_clause", "我已办研究生校外住宿，搬家后又赶上离校报备到期但不能返校，分别该补什么手续？",
        ["住宿地点变更重新申请", "请假或报备到期前再次办理相关手续"],
        [(housing, "第四条", housing_valid), (leave, "第六条", "第六条 研究生请假/报备期满应按时返校，不能按期返校的研究生，应在请假/报备到期前再次办理相关手续。")])

    add("M01", "multi_turn", "那着急要用的话，有没有加急办法？要带什么？",
        ["先在线上发起补办申请", "本人携带一卡通去长安校区行政楼720现场办理"],
        [(student, "补办注意事项（2）", "（2）如急需补办学生证，可在线上发起办理申请后本人携带一卡通前往长安校区行政楼720现场办理。")],
        history=[{"role": "user", "content": "我是研究生，学生证丢了，想补办。"}])
    add("M02", "multi_turn", "过了以后每月会变成多少？一年发几个月？",
        ["博士中期考核后2250元/月", "12个月/年", "按快照规定回答，不能当作2026实时金额核验"],
        [(stipend, "第四条标准表", table)],
        history=[{"role": "user", "content": "我是2019年入学的中国籍全日制博士，没有固定工资，正在问国家助学金。"},
                 {"role": "user", "content": "我说的考核是博士中期考核，不是课程考试。"}],
        note="金额仅来自2019制度快照；需人工核查是否有未收录的新标准。")
    add("M03", "multi_turn", "那在外期间多久要向导师汇报一次？",
        ["每周向导师汇报学习实践进展", "特殊情况及时向导师和辅导员报告"],
        [(practice, "第三章在外管理", "研究生在校外学习实践期间应按预定的区域、既定任务与时间进行，遵守国家和所在地区法律法规，遵守学校和校外单位的各项规章制度，定期与校内培养单位主动联系，每周向导师汇报学习实践进展，如遇特殊情况须及时向导师和辅导员报告。")],
        history=[{"role": "user", "content": "我是全日制非定向研究生，将去企业做两个月的校外学习实践。"}])
    add("M04", "multi_turn", "更正一下，我是自己联系去境内高校修课，不是学校交流项目。这种情况最多换多少学分？政治理论课也行吗？",
        ["自发赴境内其他高校或科研机构原则上置换总学分不超过10分", "政治理论课学分必须在本校获得"],
        [(course, "第十九条", "第十九条 自发赴境内其他高校或科研机构修读课程的同\n学，原则上置换总学分不能超过 10 分。"),
         (course, "第二十一条", "第二十一条 原则上， 转换课程不得与在我校已修的课程重\n复；政治理论课学分必须在我校获得。")],
        history=[{"role": "user", "content": "我是研究生，想了解学校组织境外交流后的课程置换。"}],
        note="保留早期错误条件，显式更正在当前问题中；只验证预设上下文检索，不声称已测生产改写。")
    add("M05", "multi_turn", "如果是限制民事行为能力，还能自己一个人住吗？",
        ["须由监护人陪同", "监护人负责生命财产安全并共同承担相应法律责任"],
        [(housing, "第十四条", "第十四条 限制民事行为能力的研究生，校外住宿须由监护人陪同，由监护人负责其生命财产安全，履行公民义务，与监护人共同承担相应法律责任。")],
        history=[{"role": "user", "content": "我想咨询研究生校外住宿的规定。"}])

    for case_id, question, missing in [
        ("U01", "按现行规定，研究生国内出差每天伙食补助准确是多少钱？", "已收录资料未核实适用的差旅伙食补助金额标准，不可用助管津贴等数字代替。"),
        ("U02", "请查一下我的研究生助管岗位申请现在审批到哪一步了。", "无个人业务系统接口、申请编号和审批记录；制度文件不能提供个人办理状态。"),
        ("U03", "2027年秋季计算机学院研究生学业奖学金申请具体几点截止？", "缺对应学院、年度的申请通知和截止时间，通用管理办法不足以回答。"),
        ("U04", "今天长安校区研究生宿舍还有哪几间空房？把房间号列给我。", "无实时宿舍房源与个人住宿数据库，公寓管理办法不提供实时空房。"),
        ("U05", "我上次博士综合考试老师给我的具体分数是多少？", "无个人考试成绩接口与成绩记录，综合考试制度不含该生分数。"),
    ]:
        add(case_id, "unanswerable", question, ["明确当前资料或接入能力不足", "不编造金额、日期、状态、房间或分数", missing], [], note=missing)

    return {"schema_version": 1, "name": "xidian-policy-pilot-v1", "split": "pilot_dev",
            "annotation_origin": "AI drafted from repository snapshots; awaiting human review",
            "corpus_checked_on": manifest["checked_on"],
            "manifest_sha256": __import__("hashlib").sha256((ROOT / "knowledge/xidian/manifest.json").read_bytes()).hexdigest(),
            "scope": "仓库快照内的资料理解；不证明制度当前有效，不作为独立测试集或正式政策咨询结论。",
            "cases": cases}


if __name__ == "__main__":
    if DATASET.exists():
        raise SystemExit("Dataset already exists; refusing to overwrite review decisions.")
    dataset = build_dataset()
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n")
    print(f"Created {len(dataset['cases'])} pending-review cases: {DATASET}")
