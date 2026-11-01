"""Author a 60-case prospective candidate set after protocol freeze.

Gold source documents are disjoint from the 75 development cases. AI authorship
and pending human review are explicit; this is not an externally certified test.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from app.policy_eval.expand import EXPANDED_DATASET
from app.policy_eval.rigorous import BASE, FROZEN_PROTOCOL, HOLDOUT
from app.policy_eval.runner import compact, digest, load_inputs, validate_dataset


def build_dataset():
    freeze = FROZEN_PROTOCOL if FROZEN_PROTOCOL.exists() else BASE / 'frozen-protocol.json'
    if not freeze.exists():
        raise ValueError('Freeze the development protocol before authoring candidates')
    original, docs = load_inputs(EXPANDED_DATASET)
    used = {g['document_id'] for c in original['cases'] for g in c['evidence']}
    cases = []

    def evidence(doc_id, section, first, last=None):
        meta, source = docs[doc_id]
        chars = [(i, c) for i, c in enumerate(source) if not c.isspace()]
        normalized = ''.join(c for _, c in chars)
        begin = compact(first)
        if normalized.count(begin) != 1:
            raise ValueError(f'Nonunique anchor in {doc_id}: {first}')
        left = normalized.index(begin)
        right = left + len(begin) if last is None else normalized.index(compact(last), left) + len(compact(last))
        start, end = chars[left][0], chars[right - 1][0] + 1
        quote = source[start:end]
        if '<!-- page:' in quote:
            raise ValueError(f'Split page-boundary evidence: {doc_id}: {section}')
        return {'document_id': doc_id, 'version': meta['version'], 'content_sha256': meta['content_sha256'],
                'source_url': meta['source_url'], 'content_path': meta['content_path'], 'section': section,
                'quote': quote, 'start_char': start, 'end_char': end,
                'line_start': source.count('\n', 0, start) + 1, 'line_end': source.count('\n', 0, end) + 1}

    def add(case_id, category, question, points, gold, family, *, history=(), behavior='answer', note='', checked=()):
        if any(g['document_id'] in used for g in gold) or any(i in used for i in checked):
            raise ValueError('Candidate gold/reference source overlaps development sources')
        cases.append({'id': case_id, 'split': 'frozen_candidate', 'category': category,
                      'question': question, 'history': [{'role': 'user', 'content': h} for h in history],
                      'answerable': bool(gold), 'answer_points': points, 'evidence': gold,
                      'required_point_indices': list(range(1, len(points) + 1)), 'optional_point_indices': [],
                      'expected_behavior': behavior, 'evidence_family': family,
                      'checked_source_ids': list(checked), 'review_status': 'pending_human',
                      'reviewer': '', 'review_note': note, 'annotation_origin': 'AI source-grounded prospective candidate'})

    e = evidence
    teaching, incident = 'gr-1039-10191', 'gr-1039-10196'
    fund, sampling = 'gr-1045-11173', 'gr-1045-14564'
    law, engineering = 'gr-1045-15344', 'gr-1045-16484'
    discipline, employment = 'ygb-1062-1639', 'ygb-1059-1589'
    research, military = 'ygb-1059-1594', 'ygb-1059-1601'
    dorm, recommend = 'ygb-1061-5463', 'jwc-1024-10930'

    add('H-D01', 'direct', '研究生课程选课人数不到5人，本学期一般还开吗？招生人数特别少的学科有例外吗？',
        ['原则上少于5人本学期停开', '个别招生少的学科除外'],
        [e(teaching, '第八条', '第八条原则上选课人数少于5人的课程本学期停开', '（个别招生少的学科除外）。')], 'course_teaching')
    add('H-D02', 'direct', '研究生教学事故按类别分为哪三类？这里问类别，不是严重程度等级。',
        ['教学类', '教学管理类', '教学保障类'],
        [e(incident, '第三条', '第三条教学事故分为教学类、教学管理类和教学保障类三类。')], 'teaching_incident')
    add('H-D03', 'direct', '按收录的2021年办法，优秀博士学位论文基金资助期最短和最长分别多久？',
        ['最短不少于1年', '最长不超过2年'],
        [e(fund, '第三条', '第三条优秀博士学位论文基金资助时间最短不少于1年，最长不超过2年。')], 'doctoral_thesis_fund')
    add('H-D04', 'direct', '学校组织的研究生论文规范性检测不通过，或者校内论文评审不合格，当次会如何处理学位授予？',
        ['校内规范性检测不通过或论文评审不合格者，当次缓授学位'],
        [e(sampling, '第四条校内结果', '西安电子科技大学组织的研究生学位论文规范性检测结果为“不通过”者或学位论文评审结果为“不合格”者，当次缓授学位。')], 'thesis_sampling')
    add('H-D05', 'direct', '按库内《学位法》，学位评定委员会至少多少人，人数必须是单数吗？',
        ['不少于九人', '必须为单数'],
        [e(law, '第十条组成', '其组成人员应当为不少于九人的单数。')], 'degree_law_boards')
    add('H-D06', 'direct', '我是2025年秋季入学的工程类专业学位博士。按2025年成果要求，用专利成果走学位论文申请路径，授权数量、排名和转化有什么要求？',
        ['发明或国防专利授权2项', '本人排名第一，或导师第一本人第二', '至少1项转化应用', '须与学位论文相关'],
        [e(engineering, '一（一）第4项', '4.获得与学位论文相关的发明/国防专利授权', '2项且至少有1项转化应用；')], 'engineering_doctoral_achievements')
    add('H-D07', 'direct', '研究生警告、严重警告、记过和留校察看的处分期限各是多少个月？',
        ['警告和严重警告均6个月', '记过9个月', '留校察看12个月'],
        [e(discipline, '第三十九条', '第三十九条警告、严重警告处分的期限为6个月，记过处分的期限为9个月，留校察看处分的期限为12个月。')], 'graduate_discipline')
    add('H-D08', 'direct', '基层就业学费补偿或助学贷款代偿，是毕业后一次发完吗？三年的比例分别是多少？',
        ['分年度，3年完成', '第一年33%', '第二年33%', '第三年34%'],
        [e(employment, '第九条', '第九条国家对到中西部地区', '3年补偿代偿完毕。')], 'grassroots_employment_compensation')
    add('H-D09', 'direct', '按库内助研办法，研究生助研津贴一年发几个月，哪两个月不发？',
        ['每年10个月', '2月、8月寒暑假不发放'],
        [e(research, '第十一条', '第十一条研究生助研津贴由设岗导师出资', '（2月、8月寒暑假不发放）。')], 'research_assistant')
    add('H-D10', 'direct', '退役后回校复学申请学费减免，应填申请表I还是II、几份，还需要哪项退役材料？',
        ['申请表II', '一式两份', '提交退役证书复印件'],
        [e(military, '第十条', '第十条退役后自愿回校复学或入学的学生', '并提交退役证书复印件。')], 'military_education_aid')
    add('H-D11', 'direct', '进学生公寓探访朋友，除了被访人同意，还要什么登记手续？访客最晚当日几点离开？',
        ['被访人允许，凭有效证件登记后进入', '当日22:00前离开'],
        [e(dorm, '第二十一条', '第二十一条访客管理。', '访客应在当日22:00前离开。')], 'dormitory_access')
    add('H-D12', 'direct', '按收录的2021版本科推免办法，学业成绩和外语成绩分别认定到推免当年的几月几日？',
        ['学业成绩7月31日', '外语成绩8月31日'],
        [e(recommend, '第十八条（一）6', '6.学业成绩认定截止时间为获得推免当年的7月31日。'),
         e(recommend, '第十八条（二）5', '5.外语成绩的认定截止时间为获得推免当年的8月31日。')], 'undergraduate_recommendation')

    add('H-C01', 'conditions', '研究生课程已经参加考试了，还能依据特殊情况申请退课吗？只按课程教学管理规定这条判断。',
        ['已参加考试的课程不允许退课'],
        [e(teaching, '第二十条', '已参加考试的课程不允许退课。')], 'course_teaching')
    add('H-C02', 'conditions', '某老师一年内已发生三次较大教学事故，之后同年第四次事故按什么等级处理？',
        ['一年内达到3次含3次较大事故，第3次之后所有事故按重大教学事故处理', '第四次应按重大教学事故处理'],
        [e(incident, '第十一条', '第十一条一年内发生3次以上（含3次）较大教学事故者，第3次之后的所有教学事故均按重大教学事故处理。')], 'teaching_incident')
    add('H-C03', 'conditions', '我的博士论文是涉密论文，或者用外文撰写，即使成果突出，能申请优秀博士学位论文资助基金吗？',
        ['涉密论文不得申请', '以外文撰写的博士学位论文不得申请'],
        [e(fund, '第五条四', '四、涉密和以外文撰写的博士学位论文不得申请本资助。')], 'doctoral_thesis_fund')
    add('H-C04', 'conditions', '陕西省硕士论文抽检有一个不合格意见，平均分正好70分，是否按“低于70分等同存在问题”这一条处理？',
        ['正好70分不属于低于70分', '低于70分不含70分才按存在问题处理', '70分属于大于等于70分分支'],
        [e(sampling, '第九条边界', '第九条陕西省学位委员会办公室组织的硕士学位论文抽检出现1篇', '等同于“存在问题”的论文。')], 'thesis_sampling')
    add('H-C05', 'conditions', '按库内《学位法》，学术博士与专业博士在独立工作能力和创新成果领域上有什么区别？',
        ['学术博士具有独立从事学术研究工作的能力', '专业博士具有独立承担专业实践工作的能力', '学术博士在学术研究领域做出创新性成果', '专业博士在专业实践领域做出创新性成果'],
        [e(law, '第二十一条（二）（三）', '（二）学术学位申请人应当具有独立从事学术研究工作的能力', '专业学位申请人应当在专业实践领域做出创新性成果。')], 'degree_law_boards')
    add('H-C06', 'conditions', '按2025年工程博士成果要求，一篇非顶尖成果类的论文有4名共同第一作者，能每人都计1篇吗？顶尖成果类共同一作是否不同？',
        ['非顶尖共同一作按1/N篇，4人各计1/4篇', '顶尖成果类共同一作按1篇计入统计数'],
        [e(engineering, '二（三）共同第一作者', 'N个共同第一作者的学术论文计入1/N篇', '按一篇计入统计数）。')], 'engineering_doctoral_achievements')
    add('H-C07', 'conditions', '研究生正在处分期限内办理休学或保留学籍，这段时间也算进处分期限吗？',
        ['休学或保留学籍时间不计入处分期限'],
        [e(discipline, '第三十六条期限', '处分期限内因故休学或保留学籍的，休学或保留学籍的时间不计入处分期限。')], 'graduate_discipline')
    add('H-C08', 'conditions', '我是全日制定向应届毕业研究生，去中西部基层服务三年以上，就一定属于基层就业补偿办法的资助对象吗？',
        ['定向学生属于排除对象', '不能仅凭全日制应届和服务年限认定符合资格'],
        [e(employment, '第三条', '第三条本办法中毕业生是指我校全日制本科生', '已享受免除学费政策的学生除外。')], 'grassroots_employment_compensation')
    add('H-C09', 'conditions', '我是一年级硕士，在导师的岗位上做助研，就不能发津贴吗？学校是否会提供配套？',
        ['导师可根据实际上岗情况参照标准发放', '一年级硕士学校不予配套', '不能把无配套理解为导师不得发放'],
        [e(research, '第十三条', '第十三条对于一年级硕士生', '学校不予配套。')], 'research_assistant')
    add('H-C10', 'conditions', '在校期间已通过其他方式免除全部学费，入伍后还能按收录的服兵役资助办法重复享受这项资助吗？',
        ['已通过其他方式免除全部学费的学生不享受该项国家资助'],
        [e(military, '第四条（一）', '第四条下列学生不享受以上国家资助：', '（一）在校期间已通过其他方式免除全部学费的学生；')], 'military_education_aid')
    add('H-C11', 'conditions', '休学后返校想重新住宿，只要提出申请就一定能安排公寓床位吗？还须满足什么条件？',
        ['须经相关部门审批同意', '须空余床位能够满足', '不能保证申请即安排'],
        [e(dorm, '第八条', '第八条特殊情况需要入住公寓的', '休学和保留学籍等原因重新返校入住公寓；')], 'dormitory_access')
    add('H-C12', 'conditions', '按2021版本科推免办法，与直系亲属合作的科研成果可以直接计入本人推免综合素质能力分吗？',
        ['仅作为参考，不纳入本人推免综合素质能力指标体系', '同等条件下可优先考虑'],
        [e(recommend, '第二十条第3项', '3.学生与直系亲属或学历、职称、职务明显高于本人者合作', '同等条件下可优先考虑。')], 'undergraduate_recommendation')

    add('H-X01', 'cross_clause', '研究生任课教师确需调课时，至少提前多久通过什么渠道申请？每门课每学期原则上最多调几次，还要通知学生吗？',
        ['提前24小时经研究生管理系统申请', '经学院审核，及时通知学生', '每门课程每学期原则上不超过3次'],
        [e(teaching, '第十五条调课', '因故确需调课的', '每门课程每学期调课次数原则上不超过3次。')], 'course_teaching')
    add('H-X02', 'cross_clause', '收录的研究生教学事故办法正文说接到通知后10日内申诉，附件告知书又写10个工作日。我能直接认定统一是10个工作日吗？',
        ['正文第十四条为接到通知后10日内', '附件告知书注为通知书发出10个工作日内', '时间单位与起算表述不一致，不能擅自统一，应说明冲突并核实主管部门'],
        [e(incident, '第十四条', '若事故责任人对事故的处理有异议', '申诉及复核期间，不影响对原处理决定的执行。'),
         e(incident, '附件3注', '注：如果当事人对处理结果有异议', '向校研究生培养指导委员会提出申诉')], 'teaching_incident', note='明确比较快照内部文字冲突，适用解释仍待业务审核，不裁定哪个期限优先。')
    add('H-X03', 'cross_clause', '普通招考博士入学第三年申请优博论文基金，第一年直接全额发吗？一年期满中期评审不合格，剩余部分怎么处理、能继续受资助吗？',
        ['第一年先按资助标准50%发放', '中期评审不合格停止资助', '不再补发首期剩余金额'],
        [e(fund, '第七条第一年资助期', '1.第一年资助期。', '不再补发首期资助经费的剩余金额。')], 'doctoral_thesis_fund')
    add('H-X04', 'cross_clause', '国务院组织的博士论文抽检不合格，作者修改后需谁评审、谁签字，多久内重新交到哪些机构？',
        ['导师指导下按评审意见修改', '学院成立不少于3名博导组成的专家组评审通过', '指导教师签字确认', '一年内向学校和国家图书馆重新提交'],
        [e(sampling, '第五条', '第五条国务院学位委员会办公室组织的博士学位论文抽检结果为“不合格”', '一年内重新提交学位论文至学校和国家图书馆。')], 'thesis_sampling')
    add('H-X05', 'cross_clause', '按《学位法》，博士学位答辩委员会至少几人，其中校外专家至少几人？答辩决议的通过票数按谁的多少比例计算？',
        ['博士答辩委员会不少于5人', '学位授予单位外专家不少于2人', '全体组成人员三分之二以上通过'],
        [e(law, '第二十六条组成', '博士学位答辩委员会组成人员应当不少于五人', '其中学位授予单位以外的专家应当不少于二人。'),
         e(law, '第二十六条表决', '答辩以投票方式表决，由全体组成人员的三分之二以上通过。')], 'degree_law_boards')
    add('H-X06', 'cross_clause', '2025年秋季入学的工程博士用实践成果申请学位，成果可是什么形式？除了满足列举的创新成果条件，还要哪些应用或效益证明？',
        ['可为重大装备、仪器设备、其他硬件产品、软件产品、设计方案、技术标准等形式之一', '须经同行专家鉴定或评审，得到推广应用或取得良好经济社会效益', '需提供推广应用、经济效益、社会效益等证明作为支撑材料'],
        [e(engineering, '一（二）形式与鉴定', '以实践成果申请学位的，实践成果应为重大装备', '且需满足以下条件之一：'),
         e(engineering, '一（二）支撑材料', '同时需提供实践成果推广应用证明', '（相关证明材料撰写参照学校模板）。')], 'engineering_doctoral_achievements')
    add('H-X07', 'cross_clause', '研究生违纪调查小组最少几人，与当事人有利害关系的人能参与吗？一般应在多久完成调查，复杂情况能延长吗？',
        ['小组不得少于2人', '有利害关系或直接涉事者回避', '一般发现违纪或收到线索后5个工作日内完成', '复杂或特殊情况可适当延长'],
        [e(discipline, '第三十一条人员', '调查小组成员不得少于两人', '应当回避。'),
         e(discipline, '第三十二条时限', '第三十二条调查小组一般应在发现研究生违纪行为', '可适当延长调查时间。')], 'graduate_discipline')
    add('H-X08', 'cross_clause', '应届毕业生去中西部基层就业，服务几年以上才符合学费补偿的服务期要求？在校有用于学费的国家助学贷款时，代偿资金先处理什么？',
        ['服务期3年以上含3年', '优先偿还用于学费的国家助学贷款本金及全部偿还前产生的利息'],
        [e(employment, '第二条', '第二条毕业生到中西部地区', '本金及其全部偿还之前产生的利息。')], 'grassroots_employment_compensation')
    add('H-X09', 'cross_clause', '研究生可以同时聘两个助研岗位吗？如果应聘其他导师的助研岗，只取得设岗导师同意就够了吗？',
        ['每学期只能聘任一个助研岗位', '其他导师岗位须本人所属导师和设岗导师均同意'],
        [e(research, '第六条', '第六条为保证助研工作质量', '须经本人所属导师和设岗导师同意。')], 'research_assistant')
    add('H-X10', 'cross_clause', '按收录的服兵役教育资助办法，补偿学费和代偿贷款金额按两者较低还是较高确定？研究生超过年度标准的部分也能报吗？',
        ['按实际缴纳学费与用于学费的国家助学贷款较高者执行', '研究生每生每年最高16000元，限收录版本标准', '超过标准部分不予补偿、代偿或减免'],
        [e(military, '第五条', '第五条学费补偿、国家助学贷款代偿以及学费减免的标准', '超出标准部分不予补偿、代偿或减免。')], 'military_education_aid')
    add('H-X11', 'cross_clause', '行动不便想暂住爱心宿舍，要提供什么证明？身体恢复后能一直住吗？家长陪住需要谁批准？',
        ['向后勤保障部申请，提交情况说明、医院证明等', '恢复后及时返回原宿舍，不得长期占用', '家长陪同须经学生社区或学院、后勤保障部审核批准'],
        [e(dorm, '第十二条', '第十二条爱心宿舍。', '住宿期间人身及财产安全由本人负责。')], 'dormitory_access')
    add('H-X12', 'cross_clause', '按收录的2021版本科推免一般类别要求，我平均学分绩78分、专业排名前18%、课程均合格，英语四级550分。只看学业和外语这两项门槛，是否达到？',
        ['学业80分及以上或专业前20%，并且课程合格，排名前18%可满足', '一般类别四级550分及以上可满足外语这一分支', '仅判断所问两项，不保证具备全部推免资格或最终获推荐'],
        [e(recommend, '第十八条（一）1', '1.“一般类别”学生平均学分绩成绩应在80分及以上或专业排名前20%，且课程合格;'),
         e(recommend, '第十八条（二）1', '1.“一般类别”学生及申请“国防科工招生单位补偿计划”的学生全国大学英语四级在550分及以上', '全国大学英语六级在425分及以上')], 'undergraduate_recommendation')

    add('H-M01', 'multi_turn', '更正，我原来的课程成绩合格，现在自愿重修。选了重修后总成绩取新成绩还是可以保留原来较高的成绩？',
        ['课程一旦选择重修，以重修之后的成绩计入总成绩和总学分', '不以原先是否合格区分，也不按两次较高分选取'],
        [e(teaching, '第三十五条重修计分', '所有课程一旦选择重修，该门课程成绩以重修之后的成绩计入总成绩与总学分。')], 'course_teaching', history=['我有门研究生课程不及格，想了解自动重修的安排。'])
    add('H-M02', 'multi_turn', '纠正，是教师的研究生教学事故，不是学生违纪。重大、较大和一般教学事故分别在哪个范围通报？',
        ['重大和较大由研究生院全校范围通报', '一般由学院或部门在本院本部门通报'],
        [e(incident, '第九条通报', '第九条研究生院对重大、较大教学事故在全校范围内通报', '在本院、本部门范围内通报。')], 'teaching_incident', history=['我想问研究生受到学生违纪处分后怎么通报。'])
    add('H-M03', 'multi_turn', '不是国家助学金，改问优秀博士学位论文基金：受资助期间完成答辩后，究竟从答辩当天还是拿到学位证书的次月停发？',
        ['从获得博士学位证书的次月起停止发放该基金资助经费'],
        [e(fund, '第十条', '第十条资助期间完成博士学位论文答辩的受资助者，从获得博士学位证书的次月起停止发放资助经费。')], 'doctoral_thesis_fund', history=['我想问研究生国家助学金毕业停发时间。'])
    add('H-M04', 'multi_turn', '更正，是陕西省硕士论文抽检结果“存在问题”。按对应的作者处理条款，修改后由谁签字、多久内交到哪里？',
        ['按评审意见修改', '指导教师签字确认', '一年内向学校提交'],
        [e(sampling, '第六条', '第六条陕西省学位委员会办公室组织的硕士学位论文抽检结果', '在一年内向学校提交学位论文。')], 'thesis_sampling', history=['我先前问的是国务院博士论文抽检不合格后的修改重交。'])
    add('H-M05', 'multi_turn', '更正，不是补学位证明书，而是按《学位法》申请硕士学位。审查是否受理的60日，从我提交当天还是申请日期截止日起算，决定后要通知吗？',
        ['自申请日期截止之日起60日内审查决定是否受理', '应通知申请人'],
        [e(law, '第二十三条受理', '学位授予单位应当自申请日期截止之日起六十日内审查决定是否受理申请，并通知申请人。')], 'degree_law_boards', history=['我在了解学位证明书补办的办结期限。'])
    add('H-M06', 'multi_turn', '改问2025年工程博士成果要求里的论文：期刊目录更新后贡献度降级，按投稿时还是更新后的等级认定？论文发表又以什么为准？',
        ['目录降级时按投稿时目录贡献度等级认定', '发表以期刊官网或指定数据库在线发表且取得DOI号为准'],
        [e(engineering, '二（四）', '（四）本规定中的发表论文以论文在线发表于期刊官方网站或指定数据库', '以投稿时《高水平期刊目录》中的贡献度等级认定。')], 'engineering_doctoral_achievements', history=['我是2025年秋季入学的工程专业博士，先前问的是专利成果数量。'])
    add('H-M07', 'multi_turn', '更正一下，我问处分解除，不是学位复核。研究生处分到期后，要在几个月内由谁申请，解除后表彰奖励还受原处分影响吗？',
        ['研究生本人在处分到期后6个月内申请', '管理单位决定解除并报党委研究生工作部备案', '解除后表彰奖励及其他权益不再受原处分影响'],
        [e(discipline, '第四十一条', '第四十一条解除处分由研究生本人在处分到期后6个月内申请', '不再受原处分的影响。')], 'graduate_discipline', history=['我想了解对学位授予决定申请复核的期限。'])
    add('H-M08', 'multi_turn', '补充一下，我的基层就业需要二次定岗。学校集中报送申请材料最迟是什么时候？和一般6月底报送有什么区别？',
        ['一般集中报送为每年6月底前', '二次定岗提交证明并经审查后，最迟当年12月底前集中报送', '具体安排按全国学生资助管理中心通知确定'],
        [e(employment, '第十二条第4项', '4.上报推荐名单。', '具体安排根据教育部全国学生资助管理中心通知确定。')], 'grassroots_employment_compensation', history=['我想了解毕业到中西部基层单位就业的代偿材料报送时间。'])
    add('H-M09', 'multi_turn', '更正，我是II类学科博士，不是一年级直博生，也不是二年级硕士。按收录助研标准每月多少、经费由谁承担，有学校配套吗？',
        ['II类博士每月800元', '导师资助800元，学校无配套', '按收录制度标准，不保证个人已实际到账'],
        [e(research, '第十二条标准表', '|博士生（一年级直博生除外）|II类|800|800|/|')], 'research_assistant', history=['我先前问的是二年级硕士I类助研津贴。'])
    add('H-M10', 'multi_turn', '更正，是因为部队编制员额缩减，经组织批准提前退役，不是拒服兵役被除名。按收录资助办法还保留受助资格吗？',
        ['部队编制员额缩减等原因，经组织批准提前退役，仍具备受助资格'],
        [e(military, '笫十八条', '笫十八条因部队编制员额缩减', '仍具备受助资格。')], 'military_education_aid', history=['我之前问学生拒服兵役被部队除名会不会取消教育资助。'])
    add('H-M11', 'multi_turn', '更正，我住的是本科生公寓，不是研究生楼。按收录公寓办法，平常几点开门、几点关门，是否可能调整？',
        ['本科生楼每日6:30开门、23:30关门', '门禁时间可按学校要求调整'],
        [e(dorm, '第十九条门禁', '第十九条门禁时间。', '门禁时间可按照学校要求做出调整。')], 'dormitory_access', history=['我先前想问研究生公寓是否设置固定门禁。'])
    add('H-M12', 'multi_turn', '更正，我走“突出特长”类别，不是一般类别。按2021版办法，仅看学业平均分和英语四级门槛，各要求什么？',
        ['突出特长平均学分绩70分及以上且课程合格', '英语四级达到学校统一授学位标准，不套用一般类别550分分支'],
        [e(recommend, '第十八条（一）4', '4.“突出特长”学生平均学分绩成绩在70分及以上，且课程合格;'),
         e(recommend, '第十八条（二）4', '4.申请参加研究生支教团的学生和“突出特长”学生，全国大学英语四级应达到学校统一规定的授学位标准;')], 'undergraduate_recommendation', history=['我先前问的是普通一般类别本科推免的成绩和外语门槛。'])

    negatives = [
        ('H-U01', '我的优秀博士论文基金申请到底审批到哪一步？请查询个人申请记录。', '无个人基金审批记录或系统接口，不能用评审程序代替实际状态。', fund, 'doctoral_thesis_fund', 'abstain'),
        ('H-U02', '我这篇博士论文最近一次国家抽检，专家实际写了什么意见、打了多少分？', '无本人论文抽检意见和分数记录，处理办法不能提供个人结果。', sampling, 'thesis_sampling', 'abstain'),
        ('H-U03', '请确认下学期王老师的研究生机器学习课，下周临时调到周几几点、哪个教室。', '无对应学期课程调课通知及课表，管理规则不能确定具体安排。', teaching, 'course_teaching', 'abstain'),
        ('H-U04', '请替我提交爱心宿舍申请并完成审批，直接回复已批准、家长今晚可以入住。', '没有提交和审批能力，不能声称已批准；可解释实际申请审核路径。', dorm, 'dormitory_access', 'abstain'),
        ('H-U05', '2027年的高水平期刊目录中IEEE Access是几类贡献度？只给确认后的具体等级。', '语料没有2027年高水平期刊目录原文，不能由工程博士成果要求反推具体期刊等级。', engineering, 'engineering_doctoral_achievements', 'abstain'),
        ('H-U06', '2027年基层就业补偿的新标准是不是已经提高到研究生每年两万元？请确认最新标准。', '缺2027年新标准文件，不能以收录旧版或问题预设确认已提高到两万元。', employment, 'grassroots_employment_compensation', 'abstain'),
        ('H-U07', '2027年全校助研津贴统一上调后的最新具体金额是多少？请确认已上调后的数额。', '缺2027年统一上调文件，不能接受已上调这一未经支持的前提或编造新数额。', research, 'research_assistant', 'abstain'),
        ('H-U08', '我从部队提前退回学校了，这种退役还能享受服兵役教育资助吗？', '缺提前退役原因、是否组织批准等关键条件，应先澄清，不能一律认定保留或取消资格。', military, 'military_education_aid', 'clarify'),
        ('H-U09', '我导师遇到论文抽检的问题了，会被暂停招生多久？', '缺国家博士/省硕士/校内抽检类型、具体结果、次数和时间，应先澄清，不能直接断言停招年限。', sampling, 'thesis_sampling', 'clarify'),
        ('H-U10', '学校对宿舍里使用违章电器统一罚款多少元？请给出确定的罚款金额。', '公寓办法规定没收和依违纪办法处理，未给统一罚款金额，不得编造金额。', dorm, 'dormitory_access', 'abstain'),
        ('H-U11', '请把我的研究生处分材料从个人档案里删干净，然后确认已经删除。', '没有修改个人档案的系统能力，不能声称已删除；处分及解除材料有真实完整归档规定。', discipline, 'graduate_discipline', 'abstain'),
        ('H-U12', '今年计算机学院本科推免每个专业最后分到几个人？把当年的具体分配表列出来。', '缺当年度学院专业具体名额表，推免办法的分配原则不足以提供实际名额。', recommend, 'undergraduate_recommendation', 'abstain'),
    ]
    for case_id, question, missing, doc, family, behavior in negatives:
        add(case_id, 'unanswerable', question,
            [missing, '可说明已知规则或核实路径，不编造个人结果、已执行操作、未来政策、具体金额或名额'],
            [], family, behavior=behavior, checked=[doc], note=missing)

    dataset = {k: v for k, v in original.items() if k not in {'cases', 'base_case_ids', 'added_case_ids', 'base_dataset_sha256', 'annotation_batch'}}
    dataset.update({'name': 'xidian-prospective-source-disjoint-60-v3', 'split': 'frozen_candidate', 'cases': cases,
                    'annotation_origin': 'AI authored after protocol freeze, all pending human source/business review',
                    'frozen_protocol_sha256': digest(freeze.read_bytes()),
                    'excluded_development_source_ids': sorted(used),
                    'source_isolation': 'gold/reference documents disjoint from 75 development gold documents; same retrieval corpus',
                    'scope': '前瞻保留候选集；60题、12证据族。未用于参数选择，仍为AI编写待人工审核，不是外部认证独立测试集。',
                    'usage_policy': 'No retrieval/generation tuning after seeing candidate outputs; resume only API or validation failures.'})
    validate_dataset(dataset, docs, dataset['manifest_sha256'])
    return dataset


def main():
    if HOLDOUT.exists():
        raise SystemExit('Candidate set exists; refusing to overwrite annotations or test decisions')
    dataset = build_dataset()
    HOLDOUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(HOLDOUT), 'cases': len(dataset['cases']),
                      'categories': Counter(c['category'] for c in dataset['cases']),
                      'families': len({c['evidence_family'] for c in dataset['cases']})}, ensure_ascii=False))


if __name__ == '__main__':
    main()
