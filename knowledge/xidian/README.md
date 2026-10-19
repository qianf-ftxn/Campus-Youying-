# 西安电子科技大学官方公开资料包

核查日期：2026-10-19。共 55 份资料；默认启用 49 份，待核对 4 份，历史归档 2 份。

资料来自西安电子科技大学研究生院、党委研究生工作部、教务处和财务处的官方公开网页或附件。这里只收集实际取得的正文；不是校内制度库的完整镜像，也不把网页近期重发的旧制度说成新制定制度。网页发布日期、生效日期和本次核查日期分别记录；有明确修订版时采用后续版。

## 文件结构

- `originals/`：下载时的官方 HTML/PDF/DOCX 原始快照。HTML 为原始页面字节，未离线打包网页图片和样式资源，核对版式请访问官网或查看 PDF。
- `texts/`：按原文提取的 UTF-8 Markdown，保留条款、PDF 页标和表格；不是 AI 改写的政策摘要。
- `manifest.json`：标题、学校、部门、来源链接、发布日期、适用范围、状态、版本、SHA-256、提取问题。
- `catalog.json`：人工筛选的公开来源目录，可用收集脚本更新已知链接；脚本不会自动发现新修订或判断制度效力。

这些文件位于 Git 跟踪范围内，与被忽略的 `data/` 数据库和 Chroma 索引分开。克隆或下载 ZIP 后原文仍存在，应用启动时从本地文本生成数据库和父子知识块；查询时不要求官网在线。

## 导入与复现

默认 `CAMPUS_NAME=西安电子科技大学`、`BUNDLED_KNOWLEDGE_ENABLED=true`。启动时校验正文、原始快照和已保存附件的哈希，然后幂等导入。重复启动不重复创建块，也不会把管理员归档的同版资料重新启用。其他学校配置会跳过西电资料，不改写成另一所学校的政策。

```bash
# 校验资料，不连接数据库和模型
.venv/bin/python scripts/import-bundled-knowledge.py --check
# 手动导入（使用 .env 的数据库与 Embedding）
.venv/bin/python scripts/import-bundled-knowledge.py
# 仅 SQL/BM25，之后可在管理端重建向量
.venv/bin/python scripts/import-bundled-knowledge.py --no-vectors
```

首次创建向量需已有的 Embedding 接口和配置；索引不直接写进 Git，因为模型、接口和数据库块 ID 可能不同。没有向量服务时按原配置降级 BM25，文档本身仍可导入和查询。大量资料首次索引可能需几分钟，一键启动等待上限已增加到 15 分钟。

## 已知边界

财务处主页要求校内访问，未获得可核实的现行差旅报销标准；不从教师个人网页或第三方转载推断学生餐补金额。可下载的工作餐规定原 PDF 已保存，但现行性未核实，标为 DRAFT，且工作餐不等于出差每日伙食补助。

GPA 计算方法 PDF 文本提取乱码、2018 年博士学制通知正文不足，以及 2020 年成果要求旧 DOC 未完成转换，均标为 DRAFT，不参与默认查询。已结束的 2025 年奖学金通知标为 ARCHIVED，保留来源供核对；长期奖助制度按主管部门现有公开制度栏目收录，具体年度补充规定仍需另行核对。

有合并单元格的网页表格已按 rowspan/colspan 展开，避免把研究生类别与金额错列。PDF 提取不提供可靠的复杂表格重建或 OCR，Word/网页图片文字不在提取正文内，详见每份资料的 extraction_warnings。

本科政策、研究生政策与年度招生通知均注明适用范围；模型仍须结合用户身份、学期、年度和条款判断，不能因检索到就自动套用。财务、图书馆开放时间等尚未取得的具体资料应继续明确提示不足。

## 更新

```bash
.venv/bin/python -m pip install -r scripts/requirements-knowledge.txt
.venv/bin/python scripts/collect-xidian-knowledge.py --checked-on 2026-10-19
.venv/bin/python scripts/import-bundled-knowledge.py --check
```

更新前先核对主管部门最新制度栏目，编辑 catalog.json 的链接、范围和状态，再运行收集。检查差异、废止关系及金额表格后提交 originals、texts、catalog 和 manifest。收集器只解析公开内容，不使用校园登录凭据，不执行网页脚本，不绕过登录或校内访问限制。

## 来源目录

| 分类 | 份数 |
|---|---:|
| 招生 | 2 |
| 培养与课程 | 8 |
| 学位与论文 | 15 |
| 违纪处分 | 1 |
| 奖助与三助 | 13 |
| 研究生管理与住宿 | 10 |
| 本科教学与学籍 | 5 |
| 财务 | 1 |

| 标题 | 官网日期 | 状态 | 官方来源 | 本地正文 |
|---|---|---|---|---|
| 2027年推荐免试研究生接收工作实施办法 | 2026-09-08 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1074/20524.htm) | [正文](texts/admissions/gr-1074-20524.md) |
| 2027年硕士研究生招生简章 | 2026-09-30 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1074/20804.htm) | [正文](texts/admissions/gr-1074-20804.md) |
| 西安电子科技大学研究生课程教学管理规定 | 2021-04-16 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1039/10191.htm) | [正文](texts/courses/gr-1039-10191.md) |
| 西安电子科技大学研究生课程考核管理办法 | 2021-04-16 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1039/10192.htm) | [正文](texts/courses/gr-1039-10192.md) |
| 西安电子科技大学研究生课程置换与学分认定管理办法 | 2021-04-16 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1039/10194.htm) | [正文](texts/courses/gr-1039-10194.md) |
| 西安电子科技大学研究生转学科（专业学位类别）、转导师管理办法（试行） | 2021-04-16 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1039/10195.htm) | [正文](texts/courses/gr-1039-10195.md) |
| 西安电子科技大学研究生教学事故认定及处理办法（试行） | 2021-04-16 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1039/10196.htm) | [正文](texts/courses/gr-1039-10196.md) |
| 西安电子科技大学博士研究生综合考试实施办法 | 2021-04-21 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1039/10205.htm) | [正文](texts/courses/gr-1039-10205.md) |
| 西安电子科技大学研究生必修环节类管理规定 | 2021-04-21 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1039/10206.htm) | [正文](texts/courses/gr-1039-10206.md) |
| 2026年秋季学期培养计划制定及选课指南 | 2026-09-04 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1073/20504.htm) | [正文](texts/courses/gr-1073-20504.md) |
| 西安电子科技大学优秀博士学位论文资助基金实施办法（2021年修订） | 2021-12-01 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1045/11173.htm) | [正文](texts/degrees/gr-1045-11173.md) |
| 西安电子科技大学研究生学位论文抽检结果处理办法（2024年修订） | 2024-07-10 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1045/14564.htm) | [正文](texts/degrees/gr-1045-14564.md) |
| 《中华人民共和国学位法》 | 2025-03-07 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1045/15344.htm) | [正文](texts/degrees/gr-1045-15344.md) |
| 全国工程专业学位研究生教育指导委员会关于发布《工程类专业学位类别硕士学位论文基本要求（试行）》的通知 | 2025-05-08 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1045/15804.htm) | [正文](texts/degrees/gr-1045-15804.md) |
| 教育部关于转发《工程类博士专业学位研究生学位论文与申请学位实践成果基本要求（试行）》的通知 | 2025-05-08 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1045/15814.htm) | [正文](texts/degrees/gr-1045-15814.md) |
| 西安电子科技大学博士学位授予实施细则（2025年修订） | 2025-06-18 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1045/16464.htm) | [正文](texts/degrees/gr-1045-16464.md) |
| 西安电子科技大学硕士学位授予实施细则（2025年修订） | 2025-06-18 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1045/16474.htm) | [正文](texts/degrees/gr-1045-16474.md) |
| 西安电子科技大学工程类专业学位博士研究生申请学位研究成果基本要求（2025年修订） | 2025-06-18 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1045/16484.htm) | [正文](texts/degrees/gr-1045-16484.md) |
| 西安电子科技大学关于研究生毕业、结业、肄业的管理规定 （2025年修订） | 2025-06-18 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1045/16494.htm) | [正文](texts/degrees/gr-1045-16494.md) |
| 西安电子科技大学研究生学位授予复核工作暂行办法 | 2025-06-18 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1045/16504.htm) | [正文](texts/degrees/gr-1045-16504.md) |
| 西安电子科技大学博士学位论文或实践成果总结报告双盲评审实施细则（2026年修订） | 2026-04-09 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1045/19204.htm) | [正文](texts/degrees/gr-1045-19204.md) |
| 西安电子科技大学研究生学位论文抽查评估实施细则 | 2016-11-10 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1045/5286.htm) | [正文](texts/degrees/gr-1045-5286.md) |
| 西安电子科技大学关于调整博士研究生学制的通知 | 2018-01-08 | DRAFT | [官网](https://gr.xidian.edu.cn/info/1045/6076.htm) | [正文](texts/degrees/gr-1045-6076.md) |
| 西安电子科技大学学位证明书管理办法 | 2019-04-17 | ACTIVE | [官网](https://gr.xidian.edu.cn/info/1045/7343.htm) | [正文](texts/degrees/gr-1045-7343.md) |
| 西安电子科技大学研究生申请学位研究成果基本要求（2020年修订） | 2020-07-07 | DRAFT | [官网](https://gr.xidian.edu.cn/info/1045/9204.htm) | [正文](texts/degrees/gr-1045-9204.md) |
| 西安电子科技大学研究生违纪处分办法 | 2026-06-15 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1062/1639.htm) | [正文](texts/discipline/ygb-1062-1639.md) |
| 研究生助研津贴发放要求及流程变动说明 | 2023-09-19 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1059/1587.htm) | [正文](texts/funding/ygb-1059-1587.md) |
| 西安电子科技大学基层就业学费补偿国家助学贷款代偿实施办法 | 2022-08-19 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1059/1589.htm) | [正文](texts/funding/ygb-1059-1589.md) |
| 西安电子科技大学研究生校长奖评选管理办法 | 2022-06-28 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1059/1590.htm) | [正文](texts/funding/ygb-1059-1590.md) |
| 西安电子科技大学研究生学业奖学金管理办法 | 2022-06-28 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1059/1591.htm) | [正文](texts/funding/ygb-1059-1591.md) |
| 西安电子科技大学研究生助管岗位设置与管理办法 | 2019-07-11 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1059/1592.htm) | [正文](texts/funding/ygb-1059-1592.md) |
| 西安电子科技大学研究生国家助学金管理办法 | 2019-05-29 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1059/1593.htm) | [正文](texts/funding/ygb-1059-1593.md) |
| 西安电子科技大学研究生助研工作管理办法 | 2019-05-29 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1059/1594.htm) | [正文](texts/funding/ygb-1059-1594.md) |
| 西安电子科技大学研究生承担本科助教工作管理办法 | 2019-05-29 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1059/1596.htm) | [正文](texts/funding/ygb-1059-1596.md) |
| 西安电子科技大学服兵役学生国家教育资助实施办法 | 2022-08-19 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1059/1601.htm) | [正文](texts/funding/ygb-1059-1601.md) |
| 关于印发《西安电子科技大学研究生国家奖学金评审管理办法》的通知 | 2024-05-14 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1059/2471.htm) | [正文](texts/funding/ygb-1059-2471.md) |
| 关于印发《西安电子科技大学家庭经济困难研究生资助管理办法》的通知 | 2024-05-14 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1059/2472.htm) | [正文](texts/funding/ygb-1059-2472.md) |
| 关于开展2025～2026学年（秋季）研究生学业奖学金评定工作的通知 | 2025-10-21 | ARCHIVED | [官网](https://ygb.xidian.edu.cn/info/1067/12483.htm) | [正文](texts/funding/ygb-1067-12483.md) |
| 关于开展2025年研究生国家奖学金评审工作的通知 | 2025-09-12 | ARCHIVED | [官网](https://ygb.xidian.edu.cn/info/1067/12503.htm) | [正文](texts/funding/ygb-1067-12503.md) |
| 西安电子科技大学研究生请销假管理办法 | 2026-06-30 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1061/13093.htm) | [正文](texts/graduate_management/ygb-1061-13093.md) |
| 西安电子科技大学研究生校外住宿管理办法 | 2026-06-30 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1061/13103.htm) | [正文](texts/graduate_management/ygb-1061-13103.md) |
| 西安电子科技大学研究生学籍管理实施细则 | 2019-04-19 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1061/1628.htm) | [正文](texts/graduate_management/ygb-1061-1628.md) |
| 关于印发《西安电子科技大学研究生管理规定》的通知 | 2018-07-23 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1061/1630.htm) | [正文](texts/graduate_management/ygb-1061-1630.md) |
| 研究生学生证新办、补办流程 | 2019-09-06 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1061/1706.htm) | [正文](texts/graduate_management/ygb-1061-1706.md) |
| 西安电子科技大学研究生学籍电子注册管理办法（试行） | 2024-10-21 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1061/3403.htm) | [正文](texts/graduate_management/ygb-1061-3403.md) |
| 西安电子科技大学研究生校外学习实践管理办法 | 2024-07-18 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1061/4703.htm) | [正文](texts/graduate_management/ygb-1061-4703.md) |
| 关于印发《西安电子科技大学学生公寓管理和服务办法》的通知 | 2025-05-16 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1061/5463.htm) | [正文](texts/graduate_management/ygb-1061-5463.md) |
| 关于印发《西安电子科技大学 学生公寓空调使用管理细则（试行）》的通知 | 2025-06-25 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1061/6353.htm) | [正文](texts/graduate_management/ygb-1061-6353.md) |
| 研究生教育管理常见手续办理流程介绍 | 2026-03-03 | ACTIVE | [官网](https://ygb.xidian.edu.cn/info/1061/9703.htm) | [正文](texts/graduate_management/ygb-1061-9703.md) |
| 关于印发《西安电子科技大学大类招生学生分流转专业实施办法（试行）》的通知 | 2021-11-10 | ACTIVE | [官网](https://jwc.xidian.edu.cn/info/1024/10929.htm) | [正文](texts/undergraduate/jwc-1024-10929.md) |
| 关于印发《西安电子科技大学推荐优秀应届本科毕业生免试攻读硕士学位研究生工作实施办法（修订）》的通知 | 2021-11-10 | ACTIVE | [官网](https://jwc.xidian.edu.cn/info/1024/10930.htm) | [正文](texts/undergraduate/jwc-1024-10930.md) |
| 关于印发《西安电子科技大学本科生平均绩点（GPA）计算方法》的通知 | 2022-09-26 | DRAFT | [官网](https://jwc.xidian.edu.cn/info/1024/12226.htm) | [正文](texts/undergraduate/jwc-1024-12226.md) |
| 西安电子科技大学学籍身份信息修改或变更管理办法 | 2023-03-14 | ACTIVE | [官网](https://jwc.xidian.edu.cn/info/1024/12457.htm) | [正文](texts/undergraduate/jwc-1024-12457.md) |
| 西安电子科技大学学籍管理规定（2019版） | 2019-12-11 | ACTIVE | [官网](https://jwc.xidian.edu.cn/info/1024/7722.htm) | [正文](texts/undergraduate/jwc-1024-7722.md) |
| 西安电子科技大学工作餐费财务报销管理规定 | 未标明 | DRAFT | [官网](https://cwc.xidian.edu.cn/__local/7/7E/69/0E110D874888E93174D33ABAB1D_75705DD5_4CE47.pdf) | [正文](texts/research_finance/cwc-work-meals.md) |

## 原文权属与来源

官方文件原文、校徽和版式等权利归原发布机构或各自权利人；本项目代码的任何许可不自动覆盖这些第三方资料。保留官方来源、原文及核查信息，用于政策查询和来源核对。文件效力与后续修订以学校主管部门的正式发布为准。
