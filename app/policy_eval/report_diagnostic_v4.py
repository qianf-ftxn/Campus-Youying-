"""Aggregate v4 contributions, grading sensitivity, and stress-test failures without editing labels."""
from __future__ import annotations
import json
import statistics
from collections import Counter
from pathlib import Path
import numpy as np
from app.policy_eval.diagnostic_v4 import BASE, HARD_DATASET, HOLDOUT, HARD_ARMS, quality_summary
from app.policy_eval.runner import digest, load_inputs, build_chunks, score_case

NAMES={'parent_child_hybrid':'现有父子混合','archived_flat_hybrid':'原始平面混合','parent_semantic_v3':'重排＋精简提示词',
 'precision_only':'仅精简提示词','rerank_only':'仅语义重排','no_context_strict':'无资料，原提示词',
 'parametric_probe':'无资料，允许已有知识','random_context':'随机无关资料','oracle_evidence':'最短标注片段',
 'oracle_source_blocks':'标准证据所在完整父块','flat_vector':'普通小块向量','parent_child_vector':'父子向量，无BM25','bm25_parent_smoke':'仅BM25映射父块'}


def read(path):return json.loads(path.read_text())
def pct(v):return '—' if v is None else f'{100*v:.2f}%'
def rate(rows,positive=True):
 r=[x for x in rows if x['answerable']==positive]
 return f"{sum(x['quality']['answer_correct'] for x in r)}/{len(r)}" if r else '—'
def mean(rows,key):
 vals=[r['quality'].get(key) for r in rows if r['quality'].get(key) is not None]
 return statistics.mean(vals) if vals else None


def paired(rows,a,b,positive=True):
 by={(r['case_id'],r['method']):r for r in rows}
 ids=sorted({r['case_id'] for r in rows if r['method']==a and r['answerable']==positive} & {r['case_id'] for r in rows if r['method']==b and r['answerable']==positive})
 pairs=[(i,int(by[i,a]['quality']['answer_correct']),int(by[i,b]['quality']['answer_correct'])) for i in ids]
 return {'cases':len(pairs),'baseline_correct':sum(x for i,x,y in pairs),'variant_correct':sum(y for i,x,y in pairs),
  'variant_only':[i for i,x,y in pairs if y and not x],'baseline_only':[i for i,x,y in pairs if x and not y],
  'delta':statistics.mean(y-x for i,x,y in pairs) if pairs else None}


def main():
 datasets={name:load_inputs(path)[0] for name,path in [('old60',HOLDOUT),('hard32',HARD_DATASET)]}
 a=read(BASE/'ablation/audited-report.json'); a0=read(BASE/'ablation/report.json')
 hard=read(BASE/'challenge/report.json'); ha=read(BASE/'challenge/audited-report.json')
 word=read(BASE/'wording/audited-report.json'); w0=read(BASE/'wording/report.json')
 components=read(BASE/'retrieval-components.json')
 for obj,path in [(a,HOLDOUT),(hard,HARD_DATASET),(ha,HARD_DATASET)]:
  if obj['dataset_sha256']!=digest(path.read_bytes()):raise ValueError('Dataset hash mismatch')
 if len(a['results'])!=324 or len(hard['results'])!=160 or len(ha['results'])!=160:raise ValueError('Incomplete result matrix')
 pairs={m:paired(a['results'],'parent_child_hybrid',m) for m in a['summary'] if m!='parent_child_hybrid'}
 hp={m:paired(ha['results'],'parent_child_hybrid',m) for m in HARD_ARMS if m!='parent_child_hybrid'}
 hard_cases={c['id']:c for c in datasets['hard32']['cases']}
 _,hard_docs=load_inputs(HARD_DATASET)
 chunks={c.id:c for c in build_chunks(hard_docs)}
 misses=[]
 for row in hard['results']:
  if row['method']=='parent_semantic_v3' and row['answerable'] and not row['retrieval']['metrics']['evidence_complete']:
   retrieved=row['retrieval'];case=hard_cases[row['case_id']]
   pool=score_case(case,[(chunks[i],0.) for i in retrieved['pre_rerank_ids']],hard_docs)
   selected=score_case(case,[(chunks[i],0.) for i in retrieved['reranked_ids']],hard_docs)
   misses.append({'case_id':case['id'],'candidate_pool_recall':pool['clause_recall_at_k'],
    'rerank_selected_recall_before_budget':selected['clause_recall_at_k'],
    'final_recall':retrieved['metrics']['clause_recall_at_k']})
 # Conditional family bootstrap for paired controls on the old60, not for the
 # hand-picked challenge where source families are heavily interconnected.
 cases={c['id']:c for c in datasets['old60']['cases']}
 rng=np.random.default_rng(20261031)
 for method,pair in pairs.items():
  by={(r['case_id'],r['method']):r for r in a['results']}
  ids=[i for i,c in cases.items() if c['answerable']]
  families=sorted({cases[i]['evidence_family'] for i in ids})
  groups=[[int(by[i,method]['quality']['answer_correct'])-int(by[i,'parent_child_hybrid']['quality']['answer_correct']) for i in ids if cases[i]['evidence_family']==f] for f in families]
  samples=[statistics.mean(v for idx in rng.integers(0,len(groups),len(groups)) for v in groups[idx]) for _ in range(5000)]
  pair['conditional_family_bootstrap_95']=np.quantile(samples,[.025,.975]).tolist()
 result={'status':'AI scoring, all source/business annotations pending human; controlled offline diagnostics',
  'dataset_audit':{n:{'cases':len(d['cases']),'answerable':sum(c['answerable'] for c in d['cases']),
   'cross_document_cases':sum(len({g['document_id'] for g in c['evidence']})>=2 for c in d['cases']),
   'gold_unit_counts':dict(Counter(len(c['evidence']) for c in d['cases'] if c['answerable'])),
   'history_lengths':dict(Counter(len(c['history']) for c in d['cases'] if c['history']))} for n,d in datasets.items()},
  'ablation_original':a0['summary'],'ablation_audited':a['summary'],'ablation_paired':pairs,
  'challenge_original':hard['summary'],'challenge_audited':ha['summary'],'challenge_paired':hp,
  'retrieval_components':components['summary'],
  'source_equivalence_audit':read(Path(__file__).with_name('source_equivalence_v4.json')),
  'remaining_rerank_misses':misses,
  'challenge_by_category':{cat:quality_summary([r for r in ha['results'] if r['category']==cat]) for cat in sorted({c['category'] for c in datasets['hard32']['cases']})},
  'wording_original':w0['original_summary'],'wording_rewritten':w0['summary'],'wording_rewritten_audited':word['summary'],
  'wording_original_audited':quality_summary([r for r in a['results'] if r['case_id'] in {v['case_id'] for v in word['results']} and r['method'] in word['summary']]),
  'label_changes':{'ablation':a['changed_labels'],'challenge':ha['changed_labels'],'wording':word['changed_labels']},
  'artifact_sha256':{str(p.relative_to(BASE)):digest(p.read_bytes()) for p in [BASE/'protocol.json',BASE/'audit-protocol.json',BASE/'ablation/inputs.json',BASE/'ablation/audited-report.json',BASE/'challenge-input.json',BASE/'challenge/report.json',BASE/'challenge/audited-report.json']}}
 (BASE/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 share=Path('docs/policy-diagnostic-v4-summary.json');share.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 lines=['# 高分来源消融与32道复杂场景压力测试','',
 '本轮先在原60题上做配对消融，再新增32题压力集。检索、生成、重排参数沿用v3，K=5、4000估算知识token、温度0、512输出token；没有依据新题输出调参或修改标准答案。生成DeepSeek，质量及证据蕴含裁判Qwen。全部题目仍待人工审核。','',
 '## 原60题：证据、已有知识和问题提示','',
 '下表用统一二次复核口径；所有初始分数和逐条变更理由另存。无资料的两组分别保持原资料边界提示词、改用允许已有知识的探测提示词。两者不是完全相同的提示词条件，不能把差值都解释为模型知识。随机资料排除指定证据及原检索文档，按块数和长度近似匹配，不能保证不存在同义规则。','',
 '| 条件 | 可回答题符合数 | 资料不足题符合数 | 可回答题要点覆盖率 |','|---|---:|---:|---:|']
 for m in ['parent_child_hybrid','no_context_strict','parametric_probe','random_context','oracle_evidence','oracle_source_blocks']:
  rows=[r for r in a['results'] if r['method']==m]
  s=a['summary'][m];lines.append(f"| {NAMES[m]} | {rate(rows)} | {rate(rows,False)} | {pct(s['point_coverage_answerable'])} |")
 lines+=['',f"在这48题、同一原回答提示词下，加入正常检索资料相对不提供资料的自动符合率差为{-100*pairs['no_context_strict']['delta']:.2f}个百分点；允许已有知识的探测和随机资料相对正常检索分别差{-100*pairs['parametric_probe']['delta']:.2f}、{-100*pairs['random_context']['delta']:.2f}个百分点。这些是固定条件下的配对差值，不是可相加的‘贡献占比’，也不等于总体模型知识率。"]
 lines+=['','“符合”指自动判定必要要点覆盖、无实质错误且行为符合预期，不是人工确认的最终准确率。无上下文组的事实忠实度不适合直接与有上下文组比较，本轮消融聚焦答案要点和行为；不将无事实回答计为忠实度1。','',
 '无资料仍能答对，可能来自模型已有知识、题面线索或猜测，这些来源不能只靠一次消融精确分开。随机资料让表现下降，说明错误上下文会干扰；这不意味着随机组与正常组的差值就是检索的可加贡献比例。只给最短原文片段会丢表头和邻近条件，因此另补完整父块的证据选择对照，二者都使用已有AI证据标注选资料，只用于诊断，不能部署为真实检索，也不是理论上的生成准确率上限。','',
 '8道已知题另做去掉候选答案/数字提示的配对措辞探测，保持决定性用户条件和标准证据；现有检索须对改写问题重新检索。原题与改写题都保存，规模小且措辞等价性待人审，不能据此宣布题面提示的百分比贡献。详见机器汇总中wording字段。','',
 f"同一复核口径下，检索组原题{result['wording_original_audited']['parent_child_hybrid']['correct_answerable']}/8、改写题{word['summary']['parent_child_hybrid']['correct_answerable']}/8；允许已有知识的无资料组原题{result['wording_original_audited']['parametric_probe']['correct_answerable']}/8、改写题{word['summary']['parametric_probe']['correct_answerable']}/8。无资料组初评和复核还出现方向不同的单题变化，说明这点差异不足以稳定归因给题面提示。",'',
 '## 新32题：复杂条件下的检索与回答','',
 '证据召回按指定文档/版本原文覆盖计分，其他来源同义规定不自动计入；该指标可能低估实际语义覆盖，不能把缺指定来源直接称为缺事实。构成：12道跨文档、8道含4～5条历史消息的对话、6道相近规则/冲突/边界辨析、6道资料或关键条件不足。26道可回答题；跨文档题均需至少2份文档，部分需3份。答案事实分散在身份、年级、时点和业务状态中。它是AI按同一语料编写的定向压力集，来源与此前题目可能重合，没有伪装成外部盲测或真实用户总体样本；不与旧60题合并报告一个总分。','',
 '| 方案 | 指定来源证据召回率 | 指定证据齐全题 | 可回答题符合数（复核） | 负例符合数（复核） | Faithfulness（初评） | Answer Relevancy |','|---|---:|---:|---:|---:|---:|---:|']
 for m in HARD_ARMS:
  rows=[r for r in ha['results'] if r['method']==m];orig=[r for r in hard['results'] if r['method']==m]
  summary=hard['summary'][m]
  complete=sum(r['retrieval']['metrics']['evidence_complete'] for r in orig if r['answerable'])
  lines.append(f"| {NAMES[m]} | {pct(summary['necessary_evidence_recall'])} | {complete}/26 | {rate(rows)} | {rate(rows,False)} | {summary['faithfulness']:.4f} | {summary['answer_relevancy']:.4f} |")
 lines+=['',f"相对现有父子方案，仅重排多通过{hp['rerank_only']['variant_correct']-hp['rerank_only']['baseline_correct']}题，仅改提示词净变化{hp['precision_only']['variant_correct']-hp['precision_only']['baseline_correct']:+d}题，两项同时改多通过{hp['parent_semantic_v3']['variant_correct']-hp['parent_semantic_v3']['baseline_correct']}题。该压力集上观察到的主要改善来自证据排序/选择，单改提示词未体现可回答题提升；仍可能存在因素交互和单次输出波动。"]
 lines+=['','后二项沿用v3响应独立提取＋Qwen闭上下文蕴含、三反推问题Embedding余弦；二次复核仅修复答案要点/行为的统一判定，不重新生成回答或改写NLI结果。个别NLI引文通过让同一裁判选择真实原文段落编号完成格式补评，程序只映射原文、不翻转支持标签；日志另存。抽取器仍可能把多项事实合为一条，主张权重及支持判定均需人审。精简提示词与原提示词、重排与不重排构成2×2对照，原始平面方案另列。API温度0仍可能有输出波动；每条件只有一次回答，不能把每个百分点当作稳定因果收益。','',
 '## 检索组件对照（同一32题，仅测检索）','',
 '| 组件 | 必要证据召回率 | 证据齐全率 |','|---|---:|---:|']
 for m,s in components['summary'].items():lines.append(f"| {NAMES[m]} | {pct(s['clause_recall_at_k'])} | {pct(s['evidence_complete_rate'])} |")
 lines+=['','普通小块向量使用公共300/40窗口，不等同于原始平面方案的512/64＋混合检索＋规则扩展。父子向量对照去除BM25并重新归一化向量权重；仅BM25映射同一父块。比较反映该条件下的检索表现，不单独证明父子结构或融合算法在所有场景的收益。','',
 '## 二次评分复核与仍需处理的问题','',
 '初评发现：旧H-X12回答首句“未达到”与后文“两项满足”矛盾被漏判；H-M09等又被要求额外重复“不保证到账”；还有裁判因自身猜测的当前年份否定固定2026-10-19快照。统一增加规则后对所有消融、压力集和措辞改写回答复核，原判定、变更理由和新判定全部保留。该修复发生在看到部分输出以后，属于透明的评分敏感性检查，并非预注册评分，更不能当人工审核。','']
 for n,changes in result['label_changes'].items():lines.append(f"- {n}：{len(changes)}个自动通过标签变化，具体题号、原判和新判见机器汇总。")
 lines+=['','二次复核仍非可靠金标准：H-M09的复核理由已称“到账免责声明属于可选补充”，结构化标签却仍把该要点判为未覆盖；H-U11虽明确拒绝删档并未声称执行，仍被判为answer而非abstain。原始标签照存、未人工改成通过。这说明增强裁判指令不能替代人工复核，1～2题差异尤其不能据此宣称性能优势。']
 lines+=['','按指定来源口径，C4-X01（自行租房＋校外实践）和C4-M06（实践单位安排住宿的排除条件）未选齐标准证据。两题在前32候选中指定证据本来齐全，重排选5块时没有全部选入，4000token预算没有再造成额外损失；不能仅凭缺指定来源就断言实际漏检。']
 lines+=['','进一步原文核对发现，实际取回的ygb-1061-9703办事流程已重复住房范围定义：C4-M06属于同义来源未计入，不应称为实际漏掉排除条件。C4-X01的住房范围和审批路径也有等价资料，但该返回片段没有标注要求的家属知情同意书上传细节，仍存在部分取证缺项。该AI来源等价审查单独保存在source-equivalence-notes.json，待人工确认；没有事后改写冻结标准证据或悄悄抬高召回分数。']
 lines+=['','生成侧也有独立问题：C4-B03的“重排＋精简提示词”已取齐证据，却只答证明书补办3个工作日，遗漏了用户同时询问的初次学位申请受理60日及截止日起算。相反，有些缺指定证据的题仍能依题面/已有知识答出参考结论，因此“答对”不等于“证据完整或全部事实有据”。下一步应同时检查子问题覆盖，不能把所有错例归因于检索。']
 lines+=['','必要证据齐全与否应和逐题回答一起看；总体Hit@5只需命中一块，无法衡量跨文档缺项。压力集的手工构造、相关题族、单次模型输出、裁判误差及业务有效性待核验，仍限制结果外推。检索查询固定拼接用户历史，未覆盖生产查询改写或完整多智能体运行。未接入生产业务库、未重建线上索引、未启用实验开关。','',
 '## 运行开销和复核文件','',
 '| 方案（32题） | 生成输入token | 重排输入token | 推理总输入token | 正常调用数 |','|---|---:|---:|---:|---:|']
 for m in HARD_ARMS:
  rows=[r for r in hard['results'] if r['method']==m]
  gen=sum(r['generation']['usage'].get('prompt_tokens',0) for r in rows)
  rr=sum(r['retrieval'].get('rerank_usage',{}).get('prompt_tokens',0) for r in rows)
  lines.append(f"| {NAMES[m]} | {gen} | {rr} | {gen+rr} | {len(rows)+(len(rows) if m in {'rerank_only','parent_semantic_v3'} else 0)} |")
 lines+=['','重排结果在两个因子实验中复用，表中重排token表示各方案单独部署的推理开销；实际实验未为两臂重复请求同一重排。token不含裁判、Embedding、失败或校验重试，不等于实际货币费用。完整父块对照首次写出目录失败导致38个完成响应未落盘、随后重新请求；这些执行开销以及中断恢复开销未计入表，没有拿重复结果择优。已有知识探测有1份512token截断，所有组保留该结果，不选择性删题；压力集截断数见机器汇总。','',
 '- [机器汇总](policy-diagnostic-v4-summary.json)：主要指标、配对差异、标签变更和产物摘要。',
 '- `app/policy_eval/challenge_v4.json`：32道新增题、原文偏移、标准要点、审核状态。',
 '- `target/policy-pilot/diagnostic-v4/ablation/`：初始消融、完整父块对照、统一评分。',
 '- `target/policy-pilot/diagnostic-v4/challenge/`：160份回答、逐块检索、独立评分和事实支持判定。',
 '- `target/policy-pilot/diagnostic-v4/wording/`：8对措辞探测。',
 '- `target/policy-pilot/diagnostic-v4/human-review.md`：新增难题匿名逐题审核包。','',
 '```bash','.venv/bin/python -m app.policy_eval.diagnostic_v4 --phase ablation',
 '.venv/bin/python -m app.policy_eval.diagnostic_v4 --phase wording',
 '.venv/bin/python -m app.policy_eval.diagnostic_v4 --phase challenge',
 '# 如出现无法逐字定位的引文格式失败，可补评后恢复同一阶段',
 '.venv/bin/python -m app.policy_eval.repair_citations_v4',
 '.venv/bin/python -m app.policy_eval.diagnostic_v4 --phase challenge',
 '.venv/bin/python -m app.policy_eval.audit_v4 --phase ablation',
 '.venv/bin/python -m app.policy_eval.audit_v4 --phase wording',
 '.venv/bin/python -m app.policy_eval.audit_v4 --phase challenge',
 '.venv/bin/python -m app.policy_eval.components_v4',
 '.venv/bin/python -m app.policy_eval.report_diagnostic_v4','```','',
 '已存在的题集不覆盖；成功请求按完整请求指纹复用。首次重现旧平面方案仍需原先备份归档。以上均为离线测试，不改业务默认。']
 Path('docs/policy-diagnostic-v4.md').write_text('\n'.join(lines)+'\n')
 # Blinded source-grounded review package, no automatic approvals.
 review=['# v4复杂场景人工审核包','','所有样本待人工确认。请先审核问题/证据/必要要点，再逐方案判定；别把模型裁判当标准答案。','']
 by={(r['case_id'],r['method']):r for r in ha['results']}
 aliases={m:f'方案{n+1}' for n,m in enumerate(sorted(HARD_ARMS,key=lambda m:digest(('v4-blind:'+m).encode())))}
 for case in datasets['hard32']['cases']:
  review += [f"## {case['id']} · {case['category']}",'',case['question'],'']
  for h in case['history']:review += [f"历史 {h['role']}：{h['content']}",'']
  review += ['预期行为：'+case['expected_behavior'],'','必要要点：','']+[f"{i}. {p}" for i,p in enumerate(case['answer_points'],1)]+['']
  for g in case['evidence']:review += [f"证据 {g['document_id']} L{g['line_start']} · [快照记录来源]({g['source_url']})",'', '> '+g['quote'].replace('\n','\n> '),'']
  for m in sorted(HARD_ARMS,key=lambda m:aliases[m]):
   row=by[case['id'],m]
   review += ['### '+aliases[m],'',row['generation']['text'],'','人工：□通过 □错例 □标注需修订；审核人：____；理由：____','']
   for i,c in enumerate(row['retrieval']['contexts'],1):review += [f"资料{i} · {c['document_id']}",'','> '+c['content'].replace('\n','\n> '),'']
 review+=['## 方案映射（审核后查看）','']+[f'{aliases[m]}：{NAMES[m]}' for m in HARD_ARMS]
 (BASE/'human-review.md').write_text('\n'.join(review)+'\n')
 print(json.dumps({'ablation':a['summary'],'challenge':ha['summary'],'artifacts':'docs/policy-diagnostic-v4.md'},ensure_ascii=False))

if __name__=='__main__':main()
