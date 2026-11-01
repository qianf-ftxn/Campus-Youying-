"""Uniform second-pass scoring audit. Original judgments are never overwritten."""
from __future__ import annotations
import argparse
import asyncio
import copy
import json
import statistics
from app.core.config import Settings
from app.policy_eval.answer_compare import run_stage, read_records
from app.policy_eval.diagnostic_v4 import (BASE, HARD_DATASET, HOLDOUT, JUDGE_MODEL, QUALITY_PROMPT,
    payload, parse_quality, freeze, evaluate_quality, quality_summary)
from app.policy_eval.runner import load_inputs, digest, build_chunks, score_case, select_context
from app.services.memory_context import estimate_tokens

AUDIT_PROMPT = QUALITY_PROMPT + '''
补充统一复核规则（应用到所有方案）：
1. 本实验的资料截止日固定为2026-10-19，这是已给定的实验条件；不得因模型自身猜测的当前年份判其虚构。
2. 必须通读回答首句、分析、结尾。若先明确断言不达标又明确断言达标，即使末尾结论正确，仍记录实质性自相矛盾；不要只看最后一句放过前文错误。
3. “不保证实际到账/最终资格”等负约束，只要回答没有作出相应无依据保证且按所问规则作答即可满足，不必额外朗读免责声明。提供数额标准不等于承诺实际到账。明确只回答部分门槛、不宣称全部资格即满足对应边界。
4. 不增设题目和参考要点未要求的强制内容。每条遗漏或实质错误给出对应依据，不能把同主题或其他身份的规定代入。
'''


def source_oracle_rows(dataset, docs, previous):
    parents = [r for r in build_chunks(docs) if r.kind == 'parent']
    full = {r['case_id']: r for r in previous['results'] if r['method'] == 'parent_child_hybrid'}
    rows=[]
    for case in dataset['cases']:
        if not case['answerable']: continue
        gold_docs={g['document_id'] for g in case['evidence']}
        pool=[r for r in parents if r.document_id in gold_docs]
        selected=[]
        # Oracle uses labels ONLY to choose genuine source blocks, with the same
        # K/budget. It is a diagnostic bound on selection, never production retrieval.
        while pool and len(selected)<5:
            before=score_case(case,selected,docs)['clause_recall_at_k']
            ranked=sorted(pool,key=lambda r:(-score_case(case,selected+[(r,1.)],docs)['clause_recall_at_k'],estimate_tokens(r.content),r.id))
            item=ranked[0]
            proposed,used=select_context(selected+[(item,1.)],5,4000)
            after=score_case(case,proposed,docs)['clause_recall_at_k']
            if after<=before: break
            selected=proposed;pool.remove(item)
            if after>=1: break
        base=copy.deepcopy(full[case['id']]['retrieval'])
        base.update(method='oracle_source_blocks',estimated_tokens=sum(estimate_tokens(r.content) for r,s in selected),
            metrics=score_case(case,selected,docs),diagnostic_only=True,
            contexts=[{'chunk_id':r.id,'document_id':r.document_id,'version':r.version,
                'heading':r.heading,'content':r.content,'score':s} for r,s in selected])
        rows.append(base)
    return rows


async def audit(rows, dataset_path, out, settings, *, write_report=True):
    dataset,docs=load_inputs(dataset_path)
    cases={c['id']:c for c in dataset['cases']}
    tasks=[]
    for row in rows:
        case=cases[row['case_id']]
        contexts=row['retrieval']['contexts']
        expanded=[{'rank':i,'title':docs[c['document_id']][0]['title'],'scope':docs[c['document_id']][0]['scope'],'content':c['content']} for i,c in enumerate(contexts,1)]
        content={'experiment_snapshot_date':'2026-10-19','question':case['question'],'history':case['history'],
            'answerable':case['answerable'],'reference_points':case['answer_points'],
            'optional_point_indices':case.get('optional_point_indices',[]),
            'reference_evidence':[g['quote'] for g in case['evidence']], 'contexts':expanded,'answer':row['generation']['text']}
        tasks.append({'case_id':case['id'],'method':row['method'],'case':case,'contexts':contexts,
            'payload':payload(settings,JUDGE_MODEL,AUDIT_PROMPT,content)})
    judgments=await run_stage('quality-audit',tasks,out/'quality-audit.jsonl',settings.openai_base_url,settings.openai_api_key,6,parse_quality)
    results=[]
    changes=[]
    for row in rows:
        item=copy.deepcopy(row)
        case=cases[row['case_id']];j=judgments[(case['id'],row['method'])]['parsed']
        required=set(case.get('required_point_indices',range(1,len(case['answer_points'])+1)))
        points=[p['covered'] for p in j['point_verdicts'] if p['index'] in required]
        safe=not j['contradictions'] and not j['fabricated_specifics']
        correct=bool(all(points) and safe and j['behavior']==case['expected_behavior'])
        old=row['quality']['answer_correct']
        if old!=correct: changes.append({'case_id':case['id'],'method':row['method'],'before':old,'after':correct,'reason':j.get('reason',j['point_verdicts'])})
        item['original_judgment']=item['judgment'];item['judgment']=j
        item['original_quality']=item['quality']
        item['quality']={'answer_correct':correct,'answer_point_coverage':statistics.mean(points),
            'safe':safe,'behavior':j['behavior']}
        results.append(item)
    report={'status':'Uniform automatic scoring audit, pending human; no model/retrieval/answer changes',
        'dataset_sha256':digest(dataset_path.read_bytes()),'audit_prompt_sha256':digest(AUDIT_PROMPT.encode()),
        'summary':quality_summary(results),'changed_labels':changes,'results':results}
    if write_report:
        (out/'audited-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'summary':report['summary'],'changed_labels':len(changes)},ensure_ascii=False),flush=True)
    return report


async def main_async(phase):
    settings=Settings()
    freeze(BASE/'audit-protocol.json',{'audit_prompt':AUDIT_PROMPT,'model':JUDGE_MODEL,
        'purpose':'Scoring consistency repair identified after first-pass outputs; not preregistered. Same fixed rule for all arms, originals retained.'})
    if phase=='ablation':
        out=BASE/'ablation'
        previous=json.loads((out/'report.json').read_text())
        dataset,docs=load_inputs(HOLDOUT)
        oracle=source_oracle_rows(dataset,docs,previous)
        freeze(out/'oracle-source-inputs.json',oracle)
        source_rows=await evaluate_quality(HOLDOUT,out/'source-oracle',oracle,settings)
        return await audit(previous['results']+source_rows,HOLDOUT,out,settings)
    if phase=='wording':
        out=BASE/'wording'
        previous=json.loads((out/'report.json').read_text())
        return await audit(previous['results'],out/'dataset.json',out,settings)
    if phase=='challenge-pregrade':
        # Answers are already complete. Independently score them while the
        # slower claim/NLI stage is running; final aggregation still waits for it.
        out=BASE/'challenge'
        dataset,_=load_inputs(HARD_DATASET)
        retrieved=json.loads((out/'retrieval.json').read_text())
        if retrieved['dataset_sha256']!=digest(HARD_DATASET.read_bytes()):
            raise ValueError('Challenge source changed')
        contexts={(r['case_id'],r['method']):r for r in retrieved['results']}
        generated={(r['case_id'],r['method']):r for r in read_records(out/'generation.jsonl').values()}
        rows=[]
        for case in dataset['cases']:
            for method in ('archived_flat_hybrid','parent_child_hybrid','rerank_only','precision_only','parent_semantic_v3'):
                retrieval_method={'precision_only':'parent_child_hybrid','rerank_only':'parent_semantic_v3'}.get(method,method)
                row={'case_id':case['id'],'method':method,'answerable':case['answerable'],'category':case['category'],
                    'generation':generated[(case['id'],method)],
                    'retrieval':copy.deepcopy(contexts[(case['id'],retrieval_method)])|{'method':method},
                    'judgment':{},'quality':{'answer_correct':None}}
                rows.append(row)
        return await audit(rows,HARD_DATASET,out,settings,write_report=False)
    out=BASE/'challenge'
    previous=json.loads((out/'report.json').read_text())
    return await audit(previous['results'],HARD_DATASET,out,settings)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['ablation','challenge','wording','challenge-pregrade'],required=True)
    asyncio.run(main_async(p.parse_args().phase))

if __name__=='__main__':main()
