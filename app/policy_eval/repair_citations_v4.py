"""Resolve failed literal citations through source-only paragraph IDs, never inferred verdicts."""
from __future__ import annotations
import asyncio
import json
import re
import httpx
from app.core.config import Settings
from app.policy_eval.diagnostic_v4 import BASE,HARD_DATASET,FAITH_PROMPT,JUDGE_MODEL,payload
from app.policy_eval.answer_compare import (read_records,parse_faithfulness,complete,fingerprint)
from app.policy_eval.runner import load_inputs,digest
from app.evaluation.judge import extract_json_object

CITATION_PROMPT = FAITH_PROMPT + '''
本次为引文格式修复，事实判据不变。除contexts外给出citations，内容全部逐字来自对应资料。
只返回单项：{"verdicts":[{"index":1,"relation":"entailed|contradicted|insufficient","citation_id":"编号或空串","reason":"简短依据"}]}。
用citation_id替代手写context_rank/support_quote。由程序按编号填回真实原文；不要再手写、改写或省略拼接引文。
若整项主张需要多个条件，须核对所有条件都在实际上下文支持，部分支持仍是insufficient；不是看到相近引文就选entailed。
资料不足时用insufficient和空编号。不能为了完成格式修复改变事实判据，也不能用外部知识或参考答案。
'''


def citations_for(contexts):
    options=[]
    for context in contexts:
        rank=context['rank']
        # Keep exact whitespace and punctuation. A paragraph may contain a table
        # or several clauses; no paraphrase or gold-source matching is used.
        paragraphs=[p for p in re.split(r'\n\s*\n',context['content']) if p.strip()]
        for n,text in enumerate(paragraphs,1):
            options.append({'citation_id':f'{rank}.{n}','context_rank':rank,'quote':text})
    return options


def map_citation(text,task,options):
    raw=extract_json_object(text)
    if len(raw.get('verdicts',[]))!=1 or raw['verdicts'][0]['index']!=1:
        raise ValueError('Expected one isolated verdict')
    verdict=raw['verdicts'][0]
    index={c['citation_id']:c for c in options}
    citation_id=verdict.get('citation_id','')
    if verdict['relation']=='insufficient' and not citation_id:
        mapped={**verdict,'context_rank':0,'support_quote':''}
    else:
        if citation_id not in index:raise ValueError('Citation ID not in actual source paragraphs')
        citation=index[citation_id]
        mapped={**verdict,'context_rank':citation['context_rank'],'support_quote':citation['quote']}
    # Preserves the returned relation; does not infer or flip an entailment.
    return parse_faithfulness(json.dumps({'verdicts':[mapped]},ensure_ascii=False),task)


async def main_async():
    settings=Settings();out=BASE/'challenge'
    errors=out/'faithfulness-errors.json'
    if not errors.exists():
        print('No unresolved citation records');return
    dataset,docs=load_inputs(HARD_DATASET)
    retrieved=json.loads((out/'retrieval.json').read_text())
    if retrieved['dataset_sha256']!=digest(HARD_DATASET.read_bytes()):raise ValueError('Dataset changed')
    by={(r['case_id'],r['method']):r for r in retrieved['results']}
    extracted={(r['case_id'],r['method']):r for r in read_records(out/'reverse-questions.jsonl').values()}
    rejected=[json.loads(l) for l in (out/'faithfulness-rejected.jsonl').read_text().splitlines()]
    async with httpx.AsyncClient(timeout=120) as client:
        for failure in json.loads(errors.read_text()):
            case_id,method=failure['case_id'],failure['method']
            retrieval_method={'precision_only':'parent_child_hybrid','rerank_only':'parent_semantic_v3'}.get(method,method)
            contexts=[{'rank':i,'title':docs[c['document_id']][0]['title'],'scope':docs[c['document_id']][0]['scope'],'content':c['content']} for i,c in enumerate(by[case_id,retrieval_method]['contexts'],1)]
            claims=extracted[case_id,method]['parsed']['claims'];options=citations_for(contexts)
            task={'claims':claims,'contexts':contexts}
            # Valid old items remain untouched. Missing/invalid ones alone are
            # rejudged in isolation; original rejected outputs remain in place.
            prior=next((extract_json_object(r['text']) for r in reversed(rejected)
                if r['case_id']==case_id and r['method']==method and r['text'].strip().endswith('}')),None)
            if prior is None:prior={'verdicts':[]}
            old={v['index']:v for v in prior.get('verdicts',[])}
            verdicts=[]
            for n,claim in enumerate(claims,1):
                single={'claims':[claim],'contexts':contexts}
                try:
                    saved=parse_faithfulness(json.dumps({'verdicts':[{**old[n],'index':1}]},ensure_ascii=False),single)['verdicts'][0]
                except (KeyError,ValueError,TypeError):
                    request=payload(settings,JUDGE_MODEL,CITATION_PROMPT,{'claims':[{'index':1,**claim}],'contexts':contexts,'citations':options})
                    for attempt in range(3):
                        result=await complete(client,request,settings.openai_base_url,settings.openai_api_key)
                        with (out/'citation-id-repair-calls.jsonl').open('a') as stream:
                            stream.write(json.dumps({'case_id':case_id,'method':method,'claim_index':n,
                                'request_payload_sha256':fingerprint(request),**result},ensure_ascii=False)+'\n')
                        try:
                            saved=map_citation(result['text'],single,options)['verdicts'][0]
                            break
                        except (KeyError,ValueError,TypeError) as exc:
                            if attempt==2:raise
                            request={**request,'messages':[*request['messages'],{'role':'user','content':
                                '格式未通过：'+str(exc)[:160]+'。请只选输入中的citation_id，不要改变事实判断来通过格式检查。'}]}
                verdicts.append({**saved,'index':n})
            raw={'verdicts':verdicts};parsed=parse_faithfulness(json.dumps(raw,ensure_ascii=False),task)
            original_request=payload(settings,JUDGE_MODEL,FAITH_PROMPT,{'claims':[{'index':i,**c} for i,c in enumerate(claims,1)],'contexts':contexts})
            sha=fingerprint({'case_id':case_id,'method':method,'endpoint':digest(settings.openai_base_url.rstrip('/').encode()),'payload':original_request})
            record={'stage':'faithfulness','case_id':case_id,'method':method,'request_sha256':sha,
                'text':json.dumps(raw,ensure_ascii=False),'parsed':parsed,'usage':{},'http_attempts':0,
                'finish_reason':'assembled_item_citation_id_repairs','source_paragraph_id_format_repair':True,
                'assembled_from_rejected_response_and_logged_item_repairs':True}
            with (out/'faithfulness.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            print(json.dumps({'case_id':case_id,'method':method,'claims':len(claims),'status':'validated'},ensure_ascii=False),flush=True)

if __name__=='__main__':asyncio.run(main_async())
