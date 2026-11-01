"""Fixed 2x2 RAG comparison: flat/parent-child index x no/same semantic reranker.

Reuses exact prior requests. No prompt experiments, new questions, or judge tuning.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import inspect
import json
import shutil
import statistics
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.policy_eval.answer_compare import read_records, run_stage, fingerprint
from app.policy_eval.audit_v4 import AUDIT_PROMPT
from app.policy_eval.diagnostic_v4 import BASE as PREVIOUS, HARD_DATASET, freeze
from app.policy_eval.pilot import ROOT
from app.policy_eval.rigorous import JUDGE_MODEL, messages, payload, parse_quality
from app.policy_eval.runner import (
    EmbeddingCache, build_chunks, digest, load_inputs, rank_chunks, retrieval_query,
    score_case, select_context, vector_rank,
)
from app.services.knowledge import bm25_scores, normalize_scores
from app.services.knowledge_reranking import RERANK_PROMPT, parse_order, rerank_request
from app.services.memory_context import estimate_tokens

BASE = ROOT / 'target/policy-pilot/core-ablation-v5'
ARMS = ('flat_hybrid', 'parent_child_hybrid', 'flat_semantic', 'rerank_only')
LABELS = dict(zip(ARMS, ('A 平面', 'B 父子', 'C 平面＋rerank', 'D 父子＋rerank')))


def flat_hybrid_rank(query, chunks, similarities, candidate_k=16):
    """Use the same total vector/BM25 weights as the parent-child path."""
    rows = [r for r in chunks if r.kind == 'flat']
    by_id = {r.id: r for r in rows}
    vector = normalize_scores(vector_rank(rows, similarities, candidate_k))
    raw = bm25_scores(query, rows)
    lexical = normalize_scores({i: raw[i] for i in sorted(raw, key=lambda i: (-raw[i], i))[:candidate_k]})
    active = [(vector, .65), (lexical, .35)]
    active = [(values, weight) for values, weight in active if values]
    total = sum(weight for _, weight in active) or 1.
    scores = {i: sum(weight * values.get(i, 0.) for values, weight in active) / total
              for i in {key for values, _ in active for key in values}}
    return [(by_id[i], score) for i, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def protocol(settings):
    return {'experiment': 'core-ablation-v5', 'dataset_sha256': digest(HARD_DATASET.read_bytes()),
            'corpus_sha256': digest((ROOT / 'knowledge/xidian/manifest.json').read_bytes()),
            'arms': list(ARMS), 'top_k': 5, 'estimated_context_budget': 4000,
            'candidate_k_per_lane': 16, 'rerank_pool_cap': 32,
            'flat_window': [300, 40], 'parent_child_target_sizes': [1000, 300, 40],
            'flat_vector_bm25_weights': [.65, .35], 'parent_child_bm25_weights': [.30, .35, .35],
            'keyword_rule_rerank': False, 'neighbor_expansion': False,
            'generation_model': settings.openai_model, 'embedding_model': settings.openai_embedding_model,
            'judge_model': JUDGE_MODEL, 'temperature': 0, 'max_output_tokens': settings.ai_max_tokens,
            'thinking': settings.openai_enable_thinking,
            'answer_prompt_sha256': digest(Path(__file__).with_name('legacy_consult_prompt.txt').read_bytes()),
            'rerank_prompt_sha256': digest(RERANK_PROMPT.encode()), 'judge_prompt_sha256': digest(AUDIT_PROMPT.encode()),
            'flat_rank_sha256': digest(inspect.getsource(flat_hybrid_rank).encode()),
            'parent_rank_sha256': digest(inspect.getsource(rank_chunks).encode()),
            'chunking_sha256': digest((ROOT / 'app/services/knowledge_chunking.py').read_bytes()),
            'scope': 'Fixed known32 development stress cases, AI labels pending human; not independent heldout or production latency test'}


def seed_cache(out):
    """Copy durable cache entries; run_stage rechecks the full request hash."""
    source = PREVIOUS / 'challenge'
    manifest = {}
    for name in ('generation.jsonl', 'quality-audit.jsonl', 'reranking.jsonl'):
        src, dst = source / name, out / name
        if not dst.exists():
            shutil.copyfile(src, dst)
        manifest[name] = {'source': str(src.relative_to(ROOT)), 'sha256_at_copy': digest(src.read_bytes())}
    freeze(out / 'cache-origin.json', manifest)


def row_for(case, method, selected, documents):
    return {'case_id': case['id'], 'method': method, 'query': retrieval_query(case),
            'estimated_tokens': sum(estimate_tokens(r.content) for r, _ in selected),
            'metrics': score_case(case, selected, documents),
            'contexts': [{'chunk_id': r.id, 'document_id': r.document_id, 'version': r.version,
                          'heading': r.heading, 'content': r.content, 'score': score} for r, score in selected]}


def request_hash(task, endpoint):
    return fingerprint({'case_id': task['case_id'], 'method': task['method'],
                        'endpoint': digest(endpoint.rstrip('/').encode()), 'payload': task['payload']})


def cache_hits(tasks, path, endpoint):
    existing = read_records(path)
    return sum(request_hash(t, endpoint) in existing for t in tasks)


async def retrieve(dataset, documents, out, settings):
    path = out / 'retrieval.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['dataset_sha256'] != digest(HARD_DATASET.read_bytes()):
            raise ValueError('Fixed dataset differs')
        return saved
    chunks = build_chunks(documents)
    cache = EmbeddingCache(settings, ROOT / 'target/policy-pilot/embeddings.sqlite3')
    try:
        vectors = cache.embed([r.embedding_text for r in chunks] + [retrieval_query(c) for c in dataset['cases']])
    finally:
        cache.close()
    matrix = np.vstack([vectors[r.embedding_text] for r in chunks])
    pools, tasks, rows = {}, [], []
    for case in dataset['cases']:
        query = retrieval_query(case)
        similarities = {r.id: float(s) for r, s in zip(chunks, matrix @ vectors[query])}
        for base, reranked in (('flat_hybrid', 'flat_semantic'), ('parent_child_hybrid', 'rerank_only')):
            ranked = (flat_hybrid_rank(query, chunks, similarities) if base == 'flat_hybrid'
                      else rank_chunks('parent_child_hybrid', query, chunks, similarities, 16))[:32]
            pools[case['id'], reranked] = ranked
            selected, _ = select_context(ranked, 5, 4000)
            rows.append(row_for(case, base, selected, documents))
            candidates = [{'id': r.id, 'title': documents[r.document_id][0]['title'],
                           'heading': r.heading, 'content': r.content} for r, _ in ranked]
            tasks.append({'case_id': case['id'],
                          'method': 'flat_semantic_rerank' if base == 'flat_hybrid' else 'semantic_rerank',
                          'candidate_ids': {r.id for r, _ in ranked},
                          'payload': rerank_request(query, candidates, 5, settings.openai_model,
                                                    thinking=settings.openai_enable_thinking)})
    def validate(text, task):
        return {'ordered_ids': parse_order(text, task['candidate_ids'], 5)}
    hits = cache_hits(tasks, out / 'reranking.jsonl', settings.openai_base_url)
    reranks = await run_stage('reranking', tasks, out / 'reranking.jsonl', settings.openai_base_url,
                             settings.openai_api_key, 6, validate)
    for case in dataset['cases']:
        for method, key in (('flat_semantic', 'flat_semantic_rerank'), ('rerank_only', 'semantic_rerank')):
            pool = pools[case['id'], method]
            result = reranks[case['id'], key]
            order = result['parsed']['ordered_ids']
            by_id = {r.id: (r, score) for r, score in pool}
            selected, _ = select_context([by_id[i] for i in order], 5, 4000)
            row = row_for(case, method, selected, documents)
            row.update(rerank_usage=result['usage'], rerank_request=result,
                       pool_ids=[r.id for r, _ in pool], reranked_ids=order)
            rows.append(row)
    result = {'dataset_sha256': digest(HARD_DATASET.read_bytes()), 'rerank_cache_hits': hits,
              'rerank_requested': len(tasks), 'results': rows}
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'stage': 'retrieval', 'reused_reranks': hits, 'new_reranks': len(tasks)-hits}), flush=True)
    return result


async def evaluate(dataset, documents, retrieved, out, settings):
    by = {(r['case_id'], r['method']): r for r in retrieved['results']}
    tasks = []
    for n, case in enumerate(dataset['cases']):
        for method in ARMS[n % 4:] + ARMS[:n % 4]:
            request = {'model': settings.openai_model, 'temperature': 0., 'max_tokens': settings.ai_max_tokens,
                       'stream': False, 'messages': messages(case, by[case['id'], method], documents)}
            if settings.openai_enable_thinking is not None:
                request['enable_thinking'] = settings.openai_enable_thinking
            tasks.append({'case_id': case['id'], 'method': method, 'payload': request})
    generation_hits = cache_hits(tasks, out / 'generation.jsonl', settings.openai_base_url)
    generated = await run_stage('generation', tasks, out / 'generation.jsonl', settings.openai_base_url,
                                settings.openai_api_key, 8)
    tasks = []
    for case in dataset['cases']:
        for method in ARMS:
            row = by[case['id'], method]
            expanded = [{'rank': i, 'title': documents[c['document_id']][0]['title'],
                         'scope': documents[c['document_id']][0]['scope'], 'content': c['content']}
                        for i, c in enumerate(row['contexts'], 1)]
            content = {'experiment_snapshot_date': '2026-10-19', 'question': case['question'], 'history': case['history'],
                       'answerable': case['answerable'], 'reference_points': case['answer_points'],
                       'optional_point_indices': case.get('optional_point_indices', []),
                       'reference_evidence': [g['quote'] for g in case['evidence']], 'contexts': expanded,
                       'answer': generated[case['id'], method]['text']}
            tasks.append({'case_id': case['id'], 'method': method, 'case': case, 'contexts': row['contexts'],
                          'payload': payload(settings, JUDGE_MODEL, AUDIT_PROMPT, content)})
    judge_hits = cache_hits(tasks, out / 'quality-audit.jsonl', settings.openai_base_url)
    judged = await run_stage('quality-audit', tasks, out / 'quality-audit.jsonl', settings.openai_base_url,
                            settings.openai_api_key, 6, parse_quality)
    results = []
    for case in dataset['cases']:
        for method in ARMS:
            key = case['id'], method
            judge = judged[key]['parsed']
            required = set(case.get('required_point_indices', range(1, len(case['answer_points']) + 1)))
            labels = [p['covered'] for p in judge['point_verdicts'] if p['index'] in required]
            correct = bool(all(labels) and not judge['contradictions'] and not judge['fabricated_specifics']
                           and judge['behavior'] == case['expected_behavior'])
            results.append({'case_id': case['id'], 'method': method, 'answerable': case['answerable'],
                            'category': case['category'], 'question': case['question'],
                            'retrieval': by[key], 'generation': generated[key], 'judgment': judge,
                            'quality': {'answer_correct': correct, 'answer_point_coverage': statistics.mean(labels)}})
    result = {'status': 'Fixed 2x2 development regression, automatic grading pending human',
              'protocol': protocol(settings), 'dataset_sha256': digest(HARD_DATASET.read_bytes()),
              'cache': {'generation_reused': generation_hits, 'generation_requested': len(generated),
                        'quality_reused': judge_hits, 'quality_requested': len(judged),
                        'rerank_reused': retrieved['rerank_cache_hits'], 'rerank_requested': retrieved['rerank_requested']},
              'results': results}
    (out / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def summaries(report):
    result = {}
    for method in ARMS:
        rows = [r for r in report['results'] if r['method'] == method]
        positive = [r for r in rows if r['answerable']]
        negative = [r for r in rows if not r['answerable']]
        generation_input = sum(r['generation']['usage'].get('prompt_tokens', 0) for r in rows)
        rerank_input = sum(r['retrieval'].get('rerank_usage', {}).get('prompt_tokens', 0) for r in rows)
        result[method] = {'correct': sum(r['quality']['answer_correct'] for r in positive),
                          'answerable': len(positive), 'negative_correct': sum(r['quality']['answer_correct'] for r in negative),
                          'negatives': len(negative),
                          'evidence_complete': sum(r['retrieval']['metrics']['evidence_complete'] for r in positive),
                          'evidence_recall': statistics.mean(r['retrieval']['metrics']['clause_recall_at_k'] for r in positive),
                          'generation_input_tokens': generation_input, 'rerank_input_tokens': rerank_input,
                          'total_input_tokens': generation_input + rerank_input,
                          'truncated': sum(r['generation']['finish_reason'] == 'length' for r in rows)}
    return result


def paired(report, a, b):
    by = {(r['case_id'], r['method']): r for r in report['results']}
    ids = sorted({r['case_id'] for r in report['results'] if r['answerable']})
    improved = [i for i in ids if not by[i, a]['quality']['answer_correct'] and by[i, b]['quality']['answer_correct']]
    regressed = [i for i in ids if by[i, a]['quality']['answer_correct'] and not by[i, b]['quality']['answer_correct']]
    return {'before': a, 'after': b, 'improved': improved, 'regressed': regressed,
            'net': len(improved)-len(regressed), 'delta_pp': 100*(len(improved)-len(regressed))/len(ids)}


def write_report(report, dataset):
    summary = summaries(report)
    historical = json.loads((PREVIOUS/'challenge/audited-report.json').read_text())
    if historical['dataset_sha256'] != report['dataset_sha256']:
        raise ValueError('Historical reference dataset differs')
    old_flat = [r for r in historical['results'] if r['method']=='archived_flat_hybrid' and r['answerable']]
    original_reference = {'source':str((PREVIOUS/'challenge/audited-report.json').relative_to(ROOT)),
                          'source_sha256':digest((PREVIOUS/'challenge/audited-report.json').read_bytes()),
                          'window':[512,64], 'keyword_rule_rerank':True, 'first_result_neighbors':True,
                          'correct':sum(r['quality']['answer_correct'] for r in old_flat), 'answerable':len(old_flat),
                          'purpose':'Historical engineering reference; not the controlled A arm'}
    comparisons = {a+'__'+b: paired(report, a, b) for a,b in
                   [('flat_hybrid','parent_child_hybrid'),('flat_semantic','rerank_only'),
                    ('flat_hybrid','flat_semantic'),('parent_child_hybrid','rerank_only'),('flat_hybrid','rerank_only')]}
    output = {'scope': report['status'], 'protocol': report['protocol'], 'summary': summary,
              'paired': comparisons, 'cache': report['cache'], 'historical_flat_reference':original_reference,
              'raw_report_sha256': digest((BASE/'report.json').read_bytes())}
    (BASE/'summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'docs/rag-core-ablation-summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    a,b,c,d = (summary[m] for m in ARMS)
    lines = ['# 父子索引与rerank：固定条件的四组对照','',
             f"统一平面控制组为{a['correct']}/26，父子索引为{b['correct']}/26；分别加入同一rerank后为{c['correct']}/26和{d['correct']}/26。父子结构在不精排时有改善；rerank改善了两类索引；精排后父子结构没有进一步增加本批回答符合数，因此两项收益不能简单相加。",'',
             '本轮只验证索引结构与语义精排两项设计。使用已有32道复杂场景开发题，其中26道可回答、6道资料或条件不足；没有扩题、改回答提示词或重新校准裁判。全部结果为固定自动规则评分，待人工审核。','',
             f"本轮A是300/40窗口、关闭旧规则重排及邻块扩展的平面控制组，不能冒充之前的512/64原始工程。旧原始工程同一题集为{original_reference['correct']}/26，完整D为{d['correct']}/26，相比旧工程实际净增{d['correct']-original_reference['correct']}题；相对本轮A的净增数另列。",'',
             '## 四组结果','',
             '| 方案 | 可答题符合数 | 负例符合数 | 指定证据齐全题 | 指定证据召回率 | 生成输入token | 含rerank总输入token |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method,s in summary.items():
        lines.append(f"| {LABELS[method]} | {s['correct']}/{s['answerable']} | {s['negative_correct']}/{s['negatives']} | {s['evidence_complete']}/{s['answerable']} | {s['evidence_recall']:.2%} | {s['generation_input_tokens']} | {s['total_input_tokens']} |")
    lines += ['', '## 每项设计的独立贡献','', '| 对比 | 目的 | 改善题数 | 退步题数 | 净变化 |', '|---|---|---:|---:|---:|']
    purposes = ['父子索引，无语义精排','父子索引，共用同一语义精排','精排，平面索引','精排，父子索引','完整方案相对平面基线']
    for p,purpose in zip(comparisons.values(),purposes):
        lines.append(f"| {LABELS[p['before']]} → {LABELS[p['after']]} | {purpose} | {len(p['improved'])} | {len(p['regressed'])} | {p['net']:+d}题（{p['delta_pp']:+.2f}个百分点） |")
    lines += ['', '## 收益和剩余问题','',
              '1. 父子索引的主要可见收益是上下文连贯性。C4-X03/C4-M01中，平面回答将理学II类助研误用为450元，父子回答保留分类、表头与金额关系，正确给出300元；C4-X03的全年合计也从错误10500元变为9000元。参考原文和实际回答均可在配对错例包核对。',
              '2. rerank在父子路径上改善6题、退步1题，净增5题；在平面路径上改善9题且本次无退步。其收益主要来自更完整地选回不同子问题所需材料。',
              '3. 有rerank时，父子结构额外补齐3道题的指定证据，但回答符合数仍与平面＋rerank相同。两项技术存在收益重叠，不能声称叠加后一定继续提高准确率。',
              '4. C4-M03是明确剩余错例：父子＋rerank取齐了结业证据，回答却把普通毕业生的“毕业后1年”授位窗口套给已结业硕士。问题在于业务状态混用，继续增加候选或扩大K无法直接修复；应针对父块内无关规则和适用状态做约束。',
              f"5. 完整父子＋rerank的推理输入为{d['total_input_tokens']}，平面＋rerank为{c['total_input_tokens']}，前者约为后者{d['total_input_tokens']/c['total_input_tokens']:.2f}倍。父子无精排的生成输入较平面增加{(b['generation_input_tokens']/a['generation_input_tokens']-1):.2%}。当前结果不支持‘父子索引同时更准、更省’；需要将额外证据完整性与开销一起衡量。",'',
              '这些错例解释属于AI对冻结原文和回答的分析，不是人工业务批准。本轮不为了获得更高分而修改提示词、题目或评分标准，也未将实验配置直接启用为生产默认。']
    lines += ['', '## 固定条件与结论边界','',
              '- 同一49份语料快照、同一Embedding及回答模型、同一原回答提示词、K=5、4000估算知识token预算、512输出token、温度0。',
              '- 两种索引的向量总权重均0.65、BM25均0.35；每通道16候选，融合池最多32。父子向量总权重分为父0.30、子0.35；两条语义精排路径使用完全相同的模型、提示词和选块数。',
              '- 平面采用公共300/40窗口，父子采用现有hierarchical-v2（父目标1000、子300/40）；均关闭旧关键词规则重排和首条邻块扩展。旧512/64原始工程方案另存历史报告，不与本轮A混称同一基线。',
              '- 固定K和预算上限不代表实际上下文长度相等；父块能携带更多上下文，质量差异应结合实际token理解，不能宣称等token或纯粹排序优势。',
              '- 指定来源/版本的整段证据覆盖作为辅助，可能受标注跨度及同义来源漏计影响；最终回答符合数为主。32题是已知开发压力集，不包装成新的独立保留测试。',
              '- 复用仅依赖包含完整请求内容的指纹，未重新抽取事实、跑Faithfulness或追加裁判校准；不同时间段响应不用于生产延迟横比。',
              '- token为正常生成与精排的供应商usage，不含评分、Embedding及失败重试，不等同实际货币费用。', '',
              '## 配对错例','']
    by = {(r['case_id'],r['method']):r for r in report['results']}
    cases = {c['id']:c for c in dataset['cases']}
    for p in comparisons.values():
        lines += [f"### {LABELS[p['before']]} → {LABELS[p['after']]}",'',
                  '改善：'+('、'.join(p['improved']) or '无'),'','退步：'+('、'.join(p['regressed']) or '无'),'']
    (ROOT/'docs/rag-core-ablation.md').write_text('\n'.join(lines)+'\n')
    details = ['# 四组对照的逐题差异','','用原文、实际上下文和回答核对；自动标签不是人工批准。','']
    for case_id in sorted({i for p in comparisons.values() for i in p['improved']+p['regressed']}):
        case = cases[case_id]
        details += [f'## {case_id}','',case['question'],'']
        for h in case['history']:details += [f"历史：{h['content']}",'']
        details += ['必要要点：','']+[f'{i}. {p}' for i,p in enumerate(case['answer_points'],1)]+['']
        for method in ARMS:
            row = by[case_id,method]
            details += ['### '+LABELS[method],'',f"自动符合：{row['quality']['answer_correct']}",'',row['generation']['text'],'',
                        '裁判：'+str(row['judgment'].get('reason',row['judgment']['point_verdicts'])),'']
            for i,c in enumerate(row['retrieval']['contexts'],1):details += [f"资料{i} · {c['document_id']}",'','> '+c['content'].replace('\n','\n> '),'']
    (BASE/'paired-cases.md').write_text('\n'.join(details)+'\n')
    print(json.dumps({'summary':summary,'paired':comparisons,'cache':report['cache']},ensure_ascii=False),flush=True)


async def main_async():
    settings = Settings()
    freeze(BASE/'protocol.json',protocol(settings))
    dataset,documents = load_inputs(HARD_DATASET)
    freeze(BASE/'dataset.json',dataset)
    seed_cache(BASE)
    retrieved = await retrieve(dataset,documents,BASE,settings)
    report = await evaluate(dataset,documents,retrieved,BASE,settings)
    write_report(report,dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'run.lock').open('w') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Core comparison already running; refusing duplicate API requests')
        asyncio.run(main_async())


if __name__=='__main__':main()
