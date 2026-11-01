"""Paired context/wording controls and a separate 32-case offline stress test.

Independent model family for judging; AI annotations remain pending human review.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import copy
import statistics
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.policy_eval.answer_compare import (
    FAITH_PROMPT, REVERSE_PROMPT, generation_messages, parse_faithfulness,
    parse_reverse, quality_metrics, run_stage, summarize_rows,
)
from app.policy_eval.pilot import ROOT
from app.policy_eval.runner import (
    EmbeddingCache, build_chunks, digest, load_inputs, rank_chunks, retrieval_query,
    score_case, select_context,
)

from app.policy_eval.rigorous import (BASE as V3_BASE, HOLDOUT, JUDGE_MODEL, QUALITY_PROMPT,
    payload, parse_quality, messages, now, retrieval)
from app.services.memory_context import estimate_tokens

BASE = ROOT / 'target/policy-pilot/diagnostic-v4'
HARD_DATASET = Path(__file__).with_name('challenge_v4.json')
LATENT_PROMPT = """你是中文校园政策问答助手。本实验不提供检索资料，允许依据你已掌握的知识作答。
请直接回答最后的用户问题；不知道学校特定规定或所问版本时明确说明，不要编造数字、个人记录或已完成操作。
题目所指资料截止2026-10-19，不能把后续未知政策当作事实。"""
HARD_ARMS = ('archived_flat_hybrid', 'parent_child_hybrid', 'rerank_only', 'precision_only', 'parent_semantic_v3')


def diagnostic_messages(case, row, documents):
    adapted = dict(row)
    if row['method'] == 'parametric_probe':
        result = generation_messages(case, row, documents)
        result[0]['content'] = LATENT_PROMPT
        return result
    adapted['method'] = 'parent_semantic_v3' if row['method'] in {'precision_only', 'parent_semantic_v3'} else 'parent_child_hybrid'
    return messages(case, adapted, documents)


def freeze(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text() != encoded:
        raise ValueError(f'Refusing changed frozen input: {path.name}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def protocol(settings):
    return {'version': 'diagnostic-v4', 'generation_model': settings.openai_model,
            'judge_model': JUDGE_MODEL, 'temperature': 0, 'top_k': 5, 'budget': 4000,
            'max_output_tokens': settings.ai_max_tokens, 'seed': 20261031,
            'heldout_sha256': digest(HOLDOUT.read_bytes()),
            'legacy_prompt_sha256': digest(Path(__file__).with_name('legacy_consult_prompt.txt').read_bytes()),
            'latent_prompt_sha256': digest(LATENT_PROMPT.encode()),
            'quality_prompt_sha256': digest(QUALITY_PROMPT.encode()),
            'reference_policy': 'Unchanged v3 labels; disputes retained; AI pending human',
            'attribution': 'Paired conditional ablations, not an additive causal decomposition',
            'challenge': 'New AI-authored stress test, same corpus; not independent human gold; no tuning after outputs'}


def ablation_rows(dataset, documents, previous):
    rng = random.Random(20261031)
    parents = [r for r in build_chunks(documents) if r.kind == 'parent']
    full = {r['case_id']: r for r in previous['results'] if r['method'] == 'parent_child_hybrid'}
    rows = []
    for case in dataset['cases']:
        base = full[case['id']]['retrieval']
        blocked = {g['document_id'] for g in case['evidence']} | set(case.get('checked_source_ids', []))
        blocked |= {c['document_id'] for c in base['contexts']}
        pool = [p for p in parents if p.document_id not in blocked]
        rng.shuffle(pool)
        # Match returned block count and approximately match total length. Gold IDs
        # are used only to construct this artificial diagnostic control.
        wanted = base['estimated_tokens'] / max(1, len(base['contexts']))
        chosen = []
        for _ in base['contexts']:
            item = min(pool, key=lambda p: abs(estimate_tokens(p.content) - wanted))
            pool.remove(item)
            chosen.append((item, 0.))
        random_selected, random_used = select_context(chosen, 5, 4000)
        gold = [({'chunk_id': -i, 'document_id': g['document_id'], 'version': g['version'],
                  'heading': g['section'], 'content': g['quote'], 'score': 1.})
                for i, g in enumerate(case['evidence'], 1)]
        random_context = [{'chunk_id': p.id, 'document_id': p.document_id, 'version': p.version,
                           'heading': p.heading, 'content': p.content, 'score': s} for p, s in random_selected]
        controls = [('no_context_strict', [], 0), ('random_context', random_context, random_used)]
        if case['answerable']:
            controls += [('oracle_evidence', gold, sum(estimate_tokens(g['content']) for g in gold)),
                         ('parametric_probe', [], 0)]
        for method, contexts, used in controls:
            row = copy.deepcopy(base)
            row.update(method=method, contexts=contexts, estimated_tokens=used)
            row['metrics'] = None  # This control is scored for answers, not inherited retrieval labels.
            row['diagnostic_only'] = True
            rows.append(row)
    return rows


async def evaluate_quality(dataset_path, out, rows, settings):
    out.mkdir(parents=True, exist_ok=True)
    dataset, documents = load_inputs(dataset_path)
    cases = {c['id']: c for c in dataset['cases']}
    tasks = []
    for row in rows:
        case = cases[row['case_id']]
        request = {'model': settings.openai_model, 'temperature': 0., 'max_tokens': settings.ai_max_tokens,
                   'stream': False, 'messages': diagnostic_messages(case, row, documents)}
        if settings.openai_enable_thinking is not None:
            request['enable_thinking'] = settings.openai_enable_thinking
        tasks.append({'case_id': case['id'], 'method': row['method'], 'payload': request})
    generated = await run_stage('generation', tasks, out / 'generation.jsonl', settings.openai_base_url, settings.openai_api_key, 8)
    tasks = []
    for row in rows:
        case = cases[row['case_id']]
        contexts = [{'rank': i, 'title': documents[c['document_id']][0]['title'],
                     'scope': documents[c['document_id']][0]['scope'], 'content': c['content']}
                    for i, c in enumerate(row['contexts'], 1)]
        content = {'question': case['question'], 'history': case['history'], 'answerable': case['answerable'],
                   'reference_points': case['answer_points'], 'optional_point_indices': case.get('optional_point_indices', []),
                   'reference_evidence': [g['quote'] for g in case['evidence']], 'contexts': contexts,
                   'answer': generated[(case['id'], row['method'])]['text']}
        tasks.append({'case_id': case['id'], 'method': row['method'], 'case': case, 'contexts': contexts,
                      'payload': payload(settings, JUDGE_MODEL, QUALITY_PROMPT, content)})
    judged = await run_stage('quality', tasks, out / 'quality.jsonl', settings.openai_base_url, settings.openai_api_key, 6, parse_quality)
    result = []
    for row in rows:
        case = cases[row['case_id']]
        key = (case['id'], row['method'])
        judge = judged[key]['parsed']
        required = set(case.get('required_point_indices', range(1, len(case['answer_points']) + 1)))
        labels = [p['covered'] for p in judge['point_verdicts'] if p['index'] in required]
        safe = not judge['contradictions'] and not judge['fabricated_specifics']
        correct = bool(all(labels) and safe and judge['behavior'] == case['expected_behavior'])
        result.append({'case_id': case['id'], 'method': row['method'], 'question': case['question'],
                       'answerable': case['answerable'], 'category': case['category'],
                       'retrieval': row, 'generation': generated[key], 'judgment': judge,
                       'quality': {'answer_correct': correct, 'answer_point_coverage': statistics.mean(labels),
                                   'behavior': judge['behavior'], 'safe': safe}})
    return result


def quality_summary(rows):
    result = {}
    for method in sorted({r['method'] for r in rows}):
        selected = [r for r in rows if r['method'] == method]
        pos = [r for r in selected if r['answerable']]
        neg = [r for r in selected if not r['answerable']]
        result[method] = {'cases': len(selected), 'answerable_cases': len(pos),
                          'correct_answerable': sum(r['quality']['answer_correct'] for r in pos),
                          'correct_negative': sum(r['quality']['answer_correct'] for r in neg),
                          'point_coverage_answerable': statistics.mean(r['quality']['answer_point_coverage'] for r in pos) if pos else None,
                          'generation_truncated': sum(r['generation']['finish_reason'] == 'length' for r in selected)}
    return result


async def ablation(settings):
    out = BASE / 'ablation'
    out.mkdir(parents=True, exist_ok=True)
    dataset, documents = load_inputs(HOLDOUT)
    previous = json.loads((V3_BASE / 'holdout/report.json').read_text())
    if previous['dataset_sha256'] != digest(HOLDOUT.read_bytes()):
        raise ValueError('V3 baseline dataset mismatch')
    rows = ablation_rows(dataset, documents, previous)
    freeze(out / 'inputs.json', rows)
    result = await evaluate_quality(HOLDOUT, out, rows, settings)
    baseline = [r for r in previous['results'] if r['method'] == 'parent_child_hybrid']
    result += baseline
    report = {'version': 'diagnostic-v4', 'status': 'AI judged pending human', 'cases': 60,
              'dataset_sha256': digest(HOLDOUT.read_bytes()), 'summary': quality_summary(result), 'results': result}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)
    return report


def question_variants(dataset):
    # Remove offered candidate answers, not decisive user facts. These are an
    # exploratory wording check on known cases, not a new independent test set.
    rewrites = {
        'H-M05': '更正，不是补学位证明书，我是在申请硕士学位。学校审查是否受理有多长时间，从什么时候起算，决定后会通知我吗？',
        'H-M03': '不是国家助学金，改问优秀博士学位论文基金：我在资助期间已完成答辩，这笔资助具体什么时候停止？',
        'H-X02': '研究生教学事故申诉期限在收录办法的正文与附件告知书中分别怎么写？如果两处说法不一致，我能自行当成同一个期限吗？',
        'H-M08': '补充一下，我的基层就业需要二次定岗。这类申请和普通申请，学校集中报送材料最迟分别是什么时候？',
        'H-D10': '我退役后回学校复学，想申请学费减免。申请表该用哪一份、提交几份，退役证明材料要交什么？',
        'H-X10': '按收录版本的服兵役教育资助办法，学费和用于学费的国家助学贷款金额不相同时如何确定补偿代偿额？研究生年度额度有限制吗，超出的部分如何处理？',
        'H-M01': '更正，我原来的课程成绩已经合格，但现在自愿选择重修。最后计算课程总成绩和总学分时，应使用哪次成绩？',
        'H-C11': '我休学后回校，想重新入住学生公寓。申请后学校依据什么条件决定是否给我安排床位？',
    }
    result = copy.deepcopy(dataset)
    result['name'] = 'v4-paired-wording-probe-8'
    result['cases'] = []
    for case in dataset['cases']:
        if case['id'] in rewrites:
            edited = copy.deepcopy(case)
            edited['original_question'] = case['question']
            edited['question'] = rewrites[case['id']]
            edited['wording_probe'] = 'Candidate answers/numerical hints removed; decisive facts preserved; AI pending human equivalence check'
            result['cases'].append(edited)
    return result


async def wording(settings):
    out = BASE / 'wording'
    out.mkdir(parents=True, exist_ok=True)
    dataset, _ = load_inputs(HOLDOUT)
    variants = question_variants(dataset)
    path = out / 'dataset.json'
    freeze(path, variants)
    previous = json.loads((BASE / 'ablation/report.json').read_text())
    keys = {(r['case_id'], r['method']): r for r in previous['results']}
    rows = [copy.deepcopy(keys[(c['id'], m)]['retrieval'])
            for c in variants['cases'] for m in ('parent_child_hybrid', 'parametric_probe')]
    # Re-run retrieval for rewritten questions: reusing old contexts would hide
    # the effect of lexical cues on the retrieval stage.
    _, docs = load_inputs(path)
    chunks = build_chunks(docs)
    cache = EmbeddingCache(settings, ROOT / 'target/policy-pilot/embeddings.sqlite3')
    try:
        vectors = cache.embed([r.embedding_text for r in chunks] + [retrieval_query(c) for c in variants['cases']])
    finally:
        cache.close()
    matrix = np.vstack([vectors[r.embedding_text] for r in chunks])
    for case in variants['cases']:
        sims = {r.id: float(s) for r, s in zip(chunks, matrix @ vectors[retrieval_query(case)])}
        selected, used = select_context(rank_chunks('parent_child_hybrid', retrieval_query(case), chunks, sims, 16), 5, 4000)
        row = next(r for r in rows if r['case_id'] == case['id'] and r['method'] == 'parent_child_hybrid')
        row.update(query=retrieval_query(case), estimated_tokens=used, metrics=score_case(case, selected, docs),
                   contexts=[{'chunk_id': r.id, 'document_id': r.document_id, 'version': r.version,
                              'heading': r.heading, 'content': r.content, 'score': s} for r, s in selected])
    freeze(out / 'inputs.json', rows)
    result = await evaluate_quality(path, out, rows, settings)
    original = [keys[(c['id'], m)] for c in variants['cases'] for m in ('parent_child_hybrid', 'parametric_probe')]
    report = {'status': 'Exploratory paired wording probe; pending human semantic equivalence review',
              'summary': quality_summary(result), 'original_summary': quality_summary(original),
              'results': result, 'original_results': original}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'original': report['original_summary'], 'rewritten': report['summary']}, ensure_ascii=False), flush=True)
    return report


async def main_async(args):
    settings = Settings()
    freeze(BASE / 'protocol.json', protocol(settings))
    if args.phase == 'ablation':
        return await ablation(settings)
    if args.phase == 'wording':
        return await wording(settings)
    if args.phase == 'challenge':
        dataset, _ = load_inputs(HARD_DATASET)
        freeze(BASE / 'challenge-input.json', dataset)
        out = BASE / 'challenge'
        out.mkdir(parents=True, exist_ok=True)
        retrieved = await retrieval(HARD_DATASET, out, settings)
        rows = list(retrieved['results'])
        for row in list(rows):
            if row['method'] == 'parent_child_hybrid':
                rows.append(copy.deepcopy(row) | {'method': 'precision_only'})
            elif row['method'] == 'parent_semantic_v3':
                rows.append(copy.deepcopy(row) | {'method': 'rerank_only'})
        retrieved = dict(retrieved, results=rows)
        return await evaluate_full(HARD_DATASET, out, retrieved, settings, HARD_ARMS)
    raise ValueError('Unknown phase')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['ablation', 'wording', 'challenge'], required=True)
    asyncio.run(main_async(parser.parse_args()))


async def evaluate_full(dataset_path, out, retrieval_report, settings, methods):
    phase = "challenge"
    dataset, documents = load_inputs(dataset_path)
    cases = dataset['cases']
    by_key = {(r['case_id'], r['method']): r for r in retrieval_report['results']}
    tasks = []
    for i, case in enumerate(cases):
        for method in methods[i % len(methods):] + methods[:i % len(methods)]:
            request = {'model': settings.openai_model, 'temperature': 0., 'max_tokens': settings.ai_max_tokens,
                       'stream': False, 'messages': diagnostic_messages(case, by_key[(case['id'], method)], documents)}
            if settings.openai_enable_thinking is not None:
                request['enable_thinking'] = settings.openai_enable_thinking
            tasks.append({'case_id': case['id'], 'method': method, 'payload': request})
    generated = await run_stage('generation', tasks, out / 'generation.jsonl', settings.openai_base_url,
                                settings.openai_api_key, 8)
    extraction, quality = [], []
    context_by_key = {}
    for case in cases:
        for method in methods:
            key = (case['id'], method)
            contexts = by_key[key]['contexts']
            expanded = [{'rank': i, 'title': documents[c['document_id']][0]['title'],
                         'scope': documents[c['document_id']][0]['scope'], 'content': c['content']}
                        for i, c in enumerate(contexts, 1)]
            context_by_key[key] = expanded
            answer = generated[key]['text']
            extraction.append({'case_id': case['id'], 'method': method, 'answer': answer,
                               'payload': payload(settings, settings.openai_model, REVERSE_PROMPT, {'answer': answer})})
            content = {'question': case['question'], 'history': case['history'], 'answerable': case['answerable'],
                       'reference_points': case['answer_points'], 'optional_point_indices': case.get('optional_point_indices', []),
                       'reference_evidence': [g['quote'] for g in case['evidence']], 'contexts': expanded, 'answer': answer}
            quality.append({'case_id': case['id'], 'method': method, 'case': case, 'contexts': contexts,
                            'payload': payload(settings, JUDGE_MODEL, QUALITY_PROMPT, content)})
    extracted = await run_stage('reverse', extraction, out / 'reverse-questions.jsonl', settings.openai_base_url,
                                settings.openai_api_key, 8, parse_reverse)
    faithful_tasks = []
    for case in cases:
        for method in methods:
            key = (case['id'], method)
            claims = extracted[key]['parsed']['claims']
            content = {'claims': [{'index': i, **claim} for i, claim in enumerate(claims, 1)], 'contexts': context_by_key[key]}
            faithful_tasks.append({'case_id': case['id'], 'method': method, 'claims': claims, 'contexts': context_by_key[key],
                                   'payload': payload(settings, JUDGE_MODEL, FAITH_PROMPT, content)})
    results = await asyncio.gather(
        run_stage('quality', quality, out / 'quality.jsonl', settings.openai_base_url, settings.openai_api_key, 4, parse_quality),
        run_stage('faithfulness', faithful_tasks, out / 'faithfulness.jsonl', settings.openai_base_url,
                  settings.openai_api_key, 4, parse_faithfulness), return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    judged, faithful = results
    cache = EmbeddingCache(settings, ROOT / 'target/policy-pilot/embeddings.sqlite3')
    try:
        vectors = cache.embed([retrieval_query(c) for c in cases] + [q for r in extracted.values() for q in r['parsed']['questions']])
    finally:
        cache.close()
    rows = []
    for case in cases:
        for method in methods:
            key = (case['id'], method)
            judge = dict(judged[key]['parsed'])
            source_claims = extracted[key]['parsed']['claims']
            judge['claims'] = [{**source_claims[v['index'] - 1], **v, 'supported': v['relation'] == 'entailed'}
                               for v in faithful[key]['parsed']['verdicts']]
            metrics = quality_metrics(case, judge)
            required = set(case.get('required_point_indices', range(1, len(case['answer_points']) + 1)))
            all_covered = all(v['covered'] for v in judge['point_verdicts'] if v['index'] in required)
            safe = not judge['contradictions'] and not judge['fabricated_specifics']
            behavior = case.get('expected_behavior', 'answer' if case['answerable'] else 'abstain')
            metrics['all_points_answer_correct'] = metrics['answer_correct']
            metrics['answer_correct'] = bool(all_covered and safe and judge['behavior'] == behavior)
            metrics['grounded_answer_correct'] = metrics['answer_correct'] and all(c['supported'] for c in judge['claims'] if not c['excluded_from_faithfulness'])
            questions = extracted[key]['parsed']['questions']
            metrics['answer_relevancy'] = statistics.mean(float(vectors[retrieval_query(case)] @ vectors[q]) for q in questions)
            rows.append({'case_id': case['id'], 'method': method, 'answerable': case['answerable'], 'category': case['category'],
                         'question': case['question'], 'history': case['history'], 'reference_points': case['answer_points'],
                         'required_point_indices': sorted(required), 'retrieval': by_key[key], 'generation': generated[key],
                         'judgment': judge, 'quality': metrics, 'reverse_questions': questions})
    report = {'status': 'independent_model_auto_judged_pending_human', 'phase': phase, 'created_at': now(),
              'dataset_sha256': digest(dataset_path.read_bytes()), 'protocol': json.loads((BASE / 'protocol.json').read_text()), 'cases': len(cases),
              'retrieval_all_cases': retrieval_report['summary'],
              'summary': {m: summarize_rows([r for r in rows if r['method'] == m]) for m in methods}, 'results': rows}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'phase': phase, 'cases': len(cases), 'summary': report['summary']}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    main()

