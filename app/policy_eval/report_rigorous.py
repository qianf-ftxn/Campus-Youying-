"""Read-only aggregation of the frozen prospective run; never changes verdicts."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from app.policy_eval.rigorous import ARMS, BASE, FROZEN_PROTOCOL, HOLDOUT, REVIEWED_DEV
from app.policy_eval.pilot import ROOT
from app.policy_eval.runner import digest

LABELS = {'archived_flat_hybrid': '原始平面混合', 'parent_child_hybrid': '现有父子混合',
          'parent_semantic_v3': '语义重排＋精简回答原型'}


def metric(row, name):
    if name == 'answer_accuracy_answerable':
        return float(row['quality']['answer_correct']) if row['answerable'] else None
    if name == 'necessary_evidence_recall':
        return row['retrieval']['metrics']['clause_recall_at_k']
    return row['quality'].get(name)


def clustered_interval(rows, dataset, baseline, candidate, name):
    """Paired bootstrap over evidence families, rather than independent questions."""
    families = sorted({case['evidence_family'] for case in dataset['cases']})
    groups = {case['id']: case['evidence_family'] for case in dataset['cases']}
    sums, counts = {}, {}
    for method in [baseline, candidate]:
        sums[method], counts[method] = [], []
        for family in families:
            values = [metric(row, name) for row in rows if row['method'] == method and groups[row['case_id']] == family]
            values = [float(v) for v in values if v is not None]
            sums[method].append(sum(values))
            counts[method].append(len(values))
    draws = np.random.default_rng(20261031).integers(0, len(families), size=(5000, len(families)))
    means = {}
    for method in [baseline, candidate]:
        denominators = np.asarray(counts[method])[draws].sum(axis=1)
        means[method] = np.divide(np.asarray(sums[method])[draws].sum(axis=1), denominators,
                                  out=np.full(5000, np.nan), where=denominators != 0)
    values = means[candidate] - means[baseline]
    values = values[np.isfinite(values)]
    point = sum(sums[candidate]) / sum(counts[candidate]) - sum(sums[baseline]) / sum(counts[baseline])
    return {'candidate_minus_baseline': point,
            'conditional_95pct_family_bootstrap_interval': [float(v) for v in np.percentile(values, [2.5, 97.5])],
            'families': len(families), 'replicates': 5000,
            'scope': 'conditional on these AI annotations and model verdicts; does not include annotation/judge uncertainty'}


def main():
    path = BASE / 'holdout/report.json'
    report = json.loads(path.read_text())
    dataset = json.loads(HOLDOUT.read_text())
    dev = json.loads((BASE / 'dev/report.json').read_text())
    freeze = json.loads(FROZEN_PROTOCOL.read_text())
    if report['dataset_sha256'] != digest(HOLDOUT.read_bytes()) or report['protocol'] != freeze['protocol']:
        raise ValueError('Frozen report inputs differ')
    if len(report['results']) != len(dataset['cases']) * len(ARMS):
        raise ValueError('Incomplete prospective comparison')
    result = {'status': report['status'], 'dataset_sha256': report['dataset_sha256'],
              'frozen_protocol_sha256': digest(FROZEN_PROTOCOL.read_bytes()),
              'prospective_cases': len(dataset['cases']), 'answerable_cases': sum(c['answerable'] for c in dataset['cases']),
              'families': len({c['evidence_family'] for c in dataset['cases']}),
              'review_status': dict(Counter(c['review_status'] for c in dataset['cases'])),
              'summary': report['summary'], 'cost': {}, 'paired_intervals': {},
              'development_regression_cases': dev['cases'], 'development_regression_summary': dev['summary']}
    for method in ARMS:
        rows = [r for r in report['results'] if r['method'] == method]
        generation_input = sum(r['generation']['usage'].get('prompt_tokens', 0) for r in rows)
        generation_output = sum(r['generation']['usage'].get('completion_tokens', 0) for r in rows)
        ranking_input = sum(r['retrieval'].get('rerank_usage', {}).get('prompt_tokens', 0) for r in rows)
        ranking_output = sum(r['retrieval'].get('rerank_usage', {}).get('completion_tokens', 0) for r in rows)
        result['cost'][method] = {'generation_input_tokens': generation_input, 'generation_output_tokens': generation_output,
                                 'reranking_input_tokens': ranking_input, 'reranking_output_tokens': ranking_output,
                                 'inference_input_total': generation_input + ranking_input,
                                 'inference_output_total': generation_output + ranking_output,
                                 'generation_tasks': len(rows), 'reranking_tasks': len(rows) if method == 'parent_semantic_v3' else 0,
                                 'excludes': ['evaluation/judge calls', 'embedding', 'failed or rejected attempts', 'currency pricing']}
    for candidate in ARMS[1:]:
        result['paired_intervals'][candidate] = {name: clustered_interval(report['results'], dataset, ARMS[0], candidate, name)
                                                for name in ['answer_accuracy_answerable', 'necessary_evidence_recall', 'faithfulness']}
    by_key = {(r['case_id'], r['method']): r for r in report['results']}
    result['paired_errors'] = {}
    for candidate in ARMS[1:]:
        result['paired_errors'][candidate] = {
            'baseline_only_correct': [case['id'] for case in dataset['cases'] if by_key[(case['id'], ARMS[0])]['quality']['answer_correct']
                                      and not by_key[(case['id'], candidate)]['quality']['answer_correct']],
            'candidate_only_correct': [case['id'] for case in dataset['cases'] if not by_key[(case['id'], ARMS[0])]['quality']['answer_correct']
                                      and by_key[(case['id'], candidate)]['quality']['answer_correct']]}
    result['returned_models'] = {}
    for file in ['generation.jsonl', 'quality.jsonl', 'faithfulness.jsonl']:
        records = [json.loads(line) for line in (BASE / 'holdout' / file).read_text().splitlines()]
        result['returned_models'][file] = dict(Counter(r.get('returned_model', 'assembled repair') for r in records))
    result['failed_cases'] = [{'case_id': r['case_id'], 'method': r['method'], 'question': r['question'],
                               'point_verdicts': r['judgment']['point_verdicts'], 'errors': r['judgment']['contradictions'],
                               'reason': r['judgment'].get('reason', '见逐要点判定'), 'behavior': r['judgment']['behavior']}
                              for r in report['results'] if not r['quality']['answer_correct']]
    (BASE / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 冻结协议后的60题三组对比', '',
             '独立模型家族自动裁判，参考标注仍待人工审核；不是外部认证的独立测试集。', '',
             '| 指标 | 原始平面混合 | 现有父子混合 | 新原型 |', '|---|---:|---:|---:|']
    for name, key, percent in [('必要证据召回率', 'necessary_evidence_recall', True), ('证据齐全率', 'evidence_complete_rate', True),
                               ('参考要点符合率（可回答题）', 'answer_accuracy_answerable', True),
                               ('所有题目符合率', 'answer_accuracy_all', True), ('Hit@5', 'hit_at_5', True), ('MRR@5', 'mrr_at_5', False),
                               ('Faithfulness', 'faithfulness', False), ('Answer Relevancy', 'answer_relevancy', False),
                               ('资料不足处理', 'unanswerable_handled_correctly', True), ('拒答正确', 'abstention_correct', True),
                               ('澄清正确', 'clarification_correct', True), ('错误拒答', 'false_refusal', True)]:
        values = [result['summary'][method][key] for method in ARMS]
        fmt = lambda v: '—' if v is None else (f'{v:.2%}' if percent else f'{v:.4f}')
        lines.append('| ' + name + ' | ' + ' | '.join(fmt(v) for v in values) + ' |')
    lines.extend(['', '## 推理token开销', '', '| 方法 | 生成输入 | 生成输出 | 重排输入 | 重排输出 | 总输入 | 总输出 |', '|---|---:|---:|---:|---:|---:|---:|'])
    for method in ARMS:
        c = result['cost'][method]
        lines.append('| ' + LABELS[method] + ' | ' + ' | '.join(str(c[k]) for k in ['generation_input_tokens', 'generation_output_tokens',
                      'reranking_input_tokens', 'reranking_output_tokens', 'inference_input_total', 'inference_output_total']) + ' |')
    lines.extend(['', '重排额外推理已计入表格。评分、Embedding、失败重试及货币费用另计，不能仅看生成token声称新方案端到端更省。', '',
                  '## 按证据族配对重采样', '', '以12个证据族为重采样单位，5000次，固定随机种子。区间仅反映这一批题和自动判定下的抽样波动，不包含人工标注或模型裁判误差。', ''])
    for method, values in result['paired_intervals'].items():
        for name, data in values.items():
            lo, hi = data['conditional_95pct_family_bootstrap_interval']
            lines.append(f"- {LABELS[method]}相对原始 / {name}：差值{data['candidate_minus_baseline']:+.4f}，条件区间[{lo:+.4f}, {hi:+.4f}]。")
    lines.extend(['', '## 需复核的错例', ''])
    for row in result['failed_cases']:
        lines.append(f"- {row['case_id']} / {LABELS[row['method']]}：{row['reason']}")
    (BASE / 'comparison.md').write_text('\n'.join(lines) + '\n')
    # Review packet uses only fixed outputs. Separate per-source-family groups
    # make shared-fact questions visible to the human reviewer.
    aliases = {m: f'方案{i+1}' for i, m in enumerate(sorted(ARMS, key=lambda m: fingerprint_label(m)))}
    review = ['# 前瞻题目与回答审核包', '', '所有题目仍待人工审核；本表勾选不回写JSON。方案对应关系保留在report.json。', '']
    for case in sorted(dataset['cases'], key=lambda c: (c['evidence_family'], c['id'])):
        review.extend([f"## {case['id']} · {case['evidence_family']}", '', case['question'], '',
                       f"预期行为：{case['expected_behavior']}。历史：{json.dumps(case['history'], ensure_ascii=False)}", '', '参考要点：', ''])
        review.extend('- ' + point for point in case['answer_points'])
        review.append('')
        for g in case['evidence']:
            review.extend([f"证据 {g['document_id']} · {g['section']} · 第{g['line_start']}行：", '',
                           f"[仓库原文]({ROOT / 'knowledge/xidian' / g['content_path']}:{g['line_start']}) · [快照记录的官方链接]({g['source_url']})", '',
                           '> ' + g['quote'].replace('\n', '\n> '), ''])
        for method in sorted(ARMS, key=lambda m: aliases[m]):
            row = by_key[(case['id'], method)]
            review.extend([f"### {aliases[method]}", '', row['generation']['text'], '',
                           f"自动裁判：{row['judgment'].get('reason', '见逐要点判定')}", '',
                           '人工判定：□通过 □错例 □标注需修订；审核人：____；依据：____', ''])
            for rank, context in enumerate(row['retrieval']['contexts'], 1):
                review.extend([f"实际取回资料{rank} · {context['document_id']}", '',
                               '> ' + context['content'].replace('\n', '\n> '), ''])
    (BASE / 'human-review.md').write_text('\n'.join(review) + '\n')
    print(json.dumps({'cases': result['prospective_cases'], 'summary': result['summary'], 'cost': result['cost']}, ensure_ascii=False))


def fingerprint_label(value):
    return digest(('rigorous-v3-blind:' + value).encode())


if __name__ == '__main__':
    main()
