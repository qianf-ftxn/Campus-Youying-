"""Retrieval-only component controls on the unchanged v4 challenge questions."""
from __future__ import annotations
import json
import numpy as np
from app.core.config import Settings
from app.policy_eval.diagnostic_v4 import BASE, HARD_DATASET
from app.policy_eval.pilot import ROOT
from app.policy_eval.runner import (EmbeddingCache, load_inputs, build_chunks, retrieval_query,
    rank_chunks, score_case, select_context, summarize, digest)


def main():
    dataset, docs = load_inputs(HARD_DATASET)
    chunks=build_chunks(docs)
    cache=EmbeddingCache(Settings(),ROOT/'target/policy-pilot/embeddings.sqlite3')
    try:
        vectors=cache.embed([c.embedding_text for c in chunks]+[retrieval_query(c) for c in dataset['cases']])
    finally:
        cache.close()
    matrix=np.vstack([vectors[c.embedding_text] for c in chunks])
    rows=[]
    for case in dataset['cases']:
        query=retrieval_query(case)
        similarities={c.id:float(s) for c,s in zip(chunks,matrix@vectors[query])}
        for method in ('flat_vector','parent_child_vector','bm25_parent_smoke','parent_child_hybrid'):
            selected,used=select_context(rank_chunks(method,query,chunks,similarities,16),5,4000)
            rows.append({'case_id':case['id'],'method':method,'answerable':case['answerable'],
                'estimated_tokens':used,'retrieval_ms':None,'metrics':score_case(case,selected,docs),
                'contexts':[{'chunk_id':c.id,'document_id':c.document_id,'content':c.content} for c,s in selected]})
    result={'dataset_sha256':digest(HARD_DATASET.read_bytes()),
        'scope':'Retrieval only; flat_vector uses common300/40 windows, NOT archived512/64 baseline. Parent-child vector control drops BM25 and renormalizes .30/.35 weights; BM25 only maps child hits to parents. No generation claims.',
        'summary':{m:summarize([r for r in rows if r['method']==m]) for m in sorted({r['method'] for r in rows})},'results':rows}
    path=BASE/'retrieval-components.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['summary'],ensure_ascii=False))

if __name__=='__main__':main()
