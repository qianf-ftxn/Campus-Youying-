import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.policy_eval.core_ablation import ARMS, cache_hits, flat_hybrid_rank, paired, request_hash
from app.policy_eval.rigorous import messages
from app.policy_eval.runner import Chunk


class CoreAblationTests(unittest.TestCase):
    def test_flat_fusion_excludes_parent_child_rows_and_preserves_common_weights(self):
        chunks = [Chunk(1,'d','v','flat','题目','甲'), Chunk(2,'d','v','flat','题目','乙'),
                  Chunk(3,'d','v','parent','题目','父'), Chunk(4,'d','v','child','题目','子',3)]
        with patch('app.policy_eval.core_ablation.bm25_scores', return_value={1:0.,2:10.}):
            ranked = flat_hybrid_rank('查询',chunks,{1:1.,2:0.,3:100.,4:100.})
        self.assertEqual([r.id for r,s in ranked],[2,1])
        self.assertAlmostEqual(ranked[0][1],.5*.65+.35)
        self.assertAlmostEqual(ranked[1][1],.65)

    def test_four_arms_use_the_identical_answer_prompt_for_the_same_context(self):
        case={'question':'怎么办？','history':[{'role':'user','content':'我是硕士'}]}
        documents={'d':({'title':'制度','scope':'研究生','version':'v'},'原文')}
        contexts=[{'document_id':'d','version':'v','content':'适用原文'}]
        requests=[messages(case,{'method':m,'contexts':contexts},documents) for m in ARMS]
        self.assertTrue(all(r==requests[0] for r in requests))
        self.assertIn('适用原文',requests[0][0]['content'])
        self.assertEqual(requests[0][-1],{'role':'user','content':'怎么办？'})

    def test_cache_reuse_requires_same_case_method_model_and_prompt(self):
        endpoint='https://example.invalid'
        payload={'model':'model-a','messages':[{'role':'user','content':'原提示'}]}
        original={'case_id':'q1','method':'parent_child_hybrid','payload':payload}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache.jsonl'
            path.write_text(json.dumps({'request_sha256':request_hash(original,endpoint)})+'\n')
            variants=[original,
                      {**original,'method':'flat_semantic'},
                      {**original,'payload':{**payload,'model':'model-b'}},
                      {**original,'payload':{**payload,'messages':[{'role':'user','content':'改过提示'}]}},
                      {**original,'case_id':'q2'}]
            self.assertEqual(cache_hits(variants,path,endpoint),1)
            self.assertEqual(cache_hits([original],path,'https://another.invalid'),0)

    def test_paired_effect_reports_both_improvements_and_regressions(self):
        rows=[]
        for i,a,b in [('1',False,True),('2',False,True),('3',True,False)]:
            for method,correct in [('flat_hybrid',a),('parent_child_hybrid',b)]:
                rows.append({'case_id':i,'method':method,'answerable':True,'quality':{'answer_correct':correct}})
        delta=paired({'results':rows},'flat_hybrid','parent_child_hybrid')
        self.assertEqual(delta['improved'],['1','2'])
        self.assertEqual(delta['regressed'],['3'])
        self.assertEqual(delta['net'],1)
        self.assertAlmostEqual(delta['delta_pp'],100/3)


if __name__=='__main__':unittest.main()
