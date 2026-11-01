# K=5 检索与回答对比：75题自动初评

本轮已在75题上完成两组检索及150份真实模型回答，并补齐单块Hit/MRR、参考答案符合度、Faithfulness、Answer Relevancy、拒答/澄清及生成token开销。结果显示：父子块方案取证稍完整、事实更贴近取回资料、输入更省，但严格参考要点符合率没有提升。下一步应优先处理排序中的相近规则混淆，以及生成时遗漏例外或扩写无依据细节。

这是固定资料快照上的开发集自动初评。75题仍全部待人工审核；独立裁判密钥未配置，本轮裁判与回答均使用已配置的 `deepseek-v4-flash-0731`，分别调用。下列分数不能作为独立测试或人工确认的最终准确率。

## 对照条件

原始方案复现源码备份中的平面混合检索：512/64窗口、向量与BM25融合、规则重排及首条邻块扩展。当前方案为hierarchical-v2父子块混合检索：父1000、子300、重叠40，父向量/子向量/子BM25权重0.30/0.35/0.35，关闭重排。比较的是两套完整检索配置，不能把全部差异单独归因于父子索引。

两组共用2026-10-19快照的49份ACTIVE资料、`text-embedding-v4`、相同问题及预设历史、每路16候选、K=5、4000估算上下文token上限。相同生成模型、CONSULT提示词、温度0和512输出token上限；全部150份回答均正常结束，没有长度截断。每组生成75次，评分修订复用原回答。

本实验控制检索到回答的链路；检索使用精确余弦扫描，多轮查询拼接用户历史，不含生产Chroma近似检索、查询改写、完整多智能体调度、在线个人业务系统或线上延迟验收。

## 总体结果

| 指标 | 原始平面混合 | 当前父子块混合 |
|---|---:|---:|
| 必要证据召回率，60题宏平均 | 95.83% | 97.78% |
| 所有必要证据齐全 | 55/60 | 57/60 |
| 单块语义Hit@5 | 60/60 | 60/60 |
| 单块语义MRR@5 | 0.9583 | 0.9597 |
| 参考要点符合率，可回答题 | 60/60 | 58/60 |
| Faithfulness，74份含政策事实的回答宏平均 | 0.8520 | 0.9342 |
| Answer Relevancy，75题宏平均 | 0.7915 | 0.8014 |
| 资料不足处理符合预期 | 14/15 | 15/15 |
| 明确需拒答的题目正确处理 | 14/14 | 14/14 |
| 缺条件需先澄清的题目正确处理 | 0/1 | 1/1 |
| 可回答题错误拒答/只追问 | 0/60 | 0/60 |
| 输入token总量，仅生成阶段，供应商usage | 176099 | 132405 |
| 输出token总量，仅生成阶段，供应商usage | 8831 | 8062 |
| 平均检索正文估算tokens | 3069.19 | 2191.05 |

父子块生成输入token减少24.81%。这是相同提示词下的供应商输入token计数，不包括裁判开销，也未换算金额。实际正文估算长度减少28.61%；两组有同一预算上限，实际输入长度不同。原始方案75题均返回5项，父子块74题返回5项、1题因预算过滤返回4项。

如果再要求“参考要点符合且所有计入的政策事实都有取回依据”，自定义组合通过数为49/75与61/75。该数比只看主结论或参考要点更严格，不是另一种公认的标准准确率。

## 扩展集是否带来同样结论

| 批次 | 原始证据召回率 | 父子块证据召回率 | 原始证据齐全 | 父子块证据齐全 |
|---|---:|---:|---:|---:|
| 原25题，20题可回答 | 90.00% | 95.83% | 16/20 | 18/20 |
| 新增50题，40题可回答 | 98.75% | 98.75% | 39/40 | 39/40 |

新增题未显示证据召回方面的优势，整体提升主要来自用于调参的原25题。因此目前有依据说开发集表现和上下文开销发生了变化，不能声称已经证明独立泛化提升。题目之间也有共享政策事实，75题不是75个相互独立的事实。

## 差异与错误定位

- **D13，父子块**：答对第二学期末分流、第三学期按专业学习，但未说个别大类经审批可调整。按当前全要点口径扣分，属于例外信息遗漏；该例外是否必须答出仍需人工审核，不能与完全答错事实混为一谈。
- **X14，父子块**：把博士“结业后两年”的条款用于“毕业后申请学位”的提问。正确的博士毕业条款在融合候选第15位，未进Top-5，属于排序失误引起的适用对象混淆。
- **U09，原始方案**：先按本科办法给出较长办理条件，末尾再追问字段与年级，没有先确认本研身份；父子块答复同时追问了本研身份和变更事项。此项只有1题，不能据此推广澄清能力的总体提升。
- **附加事实**：X02原始回答将停发助学金写为“办理保留学籍次月起”，原文只说保留学籍期间停发；U05出现“不给/不公布具体分数”的延伸推断。主结论和要点符合不等于所有附加事实都可靠，自动裁判对此仍可能漏判。

父子块未取齐指定证据的X01、X02、X14中，缺失证据在融合候选中的排名分别为8、7、15，均已被召回候选池，主要需要改进排序。X01/X02还有必要证据定义与等价依据的标注问题，当前仍保留原标注和全部样本，未按分数删题。

## 指标与评分过程

必要证据召回及齐全率继续按指定文档/版本/原文位置评分，未自动纳入所有等价依据。单块Hit/MRR由匿名裁判逐块标注：该块只要提供至少一个适用的实质事实即可相关，允许其他文档中的等价依据；MRR取第一条相关结果的倒数排名。它与旧的累计完整覆盖证据单元MRR口径不同。

参考要点符合率要求每条参考要点覆盖、裁判未发现实质冲突或编造、回答/拒答/澄清行为符合预期。它依赖AI参考标注及同模型裁判，不能证明回答中没有任何未被识别的错误，尤其是附加事实。

Faithfulness分两步：只给模型看回答提取原子事实；再让另一次调用只看这些事实及取回资料，按entailed/contradicted/insufficient判断，不提供标准答案。按每份回答被支持事实的比例宏平均。引用必须能定位到实际返回正文或同时提供的标题；只对同来源块的误编号引用校正位置，不对Hit/MRR标签跨块移动证据。

事实抽取器有时会把“未接入个人记录”和一般咨询建议也算作政策事实，因此统一执行可审计的 `policy_claim_filter-v1`：排除资料/能力边界说明与一般建议；保留建议中具体电话、系统功能、表格等可验证断言。每项排除均有标记。未过滤的原始Faithfulness也保留：原始0.8330、父子块0.9180；过滤后为表中0.8520、0.9342。无政策事实的回答为null，不计作满分，两组主指标均有74份评分样本。

Answer Relevancy只给模型看回答，反推3个问题，再计算其与固定历史拼接查询的Embedding余弦相似度均值。多轮更正前的历史也参与该固定查询，可能影响相关度。这是参考Ragas思路的自实现，未调用Ragas包；Faithfulness与相关度的定义参考 [Ragas Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) 和 [Answer Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/)。相关度不衡量事实正确性。

校准阶段发现裁判把部分支持判成完全支持，故保留旧评分并统一采用上述分离流程重评。结构、真实引文及“标签与自身理由矛盾”会触发校验和重试；反复失败的引文/命题以单条输入补评，完整记录保留。技术校验失败不记为问答错误，也不填零。仍有语义判断及事实抽取偏差，人工复核与独立模型交叉评分尚未完成。

## 产物与复现

产物目录为 `target/policy-pilot/expanded-k5/`：

- `dataset.json`、`source-audit.json`：冻结题目、资料摘要和AI自查备注，未修改人工审核状态。
- `current/report.json`、`archived-k5.json`：检索原始记录和配置指纹，保留此前25题报告。
- `remaining-evidence-errors.json`：缺失证据在候选池中的位置。
- `answers-strict/report.json`、`report.md`：全部逐题分数、输出、裁判理由、事实判定、原始/过滤后指标、分批次及分类汇总。
- `answers-strict/answer-review.md`：匿名方案的完整回答和实际上下文审核表。
- `answers-strict/generation.jsonl`、`reverse-questions.jsonl`、`judgments.jsonl`、`faithfulness.jsonl`：断点缓存；拒收输出和逐项补评调用另存JSONL。`answers/`保留校准阶段旧评分。
- `answers-strict/first-full-run-embedding-stats.json`：首次完整相关度计算的实际新增Embedding量；后续缓存汇总显示0新增，不等于第一次没有接口开销。

生成阶段实际150次调用；评分另有450个逻辑任务及引用校验/补评开销。报告分开记录生成token、评分逻辑结果及已收到响应的API用量；取消后未收到响应的请求和Embedding计费token不纳入完整费用结论，没有估算人民币或美元成本。

```bash
# 从头复现使用新目录，避免覆盖已冻结报告；Embedding复用独立评测缓存
.venv/bin/python -m app.policy_eval.runner --dataset app/policy_eval/expanded_dev.json \
  --real --top-k 5 --output target/policy-pilot/expanded-k5-rerun/current

# 需要已有 target/backups/pre-campus-youying-source.tgz，摘要记录于 archived-k5.json
.venv/bin/python - <<'PY'
from pathlib import Path
from app.policy_eval.expand import EXPANDED_DATASET
from app.policy_eval.runner import load_inputs
from app.policy_eval.compare_topk import archived_flat_reports
dataset, documents = load_inputs(EXPANDED_DATASET)
archived_flat_reports(dataset, documents, Path('target/policy-pilot/expanded-k5-rerun'),
                      dataset_path=EXPANDED_DATASET, top_ks=(5,))
PY
.venv/bin/python -m app.policy_eval.answer_compare --real --concurrency 8 \
  --output target/policy-pilot/expanded-k5-rerun/answers-strict

# 复用本轮已保存的输入和成功请求，仅重新汇总；先去掉 --real 可查看固定配置
.venv/bin/python -m app.policy_eval.answer_compare --real --concurrency 8
```

下一轮保持K=5，先改善相近规则的排序和例外信息覆盖，减少无依据扩写；同时人工复核D13、跨制度题及全部资料不足题，用独立裁判交叉检查分数，再建立按证据族隔离的新测试集。
