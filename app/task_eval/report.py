"""Write reviewable results from completed runs; never reruns candidate models."""
import json
from pathlib import Path

from app.evaluation.judge import sign_test_p_value
from app.task_eval.prepare import ROOT, encode


def main():
    out = ROOT / "target/task-success-v1/real-v2"
    report = json.loads((out / "full-report.json").read_text())
    summary, rows = report["summary"], report["results"]
    if summary["status"] != "complete_development_automatic_scoring" or len(rows) != 40:
        raise ValueError("All 40 paired tasks require valid completed grading before publishing rates")
    dataset = json.loads((out / "dataset.json").read_text())
    cases = {c["id"]: c for c in dataset["cases"]}
    single, multi = summary["arms"]["single"], summary["arms"]["multi"]
    wins, losses = summary["multi_only_success"], summary["single_only_success"]
    p = sign_test_p_value(len(wins), len(losses))
    delta = 100 * (multi["task_success_rate"] - single["task_success_rate"])
    reviews = [(r["case_id"], a) for r in rows for a, j in r["judgments"].items()
               if j.get("consistency_review", {}).get("triggered")]
    quality_only = {a: sum(all(r["checks"][a].get(k) is True for k in
                             ("required_answer_points", "no_substantive_errors", "no_forbidden_behavior",
                              "no_internal_metadata", "expected_behavior")) for r in rows) for a in ("single", "multi")}
    result = {"summary": summary, "delta_percentage_points": delta, "paired_sign_test_p": p,
              "explicit_judge_conflict_repairs": reviews, "answer_semantic_pass_count": quality_only,
              "grading": "Independent-family model scoring, known development tasks, pending human review"}
    notes_path = out / "spot-check-notes.json"
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {"flags": []}
    result["spot_check_notes"] = notes
    (out / "summary.json").write_bytes(encode(result))
    lines = ["# 单／多Agent任务成功率：40题开发对照", "",
             f"模型原始自动评分：单Agent {single['successful_tasks']}/40，多Agent {multi['successful_tasks']}/40；差值 {delta:+.1f} 个百分点。",
             "本批结果为已知开发集上的模型自动评分，不代表独立测试集、线上效果或人工业务验收。", "",
             "## 总体结果", "", "| 指标 | 单Agent（可自检） | 多Agent |", "|---|---:|---:|",
             f"| 完整任务成功 | {single['successful_tasks']}/40（{single['task_success_rate']:.1%}） | {multi['successful_tasks']}/40（{multi['task_success_rate']:.1%}） |",
             f"| 回答语义验收通过（不含路由及工具状态） | {quality_only['single']}/40 | {quality_only['multi']}/40 |",
             f"| 系统执行失败 | {single['system_failures']} | {multi['system_failures']} |",
             f"| 候选模型调用数 | {single['model_calls']} | {multi['model_calls']} |",
             f"| 候选输入token | {single['input_tokens']} | {multi['input_tokens']} |",
             f"| 候选输出token | {single['output_tokens']} | {multi['output_tokens']} |", "",
             "候选token不含答案评分、Embedding、以及原8题试跑的开销。总调用次数和实际输入输出分别比较，不用输入token单独冒充实际费用。", "",
             "## 分组结果", "", "| 分组 | 题数 | 单Agent成功 | 多Agent成功 |", "|---|---:|---:|---:|"]
    for group, value in summary["groups"].items():
        lines.append(f"| {group} | {value['tasks']} | {value['single']} | {value['multi']} |")
    lines.extend(["", "## 配对改善与退步", "",
                  f"多Agent独有成功 {len(wins)}题，单Agent独有成功 {len(losses)}题；配对双侧精确符号检验 p={p:.4f}（探索性辅助，不替代人工验收）。",
                  "", "多Agent独有成功：" + ("、".join(wins) or "无"),
                  "", "单Agent独有成功：" + ("、".join(losses) or "无"), "",
                  "## 测试口径", "",
                  "- 26道可回答政策题、6道信息不足题、8道支持／风险题，标签来源为既有AI草拟开发集，待人工复核。",
                  "- 同一deepseek-v4-flash-0731、温度0；Qwen qwen3.8-flash按相同逐要点规则盲评，不看到架构名称或工具结果。",
                  "- 两组使用同一49份政策及11份支持资料组成的语料快照、K=5、4000估算知识token、相同父子混合检索，无语义精排；每题固定同一份实际上下文及业务Skill。不能与先前RAG精排实验混报。",
                  "- 单Agent至少草拟和自检两轮；多Agent使用共享上下文适配的现有Runtime，使用实际采纳的回答，无额外重新生成。",
                  "- 两组相同模型调用与token上限；意图、风险、报告、工具执行、全部必需答案要点及行为均通过才计完整成功。",
                  "- 报告、个案、表格及通知由共同授权和执行代码处理，在临时数据库／临时文件／日志适配器中执行。日志成功不等于真实通知送达或人工接手。",
                  "- 原8题pilot发现JSON围栏解析问题，已修复并在real-v2重新运行，不将该解析缺陷当作多Agent优势。旧原始试跑单独保留。",
                  f"- 对明确布尔标签与理由相反的评分进行同规则结构一致性复核，共{len(reviews)}项；保留原判定、冲突和修复响应，不自动改为期望标签。",
                  "- 本轮没有重新测Hit@K、MRR或Faithfulness，不将旧RAG指标归因于多Agent。", "",
                  "## 结论边界", ""])
    if delta <= 0:
        lines.append("本批结果不支持在简历中声称多Agent提高任务成功率。应保留真实对照结果，按具体错例判断协作设计是否有收益。")
    elif p >= .05:
        lines.append("本批多Agent数值更高，但配对样本不足以支持显著优势。可描述已完成对照和具体开发集成绩，不应写成普遍效果提升。")
    else:
        lines.append("本批多Agent具有开发集上的配对改善证据；仍需人工核验标注与争议输出，不能外推为普遍或线上效果。")
    lines.extend(["", "原始数据：`target/task-success-v1/real-v2/full-report.json`、`summary.json`、`protocol.json`、`scoring-protocol.json`；逐次请求与输出位于`responses/`，冻结执行源码位于`source-snapshot/`。"])
    if notes["flags"]:
        lines.extend(["", "## AI抽查发现的评分争议（未改原始分数）", "",
                      "以下是AI对原始输出的抽查，不是人工批准；尚未形成完整人工复核成绩。不能将自动评分视为最终真实成功率。"])
        for flag in notes["flags"]:
            lines.extend(["", f"- {flag['case_id']} / {flag['arm']}：{flag['finding']}"])
    (ROOT / "docs/agent-task-success-results-v1.md").write_text("\n".join(lines) + "\n")
    paired = ["# 单／多Agent配对错例", ""]
    for case_id in wins + losses:
        row = next(r for r in rows if r["case_id"] == case_id)
        paired.extend(["## " + case_id, "", cases[case_id]["input"]["question"], ""])
        for arm in ("single", "multi"):
            failed = [k for k, v in row["checks"][arm].items() if v is False]
            paired.extend([f"### {arm}：{'通过' if row['task_success'][arm] else '失败'}", "",
                           "未通过项：" + ("、".join(failed) or "无"), "", row["observations"][arm].get("answer", "无有效正文"), ""])
    (out / "paired-cases.md").write_text("\n".join(paired))
    print(json.dumps({"single": single['successful_tasks'], "multi": multi['successful_tasks'],
                      "delta_pp": delta, "p": p, "judge_repairs": len(reviews)}, ensure_ascii=False))


if __name__ == "__main__": main()
