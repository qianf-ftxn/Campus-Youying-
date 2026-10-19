# MindBridge 上下文记忆与安全链路：现状、目标架构与接手指南

> 文档定位：面向第一次接触 MindBridge 的开发者、维护者和 Agent  
> 阅读目标：理解当前系统如何工作、存在什么问题、改造后为什么更可靠，以及排查问题时应该从哪里开始  
> 配套实施规格：[`context-memory-safety-redesign-spec.md`](./context-memory-safety-redesign-spec.md)

## 1. 项目概览

MindBridge 是一个面向学生的校园陪伴与心理支持系统，主要组件包括：

- FastAPI API；
- MySQL 业务数据和完整聊天记录；
- Redis 短期记忆；
- Ollama/OpenAI-compatible 模型；
- 事件驱动多 Agent 协作；
- RAG 心理知识库；
- 风险报告、个案、告警和工具队列；
- SSE 流式回复。

核心请求入口：

```text
HTTP / SSE
  ↓
ChatService
  ↓
MindBridgeAgentHarness
  ↓
EventDrivenAgentRuntimeService
  ↓
EventDrivenCoordinator + 多个 Agent
  ↓
AiClient.stream()
  ↓
保存 assistant 消息并调度工具
```

当前主要代码位置：

| 领域 | 文件 |
|---|---|
| API | `app/api/routes.py` |
| SSE 聊天 | `app/services/chat.py` |
| 一轮业务编排 | `app/agents/harness.py` |
| 多 Agent runtime | `app/agents/event_driven_runtime.py` |
| Agent 实现 | `app/agents/autonomous.py` |
| Coordinator | `app/agents/coordinator.py` |
| 黑板、任务和 artifact | `app/agents/events.py` |
| 模型调用与 Prompt | `app/services/ai.py` |
| 风险评估 | `app/services/assessment.py` |
| 短期记忆和压缩 | `app/services/memory.py` |
| 数据实体 | `app/models/entities.py` |
| 报告和个案工具 | `app/services/tools.py`、`app/services/tool_queue.py` |

## 2. 当前一轮请求如何运行

当前 `ChatService.stream_chat()` 首先调用 Harness：

```text
收到当前输入
  ↓
Harness 脱敏当前输入
  ↓
解析或创建 ChatSession
  ↓
先保存当前 user 消息并取得 message_id
  ↓
执行下一轮 token 高水位兜底
  ↓
Runtime 加载 MemorySnapshot，再运行多 Agent 协作
  ↓
ResponseAgent 生成候选正文，SafetyAgent 审查正文
  ↓
ChatService 流式传输已批准正文
  ↓
完整 assistant 消息同步落库；普通摘要维护写入异步队列
```

当前 user 消息在 runtime 前已经落库，但构造快照时通过 `current_message_id` 排除，随后只在 Prompt 末尾追加一次 `board.model_input`。这样既能让 RiskEvent 关联确切消息，又不会重复当前输入。

当前 Agent 大致职责：

```text
CoordinatorAgent
    创建任务、控制预算、采纳最终方案

UnderstandingAgent
    判断 CHAT / CONSULT / RISK

SafetyAgent
    评估当前风险、发布 SAFETY_OVERRIDE、审查候选回复

ContextAgent
    加载聊天历史、生成摘要、RAG 检索、加载 skill

ResponseAgent
    组装最终回复 Prompt
```

虽然文档经常把它画成固定顺序，实际代码是任务认领机制：Coordinator 创建开放任务，Agent 根据 capability、优先级和置信度认领。任务依赖目前不够严格，顺序可能依赖优先级排序。

## 3. 当前存储结构

### 3.1 MySQL

当前 MySQL 保存：

- `chat_sessions`：会话；
- `chat_messages`：完整 user/assistant 原文；
- `psychological_reports`：咨询/风险评估；
- `risk_cases`：中高风险个案工单；
- `case_notes`：个案备注；
- `alert_records`：告警记录；
- `tool_jobs`：异步工具任务；
- `agent_run_traces`：Agent 运行 trace。

完整消息不会因为 Prompt 压缩而删除，因此当前主要问题不是“数据库数据丢失”，而是“模型后续看不到”。

### 3.2 Redis

当前 Redis 保存：

- 每个会话最近若干条消息，默认最多 40 条；
- 每条写入时脱敏；
- 有 TTL，默认 24 小时；
- 每个 Agent 的简短私有记忆。

Redis miss 时，ContextAgent 从 MySQL 查询最后 40 条消息并尝试回填 Redis。

### 3.3 当前没有的存储

当前没有：

- 持久滚动摘要；
- 摘要覆盖到哪条消息的游标；
- 摘要版本；
- 独立 RiskEvent 事件流；
- 当前 SafetyState；
- 可被下一轮快速恢复的安全状态缓存。

## 4. 当前上下文压缩怎么工作

当前核心函数位于 `app/services/memory.py`：

```text
compact_history_for_prompt()
summarize_history_for_memory()
```

当前规则：

```text
1. 历史消息脱敏
2. 生成 memoryBrief
   - 最近4条用户消息，每条最多80字符
   - 最近3条助手消息，每条最多70字符
   - 当前输入最多80字符
   - 总长度默认最多500字符
3. 如果历史不超过8条，保留全部历史
4. 如果超过8条：
   - 一条 system “历史摘要”
   - 最近8条原文
5. 追加当前输入
6. 最后按 chat_history_limit * 2 再做消息数兜底
```

### 4.1 为什么它会遗忘

假设有 20 条交替消息：

```text
U1 A1 U2 A2 ... U10 A10
```

最近 8 条是：

```text
U7 A7 U8 A8 U9 A9 U10 A10
```

所谓摘要取最近 4 条用户消息和最近 3 条助手消息，仍然来自这 8 条附近。因此模型实际看到：

```text
最近8条的摘录
+ 最近8条的完整原文
+ 当前输入
```

U1～A6 没有进入摘要。这不是窗口外历史压缩，而是近期内容重复。

### 4.2 当前普通 CHAT 与 CONSULT/RISK 的区别

- 普通 CHAT + LOW：使用确定性规则 memoryBrief，不额外调用总结模型；
- CONSULT/RISK 或中高风险：ContextAgent 使用模型对最近 12 条生成 1～3 条记忆要点；
- 但压缩后的 `modelHistory` 头部仍然是确定性规则摘要；
- 因此支持场景可能同时出现 LLM memoryBrief、规则摘要、最近原文，信息有重复；
- LLM memoryBrief 也只看最近 12 条，不能解决长期记忆。

## 5. 当前安全链路怎么工作

### 5.1 本轮风险

SafetyAgent 调用 `PsychologicalAssessmentService`：

- 硬规则命中高风险词时直接 HIGH；
- 否则让模型返回 emotion/risk/confidence/summary；
- 模型失败时使用本地启发式；
- 本轮 HIGH 时发布 `SAFETY_OVERRIDE`。

### 5.2 报告与工单

对于非 CHAT：

- Harness 创建 `PsychologicalReport`；
- MEDIUM/HIGH 创建 `RiskCase`；
- HIGH 创建告警任务；
- 工具队列异步执行 Excel、个案和告警。

### 5.3 当前跨轮缺陷

新的用户消息会创建新的协作黑板。上一轮的 risk artifact 不会直接进入下一轮。

SafetyAgent 任务优先级通常高于 ContextAgent，所以它可能在历史加载之前完成风险评估。此时它只能评估当前输入。

例子：

```text
Turn 1：我不想活了
    → RISK / HIGH / SAFETY_OVERRIDE

Turn 2：帮我写一个 Python 排序算法
    → CHAT / LOW
```

Turn 2 的 ResponseAgent 可能还能在最近原文里看到 Turn 1，但：

- SafetyAgent 没有把它作为未解除事件使用；
- `PsychologicalReport/RiskCase` 没有被读回；
- 不会强制 HIGH 安全策略；
- 原文离开最近窗口后完全不可见。

## 6. 改造后的核心思想

改造后系统拥有两条独立记忆：

```text
普通对话记忆
    historySummary + recentMessages

安全事件记忆
    RiskEvent + SafetyState
```

普通记忆解决：

- 长对话事实不丢失；
- 最近措辞保真；
- Prompt token 可控。

安全记忆解决：

- HIGH 风险跨轮继承；
- 普通 CHAT 不会意外绕过未解除风险；
- 风险可以复核、监测、解除、纠错；
- Redis 丢失时能从 MySQL 恢复。

## 7. 改造后的普通记忆

### 7.1 三层数据

```text
chat_messages
    完整原文，永不因压缩删除

chat_memory_summaries
    窗口外历史的结构化持久摘要

recentMessages
    coveredThroughMessageId 之后的最近原文
```

### 7.2 8/12 策略

```text
压缩目标：8条原始消息
触发阈值：12条未覆盖消息
```

达到 12 条时，把最早 4 条完整消息合并进已有摘要，留下最近 8 条。

```text
压缩前：
summary_v3 + M21 ... M32

压缩后：
summary_v4 = merge(summary_v3, M21 ... M24)
recent = M25 ... M32
coveredThroughMessageId = M24.id
```

### 7.3 不需要每轮重算全部历史

每次只处理：

```text
旧摘要
+ 新移出窗口的消息
```

摘要输入大小稳定，不会随着会话无限增长。

### 7.4 token 兜底

即使不到 12 条，只要完整 Prompt 达到模型上下文高水位，也要提前压缩最早完整轮次。条数触发用于常规维护，token 触发用于真实容量保护。

## 8. 改造后的安全记忆

### 8.1 RiskEvent

记录不可变事件：

```text
何时发生
关联哪条消息
事件类型
风险等级
证据摘要
来源是规则、模型还是人工
是否触发状态转换
```

### 8.2 SafetyState

记录当前有效状态：

```text
ACTIVE
MONITORING
RESOLVED
FALSE_POSITIVE
```

并带有：

```text
effectiveRisk
latestEventId
lastHighRiskAt
lastAssessedAt
version
```

### 8.3 与 RiskCase 的区别

```text
SafetyState
    用户目前的安全生命周期

RiskCase
    工作人员/工具处理个案的工作流
```

以下推断是错误的：

```text
RiskCase = ACKNOWLEDGED
所以 SafetyState = RESOLVED
```

接手不代表风险解除。

## 9. 改造后的风险计算

每轮产生三个风险值：

```text
currentRisk
    本轮输入风险

carryoverRisk
    历史未解除状态带来的风险下限

effectiveRisk
    max(currentRisk, carryoverRisk)
```

推荐映射：

```text
ACTIVE/HIGH       → carryover HIGH
ACTIVE/MEDIUM     → carryover MEDIUM
MONITORING        → carryover 至少 MEDIUM 或专用监测策略
RESOLVED          → carryover LOW
FALSE_POSITIVE    → carryover LOW
```

关键规则：

```text
继承最高“未解除”风险
而不是继承历史上出现过的最高风险
```

### 9.1 不要把风险、生命周期和情绪混在一起

```text
风险等级：HIGH / MEDIUM / LOW
生命周期：ACTIVE / MONITORING / RESOLVED / FALSE_POSITIVE
情绪状态：焦虑 / 低落 / 正常等
```

“焦虑”不是 HIGH 和 LOW 之间的风险层级。用户可以焦虑但没有自伤风险，也可能表面平静但仍有明确计划。状态机只对风险和安全生命周期降级，情绪标签由独立评估结果描述。

## 10. 改造后的完整请求流程

```text
1. 接收、校验、脱敏当前输入

2. 解析/创建 ChatSession

3. 先保存当前 user 消息，获得 message_id

4. Runtime 调用 MemoryContextService
   - historySummary
   - recentMessages
   - SafetyState
   - activeRiskEvents

5. 必要时执行窗口/token预算兜底，生成不可变MemorySnapshot

6. 用MemorySnapshot初始化Blackboard并记录MEMORY_SNAPSHOT_LOADED事件

7. Coordinator启动协作，UnderstandingAgent和SafetyAgent并行
   - intent
   - currentRisk
   - carryoverRisk
   - effectiveRisk
   - lifecycle transition

8. 如果有新风险或状态变化
   - 写 RiskEvent
   - 更新 SafetyState
   - 提交 MySQL
   - 刷新 Redis

9. ContextAgent 聚合
   - 安全约束
   - RAG
   - skill

10. ResponseAgent 选择回复模式并生成候选正文

11. SafetyAgent 审查候选回复

12. Coordinator 采纳

13. SSE 流式输出

14. 完整 assistant 消息同步保存；中断内容标记为 INTERRUPTED，不参与压缩

15. 将普通 12→8 维护写入 memory_maintenance_jobs，由后台 worker 重试
    - 安全事件和 SafetyState 不进入异步队列
    - 下一轮模型调用前仍同步执行 token 高水位兜底
```

### 10.1 为什么不使用 MemoryAgent

记忆相关工作主要是数据库/缓存读取、版本校验、窗口判断、token 预算、摘要更新和并发控制。这些工作输入输出明确，需要事务一致性，不需要自主认领任务或开放式推理。

因此最终设计是：

```text
删除设计概念：MemoryAgent、AgentCapability.MEMORY、task:load-memory

保留设计概念：MemoryContextService、MemorySnapshot
```

即使滚动摘要内部调用 LLM，也应由 `RollingSummaryService.merge()` 作为固定 schema 的受约束转换执行，不需要包装成 Agent。

### 10.2 MemorySnapshot 如何进入 Blackboard

推荐结构：

```python
@dataclass(frozen=True)
class MemorySnapshot:
    history_summary: dict
    recent_messages: tuple[AiMessage, ...]
    safety_state: SafetyStateSnapshot | None
    active_risk_events: tuple[RiskEventSnapshot, ...]
    covered_through_message_id: int | None
    summary_version: int
    safety_state_version: int
    degraded: bool = False
    degradation_reasons: tuple[str, ...] = ()
```

Runtime 在 Coordinator 启动前执行：

```python
snapshot = memory_context_service.load_snapshot(
    user_id=user.id,
    session_id=session.id,
    current_message_id=current_message.id,
)

board = CollaborationBlackboard(
    ...,
    memory_snapshot=snapshot,
)
```

同时记录：

```text
MEMORY_SNAPSHOT_LOADED
```

事件只记录摘要版本、安全状态版本、最近消息数量和 degraded 状态，不记录敏感原文。

MemorySnapshot 应使用纯 DTO，不能直接塞入 SQLAlchemy entity。它代表本轮开始时的统一只读视图，一轮内不原地修改。SafetyAgent 本轮产生的新判断通过 risk artifact 和 RiskEvent 表示，下一轮再进入新的 snapshot。

### 10.3 Agent 前后分别做什么

Agent 开始前：

```text
加载持久摘要和最近原文
加载SafetyState和activeRiskEvents
检查缓存版本
检查未完成的12→8维护
估算Prompt token
必要时做强制兜底压缩
形成MemorySnapshot
```

回复保存后：

```text
常规检查未覆盖消息是否达到12条
达到阈值则把最早4条合并进摘要
保留最近8条
刷新摘要缓存
```

因此“每轮 Agent 前加载记忆”不等于“每轮都调用摘要模型”。前置阶段主要保证快照完整和预算可用，常规批量压缩优先放在回复完成后。

### 10.4 各 Agent 如何使用快照

```text
UnderstandingAgent
    当前输入 + historySummary + recentMessages

SafetyAgent
    当前输入 + recentMessages + SafetyState + activeRiskEvents
    → currentRisk / carryoverRisk / effectiveRisk

ContextAgent
    snapshot + intent + effectiveRisk
    → memoryBrief / RAG / skill / 最终上下文

ResponseAgent
    主要消费ContextAgent产物，并使用相同安全版本
```

ContextAgent 不再负责 Redis/MySQL 底层加载和窗口维护，它只负责需要语义判断的上下文选择与组合。

## 11. 改造前后 Agent 流程比较

| 环节 | 当前 | 改造后 |
|---|---|---|
| 当前 user 消息保存 | Agent 运行后 | Agent 运行前，先拿 message ID |
| 记忆加载 | ContextAgent 较晚执行 | Runtime/Harness 调用 MemoryContextService 前置加载 |
| Blackboard | 创建时不含统一记忆快照 | 创建时直接携带不可变 MemorySnapshot |
| Understanding 输入 | 主要是当前输入 | 当前输入 + 必要历史 |
| Safety 输入 | 经常只有当前输入 | 当前输入 + recent + SafetyState |
| 风险合并 | 只有本轮 risk | current + carryover = effective |
| SAFETY_OVERRIDE | 本轮 HIGH | effective HIGH |
| ContextAgent | 首次加载历史并做 RAG | 不负责底层加载；消费 snapshot，聚合 RAG/skill |
| Response 模式 | CHAT+LOW / support | normal/support/monitored/safety |
| 回复后维护 | 只写 Redis 最近消息 | 写消息 + 滚动摘要维护 |

## 12. 改造前后存储比较

| 数据 | 当前 | 改造后 |
|---|---|---|
| 完整消息 | MySQL | 不变 |
| Redis 最近消息 | 最多 40 条 | 继续作为缓存，不承担长期记忆 |
| 长期摘要 | 无 | MySQL `chat_memory_summaries` |
| 摘要游标 | 无 | `coveredThroughMessageId` |
| 摘要版本 | 无 | schema version + data version |
| 风险报告 | PsychologicalReport | 保留 |
| 风险工单 | RiskCase | 保留，明确为工作流 |
| 风险事件流 | 无 | MySQL RiskEvent |
| 当前安全状态 | 无 | MySQL SafetyState |
| 安全状态缓存 | 无 | Redis safety-state key |

## 13. 改造前后 Prompt 比较

### 当前普通聊天 Prompt

```text
基础 system
ResponseAgent system + memoryBrief
规则“历史摘要”
最近8条
当前输入
```

问题：

- “历史摘要”通常重复最近消息；
- 当前输入可能同时出现在 memoryBrief 和 user message；
- 安全历史没有独立约束。

### 改造后 Prompt

```text
基础 system
本轮安全约束 system
持久结构化 historySummary system
RAG/skill system
最近原文
当前输入一次
```

优点：

- 每层职责明确；
- 可分别做 token 预算；
- 容易审计；
- 安全状态不会被普通摘要覆盖；
- 当前输入不会重复。

## 14. 回复模式比较

### 当前

```python
if intent == CHAT and risk == LOW:
    normal_chat
else:
    support
```

### 改造后

```python
if intent == CHAT and effective_risk == LOW and no_active_state:
    normal_chat
elif effective_risk == HIGH:
    safety_support
elif lifecycle_status == MONITORING:
    monitored_chat
else:
    support
```

| 模式 | 典型行为 |
|---|---|
| `normal_chat` | 直接、自然回答普通问题 |
| `support` | 共情、支持、非诊断建议 |
| `monitored_chat` | 简短确认安全后处理当前问题 |
| `safety_support` | 先处理当前安全，再考虑普通请求 |

## 15. 四个典型场景

### 15.1 短普通对话

```text
消息不足12条
SafetyState 不存在或已解除
```

结果：

- 不触发压缩；
- 使用最近原文；
- CHAT + LOW 进入 normal_chat。

### 15.2 长普通对话

```text
未覆盖消息达到12条
```

结果：

- 最早4条完整消息合并进 historySummary；
- 最近8条保留原文；
- 下轮从持久摘要恢复早期事实；
- Redis 清空也不会丢长期摘要。

### 15.3 HIGH 后普通 CHAT

```text
Turn 1：明确高风险
Turn 2：普通学习/编程问题
```

改造前可能结果：

```text
Turn 2 = CHAT + LOW → normal_chat
```

改造后：

```text
currentRisk = LOW
carryoverRisk = HIGH
effectiveRisk = HIGH
responseMode = safety_support
```

系统先做必要安全确认。历史 HIGH 不会仅因本轮话题普通而消失。

### 15.4 用户明确安全

```text
用户表示当前安全、有人陪伴、已经联系可信任的人
```

系统重新评估并可能：

```text
ACTIVE → MONITORING
```

之后进入 monitored_chat，而不是每轮重复完整危机模板。只有在满足产品/专业规则或人工确认后才进入 RESOLVED。

## 16. 状态与时间的正确关系

错误做法：

```text
最后一次HIGH过去24小时
→ 自动RESOLVED
```

正确做法：

```text
最后高风险时间
+ 当前输入
+ 是否明确确认安全
+ 是否存在保护因素
+ 是否有人工确认
+ 是否出现新风险
→ 状态复核
```

时间用于决定“是否需要复核”，不是单独证明“风险已经解除”。

### 16.1 简单理解降级原则

可以把降级理解为：

```text
从严重状态向下走的每一步，都必须先完成新的安全复核；
风险越严重，需要的复核次数、保护措施和人工参与越多。
```

建议默认阶梯：

| 降级 | 建议最低条件 |
|---|---|
| `ACTIVE/HIGH → MONITORING/MEDIUM` | 至少 2 次完整复核为 LOW，明确没有当前计划/意图/危险行为，并确认至少 1 个保护因素 |
| `MONITORING/MEDIUM → MONITORING/LOW` | 至少 2 次独立且连续的 LOW 安全复核，期间无新 MEDIUM/HIGH 信号 |
| `MONITORING/LOW → RESOLVED/LOW` | 再完成 1～2 次 LOW 安全复核，最近仍明确安全，且满足人工确认策略 |
| 任意状态 → `ACTIVE/HIGH` | 出现新 HIGH 证据时立即升级，不需要逐级上升 |
| `ACTIVE → FALSE_POSITIVE` | 明确确认是关键词误判或修辞表达，并保留纠正记录 |

这里的次数是工程初始建议，不是通用临床标准，必须配置化并由项目的产品和心理安全负责人确认。

### 16.2 什么算一次有效安全复核

只有明确检查并记录以下内容，才能增加 `safeReassessmentCount`：

- 当前是否仍有自杀、自伤或伤人想法；
- 是否有计划、意图、准备行为或可获得手段；
- 当前环境是否相对安全；
- 是否存在可信保护因素；
- 是否出现新的警示信号；
- 本次 SafetyAgent 的 currentRisk、理由和置信度；
- 严重事件是否已获得必要人工确认。

保护因素可以是：

- 有可信任的人陪伴；
- 已联系辅导员、心理中心或其他专业人员；
- 当前处于相对安全环境；
- 危险手段已经妥善隔离；
- 已有可执行的安全计划。

“有人陪伴”是常见保护因素之一，不是唯一条件。

### 16.3 普通聊天轮数不等于安全复核次数

必须分别保存：

```text
normalTurnCountSinceReview
    普通聊天轮数，只决定是否该再次询问

safeReassessmentCount
    真正完成安全复核的次数，才可用于降级

consecutiveLowAssessments
    连续得到 LOW 的有效复核次数
```

例如：

```text
连续聊了10轮编程
≠
完成了10次安全复核
```

普通聊天只能说明这些消息里没有发现新的风险表达，不能证明风险已经解除。

### 16.4 时间过去很久时怎么处理

建议保存：

```text
lastHighRiskAt
lastAssessedAt
lastSafetyConfirmationAt
nextReviewAt
reassessmentRequired
```

时间或普通轮数达到阈值时只执行：

```text
reassessmentRequired = true
```

不能直接执行：

```text
ACTIVE → RESOLVED
HIGH → LOW
```

用户很久后重新出现时：

```text
加载旧 RiskEvent 和 SafetyState
→ 标记旧评估已经需要更新
→ 先做简短安全确认
→ 根据新证据决定保持、降级、解除或纠正误判
```

这样既不会永久把用户锁定为 HIGH，也不会因为沉默一段时间就错误清除风险。

### 16.5 高严重度事件不能自动计数降级

如果历史事件包含明确计划、实施意图、可获得手段、准备行为或实际尝试：

- `humanReviewRequired=true`；
- 时间和普通轮数不能触发自动降级；
- 至少需要明确安全复核和人工/专业处置结果，才能进入 MONITORING；
- `RiskCase=ACKNOWLEDGED` 仍不自动等于风险解除。

状态转换后需要重置当前阶段的连续 LOW 计数，避免同一轮复核被同时用于 `HIGH→MEDIUM`、`MEDIUM→LOW` 和 `LOW→RESOLVED`。

## 17. Redis 在改造后的定位

### 17.1 可以缓存什么

- 最近聊天消息；
- historySummary 的可重建副本；
- SafetyState；
- Agent 私有运行记忆；
- 可选的短期 session lock。

### 17.2 不能只放 Redis 的内容

- 唯一一份长期摘要；
- 唯一一份 RiskEvent；
- 唯一一份 ACTIVE SafetyState；
- 唯一一份人工状态变更记录。

### 17.3 TTL 的含义

Redis TTL 只表示缓存什么时候回源，不表示：

- 会话数据删除；
- 风险状态解除；
- 摘要失效；
- 用户已经安全。

## 18. MySQL 在改造后的定位

MySQL 是权威来源：

```text
chat_messages
chat_memory_summaries
risk_events
safety_states
psychological_reports
risk_cases
alert_records
agent_run_traces
memory_maintenance_jobs
```

当 Redis 数据与 MySQL 冲突时，以 MySQL 的较新 `version` 为准。

## 19. 为什么需要数据库迁移

当前项目启动时使用 `Base.metadata.create_all()`。

它适合：

- 空数据库初始化；
- 创建不存在的新表。

它不适合：

- 给已有表可靠增加字段；
- 修改约束；
- 变更 enum；
- 生产环境回滚；
- 审计数据库版本。

项目现在通过 `COLUMN_MIGRATIONS` 提供幂等的既有表列/索引迁移，并由 `create_all()` 创建新表；这属于轻量等价机制。正式生产环境仍建议迁移到 Alembic，以获得完整的回滚和数据库版本审计能力。只修改 SQLAlchemy entity 并重启依然不等于旧数据库已经正确升级。

## 20. 新维护者应该如何阅读代码

建议按以下顺序：

1. `app/services/chat.py`
   - 看 SSE 如何开始、输出和结束。

2. `app/agents/harness.py`
   - 看一轮请求如何解析会话、调用 runtime、保存消息、创建报告和调度工具。

3. `app/agents/event_driven_runtime.py`
   - 看 Agent 如何注册、黑板如何创建、结果如何转换。

4. `app/agents/coordinator.py`
   - 看任务如何派生、优先级和最终采纳。

5. `app/agents/autonomous.py`
   - 看每个 Agent 的 decide/act 和 Prompt payload。

6. `app/services/memory.py`
   - 看 Redis 短期记忆和当前错误压缩。

7. `app/services/assessment.py`
   - 看硬规则、模型评估和 fallback。

8. `app/models/entities.py`
   - 看数据模型和风险工单。

9. `app/services/tool_queue.py`、`app/services/tools.py`
   - 看报告如何触发个案和告警。

10. `tests/`
    - 看当前行为契约和需要扩展的多轮场景。

## 21. 改造后建议新增的模块

| 模块 | 职责 |
|---|---|
| `MemoryContextService` | 在 Runtime 启动 Coordinator 前加载记忆并构造不可变 MemorySnapshot |
| `RollingSummaryService` | 结构化摘要、游标、版本、8/12 和 token 压缩 |
| `SafetyMemoryService` | RiskEvent、SafetyState、Redis cache、状态转换 |
| `MemoryMaintenanceQueue/Worker` | assistant 保存后的可靠异步 12→8 维护、重试和重启恢复 |
| `PromptBudgetService` | token 估算和高水位处理 |
| `SafetyPolicyRenderer` | 将 effectiveRisk 转成不泄露内部标签的 Prompt 约束 |

不新增 MemoryAgent。上述服务可以先放在 `app/services/memory.py` 等现有模块中逐步拆分，但最终应避免一个文件同时承担 Redis、摘要、风险状态、Prompt 和数据库事务。

## 22. 调试一轮请求时看什么

### 22.1 普通记忆问题

检查：

```text
ChatMessage 是否完整保存
ChatMemorySummary.coveredThroughMessageId
summary version
未覆盖消息数量
是否触发 message/token compaction
摘要模型输出是否通过 schema 校验
Prompt 是否重复当前输入
```

### 22.2 风险连续性问题

检查：

```text
当前 message_id
Redis SafetyState 是否命中
MySQL SafetyState version
最近 RiskEvent
currentRisk
carryoverRisk
effectiveRisk
lifecycle status
response mode
SAFETY_OVERRIDE 是否发布
```

### 22.3 Agent 顺序问题

检查黑板事件：

```text
Runtime 是否在Coordinator之前加载MemorySnapshot
Blackboard是否在初始化时携带snapshot
MEMORY_SNAPSHOT_LOADED事件是否最先记录
intent/risk 是否在 snapshot 后产生
context 是否在 intent/risk 后产生
response 是否使用正确 safety version
review 是否审查同一个 response artifact
```

## 23. 不要这样实现

以下做法会重新引入相同问题：

1. 继续把最近 4+3 条叫作旧历史摘要；
2. 每轮重新总结全部聊天记录；
3. 只在 Redis 保存摘要或 ACTIVE 状态；
4. 只要当前输入是 CHAT 就忽略历史风险；
5. 永久使用历史最高风险约束用户；
6. 仅按时间自动解除 HIGH；
7. 把 RiskCase ACKNOWLEDGED 当成用户风险解除；
8. 让 SafetyAgent 在 memory snapshot 前执行；
9. 只修改 Prompt，不增加持久状态和测试；
10. 让普通摘要成为风险事件唯一副本；
11. 把内部风险等级和状态名直接告诉学生；
12. 依赖 `create_all()` 修改已有生产表；
13. 摘要失败后仍推进覆盖游标；
14. 压缩时切断完整 user/assistant 轮次；
15. 在模型调用期间长期持有数据库行锁。

## 24. 测试和评测入口

当前基础测试使用标准库 `unittest`：

```bash
python -m unittest discover -s tests
```

当前环境需要先安装 `requirements.txt` 中的数据库和 Redis 驱动，否则导入阶段可能因为 `pymysql` 等依赖缺失而失败。

改造后重点测试文件建议：

```text
tests/test_rolling_memory.py
tests/test_prompt_budget.py
tests/test_safety_memory.py
tests/test_safety_state_machine.py
tests/test_agent_memory_dependencies.py
tests/test_multiturn_risk_continuity.py
```

风险评测不能只用单轮数据，需要加入：

```text
HIGH → CHAT
HIGH → 明确安全
MONITORING → 新HIGH
RESOLVED → CHAT
风险消息离开最近8条
Redis miss
摘要模型失败
```

## 25. 运维和管理端变化

改造后管理端应能看到：

- 当前 SafetyState；
- RiskEvent 时间线；
- 最近一次 HIGH；
- 最近一次复核；
- 状态变化原因和来源；
- RiskCase 工单状态；
- historySummary 版本和覆盖位置；
- Redis 是否命中；
- 当前 response mode。

但学生端不得看到：

- `HIGH/MEDIUM/LOW` 后台标签；
- `ACTIVE/MONITORING` 内部状态名；
- 置信度和风险分数；
- 管理员、处理人或工单详情；
- 内部摘要 Prompt。

## 26. 改造实施顺序速查

```text
第一步：数据库迁移与新实体
    ↓
第二步：RollingSummaryService + SafetyMemoryService
    ↓
第三步：MemoryContextService + Blackboard MemorySnapshot
    ↓
第四步：调整 Runtime 前置加载和 Coordinator 任务依赖
    ↓
第五步：Understanding/Safety 使用历史
    ↓
第六步：effectiveRisk 和四种 response mode
    ↓
第七步：Harness 调整保存顺序
    ↓
第八步：12→8 和 token 兜底
    ↓
第九步：管理端、trace、指标
    ↓
第十步：shadow、灰度、正式启用
```

## 27. 接手检查清单

新维护者接手时，先回答以下问题：

- 当前运行的是哪个 Agent framework？
- user 消息是在 runtime 前还是后保存？
- Runtime 是否在 Coordinator 前调用 MemoryContextService？
- MemorySnapshot 是否已经实现？
- MemorySnapshot 是否是 Blackboard 固定只读字段，而非 MemoryAgent artifact？
- 是否错误地引入了 MemoryAgent、MEMORY capability 或 load-memory 任务？
- historySummary 是否持久化？
- `coveredThroughMessageId` 是否存在？
- 当前压缩阈值和目标分别是多少？
- token 高水位是否基于真实模型容量？
- SafetyState 的 scope 是 user 还是 session？
- SafetyState 是否以 MySQL 为权威？
- Redis miss 是否能恢复 ACTIVE？
- `effectiveRisk` 是否被 ResponseAgent 和 fallback 使用？
- ACTIVE → MONITORING/RESOLVED 的规则是谁确认的？
- 普通轮数是否只触发复核、没有直接触发降级？
- `safeReassessmentCount` 与 `normalTurnCount` 是否分开？
- 明确计划/意图/行为是否强制人工确认？
- 管理端是否支持人工纠错和解除？
- 多轮 HIGH → CHAT 测试是否通过？
- 数据库是否通过迁移升级，而非只运行 create_all？

如果其中任何一项无法回答，应先查看配套改造规格和相关 trace，不要直接修改 Prompt 猜测行为。

## 28. 一句话理解改造前后

改造前：

> 系统保存了完整聊天和风险报告，但模型主要看到最近消息；风险判断偏向每轮重新开始。

改造后：

> 系统用持久滚动摘要保持长期对话，用独立安全状态保持未解除风险，并在每轮推理前同时恢复这两类记忆。
