# MindBridge 上下文压缩与安全事件记忆改造规格

> 文档状态：设计规格，尚未实现  
> 目标读者：负责下一阶段实现、评审、测试和上线的开发者或 Agent  
> 适用仓库：`ReplyU`（GitHub: Campus-Youying-）  
> 核心目标：修复旧历史被遗忘、跨轮风险无法继承、风险记录只写不读的问题  
> 配套接手指南：[`context-memory-before-after-guide.md`](./context-memory-before-after-guide.md)

## 1. 文档目的

本文给出一套可以直接据此实施的完整改造方案，覆盖：

- 普通对话的结构化持久滚动摘要；
- 最近原文窗口与批量压缩；
- token 高水位兜底；
- 风险事件与当前安全状态的独立持久化；
- MySQL 与 Redis 的职责边界；
- `currentRisk / carryoverRisk / effectiveRisk` 的计算；
- `ACTIVE / MONITORING / RESOLVED / FALSE_POSITIVE` 状态机；
- Agent 任务依赖、Prompt 拼装、失败回退；
- 数据库迁移、并发一致性、测试和分阶段上线。

本文描述的是目标设计，不代表当前代码已经具备这些能力。实施者必须以本文的“目标状态”和“验收标准”为准，不能只修改摘要函数后就认为任务完成。

## 2. 当前问题

### 2.1 当前所谓“历史摘要”没有摘要窗口外历史

当前 `app/services/memory.py` 中：

1. `compact_history_for_prompt()` 先对全部历史调用 `summarize_history_for_memory()`；
2. `summarize_history_for_memory()` 只取最近 4 条用户消息、最近 3 条助手消息和当前输入；
3. 历史超过默认 8 条时，又保留最近 8 条原文；
4. 最终得到“最近消息的摘录 + 最近消息原文”，窗口外历史没有被摘要。

在标准的 user/assistant 交替对话中，摘要中的 4 条用户消息和 3 条助手消息几乎全部位于最近 8 条原文内。因此旧消息虽然仍在 MySQL，模型却无法继续使用它们。

### 2.2 当前没有持久滚动摘要

压缩结果只用于当前一轮 Prompt：

- 不持久化摘要；
- 不记录摘要覆盖到哪条消息；
- 下一轮重新计算；
- Redis 最多加载最近 40 条；
- MySQL 回退也只读取最后 40 条。

因此超过 Redis 窗口的历史，即使数据库仍有原文，也不会再进入模型上下文。

### 2.3 当前风险判断缺少跨轮连续性

当前每轮创建新的协作黑板。上一轮的 `intent artifact`、`risk artifact` 和 `SAFETY_OVERRIDE` 不会自动进入下一轮。

当前任务优先级还会导致 SafetyAgent 通常早于 ContextAgent 执行。SafetyAgent 调用 `_context_history()` 时，如果 ContextAgent 尚未发布 context artifact，就只能看到当前输入。

典型错误场景：

```text
上一轮：用户表达明确自杀意图 → HIGH
下一轮：用户请求普通编程帮助 → CHAT + LOW
```

系统可能直接进入 `normal_chat`。上一轮原文虽然可能仍在最近消息里，但它只作为普通文本被 ResponseAgent 偶然看到，不会作为“尚未解除的安全状态”强制影响本轮决策。

### 2.4 风险记录已经保存，但没有成为模型记忆

当前项目已有：

- `PsychologicalReport`：保存咨询或风险评估；
- `RiskCase`：保存中高风险个案；
- `AlertRecord`、`ToolJob`：负责告警和异步工具；
- SafetyAgent 私有 Redis 记忆：保存简短运行记录。

但 ContextAgent 和 SafetyAgent 下一轮不会从 `PsychologicalReport` 或 `RiskCase` 恢复“当前安全状态”。这些数据主要用于报告、后台、告警和审计，而不是跨轮安全推理。

### 2.5 现有 `RiskCase.status` 不是风险生命周期

当前 `RiskCaseStatus` 只有：

```text
OPEN
ALERT_SENT
ACKNOWLEDGED
```

这些状态描述工单是否创建、告警是否发送、是否有人接手。`ACKNOWLEDGED` 不等于用户已经安全，不能拿它替代风险生命周期。

## 3. 设计原则

### 3.1 原始数据、普通记忆和安全记忆分层

必须拆成三层：

```text
完整原始消息
    MySQL 权威保存，不因压缩删除

普通对话记忆
    结构化滚动摘要 + 最近原文

安全事件记忆
    独立 RiskEvent + SafetyState，不受普通压缩影响
```

### 3.2 Redis 只做缓存，不做权威存储

Redis 可以提升每轮加载速度，但 Redis 有 TTL、可能重启、可能不可用。因此：

- MySQL 是普通摘要、安全事件和安全状态的权威来源；
- Redis 保存可重建的缓存；
- Redis miss 时必须回退 MySQL；
- Redis 写失败不能导致 MySQL 已提交的数据回滚；
- 安全状态不能因为 Redis 过期而消失。

### 3.3 继承“最高未解除风险”，不是“历史最高风险”

曾经出现 HIGH 不代表永远 HIGH。后续约束应取：

```text
当前风险
历史 ACTIVE/MONITORING 风险产生的风险下限
```

`RESOLVED` 和 `FALSE_POSITIVE` 不再产生风险下限。

### 3.4 时间只触发复核，不单独解除风险

“超过若干小时没有消息”不能证明用户安全。时间可以决定：

- 是否需要重新询问当前安全；
- 是否到达复核时间；
- 是否降低告警展示优先级；
- 是否需要人工跟进。

但不得仅根据时间执行 `ACTIVE → RESOLVED`。

### 3.5 压缩以完整轮次为单位

“8 条消息”约等于 4 轮 user/assistant 交互。压缩时尽量移动完整轮次，不能留下孤立的 assistant 消息，也不能把尚未得到回复的当前 user 消息压入摘要。

### 3.6 安全约束高于普通意图

当：

```text
intent = CHAT
currentRisk = LOW
carryoverRisk = HIGH
```

本轮仍必须执行 HIGH 对应的安全策略。意图分类不能覆盖未解除的安全约束。

## 4. 术语和目标状态

### 4.1 普通记忆术语

| 术语 | 含义 |
|---|---|
| `historySummary` | 对最近原文窗口之前的历史所做的持久结构化摘要 |
| `recentMessages` | 尚未被摘要覆盖的最近原始消息 |
| `memoryBrief` | 当前轮为检索、规划和审计生成的临时概览，不承担历史压缩 |
| `coveredThroughMessageId` | `historySummary` 已覆盖到的最后一条消息 ID |
| `targetMessages` | 压缩后希望保留的原始消息数，默认 8 |
| `triggerMessages` | 触发批量压缩的未覆盖消息数，默认 12 |

### 4.2 安全记忆术语

| 术语 | 含义 |
|---|---|
| `RiskEvent` | 不可变的风险检测、复核、状态变化或人工覆盖事件 |
| `SafetyState` | 当前有效安全状态的可变投影 |
| `currentRisk` | 仅针对本轮输入和本轮上下文的风险评估 |
| `carryoverRisk` | 未解除历史安全状态产生的风险下限 |
| `effectiveRisk` | 决定本轮回复约束的最终风险等级 |

### 4.3 风险生命周期

| 状态 | 含义 | 是否产生风险约束 |
|---|---|---|
| `ACTIVE` | 风险尚未解除 | 是，继承事件风险等级 |
| `MONITORING` | 即时危险有所缓解但仍需关注 | 是，至少应用监测策略；默认风险下限为 MEDIUM |
| `RESOLVED` | 已有充分证据或人工确认风险解除 | 否 |
| `FALSE_POSITIVE` | 确认为误判 | 否 |

生命周期与 `RiskCaseStatus` 必须分开。前者描述用户安全状态，后者描述工单处理进度。

## 5. 目标架构

```text
用户输入
  ↓
先保存用户消息，获得 message_id
  ↓
MemoryContextService 前置加载并校验记忆
  ├─ MySQL 持久 historySummary
  ├─ 未覆盖 recentMessages
  ├─ Redis SafetyState 缓存
  └─ MySQL SafetyState / RiskEvent 回退
  ↓
必要时执行 token 高水位或未完成批量压缩兜底
  ↓
生成不可变 MemorySnapshot
  ↓
用 MemorySnapshot 初始化 CollaborationBlackboard
并记录 MEMORY_SNAPSHOT_LOADED 事件
  ↓
Coordinator 创建本轮 Agent 任务
  ↓
UnderstandingAgent ─────┐
                       ├─ 并行，均依赖 MemorySnapshot
SafetyAgent ────────────┘
  ↓
计算 currentRisk / carryoverRisk / effectiveRisk
  ↓
持久化 RiskEvent 和 SafetyState，刷新 Redis
  ↓
ContextAgent 聚合历史摘要、最近原文、安全约束、RAG、skill
  ↓
ResponseAgent 按 intent + effectiveRisk 生成候选回复
  ↓
SafetyAgent 审查候选回复
  ↓
Coordinator 采纳
  ↓
SSE 只输出已批准正文，并同步保存完整 assistant 消息
  ↓
写入 memory_maintenance_jobs，后台执行 12→8；下一轮前同步执行 token 高水位兜底
```

## 6. 普通对话滚动摘要设计

### 6.1 权威存储

继续使用 `chat_messages` 保存完整原文。压缩不得删除或覆盖原消息。

新增 `chat_memory_summaries` 表，建议字段：

| 字段 | 类型建议 | 说明 |
|---|---|---|
| `id` | Integer PK | 主键 |
| `user_id` | FK + index | 用户 |
| `session_id` | FK + unique/index | 每个会话一条摘要投影 |
| `summary_json` | Text | 结构化 JSON；为兼容测试数据库建议先用 Text |
| `summary_text` | Text | 可选，人类可读投影；也可运行时从 JSON 渲染 |
| `covered_through_message_id` | Integer nullable/index | 已覆盖到的消息 ID |
| `summary_schema_version` | Integer | 摘要结构版本 |
| `version` | Integer | 乐观并发版本 |
| `created_at` | DateTime | 创建时间 |
| `updated_at` | DateTime | 更新时间 |

建议约束：

- `session_id` 唯一；
- `covered_through_message_id` 只能单调递增；
- 更新摘要时 `version += 1`；
- 摘要生成失败时不能推进 `covered_through_message_id`。

### 6.2 结构化摘要格式

建议使用固定 JSON 结构，而不是自由文本：

```json
{
  "stableFacts": [],
  "preferences": [],
  "ongoingConcerns": [],
  "changes": [],
  "supportTried": [
    {
      "action": "",
      "result": ""
    }
  ],
  "unresolvedItems": [],
  "safetyReferences": [
    {
      "eventId": 42,
      "status": "MONITORING"
    }
  ]
}
```

约束：

- 只保留用户明确表达或可靠推导的信息；
- 推测必须标注为不确定，不能写成事实；
- 不输出诊断；
- 不在普通摘要中复制大量敏感风险原文；
- 安全事实只保存事件引用和必要概述，权威来源仍是 RiskEvent/SafetyState；
- 删除闲聊噪声和已经无关的细节；
- 保留未解决事项、时间变化、尝试过的建议及效果。

### 6.3 8/12 批量窗口

默认配置：

```text
targetMessages = 8
triggerMessages = 12
```

当未被摘要覆盖的完整消息达到 12 条时：

```text
旧 summary
+ 最早 4 条完整消息（通常 2 个完整轮次）
    ↓
新 summary

recentMessages = 最近 8 条原文
```

不要求每轮调用摘要模型。达到触发阈值时才批量更新。

### 6.4 增量压缩算法

伪代码：

```python
def compact_if_needed(session_id: int) -> CompactionResult:
    state = load_summary_state(session_id)
    uncovered = load_messages_after(
        session_id=session_id,
        message_id=state.covered_through_message_id,
    )

    complete_messages = exclude_incomplete_tail(uncovered)
    token_pressure = estimate_full_prompt_tokens(...) >= high_watermark

    if len(complete_messages) < trigger_messages and not token_pressure:
        return unchanged(state)

    evicted, remaining = split_oldest_complete_turns(
        complete_messages,
        target_messages=target_messages,
        target_tokens=target_tokens_if_needed,
    )

    if not evicted:
        return unchanged(state)

    new_summary = merge_summary(
        previous_summary=state.summary_json,
        newly_evicted_messages=evicted,
    )

    validate_summary(new_summary)

    compare_and_swap_summary(
        expected_version=state.version,
        summary_json=new_summary,
        covered_through_message_id=evicted[-1].id,
    )

    return changed(new_summary, remaining)
```

关键点：

- 输入是“旧摘要 + 本次新移出窗口的消息”；
- 不能重新摘要全部历史；
- 不能用最近 4+3 条替代旧历史；
- 当前尚未回复的 user 消息不参与常规批量压缩；
- 压缩以完整轮次为单位；
- 更新必须校验结构化 JSON；
- 只有摘要持久化成功后才推进覆盖游标。

### 6.5 摘要 Prompt 要求

建议摘要系统提示至少包含：

```text
你负责把“已有累计摘要”和“新移出近期窗口的消息”合并为新的累计摘要。
只输出符合指定 schema 的 JSON。
不得虚构；不得诊断；不得把推测写成事实。
保留稳定背景、持续问题、变化、已尝试支持及效果、未解决事项。
风险信息只保留 risk event 引用和必要概述，不得删除已有未解除引用。
```

必须对输出做：

- JSON 解析；
- schema 校验；
- 字段长度限制；
- 总字符/token 限制；
- 敏感信息脱敏；
- 失败时回退且不推进游标。

### 6.6 `memoryBrief` 与 `historySummary` 分离

现有 `memoryBrief` 可以保留，但职责改为：

- 为本轮 RAG 查询改写提供近期概览；
- 为 ResponseAgent 提供当前关注点；
- 写入 trace 便于审计。

它可以根据：

```text
historySummary
+ recentMessages
+ currentInput
```

生成，但不能作为窗口外历史的唯一副本，也不能被标成权威“历史摘要”。

### 6.7 token 高水位兜底

消息条数不能代表上下文长度。需要增加 token 预算：

```text
contextWindowTokens
reservedOutputTokens
fixedPromptTokens
ragAndSkillBudgetTokens
memoryBudgetTokens
```

建议用比例配置，而不是在代码中写死绝对值：

```text
高水位：预计输入达到模型上下文容量的 60%～70%
压缩目标：降低到 35%～45%
```

实际值必须结合本地模型上下文窗口、系统提示、RAG 和输出预留确定。

处理规则：

1. 正常情况下按 12→8 批量压缩；
2. 调用模型前估算完整 Prompt token；
3. 超过高水位时，继续移动最早的完整轮次到摘要；
4. 重新估算；
5. 仍超限时减少 RAG 片段或截断单条超长消息；
6. 所有降级都要记录日志和 trace。

### 6.8 压缩执行时机

推荐两个入口：

1. **常规维护**：assistant 回复保存后，异步或同步执行 12→8；
2. **强制兜底**：下一轮模型调用前执行 `ensure_prompt_budget()`。

如果常规异步任务尚未完成，下一轮的强制兜底必须保证上下文仍可用。

### 6.9 并发和幂等

同一会话可能快速提交多个请求。摘要更新要防止两个 worker 同时覆盖：

- 使用 `version` 做乐观锁；
- `UPDATE ... WHERE version = expected_version`；
- 更新失败则重新加载、重算或放弃本次后台更新；
- `covered_through_message_id` 只向前移动；
- 不在等待 LLM 时长期持有数据库事务锁；
- 可选：增加短期 Redis session lock，但不能把正确性只建立在锁上。

## 7. 安全事件记忆设计

### 7.1 数据模型一：RiskEvent

新增不可变 `risk_events` 表。建议字段：

| 字段 | 说明 |
|---|---|
| `id` | 主键 |
| `user_id` | 用户 ID，建立索引 |
| `session_id` | 发生事件的会话 ID，建立索引 |
| `message_id` | 关联原始用户消息，建立索引 |
| `report_id` | 可选，关联 PsychologicalReport |
| `event_kind` | `RISK_DETECTED / REASSESSMENT / STATE_TRANSITION / MANUAL_OVERRIDE` |
| `risk_type` | `SUICIDE / SELF_HARM / HARM_TO_OTHERS / IMMEDIATE_DANGER / OTHER` |
| `risk_level` | `LOW / MEDIUM / HIGH` |
| `confidence` | 评估置信度 |
| `evidence_summary` | 脱敏后的必要证据，不复制不必要的完整原文 |
| `source` | `RULE / MODEL / HUMAN` |
| `previous_status` | 状态变化前，可空 |
| `new_status` | 状态变化后，可空 |
| `created_at` | 事件时间 |

RiskEvent 记录“发生过什么”，原则上只追加、不原地改写。人工纠正通过新增 `MANUAL_OVERRIDE` 或 `STATE_TRANSITION` 事件完成。

### 7.2 数据模型二：SafetyState

新增 `safety_states` 表，作为当前状态投影。建议字段：

| 字段 | 说明 |
|---|---|
| `id` | 主键 |
| `user_id` | 默认按用户维度唯一，建立 unique/index |
| `latest_session_id` | 最近相关会话 |
| `status` | `ACTIVE / MONITORING / RESOLVED / FALSE_POSITIVE` |
| `effective_risk` | 当前历史风险下限 |
| `latest_event_id` | 最近风险事件 |
| `last_high_risk_at` | 最近 HIGH 时间 |
| `last_assessed_at` | 最近重新评估时间 |
| `last_safety_confirmation_at` | 最近一次有效安全确认时间 |
| `next_review_at` | 可选，下次应复核时间 |
| `reassessment_required` | 是否必须在继续普通回复前重新评估 |
| `normal_turn_count_since_review` | 上次安全复核后的普通对话轮数；只用于安排复核 |
| `safe_reassessment_count` | 已完成的有效安全复核次数 |
| `consecutive_low_assessments` | 连续 LOW 安全复核次数 |
| `latest_event_severity` | 最近事件证据强度，如关键词疑似、想法、计划/意图、实际行为 |
| `human_review_required` | 是否必须人工确认才能继续降级 |
| `resolution_reason` | 解除或误判原因 |
| `version` | 乐观锁版本 |
| `created_at` | 创建时间 |
| `updated_at` | 更新时间 |

默认建议按 `user_id` 维护 SafetyState，使未解除风险不会因为用户新建会话而消失。RiskEvent 仍保留原始 `session_id`。如果产品明确要求只在同一会话内继承，需要通过配置和产品评审决定，不能在实现中隐式选择。

### 7.3 与现有表的关系

| 现有/新增对象 | 职责 |
|---|---|
| `ChatMessage` | 完整原始消息 |
| `PsychologicalReport` | 某一轮咨询/风险评估报告 |
| `RiskEvent` | 风险事实、复核和状态变化的不可变事件流 |
| `SafetyState` | 当前有效安全状态 |
| `RiskCase` | 人工/工具处理工单 |
| `AlertRecord` | 告警发送记录 |

`RiskCaseStatus` 与 `SafetyLifecycleStatus` 不得复用同一个 enum。建议为 RiskCase 增加 `RESOLVED/CLOSED` 工单状态，但这仍不替代 SafetyState。

### 7.4 Redis 安全状态缓存

建议 key：

```text
mindbridge:safety-state:user:{user_id}
```

缓存内容示例：

```json
{
  "status": "ACTIVE",
  "effectiveRisk": "HIGH",
  "latestEventId": 42,
  "lastHighRiskAt": "2026-09-20T21:30:00",
  "lastAssessedAt": "2026-09-20T21:30:00",
  "version": 3
}
```

读取策略：

```text
Redis hit → 校验 schema/version 后使用
Redis miss/坏数据/超时 → 查询 MySQL → 回填 Redis
MySQL 无状态 → 返回安全默认状态，不凭空创建历史风险
```

写入策略：

```text
先提交 MySQL RiskEvent/SafetyState
再刷新或删除 Redis key
```

缓存可以设置 TTL，但 TTL 到期只意味着缓存失效，不意味着风险解除。

### 7.5 状态转换

推荐状态机：

```text
任意状态
  └─ 新的明确 HIGH 风险
       → ACTIVE/HIGH

ACTIVE
  └─ 用户明确确认当前安全，存在可信保护因素，且重新评估支持降级
       → MONITORING

MONITORING
  └─ 连续评估稳定或人工确认处理完成
       → RESOLVED

ACTIVE 或 MONITORING
  └─ 确认模型误判
       → FALSE_POSITIVE

RESOLVED 或 FALSE_POSITIVE
  └─ 出现新的风险表达
       → 创建新 RiskEvent，回到 ACTIVE
```

禁止规则：

- 仅因经过固定时间自动 `ACTIVE → RESOLVED`；
- 当前一句是普通 CHAT 就自动解除 ACTIVE；
- `RiskCase = ACKNOWLEDGED` 就自动 `SafetyState = RESOLVED`；
- 用历史最高风险永久约束用户。

### 7.6 分级降级策略

风险等级、生命周期和情绪状态必须分开：

```text
风险等级：HIGH / MEDIUM / LOW
生命周期：ACTIVE / MONITORING / RESOLVED / FALSE_POSITIVE
情绪状态：焦虑 / 低落 / 正常等
```

“焦虑”不是风险降级层级。用户可以处于焦虑情绪但风险为 LOW，也可能表面平静但仍存在 HIGH 风险。

总体原则：

```text
升级可以快速、越级；
降级必须逐步、经过有效安全复核并有明确证据；
风险越严重，降级所需证据越多；
普通聊天轮数和时间经过本身不是安全证据。
```

#### 7.6.1 什么叫有效安全复核

一次复核只有在以下信息被明确评估并持久化后，才能计入 `safe_reassessment_count`：

- 当前是否仍有自杀、自伤或伤人想法；
- 是否存在当前计划、意图、准备行为或可获得手段；
- 当前是否处于相对安全环境；
- 是否存在可信保护因素，例如可信任的人陪伴、已经联系专业人员、可执行的安全计划；
- 是否出现新的警示信号；
- SafetyAgent 对本次复核给出的 currentRisk、置信度和理由；
- 必要时的人工确认结果。

普通的编程、天气、课程或闲聊回合不能计为有效安全复核。`normal_turn_count_since_review` 只能用于决定何时再次发起复核。

#### 7.6.2 建议的默认降级阶梯

以下数字是项目的初始工程默认值，正式上线前必须由产品和心理安全负责人确认并配置化。

| 状态变化 | 最低条件 |
|---|---|
| `ACTIVE/HIGH → MONITORING/MEDIUM` | 至少 2 次完整安全复核为 LOW；明确没有当前计划、意图或危险行为；至少存在 1 个可信保护因素；没有新警示信号 |
| `MONITORING/MEDIUM → MONITORING/LOW` | 在进入 MONITORING 后至少 2 次独立、连续的 LOW 安全复核；期间没有新的 MEDIUM/HIGH 信号；保护措施仍有效 |
| `MONITORING/LOW → RESOLVED/LOW` | 再完成 1～2 次 LOW 安全复核；最近一次仍明确确认当前安全；不存在未处理的高风险个案；满足人工确认策略 |
| 任意状态 → `ACTIVE/HIGH` | 出现新的明确计划、意图、准备行为、实际行为或其他 HIGH 证据，立即升级，不要求逐级上升 |
| `ACTIVE → FALSE_POSITIVE` | 明确确认属于关键词误判、修辞表达或错误分类，并记录纠正证据和来源 |

对存在明确计划、实施意图、可获得手段或实际尝试的事件：

- 不允许依靠自动计数降级；
- `human_review_required=true`；
- 至少需要明确安全复核和人工/专业处置结果，才能进入 MONITORING；
- 时间经过和普通聊天不能绕过这一门槛。

“有人陪伴”是保护因素之一，不是唯一合法条件。其他保护因素可以包括已经联系可信任的人或专业人员、当前处于相对安全环境、危险手段已被妥善隔离、已有可执行安全计划等。具体可接受证据必须由产品和专业人员确认。

每次状态转换都要追加 RiskEvent，记录：

```text
previousStatus
newStatus
currentRisk
effectiveRisk
safeReassessmentCount
consecutiveLowAssessments
protectiveFactors
humanReviewed
reason
```

状态转换后，应重置“当前阶段”的连续复核计数，避免同一批评估结果被重复用于多个降级阶段。

### 7.7 时间和普通对话轮数的作用

时间和普通轮数只负责安排复核，不直接改变风险等级或生命周期：

```python
if now >= next_review_at:
    reassessment_required = True

if normal_turn_count_since_review >= configured_review_turns:
    reassessment_required = True
```

禁止实现：

```python
if days_since_high > 7:
    status = RESOLVED

if normal_turn_count > 10:
    effective_risk = LOW
```

如果用户在很长时间后重新出现，而旧事件仍是 ACTIVE 且从未完成复核：

1. 保留历史 RiskEvent；
2. 设置 `reassessment_required=true`；
3. 本轮不能直接进入完全无约束的 normal_chat；
4. 先执行简短、不过度重复的安全确认；
5. 根据新的有效安全复核决定 ACTIVE、MONITORING、RESOLVED 或 FALSE_POSITIVE；
6. 不能仅因旧事件时间久远继续机械重复完整危机模板，也不能自动当作已解除。

建议把“风险约束”和“回复侵入程度”分开：ACTIVE 仍保持安全门槛，但在没有新 HIGH 信号时可以使用简短确认；只有出现新的急性证据时才重复完整高风险处置内容。

### 7.8 风险合并算法

```python
def carryover_risk(state: SafetyState | None) -> RiskLevel:
    if state is None:
        return RiskLevel.LOW
    if state.status == ACTIVE:
        return state.effective_risk
    if state.status == MONITORING:
        return max_level(RiskLevel.MEDIUM, configured_monitoring_floor())
    return RiskLevel.LOW


effective_risk = max_level(
    current_risk,
    carryover_risk(previous_state),
)
```

SafetyAgent artifact 至少输出：

```json
{
  "currentRisk": "LOW",
  "carryoverRisk": "HIGH",
  "effectiveRisk": "HIGH",
  "previousStatus": "ACTIVE",
  "newStatus": "ACTIVE",
  "reason": "本轮无新高风险词，但历史HIGH尚未解除"
}
```

### 7.9 回复模式

| 模式 | 条件 | 策略 |
|---|---|---|
| `normal_chat` | CHAT + effective LOW + 无未解除事件 | 正常回答 |
| `support` | CONSULT 或 effective MEDIUM | 共情、支持、非诊断 |
| `monitored_chat` | SafetyState = MONITORING，当前无高风险 | 简短确认安全，再处理当前请求 |
| `safety_support` | effective HIGH | 当前安全优先，执行高风险规则 |

当 `intent=CHAT` 但 `effectiveRisk=HIGH` 时，必须进入 `safety_support`，不得进入 `normal_chat`。

高风险模式可以在安全优先得到满足后处理用户的普通问题，但普通回答不能掩盖、稀释或跳过必要的安全确认。

## 8. 新的 Agent 执行流程

### 8.1 Runtime 前置加载 MemorySnapshot

不新增 `MemoryAgent`，也不新增 `AgentCapability.MEMORY`。记忆加载、缓存回源、窗口检查、token 预算和摘要持久化都属于确定性基础设施，不需要 Agent 自主认领任务。

由 Harness/Runtime 在多 Agent 协作开始前同步调用：

```python
snapshot = memory_context_service.load_snapshot(
    user_id=user.id,
    session_id=session.id,
    current_message_id=current_message.id,
)
```

MemorySnapshot 建议定义为不可变 DTO：

```json
{
  "historySummary": {},
  "coveredThroughMessageId": 120,
  "recentMessages": [],
  "safetyState": {},
  "activeRiskEvents": [],
  "summaryVersion": 4,
  "safetyStateVersion": 3
}
```

实现时建议使用 frozen dataclass/Pydantic immutable model，并把 SQLAlchemy entity 转换为纯 DTO。不能把绑定数据库 Session 的 ORM 对象直接塞入 Blackboard，避免延迟加载、Session 关闭失效和 Agent 意外修改持久对象。

MemorySnapshot 表示“本轮 Agent 开始时统一看到的状态”。一轮内不原地修改。SafetyAgent 在本轮产生的新判断通过新的 risk artifact 和状态转换事件表示，持久化后供下一轮 snapshot 读取。

### 8.2 放入 CollaborationBlackboard

推荐把 MemorySnapshot 作为 Blackboard 固定字段，而不是伪装成某个 Agent 的产物：

```python
@dataclass(frozen=True)
class CollaborationBlackboard:
    ...
    memory_snapshot: MemorySnapshot | None = None
```

Runtime 初始化 Blackboard 时直接传入：

```python
board = CollaborationBlackboard(
    turn_id=turn_id,
    user_id=user.id,
    session_id=session.public_id,
    user_input=original_input,
    model_input=sanitized_input,
    memory_snapshot=snapshot,
)
```

同时追加只读审计事件：

```text
MEMORY_SNAPSHOT_LOADED
```

事件 metadata 只记录版本和状态，不记录敏感原文：

```json
{
  "summaryVersion": 4,
  "safetyStateVersion": 3,
  "recentMessageCount": 8,
  "degraded": false,
  "degradationReasons": []
}
```

如果记忆服务降级，仍要生成 `degraded=true` 的快照并记录原因；不能让 Agent 误以为“无数据”等于“无风险”。

### 8.3 Agent 任务依赖

```text
Runtime 前置条件：board.memory_snapshot 已存在
        ↓
        ├─ task:understand
        └─ task:assess-safety
                 ↓
          task:gather-context
                 ↓
          task:propose-response
                 ↓
          task:review-response
                 ↓
             FINAL_ACCEPTED
```

Coordinator 不负责创建 `task:load-memory`。Runtime 只有在 Blackboard 已经携带 MemorySnapshot 后才调用 Coordinator。Coordinator 和 Agent 应把缺少 snapshot 视为编程错误或明确降级状态，不能继续依赖任务优先级碰巧先加载历史。

### 8.4 UnderstandingAgent

输入：

- 当前输入；
- 最近原文；
- 必要的历史摘要；
- 不需要读取后台工单详情。

不能继续调用 `PromptTemplates.intent_prompt([], text)` 丢弃历史。私有记忆中的 `intent=RISK` 标签不能替代真实对话上下文。

### 8.5 SafetyAgent

输入：

- 当前输入；
- 最近原文；
- 活跃/监测风险事件；
- SafetyState；
- 必要的历史摘要。

输出：

- `currentRisk`；
- `carryoverRisk`；
- `effectiveRisk`；
- 状态转换建议；
- 是否发布 `SAFETY_OVERRIDE`；
- 风险证据和置信度。

发布 `SAFETY_OVERRIDE` 的条件改为 `effectiveRisk == HIGH`，而不是只看本轮 `currentRisk`。

### 8.6 ContextAgent

ContextAgent 消费已有 MemorySnapshot、intent 和 safety artifact，负责：

- 组装最终模型历史；
- 根据 `intent` 和 `effectiveRisk` 决定是否检索 RAG；
- 加载对应 skill；
- 生成临时 `memoryBrief`；
- 生成不向用户暴露内部标签的安全约束文本。

所有对话都获得基础历史上下文；只有 CONSULT/RISK 或 effective MEDIUM/HIGH 才需要心理知识 RAG 和相关 skill。

### 8.7 ResponseAgent

ResponseAgent 根据 `intent + effectiveRisk + lifecycleStatus` 选择回复模式。不能只判断 `intent == CHAT and risk == LOW`。

### 8.8 SafetyAgent 候选回复审查

审查使用同一个 `effectiveRisk` 和 SafetyState 版本，至少检查：

- HIGH 是否包含当前安全确认；
- 是否建议联系身边可信任的人或合适资源；
- 是否泄露 `HIGH/ACTIVE/score` 等后台标签；
- 是否出现危险操作细节；
- monitored_chat 是否进行了适度而非机械重复的确认；
- 回复是否错误地把历史事件描述为当前事实。

## 9. 消息持久化顺序

当前 Harness 是 Agent 运行后才保存 user 消息。目标流程建议调整为：

```text
1. 校验并脱敏输入
2. 解析或创建 ChatSession
3. 保存当前 user ChatMessage，获得 message_id
4. 启动 Agent runtime，并传入 current_message_id
5. SafetyAgent 如发现风险，RiskEvent 关联 current_message_id
6. 输出 assistant 流
7. 保存 assistant ChatMessage
8. 执行或调度普通记忆维护
```

读取 recent history 时要明确排除当前消息，或者将当前消息作为 recent 的最后一条且不再额外追加，避免 Prompt 中出现两次当前输入。

如果流式生成失败：

- user 消息仍保留；
- 已持久化的风险事件仍保留；
- 不创建虚假的 assistant 消息；
- 下一轮加载时要能处理未完成轮次；
- 常规压缩不能把未完成 user 尾部压入摘要。

建议为 ChatRequest 增加可选 `clientMessageId/requestId`，实现重试幂等，避免网络重试写入重复用户消息。

## 10. Prompt 拼装顺序

最终发送给回复模型的顺序建议为：

```text
1. 基础系统身份和通用安全规则
2. 本轮安全约束（根据 effectiveRisk 生成）
3. 较早历史的结构化滚动摘要
4. RAG 知识和 skill 指引
5. 最近原始消息
6. 当前用户输入（仅一次）
```

安全系统消息示例：

```text
存在尚未解除的历史安全事件。本轮必须先关注用户当前安全，再处理普通请求。
不得向用户展示风险等级、事件状态、分数、后台报告或内部标签。
不得把历史风险直接断言为当前事实，应通过关怀式询问确认当前状态。
```

普通历史摘要与安全状态必须分成不同 system message，便于审计、预算和测试。

## 11. 配置建议

在 `Settings` 中新增或替换：

```text
MEMORY_ROLLING_SUMMARY_ENABLED=true
MEMORY_RECENT_TARGET_MESSAGES=8
MEMORY_COMPACTION_TRIGGER_MESSAGES=12
MEMORY_SUMMARY_MAX_TOKENS=<按模型确定>
MEMORY_SUMMARY_SCHEMA_VERSION=1
MEMORY_PROMPT_HIGH_WATERMARK_RATIO=0.65
MEMORY_PROMPT_TARGET_RATIO=0.40
MODEL_CONTEXT_WINDOW_TOKENS=<按实际模型确定>

SAFETY_MEMORY_ENABLED=true
SAFETY_STATE_SCOPE=user
SAFETY_STATE_CACHE_TTL_SECONDS=<缓存TTL，不代表风险失效>
SAFETY_MONITORING_RISK_FLOOR=MEDIUM
SAFETY_REVIEW_INTERVAL_SECONDS=<由产品和专业人员确定>
SAFETY_REVIEW_NORMAL_TURNS=<多少普通轮次后要求复核>
SAFETY_HIGH_TO_MONITORING_MIN_SAFE_ASSESSMENTS=2
SAFETY_MEDIUM_TO_LOW_MIN_CONSECUTIVE_LOW=2
SAFETY_LOW_TO_RESOLVED_MIN_ADDITIONAL_LOW=<1或2，由产品确认>
```

旧配置迁移：

- `MEMORY_COMPACTION_RECENT_MESSAGES` 可暂时映射为 target，随后废弃；
- `MEMORY_SUMMARY_MAX_CHARS` 应迁移为 token 或 schema 字段限制；
- `REDIS_MEMORY_MAX_MESSAGES=40` 可以继续作为缓存容量，但不再决定长期记忆范围。

禁止把临床/产品含义重大的时间阈值硬编码在 Python 中。应配置化，并由产品与专业人员确认。

## 12. Redis key 规划

建议：

```text
mindbridge:chat-recent:{session_public_id}
mindbridge:chat-summary:{session_public_id}
mindbridge:safety-state:user:{user_id}
mindbridge:agent-private:{agent_name}:{session_public_id}
mindbridge:memory-lock:{session_public_id}        # 可选短锁
```

注意当前 `AgentPrivateMemory` 会把已经带前缀的 key 再传给短期记忆 store，实施时要检查最终 key 是否符合预期，避免层层拼接难以运维。

Redis payload 必须带：

- schema version；
- data version；
- 更新时间；
- 必要的 ID；
- 可验证字段。

遇到坏 JSON 或版本不兼容时应丢弃缓存并回源 MySQL。

## 13. 数据库迁移

当前实现使用 `Base.metadata.create_all()` 创建新表，并以幂等 `COLUMN_MIGRATIONS` 补充既有表字段和索引。该轻量机制已经覆盖本次改造，但不像 Alembic 一样提供完整 downgrade 和迁移版本审计。

生产化时推荐把当前轻量迁移迁入 Alembic，并保持以下迁移内容可重复执行、可审计：

1. 创建 `chat_memory_summaries`；
2. 创建 `risk_events`；
3. 创建 `safety_states`；
4. 创建 `memory_maintenance_jobs`；
5. 按需给 `psychological_reports` 增加 `message_id`；
6. 给消息请求、回复和风险评估增加幂等索引；
7. 如果扩展 `risk_cases.status`，确认历史值兼容；
8. 生产部署建议继续迁移至 Alembic，并提供 downgrade 或明确不可逆说明。

历史数据回填：

- 不要自动把所有历史 PsychologicalReport 都标成 ACTIVE；
- 可以将历史 MEDIUM/HIGH report 转成 RiskEvent；
- SafetyState 是否回填为 ACTIVE 必须采用保守、可审计的规则并由产品确认；
- 未确认时，建议历史数据只生成事件，不自动施加当前 ACTIVE 约束；
- 滚动摘要可以懒生成：会话下一次活跃时从原始历史构建。

## 14. 需要修改或新增的代码位置

### 14.1 `app/core/enums.py`

新增：

- `SafetyLifecycleStatus`；
- `RiskEventKind`；
- `RiskEventType` 或等价风险类型 enum；
- 可选 `ResponseMode`；
- 可选给 `RiskCaseStatus` 增加 `RESOLVED/CLOSED`，但不要与 SafetyLifecycleStatus 合并。

### 14.2 `app/core/config.py`

新增第 11 节配置，并保留旧配置兼容映射和弃用说明。

### 14.3 `app/models/entities.py`

新增：

- `ChatMemorySummary`；
- `RiskEvent`；
- `SafetyState`。

按需给 `PsychologicalReport` 增加 `message_id`。增加关系不是必须，但索引和约束必须明确。

### 14.4 `app/services/memory.py`

拆分职责：

- 最近消息缓存；
- 滚动摘要存储；
- 摘要 merge；
- 8/12 触发；
- token 预算；
- 结构化输出验证；
- Redis/MySQL 回退。

移除“用最近 4+3 条作为旧历史摘要”的逻辑。若保留该方法，只能作为 `memoryBrief` 的一个本地 fallback，不能用于历史压缩。

建议新增或从该文件拆出 `MemoryContextService`，统一负责：

- 在 Agent runtime 前加载 historySummary、recentMessages、SafetyState 和 activeRiskEvents；
- 构造不可变 MemorySnapshot DTO；
- `ensure_prompt_budget()` 前置兜底；
- assistant 保存后的 `compact_if_needed()` 常规维护；
- 将缓存/数据库故障转换成明确的 degraded snapshot 信息。

摘要模型即使使用 LLM，也仍由 `RollingSummaryService` 作为受约束服务调用，不需要包装成 Agent。

### 14.5 新增 `app/services/safety_memory.py`

建议集中实现：

- SafetyState load/cache；
- RiskEvent append；
- 状态转换；
- carryoverRisk；
- effectiveRisk；
- Redis 缓存刷新；
- 乐观锁；
- 人工 override API 所需服务方法。

### 14.6 `app/agents/registry.py`

无需新增 `AgentCapability.MEMORY`。记忆是所有 Agent 的前置基础设施，不参与 Agent 任务竞争。

### 14.7 `app/agents/events.py`

- 给 CollaborationBlackboard 增加只读 `memory_snapshot` 字段；
- 增加 `MEMORY_SNAPSHOT_LOADED` 事件类型；
- 事件只记录版本、消息数和降级状态，不记录敏感原文；
- 保证 Blackboard 的复制/更新方法保留同一个 snapshot。

### 14.8 `app/agents/autonomous.py`

不新增 MemoryAgent；重构：

- UnderstandingAgent 使用 MemorySnapshot；
- SafetyAgent 使用 recent history 和 SafetyState；
- SafetyAgent 输出三种 risk；
- `SAFETY_OVERRIDE` 基于 effectiveRisk；
- ContextAgent 消费 snapshot，不再自己临时决定是否先加载历史；
- ResponseAgent 根据新回复模式生成 Prompt。

### 14.9 `app/agents/coordinator.py`

在 snapshot 已由 Runtime 前置加载的基础上增加 Agent 任务依赖：

- understand/safety 都读取 `board.memory_snapshot`；
- intent 和 safety 后才 gather-context；
- context 后才 response；
- review 后才 accept。

Coordinator 不创建、认领或调度记忆加载任务。不能只依靠优先级“碰巧”得到正确顺序。

### 14.10 `app/agents/event_driven_runtime.py`

- 在创建 CollaborationBlackboard 和调用 Coordinator 之前调用 `MemoryContextService.load_snapshot()`；
- 把不可变 MemorySnapshot 直接传给 Blackboard；
- 追加 `MEMORY_SNAPSHOT_LOADED` 事件；
- 将 current message ID、MemorySnapshot 和 safety artifacts 带入结果/trace；
- fallback 也必须尊重 effectiveRisk；
- 记忆服务降级时记录明确事件。

### 14.11 `app/agents/harness.py`

- 调整 user 消息保存到 runtime 之前；
- 将 message ID 传给 runtime；
- 在风险评估后、回复前持久化风险事件；
- assistant 保存后触发普通记忆维护；
- 保证失败和重试幂等。

### 14.12 `app/services/assessment.py`

让评估输入明确接收：

- current input；
- recent conversation；
- structured safety context。

硬规则仍可立即提升 currentRisk，但最终 effectiveRisk 由 SafetyMemoryService 合并。

### 14.13 `app/services/ai.py`

增加并分离：

- rolling summary prompt；
- safety-aware assessment prompt；
- safety constraint rendering；
- monitored_chat prompt；
- structured JSON 输出校验要求。

### 14.14 `app/services/chat.py`

确保流式失败、客户端断开和 assistant 未保存时，user/risk 数据仍一致。assistant 保存成功后执行或调度压缩。

### 14.15 `app/services/report.py` 与后台页面

建议增加：

- RiskEvent 时间线；
- 当前 SafetyState；
- 状态变更来源；
- 人工标记 RESOLVED/FALSE_POSITIVE；
- 摘要版本和覆盖消息 ID；
- 禁止普通学生端接口暴露后台标签。

### 14.16 `app/services/trace.py`

trace 建议新增：

- summary version；
- coveredThroughMessageId；
- safety state version；
- current/carryover/effective risk；
- response mode；
- Redis 是否命中；
- 是否触发压缩及原因；
- token 估算和降级动作。

## 15. 失败和降级策略

### 15.1 Redis 不可用

- 普通摘要、最近消息和 SafetyState 从 MySQL 加载；
- 记录 cache miss/degraded；
- 不因 Redis 不可用清除风险约束。

### 15.2 摘要模型失败

- 不推进 `coveredThroughMessageId`；
- 保留原始消息；
- 下一次重试；
- 若 Prompt 即将超限，可执行明确记录的临时截断，但不能伪装成成功摘要。

### 15.3 SafetyState 缓存版本落后

- 以 MySQL version 为准；
- 删除/刷新旧缓存；
- 不允许旧缓存把 ACTIVE 降为 RESOLVED。

### 15.4 风险评估模型失败

- 先应用硬规则；
- 继承未解除 carryoverRisk；
- 不能因为模型调用失败把 HIGH 降为 LOW；
- 记录降级原因。

### 15.5 RAG 失败

- 不影响 SafetyState；
- 使用安全通用规则和 skill fallback；
- 不降低 effectiveRisk。

### 15.6 数据库不可用

这是权威状态不可用，不能静默当作“无风险”。应：

- 使用可验证的 Redis 快照做有限降级；
- 对安全状态标记 `source=cache/degraded`；
- 记录高优先级错误；
- 根据产品策略决定拒绝普通回答、使用保守安全回复或短暂失败。

## 16. 隐私与访问控制

- 完整原文继续在 MySQL 保存，按现有隐私政策管理；
- Redis payload 必须脱敏；
- RiskEvent 的 `evidence_summary` 只保存必要证据；
- 不在普通摘要重复身份证、手机号、邮箱等敏感信息；
- 学生端 Prompt 可以获得安全约束，但不能获得后台等级、分数、工单状态或处理人；
- 管理端查看 RiskEvent/SafetyState 需要管理员权限；
- 日志避免打印完整风险原文；
- 数据保留期限、跨会话继承范围和人工访问必须由产品/合规确认。

## 17. 测试计划

### 17.1 普通压缩单元测试

至少包含：

1. 20 条历史时，摘要输入来自窗口外历史，不是最近 8 条；
2. 达到 12 条后压缩最早 4 条并保留最后 8 条；
3. 11 条不触发常规压缩；
4. token 高水位即使不足 12 条也触发；
5. 当前未回复 user 消息不被压缩；
6. 摘要不会重复包含当前输入；
7. `coveredThroughMessageId` 单调递增；
8. LLM 输出非法 JSON 时不推进游标；
9. PII 被脱敏；
10. 并发 version 冲突不会覆盖较新摘要；
11. Redis miss 能从 MySQL 恢复；
12. 摘要中已有 safetyReferences 不会无故丢失。

### 17.2 安全状态单元测试

1. 当前 HIGH 创建 RiskEvent 和 ACTIVE SafetyState；
2. 上一轮 ACTIVE/HIGH，下一轮 CHAT/LOW，effectiveRisk 仍为 HIGH；
3. ACTIVE/HIGH + 明确安全证据可以转 MONITORING；
4. MONITORING + 新 HIGH 回到 ACTIVE；
5. RESOLVED + CHAT/LOW 不继承风险；
6. FALSE_POSITIVE 不继承风险；
7. 时间经过不会单独变 RESOLVED；
8. RiskCase ACKNOWLEDGED 不会自动变 RESOLVED；
9. Redis miss 从 MySQL 恢复 ACTIVE；
10. 旧缓存版本不能覆盖新状态；
11. 评估模型失败时保留 carryoverRisk；
12. 新会话能否继承用户级 ACTIVE，符合配置；
13. 普通聊天轮数增加不会直接推动状态降级；
14. 时间经过只设置 reassessmentRequired，不会自动 RESOLVED；
15. ACTIVE/HIGH 未获得明确安全复核和保护因素时不能进入 MONITORING；
16. MONITORING/MEDIUM 需要配置数量的连续 LOW 复核才能降到 LOW；
17. 明确计划、意图、手段或实际行为在无人审核时不能自动降级；
18. 状态转换后阶段计数正确重置，不能重复消费同一次复核。

### 17.3 Agent 集成测试

1. Runtime 在 Coordinator 启动前加载 MemorySnapshot；
2. Blackboard 固定字段携带不可变 snapshot；
3. `MEMORY_SNAPSHOT_LOADED` 事件早于 intent/risk artifact；
4. UnderstandingAgent 的 Prompt 含历史；
5. SafetyAgent 的 Prompt 含 SafetyState；
6. 一轮中所有 Agent 读取相同 snapshot 版本；
7. `SAFETY_OVERRIDE` 基于 effectiveRisk；
8. CHAT + ACTIVE/HIGH 进入 safety_support；
9. MONITORING 进入 monitored_chat；
10. 候选回复缺少必要安全确认时被拒绝；
11. 安全约束不会向用户泄露内部标签；
12. fallback response 也尊重 effectiveRisk；
13. 风险持久化发生在回复生成失败之前；
14. snapshot degraded 时不把“加载失败”解释为“无风险”；
15. 不存在 MemoryAgent、MEMORY capability 或 load-memory 任务依赖。

### 17.4 多轮验收场景

场景 A：长期普通聊天

```text
超过 40 条后，早期稳定事实仍能通过 historySummary 被模型使用。
```

场景 B：风险后转普通问题

```text
Turn 1：明确自杀表达 → ACTIVE/HIGH
Turn 2：普通编程问题 → intent CHAT，current LOW，effective HIGH
结果：先执行安全确认，不直接 normal_chat。
```

场景 C：风险缓解

```text
用户明确当前安全、有人陪伴并已求助
结果：状态从 ACTIVE 转 MONITORING；后续回复不机械重复完整危机模板。
```

场景 D：人工解除

```text
管理员确认处理完成 → RESOLVED
后续 CHAT/LOW → normal_chat
历史 RiskEvent 仍可审计。
```

场景 E：Redis 故障

```text
Redis 不可用 → MySQL 恢复摘要和 SafetyState → 行为不丢失。
```

## 18. 评测数据扩展

现有风险评测需要增加多轮样本，而不能只评估单条输入。建议新增类别：

- HIGH → 普通 CHAT；
- HIGH → 否认/模糊回应；
- HIGH → 明确安全并有保护因素；
- MONITORING → 新 HIGH；
- RESOLVED → 普通 CHAT；
- 误判纠正；
- 风险原文离开最近 8 条后仍能继承；
- Redis miss；
- 摘要模型失败。

指标至少包含：

- `currentRiskCorrect`；
- `carryoverAppliedCorrectly`；
- `effectiveRiskCorrect`；
- `stateTransitionCorrect`；
- `safetyConstraintPresent`；
- `internalLabelLeakage`；
- `oldFactRetention`；
- `summaryHallucinationRate`。

## 19. 可观测性

建议增加指标：

```text
memory_compaction_total{reason=message|token}
memory_compaction_failed_total
memory_summary_version_conflict_total
memory_prompt_tokens
memory_recent_message_count
safety_state_cache_hit_total
safety_state_cache_miss_total
safety_event_created_total{risk_level,event_kind}
safety_state_transition_total{from,to}
effective_risk_total{level}
safety_carryover_applied_total
response_mode_total{mode}
```

日志应带 session/user 的不可逆或内部 ID，但不要打印完整敏感内容。

## 20. 分阶段实施顺序

### 阶段 1：数据层和纯服务

- 加迁移框架；
- 新增三张表；
- 实现滚动摘要服务；
- 实现 SafetyMemoryService；
- 完成单元测试。

### 阶段 2：Runtime 前置 MemorySnapshot 和 Agent 依赖

- 实现 MemoryContextService 前置加载；
- 给 Blackboard 增加不可变 MemorySnapshot 字段；
- 增加 MEMORY_SNAPSHOT_LOADED 审计事件；
- 调整 Coordinator 依赖；
- Understanding/Safety 使用快照；
- 保持旧回复行为，通过 feature flag 只观察新数据。

### 阶段 3：启用普通滚动摘要

- 先 shadow 生成并比较；
- 再启用 12→8；
- 观察 token、延迟、事实保留和摘要失败率。

### 阶段 4：启用安全 carryover

- 先记录 `effectiveRisk` 不改变回复；
- 检查误报和状态转换；
- 再让 response mode 和 SAFETY_OVERRIDE 使用 effectiveRisk。

### 阶段 5：管理端和人工状态变更

- 展示 RiskEvent 时间线；
- 支持 MONITORING/RESOLVED/FALSE_POSITIVE；
- 完善审计和权限。

### 阶段 6：清理旧逻辑

- 删除旧的伪“历史摘要”；
- 弃用旧配置；
- 更新 README 和架构图；
- 补充运行手册。

## 21. 验收标准

改造完成必须同时满足：

1. 窗口外历史进入持久滚动摘要；
2. 最近 8 条原文完整保留；
3. 12 条触发批量压缩，且以完整轮次移动；
4. token 高水位能独立触发压缩；
5. 摘要有覆盖游标和版本，不会重复或漏压；
6. Redis 丢失不导致长期记忆或安全状态丢失；
7. RiskEvent 和 SafetyState 独立于普通摘要；
8. HIGH → CHAT 场景能继承未解除风险；
9. RESOLVED/FALSE_POSITIVE 不继续强制高风险回复；
10. 时间不会单独解除风险；
11. Runtime 在 Coordinator 前生成 MemorySnapshot，SafetyAgent 在评估前已获得它；
12. ResponseAgent 和 fallback 都使用 effectiveRisk；
13. 学生端不会看到后台等级、分数和状态名；
14. 新增行为有单元、集成、多轮评测覆盖；
15. 数据库升级有正式迁移，不依赖 `create_all()` 修改旧表；
16. MemorySnapshot 是 Blackboard 的不可变基础字段，同一轮所有 Agent 读取相同版本；
17. 记忆加载由 Runtime/Harness 前置调用 MemoryContextService 完成；
18. 不新增 MemoryAgent、MEMORY capability 或 load-memory Agent 任务。

## 22. 实施前必须确认的产品决策

以下决策不能由开发者静默猜测：

1. SafetyState 默认按用户还是按会话继承；本文推荐按用户；
2. 哪些证据允许 `ACTIVE → MONITORING`；
3. 哪些角色有权执行 `RESOLVED/FALSE_POSITIVE`；
4. MONITORING 状态多久复核一次；
5. monitored_chat 是每轮确认，还是按时间/轮次间隔确认；
6. 风险事件和摘要的保留期限；
7. token 阈值对应的真实模型上下文容量；
8. 数据库和 Redis 同时不可用时采用保守回复还是拒绝服务；
9. 历史数据是否回填 SafetyState，采用何种审核流程。

在这些决策未确认前，可以完成数据模型、服务接口、shadow mode 和测试框架，但不能擅自设置具有业务或安全含义的自动解除策略。
