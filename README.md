# 校园有应 · Campus Youying

基于原 MindBridge Python 改造的校园资讯与生活支持 AI 助手。保留 FastAPI、五个 Agent 的事件驱动协作、MySQL/Redis 记忆和心理安全链路，增加面向校园政策的父子双层向量检索与管理端来源追溯。

## 产品行为

| 意图 | 内容 | 行为 |
|---|---|---|
| CHAT | 闲聊、分享、学习和通用交流 | 结合历史自然回应 |
| CONSULT | 校规、研究生差旅报销、奖助、培养和校园服务信息 | 依据适用知识资料回答，来源保存在管理端；缺资料时明确提示 |
| RISK | 情绪困扰求助、危险表达 | 独立评估 LOW / MEDIUM / HIGH，保留安全支持和必要的人工预警 |

RISK 是内部业务入口，不等于 HIGH。所有输入都会进行安全评估；未解除的风险和连续复核计数仅在同一会话内继承。同一用户的新会话独立评估，返回原会话继续其原有状态。普通政策咨询不会生成心理报告或预警。

学生侧边栏按当前账号显示历史会话，支持加载更多、切换查看、继续聊天和刷新恢复上次打开的会话；恢复历史消息；来源资料仅在管理员页面逐条查看。

项目现已包含 55 份西安电子科技大学官方公开资料快照（核查日期 2026-10-19）：49 份默认启用、4 份待核对、2 份年度历史归档。默认学校为西安电子科技大学，启动时从仓库文件校验并幂等导入；克隆或下载项目后原文件依旧存在，Docker 镜像也会带上资料。目录、官方来源、适用范围与导入命令见 [官方资料包](knowledge/xidian/README.md) 和 [入库说明](docs/xidian-corpus-intake.md)。原演示政策仍不导入；现行差旅与餐补标准尚未核实，没有对应资料时明确提示，不猜测金额。独立评测中的虚构数据只用于离线测试。

输入框使用 Enter 发送、Shift+Enter 换行，中文输入法确认候选词时不会发送。

## 启动

macOS 本机可直接双击桌面的 `mindbridge.command`，自动检查 Python 依赖、启动 Docker Desktop 及本项目 MySQL/Redis、检查模型与 Embedding 连接，再启动前后端和后台任务。全部就绪后自动打开 [校园有应](http://127.0.0.1:7070)。重复双击会复用已启动的服务；保留启动终端窗口，关闭窗口或按 Ctrl+C 停止本次后端。详情见 [一键启动说明](docs/one-click-startup.md)。

也可在项目目录运行：

```bash
bash scripts/one-click.sh
```

以下为手动启动方式：

建议使用 Python 3.11 或 3.12，并在项目目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp -n .env.example .env
```

已有 `.env` 会保留。配置 MySQL、Redis 和可用的聊天/Embedding 接口后：

```bash
bash scripts/run-dev.sh
```

默认访问 [本地服务](http://127.0.0.1:8080)。脚本优先使用项目 `.venv`，模型配置由 Settings 读取 `.env`，不再用脚本默认值覆盖 API Provider。

演示账号：`student / student123`、`admin / admin123`。原账户、表名和内部 MindBridge 类名保留兼容。

使用 Docker 时先创建 `.env`，再执行：

```bash
docker compose up -d --build
```

应用环境读取 `.env`；Compose 内数据库和 Redis 地址使用容器服务名。默认端口映射：应用 8080、MySQL 13306、Redis 16379。Docker 镜像包含运行必需的 `skills/`。

## 核心链路

```text
用户消息保存
  → token 预算检查和 MemorySnapshot
  → UnderstandingAgent + SafetyAgent
  → ContextAgent：历史、领域选择、按需 RAG
  → ResponseAgent：所有模式实时生成并通过 SSE 增量输出
    （高风险先发送固定安全引导，监测状态先简短关心近况）
  → 流内安全规则：内部标签拦截，高风险未核实号码脱敏
  → SafetyAgent：完成后的规则复核
  → CoordinatorAgent：最终采纳
  → 完成 SSE，来源只保存到管理端
  → 回复保存、12→8 记忆维护、安全条件触发工具队列
```

不新增专门的政策核验 Agent，也不为每次政策回答增加模型事实打分。政策准确性通过离线评测、人工核对和知识更新后的回归保障。安全审查继续保留；当前规则检查不等于全面语义验证。

五个角色仍由任务板、共享 artifact 和事件协调。事件驱动不代表所有模型调用已异步并行；每次用户输入创建新运行，从历史记忆重建上下文。

## RAG：父子切块与检索

- 父块按主标题、完整条款或主题划分，默认不重叠；三级及以下小标题尽量留在对应父块内。
- 子块在父块内部按完整句段划分，长句使用少量重叠滑窗。Markdown 表格整体保留；PDF 目前提取文字和页码，不提供 OCR 或可靠的复杂表格重建。
- 父块、子块都生成 Embedding。模型默认配置是 `text-embedding-3-small`，可换成兼容接口支持的模型。
- 父向量、子向量、子块 BM25 分别召回；子块映射回父块，各路按父块取最高分、归一化后加权融合。
- 最终按父块去重，按预算提供完整正文，不再拼接相邻块。
- 本地规则重排为可选开关，当前默认关闭，作用于归并融合后的父块；它不是额外的神经网络模型。
- 正式/支持领域、学校、状态、生效与失效日期共同限制可检索文档。向量候选使用数据库确认的块 ID 过滤，避免旧版本混入。
- 向量不可用时，按配置降级为子块 BM25 → 父块取回，并记录实际路径。
- 重建使用新 collection 完成后原子切换 manifest，失败保留旧索引；collection 名包含模型与接口的指纹，避免更换模型后混用向量。

主要参数（均为可调起点，不宣称已在正式政策上最优）：

```env
CAMPUS_NAME=本校
KNOWLEDGE_PARENT_CHUNK_SIZE=1000
KNOWLEDGE_CHUNK_SIZE=300
KNOWLEDGE_CHUNK_OVERLAP=40
KNOWLEDGE_CANDIDATE_K=16
KNOWLEDGE_TOP_K=4
KNOWLEDGE_PARENT_VECTOR_WEIGHT=0.30
KNOWLEDGE_CHILD_VECTOR_WEIGHT=0.35
KNOWLEDGE_HYBRID_BM25_WEIGHT=0.35
KNOWLEDGE_RERANK_ENABLED=false
RAG_SKILL_BUDGET_TOKENS=4000
```

保留 `KNOWLEDGE_HYBRID_VECTOR_WEIGHT` 仅兼容旧配置；新路径实际使用父/子两项向量权重。重排开关和权重应根据本校真实问答的保留集评测调整。

## 政策导入与版本管理

管理员页面支持文字型 PDF、Markdown、UTF-8 txt（每份最多 10 MB），可填写标题、版本、领域、发布部门、原文链接、日期和状态，预览父子块，并启用/归档版本。管理端会显示有效政策文件接入情况。

真实资料接口保留：`POST /api/admin/knowledge/file` 使用 multipart `file` 和 JSON 字符串 `metadata`；`POST /api/admin/knowledge` 支持 JSON 正文；文件列表、父子块预览、版本启用与归档接口保持可用。接口详情见 [真实资料接入](docs/real-knowledge-intake.md)。

- 同一来源的新启用版本替代旧启用版本；旧文档和知识块保留。
- 同一显式版本内容不可改写，变更需新版本号。
- 未指定版本时使用内容摘要生成版本；未来生效文件保存为草稿，生效后需管理员启用。
- 失效或归档版本不参与检索。跨章节通用条件不会自动被父子检索补全，入库前需检查条款边界和关联条件。
- 启动时执行兼容迁移并重建旧平面知识的父子表示，保留旧行。MySQL 由幂等列迁移补列，已有 SQLite 开发库也支持知识列增补。

接口：

| 接口 | 用途 |
|---|---|
| `POST /api/admin/knowledge` | JSON 正文与元数据入库 |
| `POST /api/admin/knowledge/file` | multipart 文件与 `metadata` JSON 入库 |
| `GET /api/admin/knowledge/documents` | 文件与版本列表 |
| `GET /api/admin/knowledge/documents/{id}` | 父子块预览 |
| `PATCH /api/admin/knowledge/documents/{id}/status` | ACTIVE / DRAFT / ARCHIVED |
| `GET /api/admin/knowledge/status` | 父子块、向量与配置状态 |
| `POST /api/admin/knowledge/rebuild-vector` | 构建并切换索引 |
| `POST /api/admin/knowledge/backup` | 索引快照 |
| `GET /api/admin/agent-traces` | 对话、检索与协作过程 |

JSON 入库除 `source`、`content` 外，可提供 `domain`（policy/support）、`title`、`version`、`school`、`department`、`source_url`、`published_on`、`effective_on`、`expires_on`、`status`、`is_demo`。文件接口的 `metadata` 字段使用同一组元数据。

SSE 使用 `status` / `meta` / `token` / `done` / `error` 事件和心跳，不向学生发送 `sources`。CHAT、CONSULT、RISK 的所有回复模式均实时转发模型正文，包括 LOW/MEDIUM/HIGH 和监测状态；高风险发送前拦截未核实长号码、同步拦截内部标签，最终复核与记录保留。缓存及缺文件固定提示也使用 SSE，因没有模型生成过程会立即发送，不人为延迟。管理端按回复查看实际提供给模型的父块原文、文件标题、版本、章节、页码及链接；消息绑定 trace，重试不重新查询当前版本。详情见 [流式、路由与管理端溯源](docs/streaming-routing-and-admin-sources.md)。

## 记忆与安全

- MySQL 保存完整消息、累计摘要、安全事件和用户当前安全状态；Redis 为近期消息、摘要和 Agent 备注提供缓存。
- 普通摘要按会话维护，默认 12 条完整消息触发、保留最近 8 条；不是超过 8 条后每条都压缩。
- 压缩合并旧摘要和新移出消息，失败不推进游标；后台队列重试，下一轮调用前有预算兜底。
- 摘要字段兼容旧 schema，内容扩展到身份、政策主题、适用条件和更正。检索改写同时参考累计摘要和近期概览。
- 政策金额不能因为曾经回答过就成为现行事实，具体规定仍从适用资料检索。
- 保留 `currentRisk / carryoverRisk / effectiveRisk`、生命周期、人工复核和风险持久化幂等。
- 原安全审查补充未经来源核实的长联系电话检查；没有增加政策模型核验环节。
- 模型生成异常时返回明确的不可用说明或安全支持回退，并在 trace 记录 `generationError`。

普通多轮不提供跨所有聊天的通用用户画像；运行 trace 也不是进程中断后的执行恢复检查点。

## 模型与配置

保留 Ollama、本地微调模型、OpenAI-compatible API 和 mock 接入。各 Agent 仍可通过 `AGENT_MODEL_*` 配置独立模型。

兼容供应商支持时，可配置 `OPENAI_ENABLE_THINKING=false`，防止思考输出耗尽较小的输出预算而导致正文为空。此字段默认不发送，不能对不支持它的接口随意启用。更换 Embedding 模型必须重建向量。

当前机器原有 `deepseek-v4-flash` 别名返回 404，已通过模型目录和实际请求确认 `deepseek-v4-flash-0731` 可用，并更新本地配置；Embedding 保留已有 `text-embedding-v4` 配置。未将任何密钥写入文档或评测报告。

## 验证与评测

无需外部服务的回归：

```bash
DATABASE_URL=sqlite:///:memory: AI_PROVIDER=mock OPENAI_API_KEY='' KNOWLEDGE_VECTOR_ENABLED=false .venv/bin/python -m unittest discover -s tests
.venv/bin/python -m app.harness.runner
DATABASE_URL=sqlite:///:memory: .venv/bin/python -m app.campus_eval.runner
```

`app.harness.runner` 自动使用独立 SQLite、mock、关闭向量服务及本地 log 预警；输出在 `target/harness/`。

校园端到端评测使用独立临时数据库和索引，不执行预警工具：

```bash
# 使用当前配置的真实回答与 Embedding API，会产生模型调用
DATABASE_URL=sqlite:///:memory: .venv/bin/python -m app.campus_eval.runner --real
```

结果在 `target/campus-mock-eval.json`、`target/campus-real-eval.json`。样本含明确标注的虚构测试金额，验证意图、检索、多轮、无答案及安全链路，不能替代正式学校政策验收。

保留原 RAG / 回答质量 / 多 Agent 对照评测入口。原心理样本的意图标签已适配新三分类；原回答评测使用预设上下文，只能证明生成层表现，不能单独证明端到端政策准确性。正式上线前需导入本校资料、建立条款级标注与独立保留集，比较重排开关及切块/融合参数。

## 改造记录

- [改造方案与实施清单](docs/campus-youying-redesign-plan.md)
- [实施结果、验证与接入说明](docs/campus-youying-implementation.md)
- [原记忆与安全设计](docs/context-memory-safety-redesign-spec.md)

改造前源码备份在 `target/backups/pre-campus-youying-source.tgz`，不含模型、数据库或 `.env` 密钥配置。当前目录没有 Git 元数据，本次没有创建提交或 PR。
