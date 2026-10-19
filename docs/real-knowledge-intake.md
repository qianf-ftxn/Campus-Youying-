# 真实资料接入与空库行为

项目已有随仓库分发的 [西电官方资料包](../knowledge/xidian/README.md)，首次启动自动导入；下面接口仍可用于补充学校正式文件或更新版本。

更新日期：2026-10-19。

校园政策库不再预装演示资料。旧演示文档、父子块和当前 Chroma 索引中的对应向量在启动时清理；已有真实上传文件、原情绪支持资料和会话记录保留。离线评测文件不参与应用启动或业务知识库接入。

## 上传接口

管理员登录后可在“RAG 知识库维护”直接上传文字型 PDF、Markdown 或 UTF-8 txt（10 MB 内）。

- `POST /api/admin/knowledge/file`：multipart 表单，文件字段 `file`，元数据字段 `metadata`（JSON 字符串）。
- `POST /api/admin/knowledge`：JSON，包含 `source`、`content` 和同样的元数据。
- `GET /api/admin/knowledge/documents`：查看已接入文件。
- `GET /api/admin/knowledge/documents/{id}`：预览父块和子块。
- `PATCH /api/admin/knowledge/documents/{id}/status`：JSON `{"status":"ACTIVE"}`，也支持 `DRAFT`、`ARCHIVED`。
- `GET /api/admin/knowledge/status`：其中 `activePolicyDocuments` 是当前学校已启用且有效的政策文件数。

接口需要管理员 Basic Authentication。可以继续使用现有管理端，无需手动调用接口。

元数据字段：`domain`（校园政策用 `policy`）、`title`、`version`、`school`、`department`、`source_url`、`published_on`、`effective_on`、`expires_on`、`status`。学校留空时使用 `CAMPUS_NAME`，日期采用 `YYYY-MM-DD`。未来生效文件先保存为草稿；同一来源和版本内容不同会被拒绝，需使用新版本号。

## 文件未接入时

政策咨询在没有检索证据时直接提示：

> 当前尚未接入或未找到与你的问题对应的有效文件，暂时无法确认具体规则。请先添加对应资料后再查询。

此提示不让回答模型推测具体规定，也不会把历史回答当成现行文件；检索服务故障则提示稍后重试。已接入文件未覆盖的问题仍由政策提示词约束为资料不足，不能编造金额、期限、资格或链接。

原有安全评估和高危支持继续生效。校园政策缺文件不阻断情绪支持或普通闲聊。

## 输入框

Enter 发送、Shift+Enter 换行；中文输入法组合输入与 Safari 的 229 键码不会提交。空输入、长按回车和正在发送/切换会话时不会重复发送。

改动前源码及业务库备份为 `target/backups/before-remove-demo-20261019-004737.tgz` 与同名前缀 `.sql`。验证日志为 `target/no-demo-tests.log`、`target/no-demo-harness.log`；键盘回归命令为 `node --test tests/student_keyboard.test.cjs`。

本轮最终结果：128 项 Python 测试、4 项键盘测试和 6 组工程检查全部通过。真实库已确认演示文档、对应切块、对应向量均为 0，有效政策文件为 0；桌面一键启动重新打开后就绪检查通过。浏览器按键与空库提示在隔离 SQLite / mock 环境验证，没有向业务库写入测试会话。截图为 `target/no-demo-keyboard-preview.png`，业务库核验摘要为 `target/no-demo-live-verification.json`。
