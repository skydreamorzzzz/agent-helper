# Agent Runtime Pipeline 与模块边界

这份文档用中文说明当前本地个人助手 Agent 的主流程，以及各模块的大致功能和边界。目标是方便后续迭代、调试和简历项目答辩。

## 总体 Pipeline

```text
用户输入
  -> CLI 命令判断
  -> 记忆检索
  -> 请求路由
  -> 直接回答 / 单工具 Runtime / 结构化 Planner / 澄清问题
  -> 工具执行或计划状态机
  -> 最终回答
  -> 日志记录
  -> 必要时写入开发复盘文档
```

## 1. CLI 层

位置：

```text
src/cli.py
```

职责：

- 读取用户终端输入。
- 处理显式命令，例如 `/remember`、`/memories`、`/plans`、`/plan <id>`、`/exit`。
- 初始化 LLM 客户端、工具注册表、记忆服务、Planner、计划仓储。
- 对写入类工具进行人工确认。
- 把普通自然语言请求交给后续 Pipeline。

边界：

- CLI 不直接操作 SQLite 细节。
- CLI 不直接执行工具逻辑。
- CLI 不负责判断计划是否合法，只协调各模块。

## 2. 配置层

位置：

```text
src/config.py
.env
secrets/deepseek_api_key
```

职责：

- 从 `.env` 读取模型地址、模型名、超时时间、Planner 和记忆配置。
- 支持从 `LOCAL_LLM_API_KEY_FILE` 读取 API key。
- 定义项目路径，例如 `workspace/`、`logs/`、记忆数据库和计划数据库路径。

边界：

- 不在代码中写死 API key。
- 不在业务模块中散落读取环境变量的逻辑。

## 3. LLM Client

位置：

```text
src/llm/client.py
```

职责：

- 调用 OpenAI 兼容的 `/chat/completions` 接口。
- 当前默认使用 DeepSeek OpenAI 兼容接口。
- 对上层暴露简单的 `chat(messages) -> str`。

边界：

- 不解析 Agent 协议。
- 不理解工具、计划或记忆。
- 不做重试、规划或业务判断。

## 4. Agent JSON 协议层

位置：

```text
src/agent/protocol.py
```

职责：

- 定义模型输出的两类结构：`tool_call` 和 `final_answer`。
- 使用 Pydantic 校验模型输出。
- 从 Markdown 代码块或包含 `<think>` 的文本中提取 JSON 对象。

边界：

- 只负责协议解析和校验。
- 不执行工具。
- 不决定是否进入 Planner。

## 5. 工具系统

位置：

```text
src/tools/
```

核心模块：

- `base.py`：定义 Tool 抽象，包括 `name`、`description`、`argument_schema`、`risk_level`、`execute()`。
- `registry.py`：注册、查询、描述和执行工具。
- `calculator.py`：安全数学计算。
- `file_tools.py`：读取和写入 `workspace/` 内的 UTF-8 文件。
- `transform_text.py`：只读文本转换工具，由执行器调用 LLM 完成文本整理。

职责：

- 每个工具只做一件明确的事。
- 工具参数必须通过 Pydantic schema。
- 工具声明风险等级：`read_only`、`write`、`destructive`、`external`。
- 文件工具负责路径安全，禁止路径穿越。

边界：

- 工具不决定是否应该被调用。
- 工具不直接访问长期记忆。
- `write_text_file` 只写文件，不负责理解、筛选、总结文本。
- 文本理解和整理应放在 `transform_text`，这也是一次真实全链路测试后沉淀出的边界。

## 6. 记忆系统

位置：

```text
src/memory/
```

核心模块：

- `models.py`：长期记忆领域模型。
- `store.py`：SQLite 持久化和 FTS5 表。
- `retriever.py`：基于 FTS5 和关键词兜底的检索。
- `writer.py`：显式写入和候选记忆过滤。
- `summarizer.py`：会话摘要。
- `service.py`：对 Runtime 和 CLI 暴露统一记忆服务。
- `journal.py`：开发复盘记录入口。

职责：

- Working Memory：保存当前会话最近消息，并按消息数/字符数截断。
- Conversation Summary：长对话超过阈值后生成结构化摘要。
- Long-term Memory：用 SQLite 持久化用户偏好、项目事实、任务、决策、摘要。
- 每次模型调用前，按当前请求检索相关长期记忆并注入上下文。
- 长期记忆带来源字段，方便审计。

边界：

- 长期记忆不是聊天记录全文缓存。
- 不把整个数据库塞进上下文。
- 不自动保存所有对话。
- 对普通闲聊、临时信息、模型猜测和重复内容保持保守。
- 具体开发复盘条目写入 Markdown 文档，不写入长期记忆数据库。

## 7. 请求路由

位置：

```text
src/planner/router.py
```

职责：

- 判断请求属于哪类：
  - `direct_answer`
  - `single_tool`
  - `planned_task`
  - `clarification`
- 普通聊天不进入 Planner。
- 多步骤依赖任务进入 Planner。
- 缺少关键参数时要求澄清。

边界：

- Router 只分类，不执行工具。
- Router 不生成计划。
- Router 不应该猜测缺失的关键参数。

## 8. 普通 Agent Runtime

位置：

```text
src/agent/runtime.py
```

职责：

- 处理直接回答和单工具调用模式。
- 构造系统提示词、记忆上下文、会话摘要和最近消息。
- 调用模型。
- 解析 `tool_call` 或 `final_answer`。
- 执行工具并把结果交回模型。
- 记录 run log。

边界：

- 不负责复杂计划状态机。
- 不直接访问 SQLite 细节。
- 不允许模型执行任意 Shell 或 Python。

## 9. Planner

位置：

```text
src/planner/planner.py
src/planner/prompts.py
```

职责：

- 为复杂任务生成结构化 Plan。
- Planner 会看到真实工具列表、描述、风险等级和参数 schema。
- 计划中只能使用 ToolRegistry 中真实存在的工具。
- 当文本处理后还要写文件时，应生成 `read_text_file -> transform_text -> write_text_file` 这种可执行链路。

边界：

- Planner 不执行工具。
- Planner 不虚构工具结果。
- Planner 不负责最终状态更新。
- Planner 生成的计划必须经过 Validator 才能执行。

## 10. PlanValidator

位置：

```text
src/planner/validator.py
```

职责：

- 校验工具是否存在。
- 校验参数是否符合工具 schema。
- 校验依赖步骤是否存在。
- 检测循环依赖。
- 检查步骤数是否超过上限。
- 如果存在 `unresolved_questions`，阻止执行。

边界：

- Validator 不修复计划。
- Validator 不执行工具。
- Validator 只给出明确错误，让上层决定是否重新规划或要求用户澄清。

## 11. PlanExecutor 状态机

位置：

```text
src/planner/executor.py
```

职责：

- 执行已校验的计划。
- 每次只执行一个 ready 步骤。
- 根据依赖关系推进步骤状态。
- 调用工具并保存 `actual_output` 或 `error`。
- 对 placeholder 参数调用模型解析。
- 对 `transform_text` 调用模型执行文本转换。
- 对写入、破坏性和外部风险工具触发确认。
- 失败后按配置重试或触发重新规划。

边界：

- 第一版不并行。
- 第一版不后台自主运行。
- 执行器不绕过 ToolRegistry。
- 执行器不直接修改长期记忆，只在 CLI 收尾时触发开发复盘记录。

## 12. PlanRepository

位置：

```text
src/planner/repository.py
```

职责：

- 将 Plan 持久化到 SQLite。
- 支持按 `plan_id` 读取。
- 支持列出历史计划和未完成计划。

边界：

- Repository 不理解计划是否合理。
- Repository 不执行计划。
- Repository 只负责存取。

## 13. 日志与审计

位置：

```text
src/logging_utils.py
logs/
```

职责：

- 每次运行生成独立 `run_id`。
- 用 JSON 结构记录用户输入、模型原始输出、工具参数、工具结果、计划状态变化、记忆检索结果等。

边界：

- 日志用于调试和审计，不参与模型上下文检索。
- 日志不是长期记忆。

## 14. 开发复盘文档

位置：

```text
docs/development_journal.md
src/memory/journal.py
```

职责：

- 记录开发过程中有价值的架构问题、失败案例和解决方案。
- 内容用中文写，方便简历项目答辩。
- 每次完成复杂任务后，先判断是否有值得记录的问题，再决定是否写入。

边界：

- 复盘文档不是长期记忆数据库。
- 长期记忆只保存“需要做复盘”的规则和用户偏好。
- 普通流水账不写入复盘文档。

## 当前关键设计取舍

1. 不使用 LangChain/LangGraph，保持 Runtime 和 Planner 透明可调试。
2. Function Calling 不作为基础依赖，使用 JSON 协议兼容更多模型。
3. 长期记忆用 SQLite FTS5，不引入向量数据库，降低复杂度。
4. Planner 只生成计划，Validator 负责硬约束，Executor 负责状态推进。
5. 工具职责保持单一，文本整理通过 `transform_text` 建模，不混入文件写入工具。
6. 写入和高风险操作需要确认，避免模型单方面决定修改文件。

