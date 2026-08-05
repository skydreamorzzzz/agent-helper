# Local Personal Agent

这是一个本地个人助手 Agent Demo，目标是实现透明、可调试、可扩展的 Agent Runtime、工具调用系统、记忆系统和结构化 Planner。

当前不包含多 Agent、网页界面、后台自主运行或向量数据库。

## 安装

需要 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 配置

复制示例配置：

```bash
cp .env.example .env
```

`.env` 必须配置：

```bash
LOCAL_LLM_BASE_URL=http://localhost:8080/v1
LOCAL_LLM_API_KEY_FILE=secrets/deepseek_api_key
LOCAL_LLM_MODEL=deepseek-v4-flash
LOCAL_LLM_TIMEOUT=600
AGENT_MAX_TOOL_CALLS=5
WORKING_MEMORY_MAX_MESSAGES=12
WORKING_MEMORY_MAX_CHARS=12000
SUMMARY_TRIGGER_MESSAGES=16
MEMORY_RETRIEVAL_LIMIT=5
MEMORY_IMPORTANCE_THRESHOLD=0.6
PLANNER_MAX_STEPS=8
PLANNER_MAX_REPLANS=2
TOOL_MAX_RETRIES=1
CONFIRM_WRITE_ACTIONS=true
```

模型名称、地址、Key 和超时时间全部从 `.env` 读取，代码中不写死。
`LOCAL_LLM_TIMEOUT` 单位是秒，默认示例为 600 秒，适合较慢的本地模型。

当前默认配置使用 DeepSeek OpenAI 兼容接口：

```bash
LOCAL_LLM_BASE_URL=https://api.deepseek.com
LOCAL_LLM_API_KEY_FILE=secrets/deepseek_api_key
LOCAL_LLM_MODEL=deepseek-v4-flash
```

把 DeepSeek API key 写入本地文件：

```text
secrets/deepseek_api_key
```

`secrets/` 已在 `.gitignore` 中，不会提交到 Git。也可以不用 key 文件，直接在 `.env` 中设置 `LOCAL_LLM_API_KEY`。

## 启动

```bash
python -m src.cli
```

输入 `/exit` 退出。

示例：

```text
You: 计算 23.5 乘以 17，并把结果保存到 calculation.txt。
Assistant: 已计算并保存到 calculation.txt。
```

## 架构文档

模块 Pipeline 和职责边界见：

```text
docs/pipeline.md
```

## 工具调用协议

由于本地模型不一定稳定支持原生 Function Calling，本项目使用 JSON 协议。模型每轮只能输出一种结构。

调用工具：

```json
{
  "type": "tool_call",
  "tool": "calculator",
  "arguments": {
    "expression": "123 * 456"
  }
}
```

最终回答：

```json
{
  "type": "final_answer",
  "content": "最终回答内容"
}
```

模型输出会用 Pydantic 校验。非法 JSON 会先尝试从 Markdown 代码块提取 JSON，然后只向模型发送一次格式修复请求；仍失败则安全停止。

## 当前工具

- `calculator`：通过 AST 白名单安全计算基础数学表达式，不使用不受限制的 `eval`。
- `read_text_file`：读取 `workspace/` 目录中的 UTF-8 文本文件。
- `write_text_file`：向 `workspace/` 写入 UTF-8 文本，默认不覆盖已有文件，除非 `overwrite=true`。

## 三层记忆架构

### Working Memory

保存当前进程内最近若干轮消息。它受两个配置限制：

- `WORKING_MEMORY_MAX_MESSAGES`
- `WORKING_MEMORY_MAX_CHARS`

超过限制时会截断，避免把所有聊天记录无限塞进上下文。

### Conversation Summary

当当前会话消息数超过 `SUMMARY_TRIGGER_MESSAGES` 时，使用模型生成结构化摘要。摘要只保留：

- 当前目标
- 已完成事项
- 用户明确要求
- 重要决定
- 未完成问题

普通闲聊不会作为摘要重点。

### Long-term Memory

长期记忆使用 SQLite 持久化，默认数据库文件位于：

```text
workspace/memory.sqlite3
```

第一版使用 SQLite FTS5 做全文检索，不引入向量数据库。代码中检索逻辑集中在 `src/memory/retriever.py`，未来可以在相同服务边界下增加 `EmbeddingRetriever`。

## 数据库结构

长期记忆表至少包含：

- `id`
- `category`
- `content`
- `source_run_id`
- `source_message_id`
- `importance`
- `created_at`
- `updated_at`
- `last_accessed_at`
- `is_active`
- `metadata_json`

`category` 限制为：

- `user_preference`
- `personal_fact`
- `project`
- `task`
- `decision`
- `summary`

删除记忆是软删除：`is_active=false`。

## 记忆写入规则

支持两种写入方式。

显式写入：

```text
/remember 我正在开发本地个人助手项目
```

候选记忆提取：

`MemoryWriter` 可以调用模型生成候选记忆，再经过保守规则过滤。默认规则会拒绝：

- 普通闲聊
- 临时性极强的信息
- 模型猜测
- 没有用户依据的结论
- 与已有记忆高度重复的内容
- 重要性低于 `MEMORY_IMPORTANCE_THRESHOLD` 的内容

敏感、含糊或可能错误的信息不自动保存。

## 记忆检索流程

每次调用模型前：

1. 用当前用户请求作为检索查询。
2. 使用 SQLite FTS5 查找 active 记忆，中文长句不命中时使用保守 `LIKE` 兜底。
3. 根据文本相关度、重要性和时间排序。
4. 最多注入 `MEMORY_RETRIEVAL_LIMIT` 条。
5. 以独立 `Relevant Long-term Memories` 区块注入系统提示。

提示词会明确告诉模型：记忆可能过时，只是上下文，不是绝对事实；当前用户表达与旧记忆冲突时，以当前表达为准。

每次检索到的记忆都会写入 `logs/{run_id}.log`，包含记忆 ID、得分、分类和 `source_run_id`，方便审计。

## CLI 命令

- `/remember <内容>`：显式保存长期记忆
- `/memories`：查看 active 记忆，显示 `id`、`category`、`created_at`、`content`
- `/forget <memory_id>`：软删除记忆
- `/update-memory <memory_id> <新内容>`：更新记忆内容
- `/clear-session`：清空当前会话 Working Memory 和 Conversation Summary
- `/summary`：查看当前会话摘要
- `/plans`：查看历史计划
- `/plan <plan_id>`：查看计划详情、步骤状态和工具结果
- `/cancel-plan <plan_id>`：取消计划
- `/resume-plan <plan_id>`：手动恢复未完成计划
- `/history`：查看计划和记忆概览
- `/exit`：退出

## 请求路由

Planner 前有一层 `RequestRouter`，输出结构化 JSON：

```json
{
  "route": "direct_answer",
  "reason": "普通对话或可直接回答。",
  "missing_information": []
}
```

路由类型：

- `direct_answer`：普通聊天或可直接回答的任务
- `single_tool`：明显只需要一次工具调用的任务
- `planned_task`：包含多个依赖步骤的复杂任务
- `clarification`：缺少执行所必需的信息，必须先问用户

普通聊天不会进入 Planner。信息不足时不会猜文件名、目标或保存位置。

## 结构化计划

复杂任务会生成 `Plan`，持久化到：

```text
workspace/plans.sqlite3
```

`Plan` 包含 `plan_id`、`goal`、`status`、`created_at`、`updated_at`、`current_step_id`、`steps`、`assumptions`、`unresolved_questions` 和 `final_output_requirement`。

`PlanStep` 包含 `id`、`description`、`tool_name`、`arguments`、`depends_on`、`status`、`expected_output`、`actual_output`、`error` 和 `retry_count`。

计划状态：`pending`、`running`、`completed`、`failed`、`paused`、`cancelled`。

步骤状态：`pending`、`ready`、`running`、`completed`、`failed`、`skipped`。

## PlanValidator

计划执行前会经过 `PlanValidator`：

- 工具必须存在于 `ToolRegistry`
- 参数必须通过工具的 Pydantic schema
- 依赖步骤必须存在
- 依赖不能成环
- 步骤数不能超过 `PLANNER_MAX_STEPS`
- 存在 `unresolved_questions` 时不能执行

Planner 生成计划时会看到真实工具名、描述、风险等级和参数 schema，因此虚构工具或错误参数无法通过校验。

## 状态机

`PlanExecutor` 每次只执行一个 ready 步骤：

1. 找到依赖已完成的步骤。
2. 选择下一个 ready 步骤。
3. 校验和解析工具参数。
4. 根据工具风险等级决定是否需要人工确认。
5. 调用工具。
6. 保存 `actual_output` 或 `error`。
7. 更新步骤状态和计划状态。
8. 所有步骤完成后生成最终回答。

第一版不并行执行，也不做后台自主运行。`/resume-plan` 只在用户明确输入命令时恢复执行。

## 工具风险等级

工具声明 `risk_level`：

- `read_only`：可自动执行
- `write`：默认要求确认，由 `CONFIRM_WRITE_ACTIONS` 控制
- `destructive`：必须确认
- `external`：暂不自动执行，默认要求确认

当前工具风险：

- `calculator`: `read_only`
- `read_text_file`: `read_only`
- `write_text_file`: `write`

覆盖已有文件必须确认，不能只依靠模型决定。

## 重新规划

允许在工具失败、结果不符合预期、参数依赖前一步结果、方案不可执行或用户修改目标时重新规划。

当前实现提供受限重规划机制：

- 最大次数由 `PLANNER_MAX_REPLANS` 控制，默认 2
- 已完成步骤和有效结果会保留
- 原计划和修改原因会进入 `replan_history`
- 达到上限后停止，避免无限循环

第一版不会后台自动寻找新任务。

## Planner 与 Memory

规划前会检索相关长期记忆，并明确区分当前用户请求、历史长期记忆和当前计划假设。

计划完成后可以通过候选记忆机制保存长期决策、项目进展或未完成任务，但不会把每个步骤都写入长期记忆。

具体开发过程中遇到的有价值工程问题，不直接写入长期记忆数据库，而是记录到：

```text
docs/development_journal.md
```

长期记忆只保存“需要做复盘记录”的规则和偏好；复盘条目本身用中文 Markdown 保存，方便简历项目答辩时查阅。

## 安全边界

- 不允许模型直接执行 Python、Shell 或系统命令。
- 只允许调用注册过的工具。
- 工具参数由 Pydantic 校验。
- 文件工具会解析真实路径并拒绝访问 `workspace/` 之外的路径。
- 工具异常会被捕获并作为工具结果返回给模型。
- 每次运行都有唯一 `run_id`，日志保存在 `logs/{run_id}.log`。
- 最大工具调用轮数默认是 5，可用 `AGENT_MAX_TOOL_CALLS` 调整。
- 长期记忆只注入检索到的相关片段，不会把整个数据库放入上下文。
- 长期记忆带来源字段，可审计。
- Planner 只能使用 `ToolRegistry` 中真实存在的工具。
- 不允许计划或工具执行任意 Shell、Python 或系统命令。
- 写入、破坏性和外部风险操作有确认机制。

## 隐私与清除记忆

长期记忆保存在本地 SQLite 文件 `workspace/memory.sqlite3`。使用 `/forget <memory_id>` 会软删除单条记忆，让它不再参与检索。

如果需要彻底清空长期记忆，可以在退出程序后删除数据库文件：

```bash
rm workspace/memory.sqlite3
```

当前会话短期记忆和摘要可以用：

```text
/clear-session
```

## 测试

```bash
pytest
```

LLM 相关测试使用 mock，不依赖真实本地模型。

## 当前限制

- 模型必须尽量遵守 JSON 协议；Runtime 只做一次格式修复。
- 工具集很小，不包含外部网络、系统命令或复杂文件操作。
- 长期记忆检索使用关键词和 FTS5，不理解语义相似度。
- 候选记忆提取能力已在 `MemoryWriter` 中提供，但默认最可靠入口仍是 `/remember` 显式写入。
- Planner 第一版是单线程逐步执行，不支持后台自动运行。
- 复杂任务依赖本地模型生成高质量 JSON 计划；计划会被校验，但坏计划仍可能需要用户重试或补充说明。

## 下一阶段

下一阶段可以增加向量检索：保留 `MemoryService` 和 `MemoryStore`，新增 `EmbeddingRetriever`，在检索时合并 FTS5 分数、向量相似度、重要性和时间衰减分数。Planner 方向可以增加更细的结果校验、交互式改计划、计划模板和更强的重新规划策略。
