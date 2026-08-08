# 开发复盘记录

## 真实 DeepSeek 全链路测试暴露的 Planner 职责边界问题

- 来源：真实 DeepSeek 全链路测试
- 相关模块：Planner、PlanExecutor、ToolRegistry、文件工具
- 结论类型：架构演进记录

### 问题

Planner 初版会把“读取结果中抽取并整理内容”的逻辑塞进 `write_text_file` 的 `content` 参数里，例如让写文件步骤直接引用 `${step_1.result.top_task_1}` 这类并不存在的结构化字段。

这个计划看起来符合“读文件再写文件”的表面流程，但实际不可执行：`write_text_file` 只负责写入文本，不应该承担理解、筛选、总结或抽取任务。

### 解决

新增只读工具 `transform_text`，把复杂文本处理显式建模为独立步骤。

最终链路变为：

```text
read_text_file -> transform_text -> write_text_file
```

### 价值

这个问题说明 Planner 不是只要生成步骤就够了，还必须尊重工具职责边界。工具应保持单一职责：读取工具只读取，转换工具处理文本，写入工具只写入。这样计划更容易校验、执行、审计和调试，也更适合在简历项目答辩中解释架构演进。

## Tavily 深度调研工具的路径边界问题

- 来源：检查新增的 Tavily 深度调研代码并做真实端到端测试
- 相关模块：`write_cited_report`、`PlanValidator`、`PlanExecutor`
- 结论类型：安全边界修正

### 问题

`write_cited_report` 的工具描述要求报告保存到 `workspace/reports/`，执行器也会通过 `resolve_workspace_path` 防止路径穿越。但是最初的参数 schema 只要求 `report_file` 是字符串，没有强制它必须位于 `reports/` 下并以 `.md` 结尾。

这意味着模型生成的计划即使把报告写到 `workspace/` 下其他位置，也可能通过校验。虽然仍然不会逃出 workspace，但这和工具声明的职责边界不一致。

### 解决

把路径约束下沉到 `WriteCitedReportArguments` 的 Pydantic validator：

```text
report_file 必须是相对路径
report_file 必须位于 reports/ 下
report_file 必须以 .md 结尾
report_file 禁止包含 ..
```

同时补充测试，确保非法路径无法通过参数校验。

### 价值

这个问题说明安全边界不能只写在 prompt 或 README 里，必须尽量下沉到 schema、validator 和执行器中。模型负责提出计划，代码负责定义硬约束。

## GAIA 评测日志解析的轨迹统计问题

- 来源：检查新增的 GAIA benchmark harness
- 相关模块：`evals/runner.py`、`src/logging_utils.py`、`AgentRuntime`
- 结论类型：评测可信度修正

### 问题

GAIA runner 需要从 Runtime 日志里提取工具调用轨迹，例如使用了哪些工具、搜索了哪些 query。初版实现直接对日志行执行 `json.loads(line)`。

但项目日志格式是标准 logging 输出：

```text
2026-08-06 12:00:00 INFO {"event":"tool_call", ...}
```

也就是说 JSON 前面有时间戳和 level。直接 `json.loads(line)` 会失败，导致普通 Runtime 路径的工具轨迹统计为空。评测仍能跑完，但报告里的工具使用统计不可信。

### 解决

在解析前先定位日志行中的第一个 `{`，只对后面的 JSON payload 做 `json.loads`。同时补测试覆盖带 timestamp 的日志行。

### 价值

Benchmark 不只是能跑出通过率，还必须保证观测数据可信。评测代码本身也需要测试，否则会出现“模型结果是真的，但评测指标部分失真”的情况。

## Router 从单层判断演进为分层多信号决策

- 来源：检查 GAIA 评测后发现 Router 行为不稳定
- 相关模块：`src/planner/router.py`、`tests/test_router_llm.py`
- 结论类型：路由架构演进

### 问题

原 Router 把几类不同性质的判断混在一起：关键词规则、缺参判断、单工具判断和 LLM 判断。测试暴露出四个具体问题：

1. LLM 可以把“计算 23 * 7”这种强确定性单工具任务误判成 `deep_research`。
2. URL 中的 `/` 会被旧的计算规则误判为数学表达式，导致普通链接解释进入 `single_tool`。
3. 英文 `research` 关键词过宽，“research methods section” 这种普通写作请求会被误判为联网调研。
4. “latest Tavily API pricing” 这类明显需要最新外部信息的请求，在没有 LLM 时反而会走 `direct_answer`。

这些问题说明 Router 不能只是“规则或 LLM 二选一”，也不能让 LLM 覆盖硬约束。

### 解决

把 `RequestRouter` 改成分层多信号系统：

```text
ConstraintRouter -> SemanticRouter -> LLMRouter -> Rule fallback
```

- `ConstraintRouter` 负责不可违反的硬约束和强确定性场景，例如缺参、明确联网调研、最新信息、本地多步骤文件任务、明显单工具任务。
- `SemanticRouter` 负责轻量相似度匹配典型任务形态，作为未来 embedding router 的替换点。
- `LLMRouter` 只在约束层没有最终结论时理解复杂语义。
- 兜底规则负责处理 LLM 输出非法或信号不足。

### 价值

这次改动把“路由”从一个 if/else 函数提升成了可解释的决策系统。面试时可以说明：我们先用测试暴露误路由，再把路由拆成硬约束、语义相似和 LLM 理解三个层次，避免单一信号导致错误，同时为未来 embedding router 留出扩展点。
