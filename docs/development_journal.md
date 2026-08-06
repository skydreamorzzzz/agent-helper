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
