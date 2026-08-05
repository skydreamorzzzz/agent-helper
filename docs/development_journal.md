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

