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

## 外部信息请求被过粗路由到 Deep Research

- 来源：针对 Tavily 查询和 Router 的专项测试
- 相关模块：`src/planner/router.py`、`src/planner/prompts.py`、`src/cli.py`、`ToolRegistry`
- 结论类型：路由粒度和能力边界修正

### 问题

Router 之前把“需要外部信息”几乎都归到 `deep_research`。这会导致两个问题：

1. “What is the latest Tavily API pricing?” 或 “Who is the current CEO of X?” 这种只需要少量搜索即可回答的问题，被迫进入深度调研计划，成本和交互复杂度都过高。
2. `latest`、`price`、`version` 这类弱关键词如果被 ConstraintRouter 直接定死，容易误伤普通解释类问题，例如“解释一下软件版本号是什么”。

更深层的问题是，路由层同时承担了“识别实时信息需求”和“决定是否做研究报告”两个不同职责，导致粒度过粗。

### 解决

新增 `web_lookup` 路由，专门表示简单实时或外部信息查询：

```text
direct_answer     普通知识或聊天
single_tool       明确单工具任务
web_lookup        少量联网搜索后直接回答
planned_task      本地多步骤工具任务
deep_research     多来源调研、比较、报告产出
clarification     关键参数缺失
```

同时收紧 ConstraintRouter：

- 缺参、明显单工具、明确多步骤文件任务、显式“调研/研究报告”仍是硬约束。
- `latest/current/价格/版本` 这类词只作为软启发式信号，默认产出 `web_lookup`，不再直接 final 到 `deep_research`。
- 如果 LLM 把明显的简单实时查询误判成 `direct_answer`，融合层会优先 `web_lookup`，避免过时回答；如果 LLM 判断成 `deep_research` 或 `clarification`，仍允许更复杂路线覆盖。

工具能力也按 registry 分层：

```text
core_registry      本地基础工具
lookup_registry    core + search_web
research_registry  lookup + write_cited_report
```

这样普通 Runtime 默认没有联网能力，`web_lookup` 可以搜索但不能写研究报告，`deep_research` 才能搜索并生成带引用报告。

### 价值

这次问题说明 Router 不只是分类器，还承担能力授权边界。好的路由设计需要区分硬约束和软启发式，也要区分“查一下即可回答”和“需要系统研究产出”。面试时可以说明：先用失败用例证明 `deep_research` 粒度过粗，再通过新增中间路由、收紧 ConstraintRouter、分层 ToolRegistry，把成本、权限和任务复杂度对齐。

## Web Lookup 的执行闭环问题

- 来源：修复 `web_lookup` 端到端行为测试
- 相关模块：`RequestRouter`、`AgentRuntime`、`lookup_registry`、`search_web`
- 结论类型：路由结果到执行策略的闭环

### 问题

新增 `web_lookup` 后，CLI 会把请求交给带有 `search_web` 的 `lookup_runtime`。但这只解决了“模型有没有搜索能力”的问题，没有保证“模型一定会使用搜索能力”。

如果模型在第一轮直接返回：

```json
{"type":"final_answer","content":"..."}
```

Runtime 之前会直接接受最终答案。对于需要最新信息的问题，这等于绕过了 `web_lookup` 的语义，可能产生过时或未经验证的回答。

同时还发现 ConstraintRouter 的判断顺序有隐患：显式“调研最新价格并比较套餐”会先命中 current-information 软启发式，导致本应进入 `deep_research` 的请求被降级为 `web_lookup`。

### 解决

先调整 ConstraintRouter 优先级：

```text
缺参 hard constraint
-> 显式 research / 调研 / 研究报告 hard intent
-> latest/current/价格/版本 soft web_lookup signal
```

再给 `AgentRuntime.run()` 增加轻量执行策略：

```text
required_tool="search_web"
```

当 `web_lookup` 运行时：

1. system context 明确告诉模型本次请求被路由为 `web_lookup`，必须先调用 `search_web`。
2. Runtime 在代码层跟踪 `search_web` 是否成功执行。
3. 如果模型在搜索前直接返回 `final_answer`，Runtime 拒收一次，并要求模型先调用 `search_web`。
4. 只有 `search_web` 返回 `ok=true` 后，Runtime 才接受 `final_answer`。
5. 普通 core Runtime 不传 `required_tool`，仍允许首轮直接回答。

### 价值

这次问题说明 Agent 的能力控制不能停在“路由选择”和“工具可见性”两层。完整闭环应该是：

```text
route -> capability -> execution policy
```

`web_lookup` 不只是让模型“看见 search_web”，还要让执行层验证 search_web 确实发生。面试时可以把它解释为从 prompt-only 约束升级为 code-enforced policy：prompt 负责指导模型，Runtime 负责兜底执行语义。

## risk_level 只作为元数据导致 Runtime 可绕过写确认

- 来源：工具执行安全策略专项检查
- 相关模块：`AgentRuntime`、`PlanExecutor`、`ToolRegistry`、`ToolExecutionPolicy`
- 结论类型：安全策略统一

### 问题

工具已经声明了 `risk_level`，但它最初主要是 prompt 元数据。`PlanExecutor` 有自己的确认逻辑，会在写入、破坏性或外部风险工具前暂停确认；普通 `AgentRuntime` 却直接调用 `ToolRegistry.execute()`。

这意味着同一个 `write_text_file`：

```text
Planner 路径 -> 需要确认
Runtime 路径 -> 可能直接执行
```

安全策略被分散在不同执行路径里，导致行为不一致。只要模型在普通 JSON tool_call 协议里选择写文件，就可能绕过 Planner 的确认机制。

### 解决

新增统一的 `ToolExecutionPolicy`：

```text
ToolExecutionPolicy.evaluate(tool, risk_level, arguments, confirm_write_actions)
  -> allow / confirm / deny
```

第一版规则保持保守：

- `read_only` 自动允许。
- `write` 在 `CONFIRM_WRITE_ACTIONS=true` 时要求确认，关闭配置时允许。
- `destructive` 始终要求确认。
- `external` 保持原策略，要求确认。
- `write_text_file + overwrite=true` 无论全局写确认是否关闭，都要求确认。

然后让两条路径复用同一个策略：

```text
AgentRuntime
        ↘
      ToolExecutionPolicy
        ↗
PlanExecutor
```

Runtime 如果需要确认但没有 callback，或者用户拒绝，会安全停止，不执行工具。PlanExecutor 保留原状态机，只把确认原因来源切换为统一 policy。

### 价值

这次问题说明安全边界不能只靠工具声明，也不能只在某一条执行路径里实现。工具元数据只有被执行层统一解释，才会成为真正的权限策略。面试时可以强调：发现 Runtime 和 Planner 对同一风险工具行为不一致后，把风险判断抽成共享 policy，避免未来新增执行入口时再次绕过确认。

## ToolExecutionPolicy 的参数规范化和 DENY 语义收尾

- 来源：统一工具执行权限策略的最终验收
- 相关模块：`ToolRegistry`、`ToolExecutionPolicy`、`AgentRuntime`、`PlanExecutor`
- 结论类型：安全策略细节修正

### 问题

统一 policy 后还有两个细节风险：

1. Runtime 和 PlanExecutor 可能把模型给出的 raw arguments 直接传给 policy。这样 `"overwrite": "false"` 这种字符串在 Python 里如果直接 `bool("false")` 会变成 `True`，导致 policy 基于未规范化参数做出错误判断。
2. PlanExecutor 中 `PolicyAction.DENY` 如果通过异常进入普通 step failure 流程，会被当成临时工具失败处理，进而触发 retry 或 replan。但权限拒绝不是临时失败，重试和重规划都不应该绕过它。

### 解决

在 `ToolRegistry` 增加统一参数规范化入口：

```text
raw arguments
-> Pydantic schema validation
-> model_dump normalized arguments
-> ToolExecutionPolicy
-> confirmation
-> execute
```

Runtime 和 PlanExecutor 都使用同一份 normalized arguments：

- policy 看到 normalized arguments。
- confirmation 展示 normalized arguments。
- 工具执行也使用 normalized arguments。

PlanExecutor 遇到 `PolicyAction.DENY` 时直接停止当前 plan：

```text
stopped_reason = tool_policy_denied
retry_count 不增加
不触发 replan
不执行工具
```

### 价值

这次收尾说明权限策略必须建立在“已校验、已规范化”的输入上，否则安全判断会受模型 JSON 表达细节影响。同时，权限拒绝应该是终态决策，而不是普通错误恢复流程的一部分。面试时可以把它解释为：策略层不只统一入口，还明确了参数语义和失败语义。
