# 开发复盘记录

## E2E Evaluation Semantics v1.1：区分 deterministic integration 与真实 Agent 质量

- 来源：审查 `End-to-End Agent Evaluation v1` 指标语义
- 相关模块：`evals/agent/dataset.jsonl`、`evals/agent/runner.py`、`evals/agent/evaluators.py`、`evals/agent/report.py`
- 结论类型：评测语义修正

### 问题

第一版 E2E evaluation 已经能跑通完整链路：

```text
用户请求 -> Router -> Runtime / Planner -> Tools -> Policy / Confirmation -> Final Answer
```

但进一步审查发现，v1 的 `87.5%` 不能直接解释为真实 Agent task success rate，原因是：

```text
route correctness != task success
tool proposal != tool execution
permission rejection != task failure
deterministic fake-model integration != real Agent quality
```

具体来说：

- `CountingFakeLLM` 在 Router 阶段使用 `expected_route` 生成确定性输出，Planner fake 也按 case 构造预期计划，因此这个 suite 主要验证可控模型行为下的系统集成 contract。
- v1 evaluator 把 route mismatch 直接判为 task failure，但真实任务可能走了不同 route 仍然完成最终 artifact。
- v1 的 `tool_calls` 混合了模型提出工具和工具真正执行，导致 permission rejection 场景下指标不清晰。
- v1 把部分用户拒绝 confirmation 后的“无副作用安全停止”算成失败，但这类 case 的正确结果应该是 safety success。
- `known_failure_*` 和 normal tasks 混在 overall success rate 中，使得指标既不像真实任务成功率，也不像 regression pass rate。

### 解决

将当前 suite 正式定义为：

```text
deterministic_integration
```

它衡量的是：

```text
给定稳定/可控模型行为时，
Router / Runtime / Planner / Tools / Policy / Memory
之间的系统集成和执行 contract 是否成立。
```

同时预留未来独立的：

```text
live_e2e
```

用于真实 LLM/API 下的用户任务成功率评测，但两者不混合。

v1.1 的主要语义修正：

- `route_correct` 成为独立诊断字段，route mismatch 不再自动导致 `task_success=false`。
- `task_success` 只看最终 contract：状态、工具副作用、artifact、文件内容、最终回答和 permission semantics。
- permission rejection 如果符合“用户拒绝确认 -> 不产生副作用”，计为成功的安全行为。
- 工具指标拆分为：

```text
tool_proposals
tool_execution_attempts
tool_execution_successes
tool_execution_failures
tool_policy_rejections
```

- `tool_execution_success_rate` 只用真正执行过的工具计算：

```text
execution_successes / (execution_successes + execution_failures)
```

policy rejected 但未执行的工具不计入 execution success，也不计入 execution failure。

- dataset 增加 `suite`，至少区分：

```text
normal
regression
```

- memory cases 明确表述为 `memory_retrieval`，即预填充 memory 后验证检索和注入路径，不宣称覆盖完整 memory write + retrieval E2E。

### Baseline v1.1

确定性 integration baseline：

```text
overall integration pass rate = 30 / 32 = 93.8%
normal task pass rate = 28 / 29 = 96.6%
regression case pass rate = 2 / 3 = 66.7%
route accuracy = 96.9%
```

工具指标：

```text
tool proposals = 44
tool execution attempts = 40
tool execution successes = 34
tool execution failures = 6
tool policy rejections = 4
tool execution success rate = 85.0%
```

失败阶段分布：

```text
tool_execution: 2
```

代表性失败：

- `agent_007`：route mismatch 本身不再直接导致失败；真正失败原因是错误 route 进入了 bad read execution chain，没有到达预期的 write permission rejection contract。
- `agent_028`：Planner 边界 regression 仍然暴露 transform/write contract 缺失，未提出也未执行 `transform_text`。

### 价值

这次修正把 E2E benchmark 从“看起来有成功率”推进到“指标语义正确”。现在可以分别回答：

- deterministic integration contract 是否成立；
- normal task 和 regression case 各自表现如何；
- route accuracy 是多少，但它是否真的影响最终任务；
- permission rejection 是安全成功还是任务失败；
- tool proposal、execution、policy rejection 各自发生了多少。

这为后续运行真实 `live_e2e` baseline 打好了基础：live benchmark 可以复用 evaluator 和 report 语义，但不能和 deterministic integration 的分数混在一起。

## 从模块级验证转向 End-to-End Agent Evaluation

- 来源：建立 `End-to-End Agent Evaluation v1`
- 相关模块：`RequestRouter`、`AgentRuntime`、`StructuredPlanner`、`PlanExecutor`、`ToolRegistry`、`ToolExecutionPolicy`、`MemoryService`、`evals/agent`
- 结论类型：评测体系阶段演进

### 背景

此前项目已经分别建立了 Router benchmark、Tool Policy 测试、Planner 测试、Runtime 测试、Memory 测试等模块级验证。这些测试能回答“单个模块是否符合预期”，也记录了 Router 从规则、LLM 到 embedding / hybrid cascade 的工程演进。

但模块级指标无法回答最终问题：

```text
整个 Agent 能否成功完成真实用户任务？
```

一个真实任务会穿过完整链路：

```text
用户请求 -> Router -> Runtime / Planner -> Tools -> Policy / Confirmation -> Final Answer
```

任何一个环节失败，用户看到的都是任务失败。因此本阶段冻结 Router threshold、prototype 和 router-v2 的继续微调，把优化目标从局部路由指标转向端到端 task success rate 和 failure-stage attribution。

### 解决

新增 `evals/agent/`：

```text
evals/agent/
  dataset.jsonl
  runner.py
  evaluators.py
  report.py
  baselines/v1.md
```

第一版 dataset 包含 32 条确定性任务，覆盖：

```text
direct_answer
single_tool
web_lookup
planned_task
deep_research
clarification
memory-related task
tool permission / confirmation
failure / invalid input
recovery
```

评测优先使用程序化验证，不使用 LLM-as-a-Judge：

- calculator 检查最终数值。
- read/write 检查文件是否存在和内容片段。
- planned task 检查最终 artifact。
- web_lookup 检查 `search_web` 是否被实际调用。
- permission rejection 检查副作用没有发生。
- invalid input 检查系统安全停止或返回工具错误，而不是崩溃。

deterministic runner 使用 fake LLM、fake web search 和 per-case 临时 workspace，让关键链路可以在本地和 CI 中稳定重放；live evaluation 预留为后续独立模式，不和 deterministic baseline 混合。

每条 case 记录：

```text
expected_route / actual_route
tool_calls / tool_failures
retry_count / replan_count
llm_calls
final_status / stopped_reason
task_success
failure_stage
latency_ms
```

`failure_stage` 用于把失败归因到：

```text
routing
planning
argument_resolution
tool_execution
permission
recovery
final_answer
memory
runner
```

### Baseline v1

确定性 E2E baseline：

```text
overall task success = 28 / 32 = 87.5%
route accuracy = 96.9%
tool execution success rate = 85.0%
average tool calls = 1.25
average LLM calls = 2.59
total retry count = 3
total replan count = 0
```

失败阶段分布：

```text
tool_execution: 2
permission: 1
routing: 1
```

代表性失败：

- `agent_007`：写文件请求被路由为 `planned_task`，而不是单工具写入路径。
- `agent_028`：Planner 边界压力用例完成了写入，但没有经过 `transform_text`，暴露 transform/write 链的 artifact 验证价值。
- `agent_029`：required `search_web` 在空搜索结果下无法满足，Runtime 安全停止为 `required_tool_missing`。
- `agent_030`：权限拒绝导致预期 artifact 不存在，记录为 permission 阶段失败。

### 价值

这次演进把项目评测目标从“每个模块看起来能工作”推进到“真实任务是否完成”。端到端评测不只给出 success rate，还能告诉下一阶段最值得修哪里。

本轮遵守实验纪律：发现 Router、Planner、Tool、Policy、Recovery、Final Answer 的失败只记录，不立即修复。下一阶段应基于 failure-stage distribution 选择投入最高的模块，而不是继续为了局部指标微调 Router。

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

## Router 优化从手工错例转向 Benchmark 驱动

- 来源：建立独立 Router Evaluation Dataset 和评测工具
- 相关模块：`RequestRouter`、`ConstraintRouter`、`SemanticRouter`、`LLMRouter`、`evals/router`
- 结论类型：评测体系建设

### 问题

过去 Router 的修改主要由少量手工失败案例驱动。例如发现 `latest` 被过度路由到 `deep_research`，或者发现英文 `research` 会误伤普通写作请求，就针对单个问题修规则。

这种方式能快速修 bug，但无法回答几个更关键的问题：

- 哪类请求最容易误路由；
- Rule / LLM 各自贡献多少；
- 后续引入 EmbeddingRouter 是否真的改善；
- 一次规则改进是不是只修了一个 case，又破坏了另一个 case；
- 当前 failure mode 是中文、英文、缺参、多步骤还是最新事实查询导致的。

### 解决

新增独立 Router benchmark：

```text
evals/router/
  dataset.jsonl
  runner.py
  report.py
```

数据集第一版覆盖六类路由：

```text
direct_answer / single_tool / web_lookup / planned_task / deep_research / clarification
```

同时覆盖中英文、稳定知识 vs 最新事实、`web_lookup` vs `deep_research`、单工具 vs 多步骤、URL/日期/数字干扰、弱 `research` 关键词、缺参和少量 OOD 表述。

Runner 支持：

```text
rule_only   不调用 LLM，使用 deterministic + semantic + fallback
full_router 当前 RequestRouter + LLM
```

报告输出 accuracy、macro-F1、per-route precision/recall/F1、confusion matrix、category accuracy、LLM call count、LLM escalation rate 和代表性失败案例。

### 价值

这一步把 Router 迭代从 case-by-case debugging 升级为 measurable iteration。下一阶段如果引入 EmbeddingRouter，可以直接比较：

```text
Rule
vs
Embedding
vs
LLM
vs
Hybrid Cascade
```

而不是凭感觉判断“好像更聪明了”。这也更适合在简历项目中解释工程成熟度：先建立评测基线，再做模型或算法升级。

## Router Benchmark v1 固化与 Ablation Baseline

- 来源：Router benchmark v1 收尾
- 相关模块：`evals/router/dataset.jsonl`、`evals/router/runner.py`、`evals/router/baselines/v1.md`
- 结论类型：评测基线固化

### 问题

第一版 Router dataset 和 runner 已经能暴露 failure mode，但还缺少两个工程约束：

1. 没有固定 dev/test split，后续调规则或原型时容易在同一批样本上反复优化，无法区分“调参集”和“阶段验收集”。
2. 原 `rule_only` 命名不准确，因为它实际包含 `ConstraintRouter + SemanticRouter(Jaccard) + fallback`，不是纯规则约束。

此外，`llm_escalation_rate` 如果直接用 LLM 总调用次数除以样本数，未来一个样本多次调用 LLM 时会混淆“调用成本”和“升级样本比例”。

### 解决

把 60 条 dataset 固化为 `router-v1`，并在每条样本中写入稳定 split：

```text
dev  42 条，用于后续 threshold / prototype / cascade 调整
test 18 条，用于阶段性验收
```

每个 split 都覆盖六种 route：

```text
direct_answer / single_tool / web_lookup / planned_task / deep_research / clarification
```

Runner mode 改成更明确的 ablation：

```text
constraint_only  = ConstraintRouter + direct_answer fallback
lexical_baseline = RequestRouter without LLM，包含 Constraint + Jaccard Semantic + fallback
current_hybrid   = 当前 RequestRouter + LLM
```

同时区分两个 LLM 指标：

```text
llm_call_count          LLM 总调用次数
llm_escalated_examples  实际调用过 LLM 的样本数
llm_escalation_rate     llm_escalated_examples / total
```

### 价值

现在 Router benchmark v1 具备固定数据、dev/test split、ablation mode 和可提交 baseline。下一阶段引入 EmbeddingRouter 时，可以先在 dev 上调试，再用 test 做阶段验收，并且能分别比较 constraint、lexical、LLM 和未来 embedding/cascade 的真实贡献。

## EmbeddingRouter v1：低成本语义层的第一轮验证

- 来源：实现并评估第一版 `EmbeddingRouter`
- 相关模块：`src/planner/embedding_router.py`、`evals/router/runner.py`、`evals/router/baselines/embedding_v1.md`
- 结论类型：路由实验

### 问题

Router benchmark v1 的 baseline 显示：

```text
constraint_only:
accuracy 0.667
macro-F1 0.621

lexical_baseline:
accuracy 0.667
macro-F1 0.621

current_hybrid:
accuracy 0.833
macro-F1 0.812
LLM escalation rate 0.556
```

这说明当前 Jaccard `SemanticRouter` 在独立 test split 上没有带来可测增益；而 LLM Router 虽然显著提高准确率，但超过一半样本需要升级到 LLM。

因此引入 embedding 不是为了堆技术，而是尝试建立一个成本更低的中间语义层：

```text
Rule -> Embedding -> LLM
```

目标是在保持准确率的同时，降低 LLM Router 的调用比例。

### 解决

新增独立 `EmbeddingRouter`，没有直接大改 `RequestRouter`：

```text
user query
-> embedder.encode()
-> 与每个 route 的 prototype embedding 比较
-> 计算 best similarity / second similarity / margin
-> 分数和 margin 达标才返回 route
-> 不确定时交给后续 LLM 或 fallback
```

第一版使用本地、无额外依赖的 `HashingEmbedder` 作为可复现 baseline，并通过 `Embedder` 协议把 embedding provider 与 Router 解耦。prototype embedding 在初始化时缓存，避免每次请求重复编码。

评测新增两个 mode：

```text
embedding_only   ConstraintRouter -> EmbeddingRouter -> direct_answer fallback
embedding_hybrid ConstraintRouter -> EmbeddingRouter -> uncertain 时 LLMRouter
```

阈值只根据 dev split 选择：

```text
similarity_threshold = 0.32
margin_threshold     = 0.04
```

### 结果

test split 上：

```text
lexical_baseline:
accuracy 0.667
macro-F1 0.621
LLM escalation rate 0.000

embedding_only:
accuracy 0.833
macro-F1 0.813
LLM escalation rate 0.000

current_hybrid:
accuracy 0.833
macro-F1 0.812
LLM escalation rate 0.556

embedding_hybrid:
accuracy 0.778
macro-F1 0.758
LLM escalation rate 0.333
```

`embedding_only` 明显优于 Jaccard lexical baseline，并且在 test 上达到与 `current_hybrid` 接近的准确率，同时不调用 LLM。

但 `embedding_hybrid` 虽然把 LLM escalation rate 从 `0.556` 降到 `0.333`，准确率也从 `0.833` 降到 `0.778`。主要原因是 embedding 一旦做出高置信错误判断，当前 cascade 不会再交给 LLM 修正。

### 价值

这轮实验给出了三个面试时值得讲清楚的工程结论：

1. Jaccard 语义层可以被 benchmark 证明收益不足，而不是凭感觉说“不够智能”。
2. embedding 作为中间层确实能修复一批 lexical failure，例如最新股价、multi-source pricing comparison、缺参请求。
3. cascade 不是简单把 Rule / Embedding / LLM 串起来就结束；还要设计硬约束边界、置信度阈值和 LLM override policy，否则会用更低成本换来不可接受的错路由。

下一阶段更值得做的是调整 hard constraint / cascade policy，并在同一 `Embedder` 协议下接入真正的多语言 sentence embedding provider，而不是直接重构 Planner 或引入向量数据库。

## Router Embedding 实验方法收尾：元数据真实性与严格 Holdout

- 来源：Router embedding 实验可复现性收尾
- 相关模块：`src/planner/embedding_router.py`、`evals/router/runner.py`、`evals/router/dataset_v2.jsonl`
- 结论类型：实验方法修正

### 问题

上一轮 `EmbeddingRouter` v1 暴露出三个实验方法问题：

1. `HashingEmbedder` 实际是 hashed lexical vector baseline，不是 neural sentence embedding。
2. runner 里 `HashingEmbedder(model_name=embedding_model)` 会导致传入任意模型名时，实际算法仍是 hashing，但报告可能写成 `bge-m3` 或其他模型名，造成实验元数据失真。
3. `router-v1/test` 已经被用于 failure analysis 和 prototype 讨论，因此不能再被当成严格独立 holdout score。

这些问题如果不修，会让项目在面试或复盘时很难解释清楚：到底是在比较算法，还是只是在比较一个标签。

### 解决

把 embedding 配置显式拆成：

```text
embedding_provider
embedding_model
```

当前支持：

```text
provider = hashing
model    = hashing-multilingual-v1

provider = sentence_transformers
model    = BAAI/bge-small-zh-v1.5
```

未知 provider 会直接报错。`hashing` provider 也不能被任意 model 名伪装，例如不能把 hashing run 标成 `bge-m3`。

同时修正 `EmbeddingRouter` 的 cosine 语义：Router 内部负责向量归一化和 zero vector 处理，不再隐含要求 provider 返回 normalized vectors。这样后续替换真实 embedding provider 时，不需要依赖 provider 的默认 normalization 配置。

新增真实 sentence embedding provider：

```text
SentenceTransformerEmbedder
```

它通过 `sentence-transformers` 加载模型；依赖未安装时给出明确错误，不静默 fallback 到 hashing。本轮实际可复现模型为本地缓存的 `BAAI/bge-small-zh-v1.5`，并记录 `local_files_only=true`。

最后新增 `router-v2`：

```text
evals/router/dataset_v2.jsonl
```

`router-v2` 是 test-only holdout，用于第一次 untouched evaluation。流程固定为：

```text
router-v1 dev
-> 保持 prototype 不变
-> 保持 similarity_threshold = 0.32
-> 保持 margin_threshold = 0.04
-> 第一次运行 router-v2
-> 不根据 v2 failure 修改本轮配置
```

### 结果

`router-v2` 首次 holdout：

```text
lexical_baseline:
accuracy 0.361
macro-F1 0.326

hashing_only:
accuracy 0.444
macro-F1 0.446

sentence_embedding_only:
accuracy 0.583
macro-F1 0.590

current_hybrid:
accuracy 0.889
macro-F1 0.883
LLM escalation rate 0.778

sentence_embedding_hybrid:
accuracy 0.861
macro-F1 0.860
LLM escalation rate 0.444
```

### 价值

这次收尾把 embedding 实验从“看起来用了 embedding”改成了可审计实验：

- provider 和 model 与实际执行实现一致；
- hashing baseline 不再冒充 neural sentence embedding；
- cosine 计算对任意 provider 更稳健；
- `router-v2` 提供了新的 untouched holdout；
- sentence embedding hybrid 的成本/准确率 trade-off 可以量化。

当前最重要的剩余问题不是继续调 embedding threshold，而是 Router cascade policy：hard constraint 仍会抢先判定英文多步骤文件任务；embedding 的高置信错误也可能阻止 LLM 修正。这些都应该进入下一轮 Router 策略设计，而不是在本轮根据 v2 错例修规则。

## Live E2E Baseline：真实模型端到端基线

- 来源：Live E2E Baseline
- 相关模块：`evals/agent/live_dataset.jsonl`、`evals/agent/live_runner.py`、`evals/agent/report.py`
- 结论类型：真实模型 baseline，而不是优化分数

### 背景

项目已经完成了 Runtime、Router 实验、Embedding / Hybrid Routing、Router Holdout、Deterministic E2E Evaluation 和 E2E Evaluation Semantics Refinement。

上一阶段的关键修正是把：

```text
deterministic integration regression
```

和：

```text
real agent task success
```

明确分开。`CountingFakeLLM` 能稳定验证 Router / Runtime / Planner / Tools / Policy / Memory 的集成 contract，但不能代表真实模型质量。

因此本轮新增 `live_e2e`，让评测真正调用当前配置的 LLM，同时继续保持 Runtime、Router、Planner、Tool Policy、Memory 的业务行为不变。

### 实现

新增独立 live dataset：

```text
evals/agent/live_dataset.jsonl
```

覆盖：

- direct answer
- single tool calculator
- single tool file read
- planned read / transform / write
- planned calculate / write
- clarification
- write confirmation rejected
- overwrite confirmation rejected
- memory retrieval
- invalid / boundary regression

新增 runner：

```text
evals/agent/live_runner.py
```

它复用当前 CLI 的真实构造方式：

- `RequestRouter`
- `AgentRuntime`
- `StructuredPlanner`
- `PlanExecutor`
- `ToolRegistry`
- `ToolExecutionPolicy`
- `MemoryService`
- `.env` 中的 `LocalLLMClient` 配置

评测侧只增加 instrumentation：

- `CountingLLMClient` 统计 LLM calls；
- 每个 case 使用临时 workspace；
- confirmation 由 dataset 的 `confirmation` 字段控制；
- Runtime 日志解析 tool proposal / execution / policy rejection；
- Plan 执行结果记录 retry / replan / step status；
- failure stage 归因到 `router`、`planner`、`tool_execution`、`policy`、`memory`、`runtime`、`final_answer` 或 `unknown`。

### 结果

最终 live baseline：

```text
result artifact: evals/agent/results/live_20260811_230109/
dataset version: agent-live-e2e-v1
model: deepseek-v4-flash
provider: https://api.deepseek.com
overall live pass rate: 9/10 = 90.0%
normal task pass rate: 9/9 = 100.0%
regression case pass rate: 0/1 = 0.0%
route accuracy: 10/10 = 100.0%
average latency: 7384.3 ms
average LLM calls: 2.10
tool proposals: 9
tool execution attempts: 7
tool execution successes: 7
tool execution failures: 0
tool policy rejections: 2
retry count: 0
replan count: 0
```

失败分布：

```text
runtime: 1
```

唯一失败：

```text
live_010
category: failure_boundary_invalid_tool_args
route: single_tool
stage: runtime
```

Router 正确选择 `single_tool`，但普通 `single_tool` 路径并没有像 `web_lookup` 一样向 Runtime 传入 `required_tool`。真实模型直接做了安全拒绝，没有调用 `calculator`，因此没有触发 calculator 的非法表达式校验错误。用户可见行为是安全的，但不满足当前 regression contract。

第一次 live run 还观察到一次未复现的 `llm_call_failed`：文件读取工具已成功执行，但后续模型回答阶段失败。这说明 live E2E 必须继续把 provider/runtime availability failure 和 deterministic integration failure 分开看。

### 结论

这轮没有暴露出正常任务中的 Router、Planner、Tool、Policy 或 Memory 失败。当前最有价值的下一步不是继续调 Router threshold，也不是优化 embedding prototype，而是明确 Runtime 对 routed tool tasks 的 contract：

```text
single_tool 是否应该像 web_lookup 一样强制 required_tool？
```

同时 live dataset 还太小，下一阶段应该先扩展真实任务覆盖面，再根据 failure-stage distribution 决定是否优化 Runtime、Planner、Policy 或 Router。

## Evaluation Cleanup + Live Dataset Expansion

- 来源：Evaluation Cleanup + Live Dataset Expansion
- 相关模块：`evals/agent/semantics.py`、`evals/agent/evaluators.py`、`evals/agent/runner.py`、`evals/agent/live_runner.py`、`evals/agent/report.py`、`evals/agent/live_dataset.jsonl`
- 结论类型：measurement semantics cleanup，不是 Agent 行为优化

### 为什么统一 evaluation semantics

上一轮 live baseline 已经证明真实模型路径可跑通，但 deterministic 和 live 的 failure stage 命名还不完全一致：deterministic 使用 `routing` / `runner`，live 又把部分结果映射到 `router` / `runtime`。这会让两个 benchmark 在报告层看起来不可直接比较。

本轮把 failure taxonomy 收敛为：

```text
routing
planning
plan_validation
argument_resolution
tool_execution
permission
memory
recovery
final_answer
runtime
unknown
```

兼容映射保留在 evaluation 层：旧的 `router` 会归一成 `routing`，旧的 `runner` 会归一成 `runtime`。这只改变报告和归因命名，不改变 task success semantics。

### Tool proposal 语义

之前 `tool_proposals` 同时承载了两种不同含义：

- Runtime 路径中，来自模型输出的 `tool_call`；
- Planner 路径中，从 `plan.steps` 推导出的计划步骤。

这两者都对旧 expected contract 有用，但不能伪装成同一个观测指标。因此本轮新增拆分字段：

```text
model_tool_proposals    模型/runtime 实际提出的 tool_call
planned_tool_steps      Planner 产出的计划工具步骤
actual_tool_executions  真实进入 ToolRegistry 或 executor 的执行尝试
```

`tool_proposals` 暂时保留为 legacy assertion 字段，等于 `model_tool_proposals + planned_tool_steps`，用于兼容 deterministic 和 live dataset 中已有 expected contract。报告中现在同时展示 legacy 总数和拆分后的解释性指标。

### 测试补充

新增 evaluation 测试覆盖：

- live dataset schema、ID 唯一性、case 数量和覆盖类别；
- canonical failure taxonomy 与旧 stage alias；
- report generation 中的拆分工具指标；
- model proposal、planned step、actual execution 三者隔离；
- deterministic planner record 与 live planner record 不互相污染；
- runtime record 不产生 planned step。

全量测试结果：

```text
126 passed
```

### Live dataset 扩展

`evals/agent/live_dataset.jsonl` 从 10 条扩展到 35 条，dataset version 更新为：

```text
agent-live-e2e-v1.1
```

新增覆盖范围包括：

- single_tool calculator / file read / file write / nested write / no-write boundary；
- invalid tool arguments：division by zero、missing file、unsafe calculator expression、invalid calculator text；
- planned task：read-transform-write、calculate-write、extract max、short summary、multi-file merge；
- ambiguous request / missing target clarification；
- policy / confirmation：single write rejection、overwrite rejection、planned write rejection、planned overwrite rejection；
- memory retrieval：project name、language preference、no-tool memory answer；
- Router 与 Runtime contract 冲突；
- Planner / artifact failure boundary；
- model protocol failure boundary；
- 少量 web lookup 与 deep research case。

web / deep research 只占少量 case，避免外部 API 成为整个 benchmark 的主要噪声来源。

### 新 Live E2E baseline

最终运行 artifact：

```text
evals/agent/results/live_20260813_221538/
```

配置：

```text
mode: live_e2e
dataset version: agent-live-e2e-v1.1
model: deepseek-v4-flash
provider: https://api.deepseek.com
tavily_configured: true
```

结果：

```text
overall live pass rate: 28/35 = 80.0%
normal task pass rate: 24/30 = 80.0%
regression case pass rate: 4/5 = 80.0%
route accuracy: 33/35 = 94.3%
average latency: 10092.3 ms
average LLM calls: 2.43
tool proposals (legacy assertions): 48
model/runtime tool proposals: 16
planned tool steps: 32
tool execution attempts: 44
tool execution successes: 38
tool execution failures: 6
tool policy rejections: 4
retry count: 4
replan count: 0
```

失败分布：

```text
tool_execution: 4
runtime: 2
routing: 1
```

代表性失败：

- `live_010`：`single_tool` 路径没有强制 calculator 调用，模型直接 final answer，未触发 calculator invalid-argument contract。
- `live_011`：calculator boundary 期望调用 calculator，但模型直接 final answer，未执行工具。
- `live_023`：short summary artifact 缺少 expected content。
- `live_024`：merge artifact 缺少两个来源文件的 expected content。
- `live_030`：memory no-tool answer 期望 direct answer，实际 route 到 clarification。
- `live_032`：existing file write 期望不 overwrite 并返回 `FileExistsError`，实际文件被覆盖。
- `live_035`：deep research completed，但 expected report artifact 未生成。

### 下一轮最值得优化什么

不要先调 Router threshold，也不要改 embedding prototype。新的 failure distribution 指向更靠近执行 contract 的问题：

1. Runtime / tool contract：`single_tool` 是否需要 required-tool 语义，尤其 invalid args 和 calculator boundary。
2. Tool argument discipline：write overwrite 默认与模型参数之间的 contract 是否足够硬。
3. Planner artifact guarantee：`completed` 是否必须绑定 expected artifact existence，尤其 deep research。
4. Memory routing boundary：带有显式“根据记忆回答”的请求为什么会落到 clarification。

这些都应该在下一轮作为优化候选，而不是本轮为了提高 live score 立即修改 Agent 行为。

## Evaluation Semantics Finalization：task success 与 execution contract 分离

- 来源：Evaluation Semantics Finalization
- 相关模块：`evals/agent/evaluators.py`、`evals/agent/semantics.py`、`evals/agent/report.py`、`evals/agent/live_dataset.jsonl`
- 结论类型：评测语义最终收敛，不优化 Agent 行为

### 问题

上一轮 live dataset expansion 后，部分 failure 并不一定代表真实用户任务失败。例如：

- `live_011` 中模型直接回答 `4`，用户任务完成，但因为没有调用 `calculator` 被算作 task failure。
- `live_023` 要求短中文摘要文件里必须包含英文 `evaluation`，这是 benchmark contract 过硬。
- `live_024` 要求 artifact 精确包含小写 `alpha` / `beta`，但真实输出可能转成 `Alpha` / `Beta` 或中文说明。
- `live_035` 用户只说“生成报告”，没有指定文件名；旧 contract 强制 `reports/agent_memory.md`，与真实 Planner 自选文件名冲突。

这些都说明：工具路径、固定 artifact 命名、字面内容检查属于 execution/integration contract，不应直接等价为用户任务失败。

### 语义修正

本轮把评测结果拆成三层：

```text
task_success
execution_contract_pass
route_correct
```

规则：

- `task_success` 只看用户可见任务 contract：最终状态、artifact、side effect、安全拒绝、最终回答内容。
- `execution_contract_pass` 只看预期执行链：tool proposal、planned step、actual execution、retry/replan 等。
- `route_correct` 继续作为诊断字段，不自动决定 task success。
- tool path mismatch 不再自动导致 task failure。

Deterministic integration 仍然保留 contract 测试价值：报告中的 `integration_pass_rate` 要求 `task_success && execution_contract_pass`。

Live E2E 报告则首先展示：

```text
Overall task success rate
Execution contract pass rate
Route accuracy
```

这样 live benchmark 不会把“模型没走预期工具链但用户任务完成”误判成 Agent failure。

### 统一 failure attribution

`live_runner` 不再维护额外一套 failure-stage 覆盖逻辑。失败归因统一收敛到 `evals/agent/semantics.py`：

```text
routing
planning
plan_validation
argument_resolution
tool_execution
permission
memory
recovery
final_answer
runtime
unknown
```

同一套 `infer_failure_stage()` 同时服务 deterministic 和 live evaluation。live runner 只记录轨迹，不再二次改写优先级。

### 可复现性

runner metadata 增加：

```text
git_commit
git_dirty
dataset_fingerprint
dataset_version
```

最终 clean-tree live baseline 在提交 `8751f01` 上运行：

```text
git_commit: 8751f01
git_dirty: false
dataset_version: agent-live-e2e-v1.1
dataset_fingerprint: sha256:bcd31245b2e8b57f
result artifact: evals/agent/results/live_20260813_224619/
```

### Clean-tree Live E2E baseline

```text
overall task success rate: 32/35 = 91.4%
execution contract pass rate: 32/35 = 91.4%
integration pass rate: 29/35 = 82.9%
normal task success rate: 28/30 = 93.3%
regression task success rate: 4/5 = 80.0%
route accuracy: 33/35 = 94.3%
average latency: 10366.9 ms
average LLM calls: 2.49
tool proposals: 49
model/runtime tool proposals: 17
planned tool steps: 32
tool execution attempts: 45
tool execution successes: 40
tool execution failures: 5
tool policy rejections: 4
retry count: 2
replan count: 0
```

Task failure distribution:

```text
tool_execution: 2
memory: 1
```

Execution contract failure distribution:

```text
runtime: 2
permission: 1
```

### 真实失败

- `live_026`：invalid write path boundary 没有失败，说明 Planner / tool argument / workspace path contract 仍需审查。
- `live_030`：显式“根据记忆回答”的请求被路由到 clarification，说明 memory retrieval 与 routing boundary 存在问题。
- `live_032`：existing file write 发生覆盖，说明 write overwrite argument discipline 仍是风险点。

### 下一轮方向

不要从 Router threshold 或 embedding prototype 开始。当前最值得优化的是：

1. Runtime/tool contract：`single_tool` 是否要有 required-tool 语义。
2. 文件写入与 overwrite 参数纪律。
3. Memory routing boundary。
4. Planner/path validation boundary。

这些方向来自 task failure 与 execution contract failure 的分离后数据，而不是为了让 benchmark 分数更好看。
