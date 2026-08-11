# Live E2E Baseline v1

- Evaluation mode: `live_e2e`
- Dataset version: `agent-live-e2e-v1`
- Dataset size: 10 live cases
- Result artifact: `evals/agent/results/live_20260811_230109/`
- Git commit reported by runner: `60dae79`
- LLM provider/base URL: `https://api.deepseek.com`
- LLM model: `deepseek-v4-flash`
- Router configuration: current `RequestRouter`; Router tuning frozen
- Key config: `max_tool_calls=5`; `planner_max_steps=8`; `planner_max_replans=2`; `tool_max_retries=1`; `confirm_write_actions=True`; `tavily_configured=True`

This baseline calls the configured live LLM/API and runs the current Agent stack. These numbers are model/provider/config dependent and should be interpreted as a point-in-time live baseline, not a deterministic regression score.

## Summary

- Overall live pass rate: 9/10 (90.0%)
- Normal task pass rate: 9/9 (100.0%)
- Regression case pass rate: 0/1 (0.0%)
- Route accuracy: 10/10 (100.0%)
- Average latency: 7384.3 ms
- Average LLM calls: 2.10
- Total LLM calls: 21
- Total retry count: 0
- Total replan count: 0

## Tool Metrics

- Tool proposals: 9
- Tool execution attempts: 7
- Tool execution successes: 7
- Tool execution failures: 0
- Tool policy rejections: 2
- Tool execution success rate: 100.0%

Policy rejections are counted separately from tool execution. In `live_007` and `live_008`, rejected write/overwrite confirmation correctly produced no file side effect.

## Per Suite

| Suite | Total | Passed | Pass Rate |
|---|---:|---:|---:|
| normal | 9 | 9 | 100.0% |
| regression | 1 | 0 | 0.0% |

## Per Category

| Category | Total | Passed | Success Rate |
|---|---:|---:|---:|
| clarification | 1 | 1 | 100.0% |
| direct_answer | 1 | 1 | 100.0% |
| failure_boundary_invalid_tool_args | 1 | 0 | 0.0% |
| memory_retrieval | 1 | 1 | 100.0% |
| planned_calculate_write | 1 | 1 | 100.0% |
| planned_file_task | 1 | 1 | 100.0% |
| policy_confirmation_rejected | 1 | 1 | 100.0% |
| policy_overwrite_rejected | 1 | 1 | 100.0% |
| single_tool_calculator | 1 | 1 | 100.0% |
| single_tool_file_read | 1 | 1 | 100.0% |

## Failure Stage Distribution

- runtime: 1

## Representative Failure

- `live_010` (`regression`, `failure_boundary_invalid_tool_args`): Router selected `single_tool`, but Runtime did not enforce a required calculator call for the `single_tool` route. The live model returned a direct safety refusal instead of calling `calculator` and surfacing the deterministic calculator validation error. User-visible behavior was safe, but the regression contract expected tool-level invalid-argument handling.

## Failure Analysis

The final run did not expose Router, Planner, Tool, Policy, or Memory failures in the normal suite. The only contract failure is a Runtime/protocol gap: `single_tool` is routed diagnostically, but ordinary `single_tool` execution uses the core `AgentRuntime` without a `required_tool`, so the model can answer directly.

An earlier live run with the same dataset had a transient `llm_call_failed` on the file-read case after the tool call succeeded. That did not reproduce in the final baseline, but it suggests live E2E should continue separating deterministic integration failures from provider/runtime availability failures.

## Next Direction

The next most valuable improvement is not Router threshold tuning. The data points to Runtime semantics for routed tool tasks:

- decide whether `single_tool` should carry an explicit `required_tool` contract similar to `web_lookup`;
- define how boundary/safety requests should be scored when direct refusal is safer than tool execution;
- expand live dataset size before optimizing, especially with more adverse model outputs and flaky API cases.
