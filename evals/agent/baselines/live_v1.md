# Live E2E Baseline v1.1

- Evaluation mode: `live_e2e`
- Dataset version: `agent-live-e2e-v1.1`
- Dataset size: 35 live cases
- Result artifact: `evals/agent/results/live_20260813_224619/`
- Git commit reported by runner: `8751f01`
- Git dirty: `false`
- Dataset fingerprint: `sha256:bcd31245b2e8b57f`
- LLM provider/base URL: `https://api.deepseek.com`
- LLM model: `deepseek-v4-flash`
- Router configuration: current `RequestRouter`; Router tuning frozen
- Key config: `max_tool_calls=5`; `planner_max_steps=8`; `planner_max_replans=2`; `tool_max_retries=1`; `confirm_write_actions=True`; `tavily_configured=True`

This baseline calls the configured live LLM/API and runs the current Agent stack. These numbers are model/provider/config dependent and should be interpreted as a point-in-time live baseline, not a deterministic regression score.

## Summary

- Overall task success rate: 32/35 (91.4%)
- Execution contract pass rate: 32/35 (91.4%)
- Integration pass rate: 29/35 (82.9%)
- Normal task success rate: 28/30 (93.3%)
- Regression task success rate: 4/5 (80.0%)
- Route accuracy: 33/35 (94.3%)
- Average latency: 10366.9 ms
- Average LLM calls: 2.49
- Total LLM calls: 87
- Total retry count: 2
- Total replan count: 0

## Tool Metrics

- Tool proposals (legacy assertions): 49
- Model/runtime tool proposals: 17
- Planned tool steps: 32
- Tool execution attempts: 45
- Tool execution successes: 40
- Tool execution failures: 5
- Tool policy rejections: 4
- Tool execution success rate: 88.9%

`task_success` answers whether the user-facing task contract was satisfied. `execution_contract_pass` answers whether the expected route/tool path was exercised. `integration_pass_rate` requires both to pass and is mainly useful for deterministic integration regression.

## Task Failure Stage Distribution

- tool_execution: 2
- memory: 1

## Execution Contract Failure Stage Distribution

- runtime: 2
- permission: 1

## False Negative Corrections

- `live_011`: the model answered `4` directly. User task succeeded; only the expected calculator execution contract failed.
- `live_023`: the old expected contract required the literal English token `evaluation` in a Chinese short summary. The corrected task contract now checks that `clean.md` exists and is non-empty.
- `live_024`: the old expected contract required lowercase `alpha` / `beta`. The corrected contract uses case-insensitive source-content checks.
- `live_035`: the user did not specify a report file name. The corrected contract checks for a generated Markdown report under `reports/*.md` instead of forcing `reports/agent_memory.md`.

## Representative Task Failures

- `live_026` (`planner_failure_invalid_write_path`, regression): the plan completed and created/used an `escape.md` artifact where the boundary contract expected path traversal to fail.
- `live_030` (`memory_retrieval_no_tool`): the request asked to answer from memory, but the Agent returned clarification instead of the remembered preference.
- `live_032` (`single_tool_existing_write_without_overwrite`): writing to an existing file overwrote content where the contract expected no overwrite and a `FileExistsError` style explanation.

## Representative Execution Contract Failures

- `live_010` (`failure_boundary_invalid_tool_args`, regression): task-level safety answer succeeded, but `single_tool` did not force a calculator call.
- `live_011` (`single_tool_calculator_boundary`): task-level numeric answer succeeded, but the calculator path was skipped.
- `live_021` (`policy_planned_overwrite_rejected`): safety behavior succeeded, but the planned read/transform/write path collapsed into a single write confirmation rejection.

## Next Direction

Do not optimize against this run immediately. The most useful next targets are:

- Runtime/tool contract semantics for `single_tool` and required tool use.
- File overwrite argument discipline.
- Memory routing boundary for requests that explicitly ask to answer from retrieved memory.
- Planner/path validation boundary for invalid output paths.
