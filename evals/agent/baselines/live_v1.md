# Live E2E Baseline v1.1

- Evaluation mode: `live_e2e`
- Dataset version: `agent-live-e2e-v1.1`
- Dataset size: 35 live cases
- Result artifact: `evals/agent/results/live_20260813_221538/`
- Git commit reported by runner: `cafcb34`
- LLM provider/base URL: `https://api.deepseek.com`
- LLM model: `deepseek-v4-flash`
- Router configuration: current `RequestRouter`; Router tuning frozen
- Key config: `max_tool_calls=5`; `planner_max_steps=8`; `planner_max_replans=2`; `tool_max_retries=1`; `confirm_write_actions=True`; `tavily_configured=True`

This baseline calls the configured live LLM/API and runs the current Agent stack. These numbers are model/provider/config dependent and should be interpreted as a point-in-time live baseline, not a deterministic regression score.

## Summary

- Overall live pass rate: 28/35 (80.0%)
- Normal task pass rate: 24/30 (80.0%)
- Regression case pass rate: 4/5 (80.0%)
- Route accuracy: 33/35 (94.3%)
- Average latency: 10092.3 ms
- Average LLM calls: 2.43
- Total LLM calls: 85
- Total retry count: 4
- Total replan count: 0

## Tool Metrics

- Tool proposals (legacy assertions): 48
- Model/runtime tool proposals: 16
- Planned tool steps: 32
- Tool execution attempts: 44
- Tool execution successes: 38
- Tool execution failures: 6
- Tool policy rejections: 4
- Tool execution success rate: 86.4%

`tool_proposals` remains a compatibility field for existing expected contracts. For interpretation, use `model_tool_proposals` for actual runtime model tool-call proposals, `planned_tool_steps` for Planner-authored steps, and `tool_execution_attempts_by_name` for actual execution attempts.

## Failure Stage Distribution

- tool_execution: 4
- runtime: 2
- routing: 1

## Representative Failures

- `live_010` (`failure_boundary_invalid_tool_args`, regression): route was `single_tool`, but the model directly returned a final answer instead of proposing `calculator`; this preserves the known Runtime contract gap from v1.
- `live_011` (`single_tool_calculator_boundary`): route was `single_tool`, but the model answered directly instead of proposing `calculator`, so the explicit tool contract was not exercised.
- `live_023` (`planned_short_summary`): plan completed and wrote `clean.md`, but the artifact did not preserve the expected `evaluation` content.
- `live_024` (`planned_merge_files`): plan completed and wrote `merged.md`, but the artifact missed expected source content from both files.
- `live_030` (`memory_retrieval_no_tool`): routed to clarification instead of answering from retrieved memory.
- `live_032` (`single_tool_existing_write_without_overwrite`): write succeeded with overwrite semantics where the contract expected `FileExistsError`.
- `live_035` (`deep_research_small`): deep research completed without creating the expected cited report file.

## Next Direction

Do not optimize against this run immediately. The most useful next target is Runtime and tool-contract semantics, especially `single_tool` required-tool behavior, overwrite argument discipline, and whether deep-research completion should be tied to artifact existence. Router tuning remains frozen until these live failure contracts are interpreted.
