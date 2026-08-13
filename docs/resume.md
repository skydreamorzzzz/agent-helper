# Resume Notes

## Project

`agent-helper` is a local personal Agent project built around a transparent Python runtime. It focuses on debuggable Agent execution rather than a UI: routing, tool calling, structured planning, policy confirmation, memory retrieval, and quantitative evaluation.

## Architecture Highlights

- Runtime: JSON-based Agent protocol with `tool_call` and `final_answer`, Pydantic validation, one repair attempt for malformed model output, max tool-call guard, and per-run structured logs.
- Router: layered `RequestRouter` with constraint routing, lexical semantic routing, optional LLM routing, and route categories for `direct_answer`, `single_tool`, `web_lookup`, `planned_task`, `deep_research`, and `clarification`.
- Planner: `StructuredPlanner` asks the model for typed plan JSON; `PlanExecutor` validates dependencies, resolves placeholders, retries failed steps, supports research replanning, and generates a final plan answer.
- Tools: registry-based typed tools including calculator, workspace file read/write, transform text, web search, and cited research report generation.
- Policy: read-only tools are allowed, write/destructive/external actions require confirmation, overwrite is separately confirmed, and policy rejection is observable.
- Memory: working memory, conversation summary, and SQLite FTS5 long-term memory retrieval path. Current E2E coverage primarily verifies retrieval, not a full long-term autonomous memory loop.
- Evaluation: module tests, Router benchmarks, deterministic integration E2E, and live LLM E2E baseline.

## Evaluation Results

Router holdout and embedding experiments established a measurable engineering story:

- `router-v2` holdout current hybrid: accuracy `0.889`, macro-F1 `0.883`, LLM escalation rate `0.778`.
- sentence embedding hybrid reduced LLM escalation to `0.444` but accuracy dropped to `0.861`, showing that cascade policy matters more than simply adding embeddings.

Deterministic E2E v1.1:

- Mode: `deterministic_integration`
- Dataset: 32 cases
- Overall integration pass rate: 30/32 (93.8%)
- Normal task pass rate: 28/29 (96.6%)
- Regression case pass rate: 2/3 (66.7%)
- Route accuracy: 31/32 (96.9%)

Live E2E v1.1:

- Mode: `live_e2e`
- Dataset: 35 cases
- Model: `deepseek-v4-flash`
- Overall live pass rate: 28/35 (80.0%)
- Normal task pass rate: 24/30 (80.0%)
- Regression case pass rate: 4/5 (80.0%)
- Route accuracy: 33/35 (94.3%)
- Failure stages: `tool_execution` 4, `runtime` 2, `routing` 1

## Engineering Value

The project demonstrates a progression from feature implementation to measurement discipline:

1. Build the Agent runtime and tools.
2. Quantify Router behavior with benchmarks and holdouts.
3. Separate deterministic integration regression from true live LLM evaluation.
4. Attribute failures by stage instead of only reporting a success rate.
5. Use baseline data to choose the next optimization target.

## Current Limitations

- Live E2E dataset is broader but still model/provider dependent; repeated runs can shift individual case outcomes.
- `single_tool` routing does not currently enforce a required tool call in Runtime.
- Deep research depends on Tavily availability and has not yet been heavily covered in live E2E.
- Memory E2E mainly validates retrieval path; richer write/retrieve lifecycle remains future work.
- Planner robustness is measured but not yet optimized from the E2E failure distribution.
