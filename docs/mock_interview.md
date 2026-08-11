# Mock Interview Notes

## Q: What is this project?

`agent-helper` is a local personal Agent implementation. It is not just a chat wrapper: it has routing, JSON tool calling, typed tools, structured planning, policy confirmation, memory retrieval, and evaluation harnesses that measure both module-level and end-to-end behavior.

## Q: Why use a custom JSON protocol instead of native function calling?

The project targets local or OpenAI-compatible models that may not reliably support native function calling. The Runtime asks the model to output either:

```json
{"type": "tool_call", "tool": "...", "arguments": {}}
```

or:

```json
{"type": "final_answer", "content": "..."}
```

The output is parsed with Pydantic. Invalid JSON triggers one repair attempt; if repair fails, the agent safely stops. This makes failures explicit and testable.

## Q: How does routing work?

`RequestRouter` is layered:

- constraint router handles hard cases such as missing parameters, explicit research, obvious multi-step file work, and obvious single-tool tasks;
- semantic router is a lightweight lexical similarity layer;
- LLM router is used when earlier layers are non-final or inconclusive;
- fallback is direct answer.

Router tuning is currently frozen. The next stages are using E2E failure data instead of continuing threshold tweaking.

## Q: What did the Router experiments show?

The important result was not simply that embeddings helped. The holdout showed:

- current hybrid reached `0.889` accuracy and `0.883` macro-F1;
- sentence embedding hybrid lowered LLM escalation but also lowered accuracy;
- therefore cascade policy and hard-constraint boundaries matter more than chasing an embedding leaderboard.

## Q: How are tools made safe?

Tools are registered with typed Pydantic schemas and risk levels. The policy layer:

- allows read-only tools;
- requires confirmation for writes;
- requires confirmation for overwrites;
- requires confirmation for destructive or external actions;
- records rejections separately from execution failures.

The E2E semantics were refined so a rejected write with no side effect is counted as successful safety behavior.

## Q: What is the difference between deterministic E2E and live E2E?

Deterministic E2E uses fake LLM behavior and fake web search. It tests integration contracts under controlled model output. It should not be called real Agent quality.

Live E2E calls the configured real LLM and runs the same Runtime/Router/Planner/Tool/Policy path. It is a model-dependent baseline and can expose real model protocol failures, latency, and provider instability.

## Q: What are the current E2E results?

Deterministic E2E v1.1:

- 32 cases
- 30 passed
- overall integration pass rate `93.8%`
- normal task pass rate `96.6%`
- regression pass rate `66.7%`

Live E2E v1:

- 10 cases
- 9 passed
- normal task pass rate `100.0%`
- regression pass rate `0.0%`
- route accuracy `100.0%`
- main failure stage `runtime`

## Q: Where does the live Agent currently fail?

The final live baseline found one regression failure: a malicious calculator request was routed as `single_tool`, but Runtime did not enforce the calculator call. The model directly refused safely. That is acceptable user safety behavior, but it did not satisfy the regression contract that expected tool-level invalid-argument handling.

This points to a Runtime semantics question: should `single_tool` behave like `web_lookup` and require the selected tool before final answer?

## Q: What would you improve next?

I would not tune Router thresholds next. The live failure distribution points to Runtime contract semantics:

- define whether `single_tool` requires tool execution;
- expand live E2E with more boundary cases before optimizing;
- separate safe refusal success from tool-validation success in the dataset;
- add more live deep-research and provider-failure cases.

Only after the failure distribution is larger would I choose whether Planner, Runtime, Policy, or Router deserves implementation work.
