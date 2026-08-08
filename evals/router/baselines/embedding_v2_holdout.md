# Router Embedding v2 Holdout

Dataset: `router-v2`

Status: first untouched holdout evaluation for the Router embedding experiment.

## Holdout Policy

`router-v1` is now treated as the historical development benchmark. Its test split has already been used for failure analysis and prototype discussion, so it is no longer a strict holdout.

`router-v2` was added after the v1 prototype and threshold work. It contains 36 test-only examples across all six routes, with Chinese, English, mixed-language, paraphrase, and OOD-style requests. It includes the same problem types exposed by v1, but does not copy v1 test wording.

The experiment order for this run was:

```text
router-v1 dev
-> keep existing prototypes
-> keep similarity_threshold = 0.32
-> keep margin_threshold = 0.04
-> run router-v2 for the first time
-> do not change prototypes or thresholds after seeing v2 failures
```

## Run Context

- Parent commit at run time: `3348da6`
- Similarity threshold: `0.32`
- Margin threshold: `0.04`
- Hashed lexical provider: `hashing`
- Hashed lexical model: `hashing-multilingual-v1`
- Sentence embedding provider: `sentence_transformers`
- Sentence embedding model: `BAAI/bge-small-zh-v1.5`
- Sentence embedding load mode: `local_files_only=true`
- LLM model: `gemini-2.5-flash`

`BAAI/bge-small-zh-v1.5` is a real `sentence-transformers` model available in the local cache. It is Chinese-oriented and works for the mixed benchmark, but it should not be described as the strongest multilingual sentence embedding model.

## Development Check

Before running v2, the frozen configuration was checked on `router-v1/dev`:

| Mode | Split | Samples | Accuracy | Macro-F1 | LLM calls | LLM escalated examples | LLM escalation rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hashing_only` | `dev` | 42 | 0.881 | 0.877 | 0 | 0 | 0.000 |
| `sentence_embedding_only` | `dev` | 42 | 0.857 | 0.854 | 0 | 0 | 0.000 |

No threshold or prototype changes were made after this point.

## Holdout Results

| Mode | Dataset | Samples | Accuracy | Macro-F1 | LLM calls | LLM escalated examples | LLM escalation rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lexical_baseline` | `router-v2` | 36 | 0.361 | 0.326 | 0 | 0 | 0.000 |
| `hashing_only` | `router-v2` | 36 | 0.444 | 0.446 | 0 | 0 | 0.000 |
| `sentence_embedding_only` | `router-v2` | 36 | 0.583 | 0.590 | 0 | 0 | 0.000 |
| `current_hybrid` | `router-v2` | 36 | 0.889 | 0.883 | 28 | 28 | 0.778 |
| `sentence_embedding_hybrid` | `router-v2` | 36 | 0.861 | 0.860 | 16 | 16 | 0.444 |

## Per-route Recall

| Mode | direct_answer | single_tool | web_lookup | planned_task | deep_research | clarification |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lexical_baseline` | 1.000 | 0.500 | 0.167 | 0.333 | 0.167 | 0.000 |
| `hashing_only` | 1.000 | 0.500 | 0.167 | 0.333 | 0.333 | 0.333 |
| `sentence_embedding_only` | 0.833 | 0.500 | 0.667 | 0.500 | 0.833 | 0.167 |
| `current_hybrid` | 1.000 | 1.000 | 1.000 | 0.833 | 1.000 | 0.500 |
| `sentence_embedding_hybrid` | 0.833 | 1.000 | 1.000 | 0.833 | 0.833 | 0.667 |

## Comparison

### Hashing vs lexical

`hashing_only` improved over `lexical_baseline`:

```text
accuracy 0.361 -> 0.444
macro-F1  0.326 -> 0.446
```

The gain is real but limited. Hashed lexical vectors still miss many current-fact paraphrases and English multi-step file tasks.

### Sentence embedding vs hashing

`sentence_embedding_only` improved over `hashing_only`:

```text
accuracy 0.444 -> 0.583
macro-F1  0.446 -> 0.590
```

The largest recall improvements were:

- `web_lookup`: `0.167 -> 0.667`
- `deep_research`: `0.333 -> 0.833`
- `planned_task`: `0.333 -> 0.500`

This shows the real sentence embedding provider is a better semantic layer than the hashed lexical baseline on the untouched holdout.

### Hybrid cost trade-off

`sentence_embedding_hybrid` reduced LLM escalation compared with `current_hybrid`:

```text
LLM escalation rate 0.778 -> 0.444
LLM escalated examples 28 -> 16
```

Accuracy changed from `0.889` to `0.861`, and macro-F1 changed from `0.883` to `0.860`. The cascade therefore saves LLM calls with only a small accuracy drop on this holdout, but it is not strictly better than the LLM-heavy baseline.

## Representative Failures

- `route_v2_001`: `早上好，今天想简单聊两句`
  - expected `direct_answer`, `sentence_embedding_hybrid` predicted `clarification`
  - Failure mode: embedding prototype for vague/missing-input clarification is too close to casual Chinese chat.
- `route_v2_020`: `Read raw_notes.md, turn it into bullets, and save the result as bullets.md.`
  - expected `planned_task`, predicted `single_tool`
  - Failure mode: hard file-tool constraint still fires before embedding or LLM can correct English multi-step file tasks.
- `route_v2_027`: `比较 Tavily、SerpAPI 和 Exa 的最新价格、限制和适用场景`
  - expected `deep_research`, `sentence_embedding_hybrid` predicted `web_lookup`
  - Failure mode: current-price lookup signal still competes with multi-source comparison intent.
- `route_v2_033`: `把上面的结论整理成文件`
  - expected `clarification`, `sentence_embedding_hybrid` predicted `single_tool`
  - Failure mode: LLM/constraint path treats an implied prior result as writable content in standalone evaluation.
- `route_v2_035`: `做一个总结然后发给我`
  - expected `clarification`, predicted `planned_task`
  - Failure mode: multi-action wording can hide missing source and delivery target.

## Conclusion

The first strict holdout supports three conclusions:

1. The previous HashingEmbedder should be understood as a hashed lexical vector baseline, not a neural sentence embedding.
2. A real sentence-transformers provider improves the semantic layer substantially over hashing on untouched data.
3. `sentence_embedding_hybrid` materially reduces LLM escalation, but remaining errors are mostly cascade policy and hard-constraint boundary issues, not just embedding quality.

The next Router iteration should focus on hard constraint review and LLM override policy. It should not tune thresholds on `router-v2` unless a new holdout is created for the next validation cycle.
