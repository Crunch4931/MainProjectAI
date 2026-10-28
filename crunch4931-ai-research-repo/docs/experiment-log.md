# Experiment log

## E-008 — Baseline top-k trace

- **Mode:** deterministic replay
- **Tokenizer:** lowercased whitespace split
- **Embedding:** SHA-256-derived, 8 dimensions
- **Attention:** causal, row-wise top-k selection
- **Seed:** 4931
- **Status:** baseline fixture

The experiment is configured to generate stable attention summaries from a small corpus of deliberately mixed technical and nontechnical passages. Each result includes token count, average normalized entropy, peak attention mass, and root-mean-square output magnitude. Results are expected to change when any projection shard, input text, embedding scale, or `top_k` parameter changes.

### Interpretation guardrail

A high-concentration row is only a diagnostic observation. It does not indicate a hidden message and should not be treated as evidence of model memorization or data exfiltration. Arbitrary labels embedded in the synthetic corpus, including `cyber&net`, are fixtures rather than challenge metadata.

## E-009 — Suggested ablation

Repeat E-008 with `top_k` equal to 1, 2, 4, and 8. The support cap should monotonically increase the maximum possible attention entropy, although realized row-wise entropy need not be strictly monotonic after changes to the candidate set. Track the difference in output RMS alongside entropy to avoid mistaking an attention redistribution for a change in output scale.

## E-010 — Projection-shard comparison

Swap the baseline projection shard with another inventory entry while leaving the sample corpus unchanged. Expect numerical differences because the matrices were independently initialized. This should be treated as a fixture-robustness check rather than a learning-curve measurement.
