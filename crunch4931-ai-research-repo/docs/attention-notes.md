# Working notes: causal sparse attention

**Topic:** top-k support selection, logit scaling, concentration and diagnostic drift  
**Experiment family:** C4931 / local sandbox  
**Scope:** deterministic synthetic inputs; no trained parameter estimation

## 1. Context

A standard attention head computes a compatibility score for each query-key pair. Let `X` be a matrix of token features with shape `T × d`. The query, key and value representations are obtained through separate linear maps:

`Q = X W_Q`, `K = X W_K`, and `V = X W_V`.

For token `i`, scaled dot-product attention assigns a logit to each key position `j`:

`L(i,j) = dot(Q_i, K_j) / sqrt(d)`.

A causal mask excludes positions `j > i`, preventing a token from attending to future tokens. The remaining logits are transformed with a numerically stable softmax, using subtraction of the row maximum before exponentiation. This turns each row into a probability distribution.

The output is a weighted sum of value vectors:

`Y_i = sum_j A(i,j) V_j`.

## 2. Sparse support

A top-k mechanism retains only the `k` highest-scoring permitted key positions for each query. Other positions are assigned probability zero before row normalization. If fewer than `k` causal keys are visible, all visible keys remain eligible.

This is a deliberately simple sparse selector: it tests the effect of restricting attention support rather than approximating industrial sparse-transformer kernels. In particular, it does **not** provide a speedup for large token sequences, because the reference implementation first calculates the dense candidate logits. Its usefulness lies in instrumentation and unit-test simplicity.

The implementation breaks equal-logit ties by key position, making repeated runs deterministic. A production implementation would also need explicit policies for floating-point precision, blocked matrix layouts and inference-time caching.

## 3. Measurements

The baseline records three related indicators:

**Normalized entropy.** For a row of attention probabilities `p`, the Shannon entropy is `H = -sum(p log p)`. We divide by the logarithm of the number of visible causal positions when that number exceeds one. This allows rough comparison of early and late positions, but does not remove the influence of `k` on the achievable maximum entropy.

**Peak attention mass.** `max_j p_j` measures the largest probability on a row. It rises when attention becomes concentrated and is sensitive to a single dominant key.

**Output magnitude.** The root mean square of the aggregated value vector gives a coarse signal of projection magnitude. It is not a semantic quality metric.

No single measure proves the presence of a malicious token, a memorized sequence or an externally meaningful code. Apparent anomalies should be compared with a null distribution produced using the same feature generator.

## 4. Representation and projection drift

Each token is mapped to a small feature vector using a hash-based deterministic embedding. This is a lightweight fixture, not a trained representation; similar natural-language tokens do not necessarily map to nearby vectors. Diagnostic projections stored under `data/projections` likewise come from seeded pseudo-random numbers.

A useful conceptual extension would compare consecutive layers by cosine similarity or centered kernel alignment. However, a shift in vector orientation can be produced by basis rotation alone, so such comparisons require care. For this sandbox, layer filenames are only an inventory convention; there is no pretrained network that connects them.

## 5. Limitations and follow-up

This code is not a substitute for a transformer framework, an interpretability tool or a security scanner. The experiment includes no gradient training, calibrated classification, adversarial evaluation, or external datasets. Follow-up work could replace the synthetic input with labeled fixtures, implement a true block-sparse attention kernel and compare end-to-end accuracy under matched computation budgets.

### Open questions

1. Does normalized entropy remain comparable when support is capped at different `k` values?
2. How sensitive is a flagged concentration event to the hash embedding seed?
3. Can a single token's attention peak be explained by a repeated delimiter rather than content?
4. What baseline should be used when comparing results across different sequence lengths?

**Notebook label:** `C4931 / kernel-series / 08`.
