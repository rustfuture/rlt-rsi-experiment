# Design, Specified Hypotheses, and Methodological Controls

> **Not a preregistration.** These are *specified hypotheses and a fixed
> decision rule*. `DESIGN.md` and the first results were committed in the same
> commit (`161752a`), so the commit history does **not** establish that the
> hypotheses predate the experiment. The decision rule below was applied
> unchanged to every configuration and seed that this repository reports.

## Research Questions

On a small algorithmic task (binary sequence parity), we examine:

1. **Shared-Weight Recurrence vs Conventional Transformer:** Does repeatedly executing a shared transformer block alter accuracy or length generalization relative to a matched conventional one-pass transformer?
2. **Loop Scaling & Compute Trade-Off (1, 2, 4):** As iteration loops increase, how do accuracy, measured wall-clock seconds, and sequential block applications scale when architectural parameter count is held constant?
3. **Out-of-Distribution Length Generalization:** Does recurrent computation change generalization from short sequences (train/dev: 4–8) to longer sequences (held-out: 12–16)?
4. **Controlled Architectural Parameters (structural):** Within a backend, do all configurations have identical parameter counts? The NumPy and PyTorch architectures are different and their counts are not compared across backends.

*Note:* This experiment is a focused probe of iterative computation on parity. It is **not** a general reasoning benchmark.

## Specified Hypotheses and Fixed Decision Rule

The decision rule uses a fixed **operational threshold** of 0.05 (5 percentage points). It is an operational decision rule only: **not** a significance test, **not** a confidence interval, and **not** an equivalence test. With N=3 seeds and the sample sizes used here it cannot establish superiority or equivalence in a statistical sense, and the report says so explicitly.

* **H1 (Empirical Accuracy Gain) — `evaluated` in code.** For a looped configuration (1, 2 or 4 loops), the mean **paired** per-seed difference in held-out accuracy versus the same-seed baseline is `>= +0.05` → `improvement`; `<= -0.05` → `regression`; otherwise `flat`.
* **H2 (Length Generalization Preservation) — `evaluated` in code.** For each model and seed, `in_distribution_accuracy` = pooled accuracy over train+dev examples (lengths 4–8) and `length_generalization_drop = in_distribution_accuracy - heldout_accuracy` (lengths 12–16). H2 statistic = `mean_seeds[drop(baseline) - drop(looped)]`. `>= +0.05` → supported (the looped model degrades at least 5pp less); `<= -0.05` → not supported in the opposite direction; otherwise not supported within the rule.
* **H0 (Null Outcome) — `evaluated` in code.** No looped configuration meets the improvement threshold. Flat or worse performance is a valid and anticipated scientific result.

Each hypothesis is marked `evaluated` / `not evaluated` in the generated report.
Nothing in the report asserts a global conclusion that the measured values do not
support; statements are scoped to this run, this task, these seeds, these sample
sizes and this training budget.

## Task Specification and Leakage Controls

- **Task:** Binary sequence parity ($y = \bigoplus_{i=0}^{L-1} x_i$).
- **Distributions:** `train`: lengths uniformly drawn from $[4, 8]$ ($N=256$); `dev`: lengths 4–8 ($N=128$); `heldout`: lengths 12–16 ($N=128$).
- **Leakage prevention (exact-example disjointness, not just different PRNG seeds):**
  - Train and dev are drawn as **one joint pool** of unique `(length, token sequence)` examples with a single PRNG seed, then randomly partitioned (`seed + 404`). Held-out is drawn from disjoint lengths and additionally excludes every train/dev example.
  - Drawing train first and rejecting train examples from dev would let train exhaust the tiny short-length example spaces (only $2^4=16$ length-4 sequences exist), leaving dev without length-4 examples; the joint pool preserves usable length coverage in both splits.
  - `make_splits` raises rather than returning duplicates when the requested `n` exceeds the available example space.
  - The exact-example overlap (`train_dev`, `train_heldout`, `dev_heldout`) is **measured per seed, recorded in the payload, and asserted to be zero**; the experiment aborts if it is not zero.
  - Because unique examples are required, the empirical length distribution within a split deviates from uniform (short lengths saturate). Per-split `length_counts` are recorded in the split manifest for full transparency.
  - Attention padding masks prevent position leakage; classification reads out exclusively at the final valid prefix position ($L - 1$).

## Architectural Controls and Compute Measurement

| Variant | Model Class | Loop Count | Total Params | Trainable | Frozen | Sequential Block Applications | Parameter Sharing |
|---|---|---:|---:|---:|---:|---:|:---:|
| Baseline | `NumpyTransformerClassifier` | 1 | 4,681 | 25 | 4,656 | 1 | False |
| Looped-1 | `NumpyLoopedTransformerClassifier` | 1 | 4,681 | 25 | 4,656 | 1 | True |
| Looped-2 | `NumpyLoopedTransformerClassifier` | 2 | 4,681 | 25 | 4,656 | 2 | True |
| Looped-4 | `NumpyLoopedTransformerClassifier` | 4 | 4,681 | 25 | 4,656 | 4 | True |

NumPy counts: total = frozen backbone (embedding + block) 4,656 + trainable readout+bias 25. The torch model has **4,945** parameters, all trainable, and is a different architecture. `architecture_parameters()` is kept as a clearly documented alias for the **frozen backbone subtotal only** and is never reported as the total.

- **Width:** `d_model = 24`, `n_heads = 4`, `d_ff = 48`.
- **Positional Encoding:** Sinusoidal encodings added prior to the first block.
- **`estimated_block_calls` / `sequential_block_applications`:** a **structural counter** of sequential shared-block applications. It is not a FLOP count and not a latency measurement; `seconds` is the measured wall-clock time and is reported per configuration.
- **Equal epochs are not equal compute.** Each configuration trains for the same number of epochs, but a `loop_count=k` configuration performs `k` block applications per optimisation step. The report states this and reports both block applications and measured seconds.

## Backend Implementations & Hardware Environment

- **NumPy (smoke / plumbing mode):** the embedding and transformer block are deterministically seeded and **frozen**; only the binary readout + bias are trained (25 trainable, 4,656 frozen, 4,681 total). It is strictly a smoke test for determinism, split integrity, multi-seed aggregation and interfaces. Frozen-feature smoke results **do not measure whether trained recurrence helps** and are not evidence about trained recurrence.
- **PyTorch (gradient-trained mode):** `nn.MultiheadAttention` + `nn.LayerNorm`; all parameters are trained end-to-end with AdamW. Verified locally in a Python 3.12.13 virtualenv with torch 2.14.0 on an Apple M4 Pro (arm64) with Metal/MPS available.
- **Device selection:** `--device auto|cpu|mps|cuda`. `cuda`/`mps` raise a clear `RuntimeError` when unavailable — there is no silent fallback. `auto` prefers cuda, then mps, then cpu. The requested and resolved devices, dtype, torch/python/numpy versions, epochs, optimizer, learning rate, weight decay and seeds are recorded in the payload and report.
- **Reproducibility metadata:** every artifact directory records `git rev-parse HEAD`, `git status --porcelain` (dirty state), the data manifest with per-split fingerprints, per-seed raw results, paired deltas, checkpoint SHA-256 hashes, the exact rerun command, and a reload-and-re-evaluate check.

## Theory and Literature Context (scoped by their assumptions)

- **Looped / recurrent-depth transformers.** Weight-shared recurrent depth is the mechanism studied by Universal Transformers (Dehghani et al., ICLR 2019) and, in a program-execution framing, by Looped Transformers as Programmable Computers (Giannou et al., ICML 2023). *Assumption/scope:* those results are about representational capacity or designed constructions; they do not imply that a 24-dimensional, 2-layer looped model trained for 80 epochs on 256 length-4–8 parity examples will outperform a one-pass baseline.
- **Parity and transformer expressivity.** Chiang, Cholak & Pillay (ICML 2023, PMLR 202:5544–5562) characterize fixed-precision transformer encoders as recognizing only languages in uniform TC⁰ (with a counting-quantifier lower bound). Parity lies in TC⁰, so a transformer encoder can in principle represent it; **no claim is made here that transformers can never solve parity**. *Assumption/scope:* expressivity results are about representability, not about what a small model learns from finite data in 80 full-batch epochs, and hard-attention/uniform-attention constructions differ from the soft-attention, layer-normalized blocks used here.
- **Length generalization.** Anil et al. (NeurIPS 2022) document length-generalization failures and some mitigation strategies in large language models on tasks that include parity-like structure. *Assumption/scope:* those failure modes depend on training setup, scale and task formatting; they motivate RQ3 but do not predict the outcome of this small controlled run.
- **Citation corrected.** The DSPy reference is Khattab et al., *"DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"* (ICLR 2024, [arXiv:2310.03714](https://arxiv.org/abs/2310.03714)).

## Scientific Limitations and Failure Modes

1. **Parity complexity.** Parity is non-linear and sensitive to representation geometry. Random **frozen** features cannot be expected to solve it without end-to-end training, which is why the NumPy path is labeled a smoke test and no conclusion about trained recurrence is drawn from it.
2. **Compute vs parameter asymmetry.** Increasing loop count increases sequential work per step while holding parameters constant, so equal-epoch comparisons are not equal-compute comparisons. Results must be read against the measured `seconds` and block applications reported per configuration.
3. **Sample size and decision rule.** N=3 seeds and small held-out sets; the ±0.05 rule is operational only. A `flat`/`regression` label is an outcome of the rule, not evidence of equivalence or harm.
4. **No general reasoning claims.** Sequence parity is an isolated algorithmic check. Findings cannot be generalized to language modeling or multi-step reasoning agents.
5. **Backend mismatch.** NumPy and torch models differ in attention implementation, normalization, training scope and parameter count; results are comparable only within a backend.
6. **Checkpoint portability.** Saved checkpoints are local, gitignored artifacts tied to the torch build and machine that produced them; only the JSON manifest with SHA-256 hashes is tracked.
7. **Device nondeterminism (measured).** Identical re-runs of the recorded MPS command were **not** bit-reproducible: baseline and looped-2 reproduced exactly, but looped-4 moved from a paired held-out delta of +0.0417 (`flat`) to +0.0573 (`improvement`). A configuration whose delta sits near the ±0.05 threshold can therefore change its operational label between identical runs, so no label should be read as a stable property of the architecture. Use `--device cpu` and repeated trials for any comparison where exact reproducibility matters.
