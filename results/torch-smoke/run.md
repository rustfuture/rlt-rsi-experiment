# Sequence Parity: Conventional vs Shared-Weight Looped Transformer Report

- **Backend:** `torch`
- **Seeds (N=1):** `7`
- **Epochs per configuration:** 2
- **Optimizer:** AdamW | **learning rate:** 0.003 | **weight decay:** 0.0001
- **Device:** requested `cpu` -> resolved `cpu` | **dtype:** `float32`
- **Python:** `3.12.13` | **NumPy:** `2.5.3` | **PyTorch:** 2.14.0
- **Train/dev size:** 32/16 (lengths 4-8) | **Held-out size:** 16 (lengths 12-16)

> [!IMPORTANT]
> **Scope of this run.** This run trained all parameters end-to-end with AdamW for 2 epochs on device `cpu` for seeds 7. Conclusions are scoped to this run, this task (binary sequence parity), these seeds, these sample sizes and this training budget; they are not evidence about general reasoning or about recurrence at other scales.

## 1. Setup actually used

- Backend `torch`; device requested `cpu`, resolved `cpu`.
- dtype `float32`; torch version `2.14.0`; python `3.12.13`; numpy `2.5.3`.
- Epochs 2; optimizer AdamW; learning rate 0.003; weight decay 0.0001.
- Seeds `7`; gradient/training scope: all parameters (end-to-end).
- Batch: full-batch (all train examples per step).

### Parameter accounting (counted from the real model object)

| configuration | total parameters | trainable | frozen |
|---|---:|---:|---:|
| baseline (loops=1) | 4,945 | 4,945 | 0 |
| looped (loops=1) | 4,945 | 4,945 | 0 |
| looped (loops=2) | 4,945 | 4,945 | 0 |
| looped (loops=4) | 4,945 | 4,945 | 0 |

The PyTorch run reported here has no frozen parameters (`frozen = 0` and `trainable = total` for every configuration), so all parameters receive gradients. Counts are derived from `model.parameters()` and `requires_grad`.

The NumPy and PyTorch models are **different architectures** (NumPy: manual attention + RMS-style normalization + frozen backbone with a trained readout; torch: `nn.MultiheadAttention` + `LayerNorm` with full training). Their parameter counts are not expected to match and must never be compared as if they were the same model.

## 2. Summary results

| model | loops | total params | trainable | frozen | sequential block applications | train acc | dev acc | in-distribution acc (4-8) | held-out acc (12-16) | seconds (mean) | decision vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 1 | 4,945 | 4,945 | 0 | 1 | 0.594 ± 0.000 | 0.500 ± 0.000 | 0.562 ± 0.000 | 0.562 ± 0.000 | 0.0216 | baseline (reference) |
| looped | 1 | 4,945 | 4,945 | 0 | 1 | 0.594 ± 0.000 | 0.500 ± 0.000 | 0.562 ± 0.000 | 0.562 ± 0.000 | 0.0036 | flat |
| looped | 2 | 4,945 | 4,945 | 0 | 2 | 0.594 ± 0.000 | 0.500 ± 0.000 | 0.562 ± 0.000 | 0.562 ± 0.000 | 0.0046 | flat |
| looped | 4 | 4,945 | 4,945 | 0 | 4 | 0.594 ± 0.000 | 0.500 ± 0.000 | 0.562 ± 0.000 | 0.562 ± 0.000 | 0.0082 | flat |

`sequential block applications` is a structural counter (see metric notes at the end): it is not FLOPs and not a latency measurement. `seconds` is measured wall-clock training time for 2 epochs on this machine.

## 3. Paired per-seed differences vs the baseline

Differences are computed **per seed** against the baseline run with the same seed (paired), then averaged. `n_seeds` is small; the standard error is reported so it is not mistaken for a significance test.

| model | loops | metric | per-seed (looped - baseline) | mean | std | stderr |
|---|---:|---|---:|---:|---:|---:|
| looped | 1 | heldout_accuracy | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 1 | heldout_bce | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 1 | in_distribution_accuracy | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 1 | length_generalization_drop | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 1 | train_accuracy | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 2 | heldout_accuracy | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 2 | heldout_bce | -0.0037 | -0.0037 | 0.0000 | 0.0000 |
| looped | 2 | in_distribution_accuracy | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 2 | length_generalization_drop | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 2 | train_accuracy | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 4 | heldout_accuracy | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 4 | heldout_bce | +0.0148 | +0.0148 | 0.0000 | 0.0000 |
| looped | 4 | in_distribution_accuracy | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 4 | length_generalization_drop | +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 4 | train_accuracy | +0.0000 | +0.0000 | 0.0000 | 0.0000 |

### Decision rule (operational, not statistical)

The decision rule is: difference >= +0.05 -> `improvement`; difference <= -0.05 -> `regression`; otherwise `flat`. This is an **operational decision rule only**. It is not a statistical significance test, not a confidence interval, and not an equivalence test; with 1 seed(s) and held-out n=16 it cannot establish either superiority or equivalence.

## 4. Evaluation of Core Research Questions

1. **Shared-weight recurrence vs conventional single-pass transformer.** Operational decision-rule outcomes on held-out accuracy (baseline mean 0.562 ± 0.000): looped-1: `flat` (held-out delta +0.0000 ± 0.0000), looped-2: `flat` (held-out delta +0.0000 ± 0.0000), looped-4: `flat` (held-out delta +0.0000 ± 0.0000). Scoped to this run, this task (binary sequence parity), these seeds (7), these sample sizes (train 32, dev 16, held-out 16), this training budget (2 epochs, AdamW), and backend `torch` on device `cpu`. Both baselines and looped configurations were trained end-to-end under this budget.

2. **Loop scaling and compute trade-off (1, 2, 4).** Total parameter counts across the reported configurations: 4,945. Sequential block applications scale as baseline(loops=1) -> 1 block application(s), looped(loops=1) -> 1 block application(s), looped(loops=2) -> 2 block application(s), looped(loops=4) -> 4 block application(s). Measured wall-clock seconds: baseline(loops=1): 0.0216s, looped(loops=1): 0.0036s, looped(loops=2): 0.0046s, looped(loops=4): 0.0082s. Held-out accuracy by loop count: baseline(loops=1): 0.562, looped(loops=1): 0.562, looped(loops=2): 0.562, looped(loops=4): 0.562. Equal epoch counts are **not** equal compute budgets: each additional loop applies the shared block once more per optimisation step, so the higher-loop configurations perform more work per step; the measured `seconds` column is the only latency evidence in this report.

3. **Length generalization (train/dev 4-8 vs held-out 12-16).** These length ranges are disjoint and split generation is example-disjoint by construction. Length-generalization drop is defined per seed as `in_distribution_accuracy(train+dev, lengths 4-8) - heldout_accuracy(lengths 12-16)`; H2 statistic = drop(baseline) - drop(looped), evaluated against the same +/-0.05 operational rule. Results: looped-1: +0.0000 -> not supported (within the +/-0.05 rule), looped-2: +0.0000 -> not supported (within the +/-0.05 rule), looped-4: +0.0000 -> not supported (within the +/-0.05 rule).

   Exact-example overlap measured per seed (train_dev / train_heldout / dev_heldout): seed 7: 0/0/0.

4. **Controlled parameter comparison (structural).** Within backend `torch`: baseline(loops=1): total 4,945, trainable 4,945, frozen 0, looped(loops=1): total 4,945, trainable 4,945, frozen 0, looped(loops=2): total 4,945, trainable 4,945, frozen 0, looped(loops=4): total 4,945, trainable 4,945, frozen 0. Trainable parameter counts are constant across loop counts, so increasing loops adds sequential block applications but no new parameters. This is a structural check of the implementation, not a claim that the two backends have equal counts.

## 5. Hypothesis evaluation status

| id | statement | status | statistic | threshold | result |
|---|---|---|---:|---:|---|
| H1 | Looped configurations (2 or 4 loops) exceed the single-pass baseline on held-out parity accuracy by >= 0.05. | evaluated | loops=1: +0.0000; loops=2: +0.0000; loops=4: +0.0000 | 0.05 | not supported in this run |
| H2 | The accuracy drop from in-distribution lengths (4-8) to held-out lengths (12-16) is at least 0.05 smaller for a looped configuration than for the baseline. | evaluated | loops=1: +0.0000; loops=2: +0.0000; loops=4: +0.0000 | 0.05 | not supported in this run |
| H0 | No looped configuration meets the +/-0.05 improvement threshold (flat or worse held-out accuracy than the baseline). | evaluated | n/a | 0.05 | not rejected by this run's data (all configurations flat or regressing) |

Each hypothesis above is marked `evaluated` or `not evaluated`. The +/-0.05 rule is an operational decision rule (see section 3) and is applied identically to H1 and H2.

### Not evaluated by this experiment

- General reasoning ability: binary sequence parity is an isolated algorithmic check.
- Statistical significance or equivalence: the +/-0.05 rule is operational, and the seed count is small.
- Equal-compute comparisons: equal epochs are not equal compute; only measured seconds and block applications are reported.
- Trained recurrence at other scales/architectures, or on other tasks: not tested here.

## 6. Determinism and reproducibility notes

- torch.manual_seed set per run before model construction; MPS/CUDA kernels and some attention paths are not guaranteed bit-exact across runs/devices.
- num_workers=0 / single process; training is full-batch over all train examples with no DataLoader.
- accuracy metrics are computed from a single training run per seed (no repeated trials per seed).
- equal epoch counts across configurations are NOT equal compute budgets: a loop_count=k configuration performs k sequential block applications per optimisation step.
- `estimated_block_calls` counts sequential applications of the shared transformer block. It is a structural counter, not a FLOP count and not a latency measurement; see `seconds` for measured wall-clock time.

Identical initialization check (per seed, over the shared modules): the recorded initialization fingerprints are identical across the baseline and all looped configurations.
| seed | identical shared init | fingerprint (first 16 hex chars) |
|---|---|---|
| 7 | True | `5ed3a762b875fbff` |

## 7. Environment and PyTorch status (derived from this run)

PyTorch 2.14.0 is available in this environment and was used for this run on device `cpu`.

### Optimization diagnostic (separate small dataset, not the held-out split)

Overfit a tiny separate diagnostic batch of parity examples (lengths 4-8) to check that the optimizer actually reduces loss. This is a diagnostic of optimization plumbing, not a held-out claim; the diagnostic draw is never used for evaluation.
- Dataset: 32 examples, lengths 4-8, seed 9008 (a separate diagnostic draw, never the held-out split).
- Device `cpu`, 600 epochs, learning rate 0.001, optimizer AdamW, measured 1.66s.
- Train accuracy: 0.688 -> 0.812; train loss (BCE): 0.6294 -> 0.3114.
- This is a diagnostic of optimization only; it says nothing about held-out or length generalization.

## 8. Artifacts

- No checkpoint artifacts were requested for this run.

### Metric notes

- `estimated_block_calls`: `estimated_block_calls` counts sequential applications of the shared transformer block. It is a structural counter, not a FLOP count and not a latency measurement; see `seconds` for measured wall-clock time.
- `seconds`: Measured wall-clock training time (per configuration, averaged over seeds).
- `in_distribution_accuracy`: Pooled accuracy over train+dev examples (lengths 4-8).
- `length_generalization_drop`: in_distribution_accuracy - heldout_accuracy (per seed).
- `paired_delta_vs_baseline`: Per-seed difference against the same-seed baseline run, then mean/std.
- `decision`: Operational rule only: >= +0.05 improvement, <= -0.05 regression, else flat. Not a significance test and not an equivalence test.
- `parameter_counts`: total/trainable/frozen counted from the real model object (requires_grad for torch).

Historical note: `results/smoke.json` and `results/smoke.md` predate this generator and are kept as historical evidence; they are not regenerated by this report writer.

See [DESIGN.md](DESIGN.md) for the specified hypotheses and the fixed decision rule. The commit history does **not** establish preregistration: DESIGN.md and the first results were committed in the same commit (161752a), so these are specified hypotheses and a fixed decision rule, not a preregistration claim.

