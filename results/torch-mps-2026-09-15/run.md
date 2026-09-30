# Sequence Parity: Conventional vs Shared-Weight Looped Transformer Report

- **Backend:** `torch`
- **Seeds (N=3):** `7, 42, 123`
- **Epochs per configuration:** 80
- **Optimizer:** AdamW | **learning rate:** 0.003 | **weight decay:** 0.0001
- **Device:** requested `auto` -> resolved `mps` | **dtype:** `float32`
- **Python:** `3.12.13` | **NumPy:** `2.5.3` | **PyTorch:** 2.14.0
- **Train/dev size:** 256/128 (lengths 4-8) | **Held-out size:** 128 (lengths 12-16)

> [!IMPORTANT]
> **Scope of this run.** This run trained all parameters end-to-end with AdamW for 80 epochs on device `mps` for seeds 7, 42, 123. Conclusions are scoped to this run, this task (binary sequence parity), these seeds, these sample sizes and this training budget; they are not evidence about general reasoning or about recurrence at other scales.

## 1. Setup actually used

- Backend `torch`; device requested `auto`, resolved `mps`.
- dtype `float32`; torch version `2.14.0`; python `3.12.13`; numpy `2.5.3`.
- Epochs 80; optimizer AdamW; learning rate 0.003; weight decay 0.0001.
- Seeds `7, 42, 123`; gradient/training scope: all parameters (end-to-end).
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
| baseline | 1 | 4,945 | 4,945 | 0 | 1 | 0.555 ± 0.035 | 0.464 ± 0.039 | 0.524 ± 0.035 | 0.464 ± 0.083 | 0.3659 | baseline (reference) |
| looped | 1 | 4,945 | 4,945 | 0 | 1 | 0.555 ± 0.035 | 0.464 ± 0.039 | 0.524 ± 0.035 | 0.464 ± 0.083 | 0.2765 | flat |
| looped | 2 | 4,945 | 4,945 | 0 | 2 | 0.491 ± 0.006 | 0.417 ± 0.035 | 0.466 ± 0.009 | 0.471 ± 0.030 | 0.4000 | flat |
| looped | 4 | 4,945 | 4,945 | 0 | 4 | 0.586 ± 0.038 | 0.505 ± 0.052 | 0.559 ± 0.042 | 0.505 ± 0.032 | 0.6170 | flat |

`sequential block applications` is a structural counter (see metric notes at the end): it is not FLOPs and not a latency measurement. `seconds` is measured wall-clock training time for 80 epochs on this machine.

## 3. Paired per-seed differences vs the baseline

Differences are computed **per seed** against the baseline run with the same seed (paired), then averaged. `n_seeds` is small; the standard error is reported so it is not mistaken for a significance test.

| model | loops | metric | per-seed (looped - baseline) | mean | std | stderr |
|---|---:|---|---:|---:|---:|---:|
| looped | 1 | heldout_accuracy | +0.0000, +0.0000, +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 1 | heldout_bce | -0.0000, -0.0000, +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 1 | in_distribution_accuracy | +0.0000, +0.0000, +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 1 | length_generalization_drop | +0.0000, +0.0000, +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 1 | train_accuracy | +0.0000, +0.0000, +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 2 | heldout_accuracy | -0.0469, -0.0391, +0.1094 | +0.0078 | 0.0880 | 0.0508 |
| looped | 2 | heldout_bce | +0.0009, -0.0023, -0.0123 | -0.0045 | 0.0069 | 0.0040 |
| looped | 2 | in_distribution_accuracy | -0.0885, -0.0599, -0.0260 | -0.0582 | 0.0313 | 0.0181 |
| looped | 2 | length_generalization_drop | -0.0417, -0.0208, -0.1354 | -0.0660 | 0.0610 | 0.0352 |
| looped | 2 | train_accuracy | -0.1055, -0.0625, -0.0234 | -0.0638 | 0.0410 | 0.0237 |
| looped | 4 | heldout_accuracy | +0.0000, +0.0000, +0.1250 | +0.0417 | 0.0722 | 0.0417 |
| looped | 4 | heldout_bce | -0.0008, +0.0193, -0.0021 | +0.0055 | 0.0120 | 0.0069 |
| looped | 4 | in_distribution_accuracy | -0.0521, +0.0625, +0.0937 | +0.0347 | 0.0768 | 0.0443 |
| looped | 4 | length_generalization_drop | -0.0521, +0.0625, -0.0313 | -0.0069 | 0.0610 | 0.0352 |
| looped | 4 | train_accuracy | -0.0469, +0.0469, +0.0938 | +0.0312 | 0.0716 | 0.0413 |

### Decision rule (operational, not statistical)

The decision rule is: difference >= +0.05 -> `improvement`; difference <= -0.05 -> `regression`; otherwise `flat`. This is an **operational decision rule only**. It is not a statistical significance test, not a confidence interval, and not an equivalence test; with 3 seed(s) and held-out n=128 it cannot establish either superiority or equivalence.

## 4. Evaluation of Core Research Questions

1. **Shared-weight recurrence vs conventional single-pass transformer.** Operational decision-rule outcomes on held-out accuracy (baseline mean 0.464 ± 0.083): looped-1: `flat` (held-out delta +0.0000 ± 0.0000), looped-2: `flat` (held-out delta +0.0078 ± 0.0880), looped-4: `flat` (held-out delta +0.0417 ± 0.0722). Scoped to this run, this task (binary sequence parity), these seeds (7, 42, 123), these sample sizes (train 256, dev 128, held-out 128), this training budget (80 epochs, AdamW), and backend `torch` on device `mps`. Both baselines and looped configurations were trained end-to-end under this budget.

2. **Loop scaling and compute trade-off (1, 2, 4).** Total parameter counts across the reported configurations: 4,945. Sequential block applications scale as baseline(loops=1) -> 1 block application(s), looped(loops=1) -> 1 block application(s), looped(loops=2) -> 2 block application(s), looped(loops=4) -> 4 block application(s). Measured wall-clock seconds: baseline(loops=1): 0.3659s, looped(loops=1): 0.2765s, looped(loops=2): 0.4000s, looped(loops=4): 0.6170s. Held-out accuracy by loop count: baseline(loops=1): 0.464, looped(loops=1): 0.464, looped(loops=2): 0.471, looped(loops=4): 0.505. Equal epoch counts are **not** equal compute budgets: each additional loop applies the shared block once more per optimisation step, so the higher-loop configurations perform more work per step; the measured `seconds` column is the only latency evidence in this report.

3. **Length generalization (train/dev 4-8 vs held-out 12-16).** These length ranges are disjoint and split generation is example-disjoint by construction. Length-generalization drop is defined per seed as `in_distribution_accuracy(train+dev, lengths 4-8) - heldout_accuracy(lengths 12-16)`; H2 statistic = drop(baseline) - drop(looped), evaluated against the same +/-0.05 operational rule. Results: looped-1: +0.0000 -> not supported (within the +/-0.05 rule), looped-2: +0.0660 -> supported, looped-4: +0.0069 -> not supported (within the +/-0.05 rule).

   Exact-example overlap measured per seed (train_dev / train_heldout / dev_heldout): seed 7: 0/0/0; seed 42: 0/0/0; seed 123: 0/0/0.

4. **Controlled parameter comparison (structural).** Within backend `torch`: baseline(loops=1): total 4,945, trainable 4,945, frozen 0, looped(loops=1): total 4,945, trainable 4,945, frozen 0, looped(loops=2): total 4,945, trainable 4,945, frozen 0, looped(loops=4): total 4,945, trainable 4,945, frozen 0. Trainable parameter counts are constant across loop counts, so increasing loops adds sequential block applications but no new parameters. This is a structural check of the implementation, not a claim that the two backends have equal counts.

## 5. Hypothesis evaluation status

| id | statement | status | statistic | threshold | result |
|---|---|---|---:|---:|---|
| H1 | Looped configurations (2 or 4 loops) exceed the single-pass baseline on held-out parity accuracy by >= 0.05. | evaluated | loops=1: +0.0000; loops=2: +0.0078; loops=4: +0.0417 | 0.05 | not supported in this run |
| H2 | The accuracy drop from in-distribution lengths (4-8) to held-out lengths (12-16) is at least 0.05 smaller for a looped configuration than for the baseline. | evaluated | loops=1: +0.0000; loops=2: +0.0660; loops=4: +0.0069 | 0.05 | supported for at least one loop count |
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
- MPS was used for this run; MPS results are not guaranteed bit-identical to CPU runs.

Identical initialization check (per seed, over the shared modules): the recorded initialization fingerprints are identical across the baseline and all looped configurations.
| seed | identical shared init | fingerprint (first 16 hex chars) |
|---|---|---|
| 7 | True | `5ed3a762b875fbff` |
| 42 | True | `17320acad33bc4b3` |
| 123 | True | `ebe2f2c902165eea` |

## 7. Environment and PyTorch status (derived from this run)

PyTorch 2.14.0 is available in this environment and was used for this run on device `mps`.

### Optimization diagnostic (separate small dataset, not the held-out split)

Overfit a tiny separate diagnostic batch of parity examples (lengths 4-8) to check that the optimizer actually reduces loss. This is a diagnostic of optimization plumbing, not a held-out claim; the diagnostic draw is never used for evaluation.
- Dataset: 32 examples, lengths 4-8, seed 9008 (a separate diagnostic draw, never the held-out split).
- Device `mps`, 600 epochs, learning rate 0.001, optimizer AdamW, measured 3.82s.
- Train accuracy: 0.688 -> 0.781; train loss (BCE): 0.6294 -> 0.3306.
- This is a diagnostic of optimization only; it says nothing about held-out or length generalization.

## 8. Artifacts

- Directory: `results/torch-mps-2026-09-15` (run.json, run.md, checkpoint_manifest.json, checkpoints/).
- Exact rerun command: `python -m rlt_rsi.train --backend torch --device auto --seeds 7,42,123 --epochs 80 --artifacts-dir results/torch-mps-2026-09-15`
- Checkpoint manifest SHA-256: `f895565dbe9f4414f5f723df36d4feb89b0ec91b4b925018f922970d766b1fc0`
- Checkpoints verified by reload-and-re-evaluate: 12/12 matched the recorded metrics (tolerance 0.0001).
- Portability limit: Checkpoint .pt files are gitignored local artifacts (12 files, 25757-25905 bytes each) and are NOT portable; only checkpoint_manifest.json (with SHA-256 hashes) is tracked. Regenerate checkpoints with the rerun command on a matching torch build.

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


---

## Editorial note added 2026-09-30 (the numbers above are unchanged)

This note was added after the run; nothing above it was edited.

1. **`in_distribution_accuracy` and H2 use an older definition.** Section 2, section 4 (item 3), section 5 (H2) and the Metric notes above define `in_distribution_accuracy` as accuracy pooled over train+dev examples (lengths 4-8). The current code (`rlt_rsi/train.py`) and [`docs/reference.md`](../../docs/reference.md) define it as dev examples only. The corrected-definition reference run is [`results/torch-cpu-dev-reference/`](../torch-cpu-dev-reference/run.md). The H2 result "supported for at least one loop count" in this file rests on the pooled definition. Recomputed from the per-seed values in `run.json` with the dev-only definition (`drop = dev_accuracy - heldout_accuracy`), the H2 statistic is +0.0000 for looped-1, +0.0547 for looped-2 and +0.0000 for looped-4; the looped-2 value comes from a dev accuracy (0.417) that is below 0.5, not from better held-out accuracy (0.471 vs 0.464 for the baseline).
2. **All configurations are near chance.** Every held-out accuracy above is between 0.464 and 0.505, train accuracy is between 0.49 and 0.59, and per-seed held-out BCE is 0.692-0.721 (ln 2 = 0.693 is the loss of a constant 0.5 prediction). These runs do not show that any configuration learned parity, so the H2 "supported" verdict reflects differences between numbers that are close to chance, on 3 seeds and 128 examples.
3. **No MPS repeat is committed.** This file records one MPS run, in which looped-4 held-out accuracy is 0.5052 (paired delta +0.0417, `flat`). The looped-4 value 0.5208 (paired delta +0.0573, `improvement`) is committed only for the `--device cpu` run in [`results/torch-cpu-dev-reference/`](../torch-cpu-dev-reference/run.md). The claim that a repeated MPS run moved from 0.5052 to 0.5208 has no committed artifact, and no test or artifact verifies bit-exact reproducibility on any device.
