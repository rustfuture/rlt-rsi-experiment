# Sequence Parity: Conventional vs Shared-Weight Looped Transformer Report

- **Backend:** `numpy`
- **Seeds (N=3):** `7, 42, 123`
- **Epochs per configuration:** 40
- **Optimizer:** numpy full-batch gradient descent on readout+bias only | **learning rate:** 0.08 | **weight decay:** 0.0001
- **Device:** requested `auto` -> resolved `cpu` | **dtype:** `float64`
- **Python:** `3.12.13` | **NumPy:** `2.5.3` | **PyTorch:** 2.14.0
- **Train/dev size:** 256/128 (lengths 4-8) | **Held-out size:** 128 (lengths 12-16)

> [!IMPORTANT]
> **Scope of this run.** This run used the NumPy frozen-feature smoke path: the transformer block and embeddings are frozen and only the linear readout + bias are trained. Frozen-feature smoke results do not measure whether trained recurrence helps, and nothing here is evidence about trained recurrence or general reasoning.

## 1. Setup actually used

- Backend `numpy`; device requested `auto`, resolved `cpu`.
- dtype `float64`; torch version `2.14.0`; python `3.12.13`; numpy `2.5.3`.
- Epochs 40; optimizer numpy full-batch gradient descent on readout+bias only; learning rate 0.08; weight decay 0.0001.
- Seeds `7, 42, 123`; gradient/training scope: readout + bias only; embedding and transformer block are frozen.
- Batch: full-batch (all train examples per step).

### Parameter accounting (counted from the real model object)

| configuration | total parameters | trainable | frozen |
|---|---:|---:|---:|
| baseline (loops=1) | 4,681 | 25 | 4,656 |
| looped (loops=1) | 4,681 | 25 | 4,656 |
| looped (loops=2) | 4,681 | 25 | 4,656 |
| looped (loops=4) | 4,681 | 25 | 4,656 |

The NumPy model freezes the embedding and transformer block and trains **only the linear readout + bias**. For the configurations above: total 4,681 = frozen 4,656 + trainable 25 (trainable = d_model + 1 with d_model=24). Earlier reports quoted the frozen backbone subtotal (embedding + block, excluding the readout and bias) as if it were the total; that was wrong and is corrected here.

The NumPy and PyTorch models are **different architectures** (NumPy: manual attention + RMS-style normalization + frozen backbone with a trained readout; torch: `nn.MultiheadAttention` + `LayerNorm` with full training). Their parameter counts are not expected to match and must never be compared as if they were the same model.

## 2. Summary results

| model | loops | total params | trainable | frozen | sequential block applications | train acc | dev acc | in-distribution acc (4-8) | held-out acc (12-16) | seconds (mean) | decision vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 1 | 4,681 | 25 | 4,656 | 1 | 0.522 ± 0.002 | 0.479 ± 0.023 | 0.508 ± 0.007 | 0.445 ± 0.021 | 0.0029 | baseline (reference) |
| looped | 1 | 4,681 | 25 | 4,656 | 1 | 0.522 ± 0.002 | 0.479 ± 0.023 | 0.508 ± 0.007 | 0.445 ± 0.021 | 0.0026 | flat |
| looped | 2 | 4,681 | 25 | 4,656 | 2 | 0.483 ± 0.058 | 0.419 ± 0.045 | 0.462 ± 0.051 | 0.461 ± 0.023 | 0.0043 | flat |
| looped | 4 | 4,681 | 25 | 4,656 | 4 | 0.490 ± 0.010 | 0.516 ± 0.008 | 0.498 ± 0.009 | 0.503 ± 0.058 | 0.0071 | improvement |

`sequential block applications` is a structural counter (see metric notes at the end): it is not FLOPs and not a latency measurement. `seconds` is measured wall-clock training time for 40 epochs on this machine.

## 3. Paired per-seed differences vs the baseline

Differences are computed **per seed** against the baseline run with the same seed (paired), then averaged. `n_seeds` is small; the standard error is reported so it is not mistaken for a significance test.

| model | loops | metric | per-seed (looped - baseline) | mean | std | stderr |
|---|---:|---|---:|---:|---:|---:|
| looped | 1 | heldout_accuracy | +0.0000, +0.0000, +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 1 | heldout_bce | +0.0000, +0.0000, +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 1 | in_distribution_accuracy | +0.0000, +0.0000, +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 1 | length_generalization_drop | +0.0000, +0.0000, +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 1 | train_accuracy | +0.0000, +0.0000, +0.0000 | +0.0000 | 0.0000 | 0.0000 |
| looped | 2 | heldout_accuracy | +0.0000, -0.0156, +0.0625 | +0.0156 | 0.0413 | 0.0239 |
| looped | 2 | heldout_bce | +0.0002, +0.0010, -0.0044 | -0.0011 | 0.0029 | 0.0017 |
| looped | 2 | in_distribution_accuracy | -0.0521, +0.0026, -0.0885 | -0.0460 | 0.0459 | 0.0265 |
| looped | 2 | length_generalization_drop | -0.0521, +0.0182, -0.1510 | -0.0616 | 0.0850 | 0.0491 |
| looped | 2 | train_accuracy | -0.0547, +0.0273, -0.0898 | -0.0391 | 0.0601 | 0.0347 |
| looped | 4 | heldout_accuracy | -0.0234, +0.0938, +0.1016 | +0.0573 | 0.0700 | 0.0404 |
| looped | 4 | heldout_bce | +8.9168, +1.2407, +1.6996 | +3.9524 | 4.3055 | 2.4858 |
| looped | 4 | in_distribution_accuracy | -0.0052, -0.0208, -0.0026 | -0.0095 | 0.0099 | 0.0057 |
| looped | 4 | length_generalization_drop | +0.0182, -0.1146, -0.1042 | -0.0668 | 0.0739 | 0.0426 |
| looped | 4 | train_accuracy | -0.0234, -0.0391, -0.0352 | -0.0326 | 0.0081 | 0.0047 |

### Decision rule (operational, not statistical)

The decision rule is: difference >= +0.05 -> `improvement`; difference <= -0.05 -> `regression`; otherwise `flat`. This is an **operational decision rule only**. It is not a statistical significance test, not a confidence interval, and not an equivalence test; with 3 seed(s) and held-out n=128 it cannot establish either superiority or equivalence.

## 4. Evaluation of Core Research Questions

1. **Shared-weight recurrence vs conventional single-pass transformer.** Operational decision-rule outcomes on held-out accuracy (baseline mean 0.445 ± 0.021): looped-1: `flat` (held-out delta +0.0000 ± 0.0000), looped-2: `flat` (held-out delta +0.0156 ± 0.0413), looped-4: `improvement` (held-out delta +0.0573 ± 0.0700). Scoped to this run, this task (binary sequence parity), these seeds (7, 42, 123), these sample sizes (train 256, dev 128, held-out 128), this training budget (40 epochs, numpy full-batch gradient descent on readout+bias only), and backend `numpy` on device `cpu`. Because this is the frozen-feature NumPy smoke path, it does not measure whether trained recurrence helps; only the readout+bias are trained.

2. **Loop scaling and compute trade-off (1, 2, 4).** Total parameter counts across the reported configurations: 4,681. Sequential block applications scale as baseline(loops=1) -> 1 block application(s), looped(loops=1) -> 1 block application(s), looped(loops=2) -> 2 block application(s), looped(loops=4) -> 4 block application(s). Measured wall-clock seconds: baseline(loops=1): 0.0029s, looped(loops=1): 0.0026s, looped(loops=2): 0.0043s, looped(loops=4): 0.0071s. Held-out accuracy by loop count: baseline(loops=1): 0.445, looped(loops=1): 0.445, looped(loops=2): 0.461, looped(loops=4): 0.503. Equal epoch counts are **not** equal compute budgets: each additional loop applies the shared block once more per optimisation step, so the higher-loop configurations perform more work per step; the measured `seconds` column is the only latency evidence in this report.

3. **Length generalization (train/dev 4-8 vs held-out 12-16).** These length ranges are disjoint and split generation is example-disjoint by construction. Length-generalization drop is defined per seed as `in_distribution_accuracy(train+dev, lengths 4-8) - heldout_accuracy(lengths 12-16)`; H2 statistic = drop(baseline) - drop(looped), evaluated against the same +/-0.05 operational rule. Results: looped-1: +0.0000 -> not supported (within the +/-0.05 rule), looped-2: +0.0616 -> supported, looped-4: +0.0668 -> supported.

   Exact-example overlap measured per seed (train_dev / train_heldout / dev_heldout): seed 7: 0/0/0; seed 42: 0/0/0; seed 123: 0/0/0.

4. **Controlled parameter comparison (structural).** Within backend `numpy`: baseline(loops=1): total 4,681, trainable 25, frozen 4,656, looped(loops=1): total 4,681, trainable 25, frozen 4,656, looped(loops=2): total 4,681, trainable 25, frozen 4,656, looped(loops=4): total 4,681, trainable 25, frozen 4,656. Trainable parameter counts are constant across loop counts, so increasing loops adds sequential block applications but no new parameters. This is a structural check of the implementation, not a claim that the two backends have equal counts.

## 5. Hypothesis evaluation status

| id | statement | status | statistic | threshold | result |
|---|---|---|---:|---:|---|
| H1 | Looped configurations (2 or 4 loops) exceed the single-pass baseline on held-out parity accuracy by >= 0.05. | evaluated | loops=1: +0.0000; loops=2: +0.0156; loops=4: +0.0573 | 0.05 | supported for at least one loop count |
| H2 | The accuracy drop from in-distribution lengths (4-8) to held-out lengths (12-16) is at least 0.05 smaller for a looped configuration than for the baseline. | evaluated | loops=1: +0.0000; loops=2: +0.0616; loops=4: +0.0668 | 0.05 | supported for at least one loop count |
| H0 | No looped configuration meets the +/-0.05 improvement threshold (flat or worse held-out accuracy than the baseline). | evaluated | n/a | 0.05 | rejected (at least one configuration met the improvement threshold) |

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
- NumPy backend uses float64 on CPU and trains only the readout+bias via deterministic full-batch gradient descent; the frozen features make it a plumbing check.

Identical initialization check (per seed, over the shared modules): the recorded initialization fingerprints are identical across the baseline and all looped configurations.
| seed | identical shared init | fingerprint (first 16 hex chars) |
|---|---|---|
| 7 | True | `f2c025f7133076ad` |
| 42 | True | `44ee10a41d816fec` |
| 123 | True | `1b2c241d3182d27c` |

## 7. Environment and PyTorch status (derived from this run)

PyTorch 2.14.0 is available in this environment, but this run used the NumPy backend on CPU; no torch device was used.

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

