# Experiment Reference and Methodological Notes

This document collects architectural details, operational rules, artifact descriptions, and design references relocated from the main repository README.

## Run Configurations

The runner evaluates 4 configurations across 3 seeds (`7, 42, 123`):

- `baseline` (loops=1): single-pass reference.
- `looped` (loops=1): shared-weight model, single pass.
- `looped` (loops=2): 2 sequential block applications per step.
- `looped` (loops=4): 4 sequential block applications per step.

The project is an experimental research prototype (v0.3.0). NumPy and PyTorch runs, manifests, and split integrity checks are documented in the committed results.

## Iterative Adaptation

`rlt_rsi.train_rsi` provides bounded iterative adaptation over the same parity task. The search changes only the loop schedule, selects with the validation split, and reads the held-out split once after all generations complete.

```text
current loop count
→ bounded candidate loop counts (current, current + 1, current - 1, clamped to 1-8)
→ train each candidate on the train split for --rsi-epochs-per-gen
→ evaluate each candidate on the dev split
→ select the candidate with the best dev accuracy (ties broken by lower dev BCE)
→ carry the selected loop count into the next generation
→ final held-out evaluation of the selected lineage (post-selection only)
```

The held-out split is never used during candidate generation, scoring, or selection. Each `lineage` entry records candidate proposals, the accepted loop count, post-training train BCE, and dev accuracy/BCE; the run reports a finite `final_train_bce`.

```bash
# NumPy backend (frozen features; fast CPU check)
python3 -m rlt_rsi.train_rsi --backend numpy --seeds 7,42,123 \
    --rsi-generations 5 --rsi-epochs-per-gen 8 --output results/rsi_smoke.json

# PyTorch backend (end-to-end training; device auto -> cuda, else mps, else cpu)
python3 -m rlt_rsi.train_rsi --backend torch --device auto --seeds 7,42,123 \
    --rsi-generations 5 --rsi-epochs-per-gen 8 --output results/rsi_torch.json
```

These commands write `results/rsi_smoke.json` and `results/rsi_torch.json` locally; neither file is committed, and no RSI-adaptation result is committed to this repository yet.

## Backend Architecture Comparison

| | NumPy (`--backend numpy`) | PyTorch (`--backend torch`) |
|---|---|---|
| Attention | manual scaled dot-product | `nn.MultiheadAttention` |
| Normalization | RMS-style | `nn.LayerNorm` |
| Training scope | **frozen** embedding + block; **only** the linear readout + bias are trained | **all** parameters, end-to-end |
| Parameters | total 4,681 = frozen 4,656 + trainable 25 ([`tests/test_models.py`](../tests/test_models.py)) | total 4,945, all trainable ([`tests/test_torch_training.py`](../tests/test_torch_training.py)) |
| dtype / device | float64 / CPU | float32 / resolved torch device |
| Purpose | smoke + plumbing check; **not** evidence about trained recurrence | intended research run |

The two architectures are **not** parameter-identical, so their parameter counts are not comparable across backends. `architecture_parameters()` is retained for backward compatibility and returns the *frozen backbone subtotal only* (embedding + block); it is never the total. Use `total_parameters()` / `trainable_parameters()` / `frozen_parameters()`.

## Device Selection (Torch)

`--device auto|cpu|mps|cuda` (default `auto`):

- `cpu` -> CPU
- `cuda` -> requires `torch.cuda.is_available()`, otherwise raises `RuntimeError`
- `mps` -> requires `torch.backends.mps.is_available()`, otherwise raises `RuntimeError`
- `auto` -> cuda if available, else mps if available, else cpu

There is **no silent fallback**; the requested and resolved devices are recorded in the payload and report.

## Interpreting Results and Operational Rules

- `estimated_block_calls` counts sequential applications of the shared transformer block. It is a structural counter — **not** a FLOP count and **not** a latency measurement. Measured wall-clock training time is reported separately as `seconds`.
- **Equal epoch counts are not equal compute budgets.** A `loop_count=k` configuration performs `k` block applications per optimisation step, so higher loop counts do more work per step. Compare the measured `seconds` column (hardware-specific).
- **The ±5 percentage-point rule is an operational decision rule only.** It is not a statistical significance test or equivalence test, and it does not use the reported 95% interval. Results are scoped to this run, task, seeds, and sample sizes.
- **Paired per-seed deltas** (`paired_delta_vs_baseline`, with mean/std/stderr and `ci95_low`/`ci95_high`) are reported per configuration alongside aggregate means. The interval is a two-sided 95% Student-t interval for the mean of the per-seed differences (looped minus baseline, same seed), computed by `rlt_rsi.stats.mean_ci95`; it is `null` (reported as `n/a`) for fewer than two seeds. With the default 3 seeds it uses t = 4.303 and is very wide.
- **Near-chance results.** In every committed run all configurations are near chance (held-out accuracy 0.464–0.521, held-out BCE about ln 2). H1/H2 verdicts in those reports compare numbers that are close to chance and are not evidence that the models learned parity or that looping helps.
- **H1, H2 and H0 are operational rules.** H2 uses `drop = dev_accuracy(lengths 4–8) - heldout_accuracy(lengths 12–16)`, excluding training examples. A smaller gap can reflect worse dev performance, so it is exploratory and must be read alongside absolute accuracies. The corrected reference is [`results/torch-cpu-dev-reference/`](../results/torch-cpu-dev-reference/); the older [MPS run](../results/torch-mps-2026-09-15/run.md) used pooled train+dev accuracy for this quantity, so its H2 verdict is not comparable (see the dated note at the end of its `run.md`).
- **Splits are example-disjoint by construction.** Train/dev are drawn as one joint pool of unique examples and randomly partitioned; held-out uses disjoint lengths and excludes train/dev examples. The overlap count per seed is recorded and asserted to be zero; `make_splits` raises rather than returning overlapping examples.
- Because unique examples are required, the empirical length distribution deviates from uniform (the short-length example spaces are tiny); per-split `length_counts` are recorded in the split manifest.

## Artifacts and Historical Runs

| Path | Contents |
|---|---|
| [`results/smoke.json`](../results/smoke.json), [`results/smoke.md`](../results/smoke.md) | NumPy smoke output (frozen features; plumbing check only) |
| [`results/archive-v1/`](../results/archive-v1/) | Superseded v1 smoke output, retained as evidence |
| [`results/torch-smoke/`](../results/torch-smoke/) | Tiny CPU torch smoke run (`--epochs 2`) |
| [`results/torch-cpu-dev-reference/`](../results/torch-cpu-dev-reference/) | Corrected CPU torch reference (JSON + manifest + report) |
| [`results/torch-mps-2026-09-15/`](../results/torch-mps-2026-09-15/) | Recorded MPS run. Its `in_distribution_accuracy` and H2 use the older pooled train+dev definition (see the dated note at the end of its `run.md`); all configurations are near chance |

Checkpoint `.pt` files are small but **not portable** (machine/torch-build specific) and are gitignored under `results/*/checkpoints/`; only the JSON manifest with SHA-256 hashes is tracked. Regenerate them with the rerun command recorded in the manifest.

## Theoretical References

- Khattab et al. (2024). *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines.* ICLR 2024. [arXiv:2310.03714](https://arxiv.org/abs/2310.03714).
- Chiang, Cholak & Pillay (2023). *Tighter Bounds on the Expressivity of Transformer Encoders.* ICML 2023, PMLR 202:5544–5562. [link](https://proceedings.mlr.press/v202/chiang23a.html). Fixed-precision transformer encoders recognize only languages in uniform TC⁰; parity is in TC⁰, so no claim is made that transformers can *never* represent parity.
- Anil et al. (2022). *Exploring Length Generalization in Large Language Models.* NeurIPS 2022. [link](https://mlanthology.org/neurips/2022/anil2022neurips-exploring/).
- Dehghani et al. (2019). *Universal Transformers.* ICLR 2019. [arXiv:1807.03819](https://arxiv.org/abs/1807.03819).
- Giannou et al. (2023). *Looped Transformers as Programmable Computers.* ICML 2023. [link](https://collaborate.princeton.edu/en/publications/looped-transformers-as-programmable-computers/).
