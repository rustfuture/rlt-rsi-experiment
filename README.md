
# RLT-RSI Experiment: Looped Transformers, Length Generalization, and RSI-Style Iterative Adaptation

Reproducible minimal experiment comparing a conventional transformer with a
shared-weight recurrent/looped transformer on binary sequence parity. The
experiment probes iterative computation, compute scaling, and length
generalization; it does not test or claim general reasoning ability.

A separate mode adds **RSI-style iterative adaptation** over the recurrent loop
schedule (`rlt_rsi.train_rsi`). It is a bounded engineering loop — bounded
candidate loop counts, dev-set selection, and a final post-selection held-out
evaluation — not unrestricted or general Recursive Self-Improvement, and it
makes no claim of intelligence improvement.

## Research Questions

1. **Shared-weight recurrence vs conventional transformer:** Does applying a transformer block repeatedly with shared weights alter accuracy or length generalization relative to a single-pass transformer under identical initialization?
2. **Loop count scaling (1, 2, 4) and compute cost:** How do accuracy, measured wall-clock seconds, and sequential block applications scale as iteration loops increase while architectural parameters stay strictly constant?
3. **Length generalization:** Does recurrent processing change generalization to longer sequence lengths (held-out lengths 12–16 vs train/dev lengths 4–8)?
4. **Controlled parameter comparison (structural, per backend):** Within a backend, the baseline and every looped configuration have identical parameter counts; adding loops adds block applications, not parameters. The NumPy and PyTorch models are **different architectures** and their parameter counts are never claimed to be equal.

## Quickstart

```bash
# Setup environment
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'

# Optional: real training backend (not required for the NumPy smoke path)
python -m pip install -e '.[torch]'

# Run the NumPy smoke pipeline (frozen features; plumbing check only)
python -m rlt_rsi.train --backend numpy --seeds 7,42,123 --output results/smoke.json

# Run the full trainable torch pipeline (device auto -> cuda, else mps, else cpu)
python -m rlt_rsi.train --backend torch --device auto --seeds 7,42,123 --epochs 80 \
    --artifacts-dir results/torch-<device>-<date>

# Run test suite
pytest -q
```

The local environment used for this correction round: python 3.12.13, numpy
2.5.3, torch 2.14.0, Apple M4 Pro (arm64) with Metal/MPS available.

The runner evaluates 4 configurations across 3 seeds (`7, 42, 123`):
- `baseline` (loops=1): single-pass reference
- `looped` (loops=1): shared-weight model, single pass
- `looped` (loops=2): shared-weight model, 2 sequential block applications per step
- `looped` (loops=4): shared-weight model, 4 sequential block applications per step

## What each backend actually does

| | NumPy (`--backend numpy`) | PyTorch (`--backend torch`) |
|---|---|---|
| Attention | manual scaled dot-product | `nn.MultiheadAttention` |
| Normalization | RMS-style | `nn.LayerNorm` |
| Training scope | **frozen** embedding + block; **only** the linear readout + bias are trained | **all** parameters, end-to-end |
| Parameters | total 4,681 = frozen 4,656 + trainable 25 (d_model + 1) | total 4,945, all trainable |
| dtype / device | float64 / CPU | float32 / resolved torch device |
| Purpose | smoke + plumbing check; **not** evidence about trained recurrence | intended research run |

The two architectures are **not** parameter-identical, so their parameter counts
are not comparable across backends. `architecture_parameters()` is retained for
backward compatibility and returns the *frozen backbone subtotal only*
(embedding + block); it is never the total. Use
`total_parameters()` / `trainable_parameters()` / `frozen_parameters()`.

## Device selection (torch)

`--device auto|cpu|mps|cuda` (default `auto`):

- `cpu` → CPU.
- `cuda` → requires `torch.cuda.is_available()`, otherwise raises `RuntimeError`.
- `mps` → requires `torch.backends.mps.is_available()`, otherwise raises `RuntimeError`.
- `auto` → cuda if available, else mps if available, else cpu.

There is **no silent fallback**; the requested and resolved devices are recorded
in the payload and report.

## Reading the results honestly

- **`estimated_block_calls`** counts sequential applications of the shared transformer block. It is a structural counter: **not** a FLOP count and **not** a latency measurement. Measured wall-clock training time is reported separately as `seconds` per configuration.
- **Equal epoch counts are not equal compute budgets.** A `loop_count=k` configuration performs `k` block applications per optimisation step, so higher loop counts do more work per step. Compare the measured `seconds` column (and remember it is hardware-specific).
- **The ±5 percentage point rule is an operational decision rule only.** It is not a statistical significance test, not a confidence interval, and not an equivalence test. Results are scoped to this run, this task, these seeds, and these sample sizes.
- **Paired per-seed deltas** (`paired_delta_vs_baseline`, with mean/std/stderr) are reported per configuration alongside aggregate means.
- **H1, H2 and H0 are operational rules.** H2 uses `drop = dev_accuracy(lengths 4–8) - heldout_accuracy(lengths 12–16)`, excluding training examples. A smaller gap can reflect worse dev performance, so it is exploratory and must be read alongside absolute accuracies. Earlier artifacts using pooled train+dev remain historical; the corrected reference is `results/torch-cpu-dev-reference/`.
- **Splits are example-disjoint by construction.** Train/dev are drawn as one joint pool of unique examples and randomly partitioned; held-out uses disjoint lengths and excludes train/dev examples. The exact-example overlap count per seed is recorded in the payload and asserted to be zero; `make_splits` raises rather than returning overlapping examples.
- Because unique examples are required, the empirical length distribution deviates from uniform (the short-length example spaces are tiny); the per-split `length_counts` are recorded in the split manifest.

## Artifacts

| Path | Contents |
|---|---|
| `results/smoke.json`, `results/smoke.md` | Current NumPy smoke output (frozen features; plumbing check only), generated by the fixed report writer from example-disjoint splits. |
| `results/archive-v1/` | Superseded v1 NumPy smoke output from the previous split generator and report writer. Retained as evidence, including the note on why it was superseded. |
| `results/torch-smoke/` | Tiny CPU torch smoke run (`--epochs 2`) used to validate the torch report path. |
| `results/torch-<device>-<date>/` | Multi-seed torch comparison: `run.json`, `run.md`, `checkpoint_manifest.json` (per-file SHA-256/size, source commit + dirty state, data manifest + fingerprints, exact rerun command, per-seed raw results, paired summary, reload-and-re-evaluate checks). |

Checkpoint `.pt` files are small but **not portable** (machine/torch-build
specific) and are gitignored under `results/*/checkpoints/`; only the JSON
manifest with SHA-256 hashes is tracked. Regenerate them with the rerun command
recorded in the manifest.

## Scientific integrity notes

- **Measured MPS run-to-run instability (this round).** Re-running the exact
  recorded command (`--backend torch --device auto --seeds 7,42,123 --epochs 80`)
  reproduced the baseline and looped-2 results but changed looped-4: held-out
  accuracy 0.5052 -> 0.5208 and paired delta +0.0417 -> +0.0573, which flips the
  operational label from `flat` to `improvement`. MPS kernels are not bit-exact,
  so the ±0.05 label for a configuration near the threshold is not stable across
  identical re-runs. This is reported as a limitation of the run, not hidden.
- The hypotheses and decision rule are **specified in advance of the run** but are **not** a preregistered protocol: `DESIGN.md` and the first results were committed in the same commit (`161752a`), so commit history does not establish precedence of the hypotheses over the experiment.
- NumPy frozen-feature smoke results **do not measure whether trained recurrence helps**; they only verify plumbing, determinism, and aggregation.
- Torch runs train end-to-end, but with 3 seeds and small sample sizes the ±0.05 rule cannot establish significance or equivalence. Any "improvement" label is an operational-rule outcome for this run only.
- CI runs `pytest -q` (torch tests skip cleanly via `pytest.importorskip`) and a NumPy smoke run on python 3.11 without torch.

## References (scoped theory context)

- Khattab et al. (2024). *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines.* ICLR 2024. [arXiv:2310.03714](https://arxiv.org/abs/2310.03714). (Correct title; earlier drafts of this repo referred to it imprecisely.)
- Chiang, Cholak & Pillay (2023). *Tighter Bounds on the Expressivity of Transformer Encoders.* ICML 2023, PMLR 202:5544–5562. [link](https://proceedings.mlr.press/v202/chiang23a.html). Fixed-precision transformer encoders recognize only languages in uniform TC⁰ — parity is in TC⁰, so no claim that transformers can *never* represent parity is made here.
- Anil et al. (2022). *Exploring Length Generalization in Large Language Models.* NeurIPS 2022. [link](https://mlanthology.org/neurips/2022/anil2022neurips-exploring/). Reported length-generalization failures motivate RQ3, but failure modes depend on training setup and are not universal.
- Dehghani et al. (2019). *Universal Transformers.* ICLR 2019. [arXiv:1807.03819](https://arxiv.org/abs/1807.03819). Recurrent depth with weight sharing; our looped block is a minimal instance.
- Giannou et al. (2023). *Looped Transformers as Programmable Computers.* ICML 2023. [link](https://collaborate.princeton.edu/en/publications/looped-transformers-as-programmable-computers/). Looped depth as a computational resource; our `estimated_block_calls` is a structural depth proxy only.

## RSI-Style Iterative Adaptation

`rlt_rsi.train_rsi` adds a bounded iterative adaptation mode over the same parity
task. It is deliberately narrow: the search only mutates the recurrent loop
schedule, selection uses the `dev` split, and the held-out split is read only
once at the end as a post-selection evaluation. It is **not** a general or
unrestricted Recursive Self-Improvement system and makes no claim of
intelligence improvement.

Mechanism, repeated for `--rsi-generations` generations:

```text
current loop count
→ bounded candidate loop counts (current, current + 1, current - 1, clamped to 1-8)
→ train each candidate on the train split for --rsi-epochs-per-gen
→ evaluate each candidate on the dev split
→ select the candidate with the best dev accuracy (ties broken by lower dev BCE)
→ carry the selected loop count into the next generation
→ final held-out evaluation of the selected lineage (post-selection only)
```

The held-out split is never used during candidate generation, scoring, or
selection. Each `lineage` entry records the candidate proposals, the accepted
loop count, the post-training train BCE, and dev accuracy/BCE; the run reports a
finite `final_train_bce`.

Real execution examples:

```bash
# NumPy backend (frozen features; plumbing check, no torch required)
python -m rlt_rsi.train_rsi --backend numpy --seeds 7,42,123 \
    --rsi-generations 5 --rsi-epochs-per-gen 8 --output results/rsi_smoke.json

# Torch backend (end-to-end training; device auto -> cuda, else mps, else cpu)
python -m rlt_rsi.train_rsi --backend torch --device auto --seeds 7,42,123 \
    --rsi-generations 5 --rsi-epochs-per-gen 8 --output results/rsi_torch.json
```

The RLT baseline documentation, research questions, and scientific limitations
above still apply to this mode: it uses the same task, seeds, and small sample
sizes, and its operational labels are not significance results.
