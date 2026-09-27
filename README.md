# RLT-RSI Experiment

A reproducible experimental benchmark comparing conventional and shared-weight looped transformers on binary sequence parity, built for machine learning researchers studying recurrent depth, length generalization, and bounded loop-schedule adaptation.

<p align="left">
  <a href="https://github.com/rustfuture/rlt-rsi-experiment/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/rustfuture/rlt-rsi-experiment/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-green?style=flat-square"></a>
</p>

**Status:** Experimental research prototype (v0.3.0). Evaluates small-scale algorithmic parity across NumPy smoke tests and PyTorch models with committed run manifests and verified split integrity.

- **Baseline vs. Looped Models:** Evaluates a single-pass conventional transformer against a shared-weight looped architecture (1, 2, and 4 iterations) on binary sequence parity.
- **Length Generalization:** Tests out-of-distribution transfer from short train/dev sequences (lengths 4–8) to longer held-out sequences (lengths 12–16) across fixed seeds (`7, 42, 123`).
- **Data Leakage Controls:** Asserts zero exact-example overlap between train, dev, and held-out splits with automated per-seed assertions.
- **RSI Iterative Adaptation:** Implements bounded loop-schedule search (`rlt_rsi.train_rsi`) that mutates candidate loop counts (clamped to 1–8) with dev-set selection and post-selection held-out evaluation.
- **Dual Execution Backends:** Provides a fast, deterministic NumPy CPU smoke runner (frozen backbone) for pipeline checks and an end-to-end PyTorch training pipeline with explicit device resolution (`--device auto|cpu|mps|cuda`).

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#how-the-rsi-loop-works">How RSI Works</a> ·
  <a href="#backend-scope">Backend Scope</a> ·
  <a href="#interpreting-results">Results</a> ·
  <a href="#rsi-style-iterative-adaptation">RSI Adaptation</a> ·
  <a href="#artifacts">Artifacts</a> ·
  <a href="#scope-and-limitations">Limitations</a>
</p>

## At a Glance

| | |
|---|---|
| Task | Binary sequence parity; train/dev lengths 4–8, held-out lengths 12–16 |
| Models | Conventional transformer vs. shared-weight looped transformer |
| Backends | NumPy (frozen features) and PyTorch (end-to-end) |
| RSI mode | Bounded loop-schedule search, dev selection, post-selection held-out |
| Splits | Example-disjoint by construction; zero-overlap asserted per seed |
| Tests | Full suite in CI (a NumPy job and a Torch-CPU job) |

## Two Paths

**RLT baseline** (`rlt_rsi.train`)

- conventional single-pass `baseline` reference
- shared-weight `looped` model at loop counts 1, 2, 4
- loop-count scaling vs. measured wall-clock compute
- length generalization to held-out lengths 12–16

**RSI-style iterative adaptation** (`rlt_rsi.train_rsi`)

- bounded candidate loop counts
- per-candidate training on the train split
- dev-set evaluation and selection
- explicit lineage per generation
- final held-out evaluation, post-selection only

## How the RSI Loop Works

```mermaid
flowchart TD
    K["Current loop count"] --> G["Generate k-1, k, k+1 - clamped to 1-8"]
    G --> T["Train each candidate on train"]
    T --> D["Evaluate on dev"]
    D --> S["Select best dev accuracy - tie-break lower dev BCE"]
    S --> N["Next generation"]
    N -->|more generations| G
    N -->|done| H["Final held-out evaluation - post-selection only"]
```

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# Run test suite
pytest -q

# NumPy smoke pipeline (frozen features; fast CPU check)
python3 -m rlt_rsi.train --backend numpy --seeds 7,42,123 --output results/smoke.json

# RSI iterative adaptation loop (NumPy backend)
python3 -m rlt_rsi.train_rsi --backend numpy --seeds 7,42,123 --rsi-generations 2 --rsi-epochs-per-gen 2 --output results/rsi_smoke.json

# Optional: PyTorch end-to-end training (requires 'pip install -e .[torch]')
python3 -m rlt_rsi.train --backend torch --device auto --seeds 7,42,123 --epochs 80 \
    --artifacts-dir results/torch-run
```

Prefer no local setup? Open [`notebooks/rlt_rsi_colab.ipynb`](notebooks/rlt_rsi_colab.ipynb)
in Colab. Stage A is a CPU-safe NumPy smoke and RSI run; Stage B runs the Torch path when a GPU is available.

The runner evaluates 4 configurations across 3 seeds (`7, 42, 123`):

- `baseline` (loops=1): single-pass reference
- `looped` (loops=1): shared-weight model, single pass
- `looped` (loops=2): 2 sequential block applications per step
- `looped` (loops=4): 4 sequential block applications per step

## Backend Scope

| | NumPy (`--backend numpy`) | PyTorch (`--backend torch`) |
|---|---|---|
| Attention | manual scaled dot-product | `nn.MultiheadAttention` |
| Normalization | RMS-style | `nn.LayerNorm` |
| Training scope | **frozen** embedding + block; **only** the linear readout + bias are trained | **all** parameters, end-to-end |
| Parameters | total 4,681 = frozen 4,656 + trainable 25 ([`tests/test_models.py`](tests/test_models.py)) | total 4,945, all trainable ([`tests/test_torch_training.py`](tests/test_torch_training.py)) |
| dtype / device | float64 / CPU | float32 / resolved torch device |
| Purpose | smoke + plumbing check; **not** evidence about trained recurrence | intended research run |

The two architectures are **not** parameter-identical, so their parameter counts are not
comparable across backends. `architecture_parameters()` is retained for backward
compatibility and returns the *frozen backbone subtotal only* (embedding + block); it is
never the total. Use `total_parameters()` / `trainable_parameters()` / `frozen_parameters()`.

## Device Selection (Torch)

`--device auto|cpu|mps|cuda` (default `auto`):

- `cpu` → CPU
- `cuda` → requires `torch.cuda.is_available()`, otherwise raises `RuntimeError`
- `mps` → requires `torch.backends.mps.is_available()`, otherwise raises `RuntimeError`
- `auto` → cuda if available, else mps if available, else cpu

There is **no silent fallback**; the requested and resolved devices are recorded in the
payload and report.

## Interpreting Results

- **`estimated_block_calls`** counts sequential applications of the shared transformer
  block. It is a structural counter — **not** a FLOP count and **not** a latency
  measurement. Measured wall-clock training time is reported separately as `seconds`.
- **Equal epoch counts are not equal compute budgets.** A `loop_count=k` configuration
  performs `k` block applications per optimisation step, so higher loop counts do more
  work per step. Compare the measured `seconds` column (hardware-specific).
- **The ±5 percentage-point rule is an operational decision rule only.** It is not a
  statistical significance test, confidence interval, or equivalence test. Results are
  scoped to this run, task, seeds, and sample sizes.
- **Paired per-seed deltas** (`paired_delta_vs_baseline`, with mean/std/stderr) are
  reported per configuration alongside aggregate means.
- **H1, H2 and H0 are operational rules.** H2 uses
  `drop = dev_accuracy(lengths 4–8) - heldout_accuracy(lengths 12–16)`, excluding training
  examples. A smaller gap can reflect worse dev performance, so it is exploratory and must
  be read alongside absolute accuracies. The corrected reference is
  [`results/torch-cpu-dev-reference/`](results/torch-cpu-dev-reference/).
- **Splits are example-disjoint by construction.** Train/dev are drawn as one joint pool
  of unique examples and randomly partitioned; held-out uses disjoint lengths and excludes
  train/dev examples. The overlap count per seed is recorded and asserted to be zero;
  `make_splits` raises rather than returning overlapping examples.
- Because unique examples are required, the empirical length distribution deviates from
  uniform (the short-length example spaces are tiny); per-split `length_counts` are
  recorded in the split manifest.

## RSI-Style Iterative Adaptation

`rlt_rsi.train_rsi` adds a bounded iterative adaptation mode over the same parity task.
The search only mutates the recurrent loop schedule, selection uses the `dev` split, and
the held-out split is read only once at the end as a post-selection evaluation.

```text
current loop count
→ bounded candidate loop counts (current, current + 1, current - 1, clamped to 1-8)
→ train each candidate on the train split for --rsi-epochs-per-gen
→ evaluate each candidate on the dev split
→ select the candidate with the best dev accuracy (ties broken by lower dev BCE)
→ carry the selected loop count into the next generation
→ final held-out evaluation of the selected lineage (post-selection only)
```

The held-out split is never used during candidate generation, scoring, or selection. Each
`lineage` entry records the candidate proposals, the accepted loop count, the post-training
train BCE, and dev accuracy/BCE; the run reports a finite `final_train_bce`.

```bash
# NumPy backend (frozen features; plumbing check, no torch required)
python3 -m rlt_rsi.train_rsi --backend numpy --seeds 7,42,123 \
    --rsi-generations 5 --rsi-epochs-per-gen 8 --output results/rsi_smoke.json

# Torch backend (end-to-end training; device auto -> cuda, else mps, else cpu)
python3 -m rlt_rsi.train_rsi --backend torch --device auto --seeds 7,42,123 \
    --rsi-generations 5 --rsi-epochs-per-gen 8 --output results/rsi_torch.json
```

## Artifacts

| Path | Contents |
|---|---|
| [`results/smoke.json`](results/smoke.json), [`results/smoke.md`](results/smoke.md) | NumPy smoke output (frozen features; plumbing check only) |
| [`results/archive-v1/`](results/archive-v1/) | Superseded v1 smoke output, retained as evidence |
| [`results/torch-smoke/`](results/torch-smoke/) | Tiny CPU torch smoke run (`--epochs 2`) |
| [`results/torch-cpu-dev-reference/`](results/torch-cpu-dev-reference/) | Corrected CPU torch reference (JSON + manifest + report) |
| [`results/torch-mps-2026-09-15/`](results/torch-mps-2026-09-15/) | Recorded MPS run discussed in the evaluation notes |

Checkpoint `.pt` files are small but **not portable** (machine/torch-build specific) and
are gitignored under `results/*/checkpoints/`; only the JSON manifest with SHA-256 hashes
is tracked. Regenerate them with the rerun command recorded in the manifest.

## Scope and Limitations

- **Algorithmic Parity Scope (No General Reasoning):** Sequence parity is a narrow algorithmic check on binary sequences; findings do not evaluate language modeling, multi-step reasoning, or general cognitive capabilities.
- **Bounded RSI Adaptation Scope:** The RSI mode is strictly a bounded heuristic search over integer loop schedules (clamped to 1–8) using dev-set selection; it is not general Recursive Self-Improvement and makes no claim of intelligence improvement.
- **Operational Decision Threshold (±0.05):** The ±0.05 held-out accuracy threshold across N=3 seeds is an operational decision heuristic, not a formal statistical significance or equivalence test.
- **Backend Comparability:** The NumPy backend uses a frozen backbone with trained readout (total 4,681 parameters) as a fast smoke test; PyTorch trains all parameters end-to-end (total 4,945 parameters). Architectures differ and are not directly comparable across backends.
- **Hardware Nondeterminism & Stability:** Re-running identical PyTorch MPS runs on Apple Silicon revealed kernel-level variation where looped-4 held-out accuracy moved from 0.5052 (`flat`) to 0.5208 (`improvement`), crossing the operational boundary (documented in [`results/torch-mps-2026-09-15/run.md`](results/torch-mps-2026-09-15/run.md)). For bit-exact reproducibility, use `--device cpu`.
- **Checkpoint Portability:** Checkpoint files (`.pt`) are local machine-specific artifacts and gitignored; only the JSON manifest with SHA-256 hashes is tracked in version control.
- **Specified Hypotheses, Not Preregistration:** Hypotheses and decision rules were specified in advance of analysis, but because [`DESIGN.md`](DESIGN.md) and initial results were committed in the same commit (`161752a`), the repository documents this as specified hypotheses rather than claiming external preregistration.
- **CI Test Coverage:** CI runs `pytest -q` plus a NumPy smoke in a Python 3.11 job without torch, and a Torch-CPU job installs CPU torch and runs the full suite including the Torch RSI regression test.

<details>
<summary>Design references (scoped theory context)</summary>

- Khattab et al. (2024). *DSPy: Compiling Declarative Language Model Calls into
  Self-Improving Pipelines.* ICLR 2024. [arXiv:2310.03714](https://arxiv.org/abs/2310.03714).
- Chiang, Cholak & Pillay (2023). *Tighter Bounds on the Expressivity of Transformer
  Encoders.* ICML 2023, PMLR 202:5544–5562.
  [link](https://proceedings.mlr.press/v202/chiang23a.html). Fixed-precision transformer
  encoders recognize only languages in uniform TC⁰; parity is in TC⁰, so no claim is made
  that transformers can *never* represent parity.
- Anil et al. (2022). *Exploring Length Generalization in Large Language Models.* NeurIPS
  2022. [link](https://mlanthology.org/neurips/2022/anil2022neurips-exploring/).
- Dehghani et al. (2019). *Universal Transformers.* ICLR 2019.
  [arXiv:1807.03819](https://arxiv.org/abs/1807.03819).
- Giannou et al. (2023). *Looped Transformers as Programmable Computers.* ICML 2023.
  [link](https://collaborate.princeton.edu/en/publications/looped-transformers-as-programmable-computers/).

</details>

## Repository Map

| Path | Contents |
|---|---|
| [`rlt_rsi/train.py`](rlt_rsi/train.py) | Baseline and looped training, backends, evaluation |
| [`rlt_rsi/train_rsi.py`](rlt_rsi/train_rsi.py) | RSI-style iterative adaptation CLI and reporting |
| [`rlt_rsi/rsi.py`](rlt_rsi/rsi.py) | RSI loop (`run_rsi_numpy`, `run_rsi_torch`) |
| [`rlt_rsi/data.py`](rlt_rsi/data.py) | Example-disjoint split generation and manifests |
| [`rlt_rsi/model.py`](rlt_rsi/model.py) | NumPy model and configuration |
| [`rlt_rsi/diagnostics.py`](rlt_rsi/diagnostics.py) | Overfit diagnostic routines for optimization checks |
| [`tests/`](tests/) | Data, model, generalization, RSI, and torch tests |
| [`results/`](results/) | Committed experiment artifacts |
| [`notebooks/rlt_rsi_colab.ipynb`](notebooks/rlt_rsi_colab.ipynb) | CPU smoke + optional GPU walkthrough |
| [`DESIGN.md`](DESIGN.md) | Architectural controls, research questions, and hypotheses |

## License

MIT — see [LICENSE](LICENSE).
