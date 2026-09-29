# RLT-RSI Experiment

This project trains and compares computer models that reuse layers to classify whether an on/off sequence contains an odd or even number of on values.

[![CI](https://github.com/rustfuture/rlt-rsi-experiment/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rustfuture/rlt-rsi-experiment/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Status:** Experimental research prototype (v0.3.0).

- Compares standard models with models that repeat the same layer 1, 2, or 4 times.
- Checks whether models trained on short sequences work on longer ones.
- Keeps training, validation, and test examples separate.
- Includes a bounded search that adjusts how many times a layer runs.
- Offers a quick NumPy check and full PyTorch training.

## Quick start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# Run test suite
pytest -q

# NumPy smoke pipeline (frozen features; fast CPU check)
python3 -m rlt_rsi.train --backend numpy --seeds 7,42,123 --output results/smoke.json

# Adaptation run (NumPy backend)
python3 -m rlt_rsi.train_rsi --backend numpy --seeds 7,42,123 --rsi-generations 2 --rsi-epochs-per-gen 2 --output results/rsi_smoke.json

# Optional: PyTorch end-to-end training (requires 'pip install -e .[torch]')
python3 -m rlt_rsi.train --backend torch --device auto --seeds 7,42,123 --epochs 80 \
    --artifacts-dir results/torch-run
```

To run in a browser without local setup, open [`notebooks/rlt_rsi_colab.ipynb`](notebooks/rlt_rsi_colab.ipynb) in Google Colab.

The default run compares a single-pass baseline with looped models. See [the reference notes](docs/reference.md#run-configurations) for configurations and extended commands.

## How it works

- The data code creates separate training and validation examples with lengths 4–8, then held-out examples with lengths 12–16.
- The model turns each input into a vector, processes all positions with one shared layer, then predicts whether the number of on values is odd or even.
- The baseline runs the block once; looped models repeat the same block with shared weights.
- Baseline and looped runs fit on training examples and evaluate on held-out examples; RSI additionally selects loop counts using validation data.
- RSI (recursive self-improvement style search) tests nearby loop counts and evaluates the final choice on held-out examples after selection. Details are in [the adaptation notes](docs/reference.md#iterative-adaptation).

## Tests

```bash
# PyTorch CPU job
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -c "import torch; print(torch.__version__)"
pytest -q

# NumPy test job
python -m pip install -e '.[dev]'
pytest -q
python -m rlt_rsi.train --backend numpy --seeds 7,42,123 --train-size 64 --dev-size 32 --heldout-size 32 --epochs 4 --output results/ci.json
python -m rlt_rsi.train_rsi --backend numpy --seeds 7,42,123 --train-size 64 --dev-size 32 --heldout-size 32 --epochs 4 --rsi-generations 2 --rsi-epochs-per-gen 2 --output results/rsi_ci.json
```

The NumPy CI job runs on Python 3.11; a separate job installs CPU PyTorch. Tests cover split isolation, model parameter counts, length generalization, reports, and training and adaptation on both backends.

## RSI-Style Iterative Adaptation

`rlt_rsi.train_rsi` changes only the loop schedule, selects candidates on validation data, and uses held-out data once after search. The [full adaptation flow and commands](docs/reference.md#iterative-adaptation) document this process.

## Scope and Limitations

- The task covers sequence parity only; it does not test language modeling or general reasoning.
- Adaptation is a bounded heuristic over loop counts from 1–8, selected on validation data. It is not general Recursive Self-Improvement.
- The ±0.05 held-out accuracy threshold across N=3 seeds is an operational rule, not a statistical significance test. Reports include paired 95% t-intervals; with N=3 they are very wide (t=4.30), so pass more seeds (for example `--seeds 1,2,3,4,5,6,7,8,9,10`) before reading anything into a label.
- The NumPy and PyTorch backends use different architectures and parameter counts; see the [backend comparison](docs/reference.md#backend-architecture-comparison) and its [model test](tests/test_models.py) and [PyTorch test](tests/test_torch_training.py).
- On repeated Apple Silicon MPS runs, looped-4 held-out accuracy moved from 0.5052 (`flat`) to 0.5208 (`improvement`) ([run details](results/torch-mps-2026-09-15/run.md)). Use `--device cpu` for bit-exact reproducibility.
- The hypotheses were specified before analysis, but are not described as preregistered because [`DESIGN.md`](DESIGN.md) and initial results were committed together (`161752a`).

Checkpoint `.pt` files are machine-specific and gitignored; only JSON manifests with SHA-256 hashes are tracked. More interpretation rules and artifact details are in [the reference notes](docs/reference.md).

## Documentation and Artifacts

| Path | Contents |
|---|---|
| [`docs/reference.md`](docs/reference.md) | Relocated backend tables, operational rules, artifact notes, and citations |
| [`DESIGN.md`](DESIGN.md) | Architectural controls, research questions ([`DESIGN.md#research-questions`](DESIGN.md#research-questions)), and hypotheses |

## Repository Map

| Path | Contents |
|---|---|
| [`rlt_rsi/train.py`](rlt_rsi/train.py) | Baseline and looped training, backends, evaluation |
| [`rlt_rsi/train_rsi.py`](rlt_rsi/train_rsi.py) | RSI-style iterative adaptation CLI and reporting |
| [`rlt_rsi/rsi.py`](rlt_rsi/rsi.py) | RSI adaptation loop (`run_rsi_numpy`, `run_rsi_torch`) |
| [`rlt_rsi/data.py`](rlt_rsi/data.py) | Example-disjoint split generation and manifests |
| [`rlt_rsi/model.py`](rlt_rsi/model.py) | NumPy model and configuration |
| [`rlt_rsi/diagnostics.py`](rlt_rsi/diagnostics.py) | Overfit diagnostic routines for optimization checks |
| [`tests/`](tests/) | Data, model, generalization, RSI, and torch tests |
| [`results/`](results/) | Committed experiment artifacts |
| [`notebooks/rlt_rsi_colab.ipynb`](notebooks/rlt_rsi_colab.ipynb) | CPU smoke and GPU walkthrough |
| [`DESIGN.md`](DESIGN.md) | Architectural controls and research questions |

## License

MIT — see [LICENSE](LICENSE).
