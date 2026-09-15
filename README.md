# rlt-rsi-experiment

Reproducible minimal experiment comparing a conventional transformer with a
shared-weight recurrent/looped transformer on binary sequence parity. The
experiment probes iterative computation, compute scaling, and length generalization;
it does not test or claim general reasoning ability.

## Research Questions

1. **Shared-weight recurrence vs conventional transformer:** Does applying a transformer block repeatedly with shared weights provide advantages over a single-pass transformer under identical initialization?
2. **Loop count scaling (1, 2, 4) & compute cost:** How do accuracy and computational complexity (`estimated_block_calls`) scale as iteration loops increase while keeping architectural parameters strictly constant?
3. **Length generalization:** Does recurrent processing enable generalization to longer sequence lengths (held-out lengths 12–16 vs train/dev lengths 4–8)?
4. **Controlled parameter comparison:** Controlled comparison where baseline and looped models share the exact same parameter count (4,656 architecture parameters).

## Quickstart & Reproduce the CPU Smoke Run

```bash
# Setup environment
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'

# Run the multi-seed smoke pipeline (N=3 seeds: 7, 42, 123)
python -m rlt_rsi.train --backend numpy --seeds 7,42,123 --output results/smoke.json

# Run test suite
pytest -q
```

The runner evaluates 4 configurations across 3 seeds (`7, 42, 123`):
- `baseline` (loops=1): Single-pass reference (4,656 params, 1 block call)
- `looped` (loops=1): Recurrent model single pass (4,656 params, 1 block call)
- `looped` (loops=2): Recurrent model 2 iterations (4,656 params, 2 block calls)
- `looped` (loops=4): Recurrent model 4 iterations (4,656 params, 4 block calls)

Results are automatically saved to `results/smoke.json` and summarized with mean ± std in `results/smoke.md`.

## PyTorch GPU Experiment (Colab Pro / NVIDIA L4)

> [!WARNING]
> **Environment Blocker:** PyTorch is not installed in the local Python 3.14 environment. Calling `--backend torch` locally raises an informative `RuntimeError`. The local run strictly uses the labeled NumPy fallback for pipeline verification.

To run the full trainable, gradient-updated experiment on an NVIDIA L4 GPU instance (e.g. Google Colab Pro):

```bash
# 1. Install dependencies with PyTorch
python -m pip install -e '.[torch]'

# 2. Run multi-seed experiment on GPU
python -m rlt_rsi.train --backend torch --seeds 7,42,123 --epochs 80 --output results/torch-l4.json
```

## Interpreting Results & Scientific Integrity

- **NumPy Fallback Labeling:** The NumPy fallback freezes the transformer feature weights and trains only the linear readout layer. It is strictly a **smoke and plumbing verification test**, not a substitute for the full trainable PyTorch experiment.
- **Hypothesis Decisioning:** The runner automatically compares looped models against baseline on held-out sequences (lengths 12–16) and tags each condition as `improvement` (>= +5%), `regression` (<= -5%), or `flat`.
- **Honest Scientific Reporting:** On frozen features, recurrence yields a **flat** outcome (~50% accuracy on held-out lengths). Do **not** claim recurrence improves reasoning. Flat or regressed results are fully valid scientific outcomes.

See [DESIGN.md](DESIGN.md) for preregistered hypotheses, leakage controls, and failure mode documentation.
