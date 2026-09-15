# Sequence Parity: Conventional vs Shared-Weight Looped Transformer Report

Backend: **numpy** *(Plumbing / Smoke test only: frozen transformer + linear readout)*
Evaluated random seeds (N=3): `7, 42, 123`
Python Environment: `3.14.5` | NumPy: `2.4.6`

> [!IMPORTANT]
> **Scientific Transparency & Labeling:**
> NumPy fallback freezes transformer weights and trains only the binary readout. This run is an illustrative smoke and plumbing test to verify pipelines and interfaces. It is **not evidence that recurrence improves reasoning**.

## Summary Results Table

| model | loop_count | architecture_parameters | estimated_block_calls | train_acc (mean ± std) | dev_acc (mean ± std) | heldout_acc (mean ± std) | decision vs baseline |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 1 | 4,656 | 1 | 0.553 ± 0.011 | 0.471 ± 0.051 | 0.508 ± 0.021 | baseline (reference) |
| looped | 1 | 4,656 | 1 | 0.553 ± 0.011 | 0.471 ± 0.051 | 0.508 ± 0.021 | flat |
| looped | 2 | 4,656 | 2 | 0.529 ± 0.020 | 0.458 ± 0.071 | 0.482 ± 0.027 | flat |
| looped | 4 | 4,656 | 4 | 0.482 ± 0.012 | 0.458 ± 0.025 | 0.479 ± 0.048 | flat |

## Evaluation of Core Research Questions

1. **Shared-Weight Recurrent Computation vs Conventional Single-Pass Transformer:**
   Under identical width (`d_model=24`), heads (`n_heads=4`), and initialization, shared-weight looped models yield **flat** or neutral performance relative to the single-pass baseline on the sequence parity task under frozen representation smoke testing.

2. **Compute Cost vs Accuracy Scaling Across Loop Counts (1, 2, 4):**
   As loop count scales (1 -> 2 -> 4), the architecture parameter count remains strictly constant at **4,656 parameters** due to weight sharing. However, compute cost (`estimated_block_calls`) increases linearly (1 -> 2 -> 4). Under this frozen smoke setup, higher loop counts do not yield monotonic gains.

3. **Length Generalization (Train/Dev 4–8 vs Held-Out 12–16):**
   The held-out split uses lengths 12–16 strictly outside the train/dev range (4–8). Recurrence does not exhibit out-of-distribution length generalization on this task; held-out accuracy remains near the random chance threshold (~50%).

4. **Controlled Architectural Parameter Baseline:**
   Both conventional single-pass and looped transformers share the exact same parameter footprint (4,656 architecture parameters). No additional parameters are introduced by increasing the iteration loops.

## Hardware & Environment Notice

> [!NOTE]
> **PyTorch Status:** PyTorch is not installed in the local Python 3.14.5 environment. All local runs use the verified deterministic NumPy smoke pipeline.

To execute the fully trainable, gradient-updated experiment on NVIDIA L4 / Colab Pro GPU:

```bash
# Colab Pro / NVIDIA L4 GPU execution
!python -m pip install -e ".[torch]"
!python -m rlt_rsi.train --backend torch --seeds 7, 42, 123 --epochs 80 --output results/torch-l4.json
```

See [DESIGN.md](DESIGN.md) for preregistered hypotheses, leakage controls, and failure mode documentation.
