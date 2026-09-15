# Design, Preregistered Questions, and Methodological Controls

## Research Questions

On a small algorithmic task (binary sequence parity), we examine:
1. **Shared-Weight Recurrence vs Conventional Transformer:** Does repeatedly executing a shared transformer block alter accuracy or length generalization relative to a matched conventional one-pass transformer?
2. **Loop Scaling & Compute Trade-Off (1, 2, 4):** As iteration loops increase, how do classification accuracy and computational complexity (`estimated_block_calls`) scale when architectural parameter count is strictly constrained?
3. **Out-of-Distribution Length Generalization:** Does recurrent computation facilitate generalization from short sequences (train/dev: 4–8) to longer sequences (held-out: 12–16)?
4. **Controlled Architectural Parameters:** Isolating the effect of compute iterations from model capacity by strictly matching architecture parameter counts (4,656 parameters across all models).

*Note:* This experiment is a focused probe of iterative computation on parity. It is **not** a general reasoning benchmark.

## Falsifiable Hypotheses & Preregistered Decision Rules

* **H1 (Empirical Accuracy Gain):** At identical model width and training budget, the looped model (2 or 4 loops) exceeds the single-pass baseline on held-out parity accuracy by at least 5 percentage points ($\Delta \ge +0.05$). Evaluated as **`improvement`**.
* **H2 (Length Generalization Preservation):** The accuracy degradation from train/dev lengths (4–8) to held-out lengths (12–16) is at least 5 percentage points smaller in the looped model than in the baseline.
* **H0 (Null Outcome):** Neither threshold is met ($|\Delta| < 0.05$ or $\Delta \le -0.05$). Flat or worse performance is a valid and anticipated scientific result. Evaluated as **`flat`** or **`regression`**.

All decisions are computed across at least 3 independent seeds (`7, 42, 123`), reporting mean and sample standard deviation (`ddof=1`).

## Task Specification and Leakage Controls

- **Task:** Binary sequence parity ($y = \bigoplus_{i=0}^{L-1} x_i$).
- **Distributions:**
  - `train`: Lengths uniformly sampled from $[4, 8]$ ($N=256$).
  - `dev`: Lengths uniformly sampled from $[4, 8]$ ($N=128$).
  - `heldout`: Lengths uniformly sampled from $[12, 16]$ ($N=128$).
- **Leakage Prevention:**
  - Length distributions are strictly disjoint: $\text{lengths}(\text{train/dev}) \cap \text{lengths}(\text{heldout}) = \emptyset$.
  - Split generation uses independent PRNG seeds: `seed + 101` (train), `seed + 202` (dev), `seed + 303` (heldout).
  - Attention padding masks prevent position leakage; classification reads out exclusively at the final valid prefix position ($L - 1$).

## Architectural Controls and Compute Proxy

| Variant | Model Class | Loop Count | Architecture Parameters | Estimated Block Calls | Parameter Sharing |
|---|---|---:|---:|---:|:---:|
| Baseline | `NumpyTransformerClassifier` | 1 | 4,656 | 1 | False |
| Looped-1 | `NumpyLoopedTransformerClassifier` | 1 | 4,656 | 1 | True (1 pass) |
| Looped-2 | `NumpyLoopedTransformerClassifier` | 2 | 4,656 | 2 | True (2 passes) |
| Looped-4 | `NumpyLoopedTransformerClassifier` | 4 | 4,656 | 4 | True (4 passes) |

- **Width:** `d_model = 24`, `n_heads = 4`, `d_ff = 48`.
- **Positional Encoding:** Sinusoidal encodings added prior to the first block.
- **Compute Proxy:** `estimated_block_calls` tracks sequential FLOPs/depth without increasing parameter footprint.

## Backend Implementations & Hardware Environment

- **NumPy Fallback (Smoke / Plumbing Mode):**
  - Used in local macOS development. The transformer parameters are deterministically seeded and frozen; only the binary readout vector and bias are trained.
  - **Critical Labeling:** Strictly designated as a smoke test for verifying determinism, multi-seed aggregation, and interfaces. It is **not** evidence of reasoning capability.
- **PyTorch Backend (Gradient-Trained Mode):**
  - Trains all model parameters end-to-end with AdamW.
  - **Local Hardware Blocker:** PyTorch is not installed in the local Python 3.14 environment. Attempting to use `--backend torch` locally raises a clear `RuntimeError`.
  - **Full GPU Run Specification (Colab Pro / NVIDIA L4):**
    ```bash
    pip install -e ".[torch]"
    python -m rlt_rsi.train --backend torch --seeds 7,42,123 --epochs 80 --output results/torch-l4.json
    ```

## Scientific Limitations and Failure Modes

1. **Parity Complexity:** Parity is non-linear and sensitive to representation geometry. Random frozen features cannot solve arbitrary XOR without end-to-end training; ~50% accuracy is expected.
2. **Compute vs Parameter Asymmetry:** Increasing loop count increases runtime latency and FLOPs while holding parameters constant. Any observed performance changes must be evaluated against this compute cost.
3. **No General Reasoning Claims:** Sequence parity is an isolated algorithmic check. Findings cannot be generalized to language modeling or multi-step reasoning agents.
