# Changelog

## 0.2.0

* Added `pythonpath = ["."]` under `[tool.pytest.ini_options]` in `pyproject.toml` for seamless `pytest -q` execution.
* Added multi-seed execution support (`--seeds` flag, defaulting to seeds 7, 42, 123) with mean ± std aggregation across accuracy, cross-entropy loss, accuracy-by-length, and runtime.
* Added loop scaling ablation evaluating `baseline (loops=1)`, `looped (loops=1)`, `looped (loops=2)`, and `looped (loops=4)`.
* Verified architectural parameter invariance (strictly 4,656 parameters across all models) while tracking linear compute scaling via `estimated_block_calls` (1, 1, 2, 4).
* Implemented preregistered hypothesis decision rules (`baseline (reference)`, `improvement`, `flat`, `regression`) based on held-out length performance.
* Strengthened test suite (`tests/test_data.py`, `tests/test_models.py`) with checks for length disjointness, absence of data leakage, bitwise determinism, parameter invariance, and PyTorch environment blocker errors.
* Updated `results/smoke.json` and `results/smoke.md` with multi-seed benchmark tables, length-wise metrics, and transparency notices.
* Updated CI workflow (`.github/workflows/ci.yml`) to execute `pytest -q` and the multi-seed smoke experiment.
* Documented Colab Pro / NVIDIA L4 execution instructions and documented the Python 3.14 PyTorch blocker.

## 0.1.0

* Added leakage-safe parity data generation with train/dev/held-out lengths.
* Added conventional and shared-weight looped transformer implementations.
* Added deterministic NumPy smoke backend and full optional PyTorch backend.
* Added machine-readable metrics, report generation, tests, and CI.
