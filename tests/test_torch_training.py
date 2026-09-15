"""Torch-only training tests.

The whole module skips cleanly when torch is unavailable (e.g. the CI job on
python 3.11), so the NumPy-only environment stays green.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rlt_rsi.data import make_split, make_splits
from rlt_rsi.diagnostics import overfit_diagnostic
from rlt_rsi.model import NumpyConfig
from rlt_rsi.train import (
    _reload_and_reevaluate,
    _torch_batch,
    _torch_parameter_counts,
    _torch_state_fingerprint,
    build_torch_model,
    resolve_device,
    run_torch,
)


def test_single_optimizer_step_is_finite_and_changes_every_parameter():
    """Loss finite, all gradients finite, and every optimized tensor actually changes."""
    split = make_split(n=32, min_length=4, max_length=8, seed=6)
    cfg = NumpyConfig(seed=2)
    model = build_torch_model(cfg, "looped", 2, seed=7)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    tx, tl, ty = _torch_batch(torch, split, torch.device("cpu"))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    optim.zero_grad(set_to_none=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tx, tl), ty)
    assert bool(torch.isfinite(loss))
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert bool(torch.isfinite(param.grad).all()), f"{name} has non-finite gradients"

    optim.step()
    unchanged = [name for name, param in model.named_parameters() if torch.equal(before[name], param.detach())]
    assert unchanged == [], f"parameters unchanged after the optimizer step: {unchanged}"


def test_parameter_counts_are_constant_across_loop_counts_and_total_is_the_sum():
    cfg = NumpyConfig(d_model=24, n_heads=4, d_ff=48, seed=7)
    counts_by_loops = {}
    for loops in (1, 2, 4):
        counts = _torch_parameter_counts(build_torch_model(cfg, "looped", loops, seed=7))
        assert counts["total"] == counts["trainable"] + counts["frozen"]
        counts_by_loops[loops] = counts
    baseline_counts = _torch_parameter_counts(build_torch_model(cfg, "baseline", 1, seed=7))

    assert counts_by_loops[1] == counts_by_loops[2] == counts_by_loops[4] == baseline_counts
    # All torch parameters are trained end-to-end, and the torch architecture is
    # deliberately not parameter-identical to the NumPy one.
    assert counts_by_loops[1]["frozen"] == 0
    assert counts_by_loops[1]["trainable"] == counts_by_loops[1]["total"] == 4945


def test_baseline_and_looped_share_identical_initialization_for_a_seed():
    cfg = NumpyConfig(d_model=24, n_heads=4, d_ff=48, seed=42)
    reference = _torch_state_fingerprint(torch, build_torch_model(cfg, "baseline", 1, seed=42))
    for loops in (1, 2, 4):
        candidate = _torch_state_fingerprint(torch, build_torch_model(cfg, "looped", loops, seed=42))
        assert candidate == reference, f"looped-{loops} did not start from the baseline initialization"


def test_resolve_device_never_silently_falls_back():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "mps", "cuda"}
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="cuda"):
            resolve_device("cuda")
    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="mps"):
            resolve_device("mps")
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("tpu")


def test_run_torch_smoke_records_real_setup_and_evaluates_all_splits():
    splits = make_splits(train_n=32, dev_n=16, heldout_n=16, seed=7)
    cfg = NumpyConfig(seed=7)
    result = run_torch(
        "looped", 2, splits, cfg, epochs=2, learning_rate=3e-3, weight_decay=1e-4,
        seed=7, device=resolve_device("cpu"), device_requested="cpu",
    )

    assert result["device"] == "cpu"
    assert result["dtype"] == "float32"
    assert result["optimizer"] == "AdamW"
    assert result["weight_decay"] == 1e-4
    assert result["torch_version"] == torch.__version__
    assert result["sequential_block_applications"] == 2
    assert result["estimated_block_calls"] == 2
    counts = result["parameter_counts"]
    assert counts["total"] == counts["trainable"] + counts["frozen"]
    assert np.isfinite(result["final_train_bce"])
    for split_name in ("train", "dev", "heldout"):
        assert np.isfinite(result[split_name]["bce"])
        assert 0.0 <= result[split_name]["accuracy"] <= 1.0


def test_checkpoint_reload_and_reevaluate_matches_recorded_metrics(tmp_path):
    splits = make_splits(train_n=32, dev_n=16, heldout_n=16, seed=11)
    cfg = NumpyConfig(seed=11)
    checkpoint = tmp_path / "seed11-looped-loop2.pt"
    run = run_torch(
        "looped", 2, splits, cfg, epochs=2, learning_rate=3e-3, weight_decay=1e-4,
        seed=11, device=resolve_device("cpu"), device_requested="cpu", checkpoint_path=checkpoint,
    )
    assert checkpoint.exists()

    record = {
        "seed": 11,
        "model": "looped",
        "loop_count": 2,
        "recorded_metrics": {name: run[name] for name in ("train", "dev", "heldout")},
    }
    check = _reload_and_reevaluate(checkpoint, record, splits, cfg, resolve_device("cpu"))
    assert check["matches_recorded"] is True
    assert check["max_abs_accuracy_diff"] <= check["tolerance"]


def test_overfit_diagnostic_on_a_separate_dataset_reduces_loss():
    diag = overfit_diagnostic(seed=3, epochs=300, device="cpu", learning_rate=1e-3)

    assert diag["not_heldout"] is True
    assert diag["dataset"]["n"] == 32
    assert all(np.isfinite(diag["loss_trajectory"]))
    assert all(np.isfinite(diag["accuracy_trajectory"]))
    assert diag["final_loss"] < diag["initial_loss"]
    assert diag["best_train_accuracy"] >= diag["initial_train_accuracy"]
    assert diag["seconds"] < 60.0
