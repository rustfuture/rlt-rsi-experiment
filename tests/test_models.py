import json
import pytest
import numpy as np

from rlt_rsi.data import make_split
from rlt_rsi.model import (
    NumpyConfig,
    NumpyLoopedTransformerClassifier,
    NumpyTransformerClassifier,
)
from rlt_rsi.train import (
    _torch_components,
    evaluate_numpy,
    fit_numpy,
    main,
    run_numpy,
)


def test_shapes_and_loop_count():
    split = make_split(n=8, min_length=4, max_length=8, seed=4)
    cfg = NumpyConfig(d_model=12, n_heads=3, d_ff=16, seed=5)
    baseline = NumpyTransformerClassifier(cfg)
    looped = NumpyLoopedTransformerClassifier(cfg, 3)
    assert baseline.features(split.tokens, split.lengths).shape == (8, 12)
    assert looped.features(split.tokens, split.lengths).shape == (8, 12)
    assert looped.block is not None
    assert looped.loop_count == 3


def test_readout_smoke_is_finite():
    split = make_split(n=32, min_length=4, max_length=8, seed=6)
    model = NumpyTransformerClassifier(NumpyConfig(seed=2))
    result = fit_numpy(model, split, epochs=3, learning_rate=0.05, l2=1e-4)
    assert np.isfinite(result["final_train_bce"])


def test_shared_weights_architecture_parameters_constant_and_block_calls_scale():
    """Verify that architecture parameters remain strictly identical across loop counts (1, 2, 4)
    while estimated_block_calls increases linearly with loop count."""
    cfg = NumpyConfig(d_model=24, n_heads=4, d_ff=48, seed=7)
    baseline = NumpyTransformerClassifier(cfg)
    looped_1 = NumpyLoopedTransformerClassifier(cfg, 1)
    looped_2 = NumpyLoopedTransformerClassifier(cfg, 2)
    looped_4 = NumpyLoopedTransformerClassifier(cfg, 4)

    # All architectures must share the exact same parameter count
    p_base = baseline.architecture_parameters()
    p_l1 = looped_1.architecture_parameters()
    p_l2 = looped_2.architecture_parameters()
    p_l4 = looped_4.architecture_parameters()

    assert p_base == 4656
    assert p_l1 == p_base
    assert p_l2 == p_base
    assert p_l4 == p_base

    # Block calls scale linearly with loop count
    assert baseline.estimated_block_calls == 1
    assert looped_1.estimated_block_calls == 1
    assert looped_2.estimated_block_calls == 2
    assert looped_4.estimated_block_calls == 4

    # Invalid loop count raises ValueError
    with pytest.raises(ValueError, match="loop_count must be >= 1"):
        NumpyLoopedTransformerClassifier(cfg, 0)


def test_determinism_and_reproducibility():
    """Verify that two independent runs with identical random seed produce bitwise identical results."""
    cfg1 = NumpyConfig(d_model=24, n_heads=4, d_ff=48, seed=42)
    cfg2 = NumpyConfig(d_model=24, n_heads=4, d_ff=48, seed=42)

    model1 = NumpyLoopedTransformerClassifier(cfg1, 2)
    model2 = NumpyLoopedTransformerClassifier(cfg2, 2)

    split1 = make_split(n=64, min_length=4, max_length=8, seed=123)
    split2 = make_split(n=64, min_length=4, max_length=8, seed=123)

    feat1 = model1.features(split1.tokens, split1.lengths)
    feat2 = model2.features(split2.tokens, split2.lengths)
    np.testing.assert_array_equal(feat1, feat2)

    fit1 = fit_numpy(model1, split1, epochs=10, learning_rate=0.05, l2=1e-4)
    fit2 = fit_numpy(model2, split2, epochs=10, learning_rate=0.05, l2=1e-4)
    assert fit1["final_train_bce"] == fit2["final_train_bce"]
    np.testing.assert_array_equal(model1.readout, model2.readout)
    assert model1.bias == model2.bias

    ev1 = evaluate_numpy(model1, split1)
    ev2 = evaluate_numpy(model2, split2)
    assert ev1["accuracy"] == ev2["accuracy"]
    assert ev1["bce"] == ev2["bce"]


def test_torch_backend_unavailable_raises_runtime_error():
    """Verify that attempting to invoke PyTorch backend when torch is absent raises a descriptive RuntimeError."""
    with pytest.raises(RuntimeError) as exc_info:
        _torch_components()

    msg = str(exc_info.value)
    assert "PyTorch backend requested" in msg
    assert "PyTorch is not installed" in msg
    assert "Python 3.14" in msg
    assert "Colab Pro / NVIDIA L4" in msg


def test_multi_seed_cli_runner(tmp_path):
    """Test CLI runner with multi-seed flag end-to-end on temporary output paths."""
    out_json = tmp_path / "test_run.json"
    out_md = tmp_path / "test_run.md"

    code = main([
        "--backend", "numpy",
        "--seeds", "7,42",
        "--train-size", "32",
        "--dev-size", "16",
        "--heldout-size", "16",
        "--epochs", "2",
        "--output", str(out_json),
        "--report", str(out_md),
    ])

    assert code == 0
    assert out_json.exists()
    assert out_md.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["backend"] == "numpy"
    assert data["seeds"] == [7, 42]
    assert len(data["results"]) == 4  # baseline-1, looped-1, looped-2, looped-4

    md_content = out_md.read_text(encoding="utf-8")
    assert "baseline (reference)" in md_content
    assert "4,656" in md_content
