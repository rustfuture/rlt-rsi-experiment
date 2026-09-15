import json
import platform
import sys

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


def test_numpy_parameter_accounting_reports_total_trainable_and_frozen():
    cfg = NumpyConfig(d_model=24, n_heads=4, d_ff=48, seed=7)
    model = NumpyTransformerClassifier(cfg)
    counts = model.parameter_counts()

    # The transformer + embedding subtotal is 4,656; it is NOT the total because the
    # trainable readout (d_model) and bias (1) are separate parameters.
    assert counts["frozen"] == 4656
    assert counts["trainable"] == cfg.d_model + 1 == 25
    assert counts["total"] == 4681
    assert counts["total"] == counts["trainable"] + counts["frozen"]

    # architecture_parameters() is the documented frozen-backbone subtotal only.
    assert model.architecture_parameters() == counts["frozen"]
    assert model.architecture_parameters() != counts["total"]


def test_parameter_counts_constant_across_loop_counts_and_block_calls_scale():
    """Architecture parameter counts stay constant across loops; block applications scale 1/1/2/4."""
    cfg = NumpyConfig(d_model=24, n_heads=4, d_ff=48, seed=7)
    models = [
        NumpyTransformerClassifier(cfg),
        NumpyLoopedTransformerClassifier(cfg, 1),
        NumpyLoopedTransformerClassifier(cfg, 2),
        NumpyLoopedTransformerClassifier(cfg, 4),
    ]

    first = models[0].parameter_counts()
    for model in models:
        assert model.parameter_counts() == first
    assert first == {"total": 4681, "trainable": 25, "frozen": 4656}

    assert [m.estimated_block_calls for m in models] == [1, 1, 2, 4]
    # estimated_block_calls is a structural counter, not FLOPs or latency.
    block_call_doc = NumpyTransformerClassifier.estimated_block_calls.__doc__ or ""
    assert "structural counter" in block_call_doc
    assert "FLOP" in block_call_doc and "latency" in block_call_doc

    with pytest.raises(ValueError, match="loop_count must be >= 1"):
        NumpyLoopedTransformerClassifier(cfg, 0)


def test_baseline_and_looped_start_from_identical_shared_parameters_for_a_seed():
    cfg = NumpyConfig(d_model=24, n_heads=4, d_ff=48, seed=42)
    baseline = NumpyTransformerClassifier(cfg)
    for loops in (1, 2, 4):
        looped = NumpyLoopedTransformerClassifier(cfg, loops)
        np.testing.assert_array_equal(baseline.embedding, looped.embedding)
        for name in ("wq", "wk", "wv", "wo", "w1", "w2"):
            np.testing.assert_array_equal(getattr(baseline.block, name), getattr(looped.block, name))
        np.testing.assert_array_equal(baseline.readout, looped.readout)
        assert baseline.bias == looped.bias


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


def test_torch_missing_error_is_dynamic_and_has_no_stale_version_claims(monkeypatch):
    """The import error must name the *actual* python version, not a hardcoded environment."""
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError) as exc_info:
        _torch_components()

    msg = str(exc_info.value)
    assert "PyTorch is not importable" in msg
    assert platform.python_version() in msg
    assert "3.14" not in msg.replace(platform.python_version(), "")
    assert "colab" not in msg.lower()
    assert "nvidia" not in msg.lower()


def test_multi_seed_cli_runner(tmp_path):
    """CLI runner writes a payload and a report derived from the payload."""
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

    counts = data["results"][0]["parameter_counts"]
    assert counts["total"] == counts["trainable"] + counts["frozen"]

    overlap = data["data"]["example_overlap_by_seed"]
    assert overlap and all(v == 0 for per_seed in overlap.values() for v in per_seed.values())

    md_content = out_md.read_text(encoding="utf-8")
    assert "baseline (reference)" in md_content
    assert f"{counts['total']:,}" in md_content
    assert "| H1 |" in md_content
    assert "evaluated" in md_content
    assert "does **not** establish preregistration" in md_content
    assert "preregistered" not in md_content.lower()
