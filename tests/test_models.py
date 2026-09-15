import numpy as np

from rlt_rsi.data import make_split
from rlt_rsi.model import NumpyConfig, NumpyLoopedTransformerClassifier, NumpyTransformerClassifier
from rlt_rsi.train import fit_numpy


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
