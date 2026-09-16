import pytest
import numpy as np

from rlt_rsi.data import make_splits
from rlt_rsi.model import NumpyConfig
from rlt_rsi.rsi import run_rsi_numpy

def test_rsi_leakage_and_lineage():
    # Generate splits
    splits = make_splits(
        train_n=16, dev_n=16, heldout_n=16, seed=42, train_lengths=(4, 8), heldout_lengths=(12, 16)
    )
    
    cfg = NumpyConfig(seed=42)
    # Run RSI
    res = run_rsi_numpy(
        splits, cfg, generations=2, epochs_per_gen=2, learning_rate=0.08, weight_decay=1e-4, seed=42
    )
    
    assert "lineage" in res
    assert len(res["lineage"]) == 2
    
    # Check that lineage only contains generation info
    for gen_info in res["lineage"]:
        assert "proposals" in gen_info
        assert "accepted_loop_count" in gen_info
        assert "dev_accuracy" in gen_info
        # It must NOT contain heldout
        assert "heldout_accuracy" not in gen_info
        
    assert res["model"] == "rsi_looped"
    
    assert "heldout" in res
    assert "train" in res
    assert "dev" in res

def test_rsi_deterministic():
    splits = make_splits(
        train_n=8, dev_n=8, heldout_n=8, seed=42, train_lengths=(4, 8), heldout_lengths=(12, 16)
    )
    cfg = NumpyConfig(seed=42)
    
    res1 = run_rsi_numpy(
        splits, cfg, generations=2, epochs_per_gen=2, learning_rate=0.08, weight_decay=1e-4, seed=42
    )
    res2 = run_rsi_numpy(
        splits, cfg, generations=2, epochs_per_gen=2, learning_rate=0.08, weight_decay=1e-4, seed=42
    )
    
    assert res1["heldout"]["accuracy"] == res2["heldout"]["accuracy"]
    assert res1["final_loop_count"] == res2["final_loop_count"]
    assert res1["lineage"] == res2["lineage"]
