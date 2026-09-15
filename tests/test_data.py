import json
import pytest
import numpy as np

from rlt_rsi.data import (
    DatasetSplit,
    count_overlap,
    example_keys,
    make_split,
    make_splits,
    select_rows,
    split_manifest,
)


def test_generation_is_deterministic_and_parity_is_correct():
    a = make_split(n=20, min_length=3, max_length=7, seed=11)
    b = make_split(n=20, min_length=3, max_length=7, seed=11)
    assert a.fingerprint() == b.fingerprint()
    expected = np.array([np.bitwise_xor.reduce(row[:length]) for row, length in zip(a.tokens, a.lengths)])
    np.testing.assert_array_equal(a.labels, expected)


def test_length_ranges_are_respected():
    split = make_split(n=100, min_length=12, max_length=16, seed=3)
    assert split.tokens.shape == (100, 16)
    assert split.lengths.min() >= 12
    assert split.lengths.max() <= 16


def test_heldout_lengths_are_strictly_disjoint_from_train_and_dev():
    seed = 42
    train = make_split(n=256, min_length=4, max_length=8, seed=seed + 101)
    dev = make_split(n=128, min_length=4, max_length=8, seed=seed + 202)
    heldout = make_split(n=128, min_length=12, max_length=16, seed=seed + 303)

    train_lengths = set(train.lengths.tolist())
    dev_lengths = set(dev.lengths.tolist())
    heldout_lengths = set(heldout.lengths.tolist())

    # Check boundaries
    assert train_lengths.issubset(set(range(4, 9)))
    assert dev_lengths.issubset(set(range(4, 9)))
    assert heldout_lengths.issubset(set(range(12, 17)))

    # Strict disjointness - zero overlap between training distribution and held-out distribution
    assert train_lengths.isdisjoint(heldout_lengths), "Data leakage: train and held-out lengths overlap!"
    assert dev_lengths.isdisjoint(heldout_lengths), "Data leakage: dev and held-out lengths overlap!"


def test_no_data_leakage_and_splits_are_distinct():
    seed = 7
    train = make_split(n=128, min_length=4, max_length=8, seed=seed + 101)
    dev = make_split(n=128, min_length=4, max_length=8, seed=seed + 202)
    heldout = make_split(n=128, min_length=12, max_length=16, seed=seed + 303)

    # Fingerprints must be distinct
    fps = {train.fingerprint(), dev.fingerprint(), heldout.fingerprint()}
    assert len(fps) == 3, "Splits must have unique, independent fingerprints"

    # Verify manifest JSON serialization
    manifest_str = split_manifest({"train": train, "dev": dev, "heldout": heldout})
    parsed = json.loads(manifest_str)
    assert set(parsed.keys()) == {"train", "dev", "heldout"}
    assert parsed["train"]["n"] == 128
    assert parsed["heldout"]["min_length"] == 12


def test_invalid_parameters_raise_value_error():
    with pytest.raises(ValueError, match="n > 0"):
        make_split(n=0, min_length=4, max_length=8, seed=1)

    with pytest.raises(ValueError, match="min_length <= max_length"):
        make_split(n=10, min_length=10, max_length=5, seed=1)

    with pytest.raises(ValueError, match="0 < min_length <= max_length"):
        make_split(n=10, min_length=0, max_length=5, seed=1)


def test_example_keys_are_padding_independent_and_unique_by_construction():
    split = make_split(n=64, min_length=4, max_length=8, seed=5)
    keys = example_keys(split)
    assert len(keys) == 64, "rejection sampling must produce 64 distinct examples"
    subset = select_rows(split, np.arange(8), seed=99)
    assert example_keys(subset) <= keys


def test_count_overlap_counts_exact_shared_examples():
    a = make_split(n=64, min_length=4, max_length=8, seed=1)
    b = select_rows(a, np.arange(0, 64, 2), seed=2)
    assert count_overlap(a, b) == 32
    assert count_overlap(a, a) == 64


def test_experiment_splits_are_example_disjoint_for_every_documented_seed():
    """train/dev/held-out must share zero exact examples (same length + same tokens)."""
    for seed in (7, 42, 123):
        splits = make_splits(train_n=256, dev_n=128, heldout_n=128, seed=seed)
        assert count_overlap(splits["train"], splits["dev"]) == 0, f"seed {seed}: train/dev overlap"
        assert count_overlap(splits["train"], splits["heldout"]) == 0, f"seed {seed}: train/held-out overlap"
        assert count_overlap(splits["dev"], splits["heldout"]) == 0, f"seed {seed}: dev/held-out overlap"
        assert len(example_keys(splits["train"])) == 256
        assert len(example_keys(splits["dev"])) == 128


def test_joint_pool_keeps_short_lengths_in_both_train_and_dev():
    """Drawing train first would exhaust the 2**4=16 length-4 examples; the joint pool avoids that."""
    splits = make_splits(train_n=256, dev_n=128, heldout_n=128, seed=7)
    train_lengths = set(splits["train"].lengths.tolist())
    dev_lengths = set(splits["dev"].lengths.tolist())
    assert train_lengths == {4, 5, 6, 7, 8}
    assert dev_lengths == {4, 5, 6, 7, 8}


def test_split_generation_fails_loudly_when_example_space_is_exhausted():
    with pytest.raises(ValueError, match="example space is too small"):
        make_split(n=1000, min_length=2, max_length=2, seed=1)

    with pytest.raises(ValueError, match="example space is too small"):
        make_splits(train_n=400, dev_n=200, heldout_n=8, seed=1)
