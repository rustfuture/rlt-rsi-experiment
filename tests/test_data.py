import numpy as np

from rlt_rsi.data import make_split


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
