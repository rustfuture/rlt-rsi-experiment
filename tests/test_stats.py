import math

import pytest

from rlt_rsi.stats import mean_ci95, t_critical_95


def test_t_critical_known_values():
    assert t_critical_95(2) == pytest.approx(4.303)
    assert t_critical_95(9) == pytest.approx(2.262)
    assert t_critical_95(1000) == pytest.approx(1.960)


def test_t_critical_is_monotone_non_increasing():
    values = [t_critical_95(df) for df in range(1, 200)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_t_critical_rejects_zero_df():
    with pytest.raises(ValueError):
        t_critical_95(0)


def test_mean_ci95_matches_hand_computation():
    low, high = mean_ci95([0.0, 0.1, 0.2])
    half = 4.303 * math.sqrt(0.01 / 3)
    assert low == pytest.approx(0.1 - half)
    assert high == pytest.approx(0.1 + half)


def test_mean_ci95_undefined_for_single_value():
    assert mean_ci95([0.3]) == (None, None)
