"""Small-sample statistics helpers (no SciPy dependency)."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

# Two-sided 95% Student-t critical values, indexed by degrees of freedom.
_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
    10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
    18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_critical_95(df: int) -> float:
    """Two-sided 95% t critical value; normal approximation beyond df=30."""
    if df < 1:
        raise ValueError("degrees of freedom must be >= 1")
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    for limit, value in ((40, 2.021), (60, 2.000), (120, 1.980)):
        if df <= limit:
            return value
    return 1.960


def mean_ci95(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """95% t-interval for the mean of ``values``.

    Returns ``(None, None)`` for fewer than two values, where no interval exists.
    """
    n = len(values)
    if n < 2:
        return (None, None)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    half = t_critical_95(n - 1) * math.sqrt(var / n)
    return (mean - half, mean + half)
