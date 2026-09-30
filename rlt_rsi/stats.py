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


# Tabulated values above df=30. Each value applies from its own df upward until the next entry,
# so between table rows the critical value is rounded up (a slightly wider, conservative interval)
# rather than down.
_T_CRIT_95_COARSE = ((120, 1.980), (60, 2.000), (40, 2.021))


def t_critical_95(df: int) -> float:
    """Two-sided 95% t critical value.

    Exact (3 decimals) for df <= 30. For larger df the value of the nearest tabulated df at or
    below ``df`` is used (never narrower than the true value), and 1.960 beyond df=1000.
    """
    if df < 1:
        raise ValueError("degrees of freedom must be >= 1")
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    if df >= 1000:
        return 1.960
    for limit, value in _T_CRIT_95_COARSE:
        if df >= limit:
            return value
    return _T_CRIT_95[30]


def mean_ci95(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """95% t-interval for the mean of ``values``.

    Returns ``(None, None)`` for fewer than two values, where no interval exists. With zero
    sample variance (all values equal) the interval collapses to the mean; that reflects the
    observed values only and is not evidence of zero uncertainty.
    """
    n = len(values)
    if n < 2:
        return (None, None)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    half = t_critical_95(n - 1) * math.sqrt(var / n)
    return (mean - half, mean + half)
