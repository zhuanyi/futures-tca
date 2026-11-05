"""Statistical helpers.

Confidence intervals, decile binning, R² and variance decomposition for
chart (e). Used by report.py and tests.
"""

from __future__ import annotations

import numpy as np


def ci_mean(
    x: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Compute mean and (1-alpha) confidence interval.

    Returns (mean, lower, upper).
    """
    n = len(x)
    if n == 0:
        return (0.0, 0.0, 0.0)
    from scipy import stats

    m = float(np.mean(x))
    se = float(stats.sem(x))
    if se == 0 or n < 2:
        return (m, m, m)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return (m, m - t_crit * se, m + t_crit * se)


def ci_median(
    x: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int = 20261101,
) -> tuple[float, float, float]:
    """Bootstrap CI for the median.

    Returns (median, lower, upper).
    """
    if len(x) == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    med = float(np.median(x))
    n = len(x)
    boot_medians = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(x, size=n, replace=True)
        boot_medians[i] = np.median(sample)
    lower = float(np.percentile(boot_medians, 100 * alpha / 2))
    upper = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
    return (med, lower, upper)


def decile_bins(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Assign each value to a decile bin (0 to n_bins-1).

    Uses quantile-based binning so each bin has roughly equal count.
    """
    if len(x) == 0:
        return np.empty(0, dtype=int)
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(x, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    # Remove duplicate edges
    edges = np.unique(edges)
    return np.digitize(x, edges[1:-1])


def r_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² (coefficient of determination).

    R² = 1 - SS_res / SS_tot
    """
    if len(y) == 0:
        return 0.0
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def variance_decomposition(
    y: np.ndarray,
    X: np.ndarray,
) -> dict[str, float]:
    """Decompose variance of y into contributions from each column of X.

    Uses OLS regression: y = X @ beta + epsilon.
    Returns dict with the overall R² and, per column ``i``, the univariate R² of a
    regression of y on that column alone (key ``r2_x{i}``; not a marginal
    contribution given the other columns).
    """
    if len(y) == 0 or X.shape[0] == 0:
        return {"overall_r2": 0.0}

    n, k = X.shape

    # Add constant
    X_const = np.column_stack([np.ones(n), X])

    # OLS
    try:
        beta, _, _, _ = np.linalg.lstsq(X_const, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"overall_r2": 0.0}

    y_pred = X_const @ beta
    overall_r2 = r_squared(y, y_pred)

    result = {"overall_r2": overall_r2}

    # Univariate R² for each predictor
    for i in range(k):
        X_marginal = np.column_stack([np.ones(n), X[:, i]])
        try:
            beta_m, _, _, _ = np.linalg.lstsq(X_marginal, y, rcond=None)
        except np.linalg.LinAlgError:
            result[f"r2_x{i}"] = 0.0
            continue
        y_pred_m = X_marginal @ beta_m
        result[f"r2_x{i}"] = r_squared(y, y_pred_m)

    return result
