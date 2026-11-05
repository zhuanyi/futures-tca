"""Tests for stats.py: CI helpers, decile binning, R², variance decomposition."""

from __future__ import annotations

import numpy as np

from futures_tca.stats import (
    ci_mean,
    ci_median,
    decile_bins,
    r_squared,
    variance_decomposition,
)


class TestCIMean:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m, lo, hi = ci_mean(x)
        assert m == 3.0
        assert lo < 3.0
        assert hi > 3.0

    def test_empty(self):
        m, lo, hi = ci_mean(np.array([]))
        assert m == 0.0

    def test_constant(self):
        x = np.array([5.0, 5.0, 5.0])
        m, lo, hi = ci_mean(x)
        assert m == 5.0
        assert lo == 5.0
        assert hi == 5.0


class TestCIMedian:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        med, lo, hi = ci_median(x, n_bootstrap=500)
        assert abs(med) < 0.2
        assert lo < med
        assert hi > med

    def test_empty(self):
        med, lo, hi = ci_median(np.array([]))
        assert med == 0.0


class TestDecileBins:
    def test_basic(self):
        x = np.arange(100)
        bins = decile_bins(x, n_bins=10)
        assert len(bins) == 100
        assert bins.min() >= 0
        assert bins.max() <= 9

    def test_empty(self):
        bins = decile_bins(np.array([]))
        assert len(bins) == 0


class TestRSquared:
    def test_perfect_fit(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y.copy()
        assert r_squared(y, y_pred) == 1.0

    def test_mean_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y, np.mean(y))
        assert abs(r_squared(y, y_pred)) < 1e-10

    def test_empty(self):
        assert r_squared(np.array([]), np.array([])) == 0.0


class TestVarianceDecomposition:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = 2 * x1 + 0.5 * x2 + rng.normal(0, 0.1, n)
        X = np.column_stack([x1, x2])
        result = variance_decomposition(y, X)
        assert result["overall_r2"] > 0.9
        assert result["r2_x0"] > 0.5  # x1 explains most variance
        assert result["r2_x1"] > 0.1  # x2 explains some

    def test_empty(self):
        result = variance_decomposition(np.array([]), np.empty((0, 2)))
        assert result["overall_r2"] == 0.0
