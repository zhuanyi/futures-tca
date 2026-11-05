"""Committed report outputs against the code, and chart contracts."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from futures_tca.report import (
    build_summary,
    chart_a_queue_lottery,
    chart_b_markouts,
    chart_metric_rows,
    write_summary,
)
from futures_tca.roll import liquidity_migration, roll_thresh, simulate_roll_episode
from futures_tca.spec import ContractSpec

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def _committed() -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / "summary.csv", keep_default_na=True)


def _metric(policy: str, metric: str, regime: str | None = None) -> float:
    df = _committed()
    rows = df[(df["policy"] == policy) & (df["metric"] == metric)]
    if regime is not None:
        rows = rows[rows["regime"] == regime]
    assert len(rows) == 1, (policy, metric, regime)
    return float(rows["value"].iloc[0])


class TestGolden:
    def test_committed_cost_rows_match_the_code(self, tmp_path):
        """Fast golden test: rerun the 50-order summary and compare with reports/summary.csv."""
        fresh = build_summary(ContractSpec(), out=str(tmp_path), n_episodes=50)
        committed = _committed()
        committed = committed[committed["metric"].isna()].reset_index(drop=True)
        assert list(committed["regime"]) == list(fresh["regime"])
        assert list(committed["policy"]) == list(fresh["policy"])
        numeric = fresh.select_dtypes("number").columns
        for col in numeric:
            assert committed[col].to_numpy(dtype=float) == pytest.approx(
                fresh[col].to_numpy(dtype=float), rel=1e-9, abs=1e-12
            ), col


class TestSummaryColumns:
    def test_names(self, tmp_path):
        df = build_summary(ContractSpec(), out=str(tmp_path), n_episodes=3)
        for col in ("total_perf_bps", "time_to_fill_s", "passive_fills"):
            assert col in df.columns
        for col in ("tcost_perf_bps", "fill_time", "n_partial"):
            assert col not in df.columns

    def test_time_to_fill_is_relative_to_arrival(self, tmp_path):
        df = build_summary(ContractSpec(), out=str(tmp_path), n_episodes=5)
        assert (df[df["policy"] == "AGG"]["time_to_fill_s"] == 0.0).all()
        assert (df[df["policy"] == "PASS_30"]["time_to_fill_s"].between(0.0, 30.0)).all()

    def test_metric_rows_are_written_after_cost_rows(self, tmp_path):
        costs = build_summary(ContractSpec(), out=str(tmp_path), n_episodes=3)
        metrics = pd.DataFrame(
            [("large", "chart_a", "break_even_ahead", 360.0)],
            columns=["regime", "policy", "metric", "value"],
        )
        write_summary(costs, metrics, out=str(tmp_path))
        back = pd.read_csv(tmp_path / "summary.csv")
        assert back["metric"].isna().sum() == len(costs)
        assert back["metric"].iloc[-1] == "break_even_ahead"


@pytest.fixture(scope="module")
def chart_a_res(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("a"))
    return chart_a_queue_lottery(ContractSpec(), n_episodes=400, out=out)


@pytest.fixture(scope="module")
def chart_b_res(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("b"))
    return chart_b_markouts(ContractSpec(), n_episodes=400, out=out)


class TestChartA:
    def test_every_point_has_a_real_queue_and_paired_agg(self, chart_a_res):
        rows = chart_a_res["chart_a"]
        assert rows[0]["ahead"] == 0 and rows[0]["n_paths"] == 400
        counts = [r["n_paths"] for r in rows]
        assert counts == sorted(counts, reverse=True)
        assert all(r["n_paths"] >= 30 for r in rows)
        for r in rows:
            assert r["diff"] == pytest.approx(r["mean_cost"] - r["agg_cost"])

    def test_break_even_definitions(self, chart_a_res):
        rows = {r["ahead"]: r for r in chart_a_res["chart_a"]}
        be = chart_a_res["break_even_ahead"]
        assert be is not None
        assert rows[be]["diff"] > 0
        assert all(r["diff"] <= 0 for a, r in rows.items() if a < be)
        be_ci = chart_a_res["break_even_ci_ahead"]
        assert be_ci is None or rows[be_ci]["diff_ci_lo"] > 0

    def test_tiny_run_keeps_ahead_zero(self, tmp_path):
        res = chart_a_queue_lottery(ContractSpec(), n_episodes=5, out=str(tmp_path))
        assert res["chart_a"][0]["ahead"] == 0


class TestChartB:
    def test_last_in_queue_carries_the_level_move_from_the_first_horizon(self, chart_b_res):
        assert chart_b_res["horizons"][0] == 1.0
        assert chart_b_res["chart_b"]["passive, last in queue"][0] > 0.3
        assert abs(chart_b_res["chart_b"]["passive, queue behind"][0]) < 0.05

    def test_one_order_per_episode(self, chart_b_res):
        n = chart_b_res["chart_b_n"]
        assert n["aggressive"][0] == 400
        passive = n["passive, queue behind"][0] + n["passive, last in queue"][0]
        assert 0 < passive <= 400

    def test_every_mean_has_a_ci(self, chart_b_res):
        for label, means in chart_b_res["chart_b"].items():
            for m, (lo, hi) in zip(means, chart_b_res["chart_b_ci"][label], strict=True):
                assert lo <= m <= hi

    # F2 amendment: chart_b_n[label] is constant across horizons for every label
    def test_counts_are_constant_across_horizons(self, chart_b_res):
        for label, counts in chart_b_res["chart_b_n"].items():
            assert len(set(counts)) == 1, f"{label} counts vary: {counts}"

    # F1 amendment: empty group has NaN mean and NaN CI bounds, not 0.0
    def test_empty_group_has_nan_not_zero(self, tmp_path):
        res = chart_b_markouts(ContractSpec(), n_episodes=2, out=str(tmp_path))
        for label, means in res["chart_b"].items():
            for m, (lo, hi) in zip(means, res["chart_b_ci"][label], strict=True):
                if np.isnan(m):
                    assert np.isnan(lo) and np.isnan(hi)


class TestReadme:
    def test_table_matches_summary(self):
        df = _committed()
        df = df[df["metric"].isna()]
        for _, r in df.iterrows():
            row = (
                f"| {r['regime']} | {r['policy']} | {r['total_ticks']:.2f} | "
                f"{r['spread_ticks']:.2f} | {r['selection_ticks']:.2f} | "
                f"{r['fees_ticks']:.2f} | {r['passive_fills']:.2f} |"
            )
            assert row in README, row

    @pytest.mark.parametrize(
        "policy, metric, regime, fmt",
        [
            ("chart_a", "mean_cost_ahead_0", None, "{:+.2f}"),
            ("chart_a", "fill_rate_ahead_0", None, "{:.2f}"),
            ("chart_a", "break_even_ahead", None, "{:.0f}"),
            ("chart_a", "diff_ahead_360", None, "{:+.2f}"),
            ("chart_a", "diff_ci_lo_ahead_360", None, "{:+.2f}"),
            ("chart_a", "diff_ci_hi_ahead_360", None, "{:+.2f}"),
            ("chart_a", "break_even_ci_ahead", None, "{:.0f}"),
            ("chart_a", "diff_ahead_420", None, "{:+.2f}"),
            ("chart_a", "diff_ci_lo_ahead_420", None, "{:+.2f}"),
            ("chart_a", "diff_ci_hi_ahead_420", None, "{:+.2f}"),
            ("chart_a", "diff_ahead_600", None, "{:+.2f}"),
            ("chart_a", "n_paths_ahead_600", None, "{:.0f}"),
            ("chart_a", "fill_rate_ahead_600", None, "{:.2f}"),
            ("chart_b", "passive_last_in_queue_mean_1s", None, "{:+.2f}"),
            ("chart_b", "passive_last_in_queue_ci_lo_1s", None, "{:+.2f}"),
            ("chart_b", "passive_last_in_queue_ci_hi_1s", None, "{:+.2f}"),
            ("chart_b", "passive_last_in_queue_mean_300s", None, "{:+.2f}"),
            ("chart_b", "passive_last_in_queue_n_1s", None, "{:.0f}"),
            ("chart_b", "passive_queue_behind_mean_300s", None, "{:+.2f}"),
            ("chart_b", "passive_queue_behind_n_1s", None, "{:.0f}"),
            ("chart_b", "aggressive_mean_300s", None, "{:+.2f}"),
            ("chart_c", "thresh_saving_mean_sd_0", None, "{:.2f}"),
            ("chart_c", "thresh_saving_mean_sd_0.25", None, "{:.2f}"),
            ("chart_c", "thresh_saving_mean_sd_1", None, "{:+.2f}"),
            ("chart_c", "thresh_saving_ci_lo_sd_1", None, "{:+.2f}"),
            ("chart_c", "thresh_saving_ci_hi_sd_1", None, "{:+.2f}"),
            ("chart_e", "r2_queue_ahead", "large", "{:.2f}"),
            ("chart_e", "r2_spread_imbalance", "large", "{:.2f}"),
            ("chart_e", "r2_queue_ahead", "medium", "{:.2f}"),
            ("chart_e", "r2_spread_imbalance", "medium", "{:.2f}"),
            ("chart_e", "r2_queue_ahead", "small", "{:.2f}"),
            ("chart_e", "r2_spread_imbalance", "small", "{:.2f}"),
        ],
    )
    def test_chart_values_in_prose(self, policy, metric, regime, fmt):
        value = fmt.format(_metric(policy, metric, regime))
        assert value in README, (policy, metric, regime, value)

    def test_stale_claims_are_gone(self):
        lowered = README.lower()
        for phrase in (
            "from\npublished papers",
            "published papers or stated assumptions",
            "true roll\nrichness",
            "no break-even",
            "negative at sd 1.0",
            "intended v1 behaviour",
            "mechanically the half-spread",
        ):
            assert phrase.lower() not in lowered, phrase

    def test_missing_chart_d_is_explained(self):
        assert re.search(r"no chart \(d\)", README, re.IGNORECASE)

    def test_roll_assumption_table_matches_defaults(self):
        ep = inspect.signature(simulate_roll_episode).parameters
        mig = inspect.signature(liquidity_migration).parameters
        thresh = inspect.signature(roll_thresh).parameters
        expected = {
            "carry": ep["carry"].default,
            "richness half-life": ep["richness_half_life"].default / 3600,
            "richness sd": ep["richness_sd"].default,
            "skew half-life": ep["skew_half_life"].default / 60,
            "skew sd": ep["skew_sd"].default,
            "migration midpoint": mig["d_roll"].default,
            "migration width": mig["w"].default,
            "forced roll day": thresh["forced_day"].default,
        }
        for name, value in expected.items():
            assert re.search(rf"\| {name} \| {value:g}\b", README), (name, value)

    # F3 amendment: assert specific formatted strings verbatim
    def test_chart_e_r2_strings(self):
        df = _committed()
        for regime in ("large", "medium", "small"):
            q = float(
                df[(df["policy"] == "chart_e")
                   & (df["metric"] == "r2_queue_ahead")
                   & (df["regime"] == regime)]["value"].iloc[0]
            )
            s = float(
                df[(df["policy"] == "chart_e")
                   & (df["metric"] == "r2_spread_imbalance")
                   & (df["regime"] == regime)]["value"].iloc[0]
            )
            assert f"{q:.2f} / {s:.2f}" in README

    def test_chart_a_360_claim_string(self):
        df = _committed()
        d = float(
            df[(df["policy"] == "chart_a")
               & (df["metric"] == "diff_ahead_360")]["value"].iloc[0]
        )
        lo = float(
            df[(df["policy"] == "chart_a")
               & (df["metric"] == "diff_ci_lo_ahead_360")]["value"].iloc[0]
        )
        hi = float(
            df[(df["policy"] == "chart_a")
               & (df["metric"] == "diff_ci_hi_ahead_360")]["value"].iloc[0]
        )
        assert f"({d:+.2f} ticks, 95% CI {lo:+.2f} to {hi:+.2f})" in README


def test_chart_metric_rows_handles_no_break_even():
    a = {
        "chart_a": [{"ahead": 0, "n_paths": 5, "mean_cost": 0.0, "agg_cost": 0.0, "diff": 0.0,
                     "diff_ci_lo": 0.0, "diff_ci_hi": 0.0, "fill_rate": 1.0}],
        "break_even_ahead": None,
        "break_even_ci_ahead": None,
    }
    b = {"chart_b": {"aggressive": [0.0]}, "chart_b_ci": {"aggressive": [(0.0, 0.0)]},
         "chart_b_n": {"aggressive": [1]}, "horizons": [1.0]}
    c = {"thresh_saving": {0.0: {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}}}
    e = {"chart_e": {"large": {"r2_queue_ahead": 0.0, "r2_spread_imbalance": 0.0,
                               "passive_share": 0.0, "n": 1}}}
    df = chart_metric_rows(a, b, c, e)
    assert list(df.columns) == ["regime", "policy", "metric", "value"]
    assert np.isnan(df[df["metric"] == "break_even_ahead"]["value"].iloc[0])
