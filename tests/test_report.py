"""Tests for report.py: chart generation and summary.csv.

Test numbers: 25-26.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from futures_tca.report import (
    build_summary,
    chart_a_queue_lottery,
    chart_b_markouts,
    chart_c_roll_cost,
    chart_e_tick_regimes,
)
from futures_tca.spec import REGIMES, ContractSpec


@pytest.fixture
def tmp_reports(tmp_path):
    """Create a temporary reports directory."""
    return str(tmp_path / "reports")


class TestReportFast:
    def test_charts_generated(self, tmp_reports):
        """--fast run writes all four PNGs (a, b, c, e) and summary.csv."""
        contract = ContractSpec()

        chart_a_queue_lottery(contract, n_episodes=5, out=tmp_reports)
        chart_b_markouts(contract, n_episodes=5, out=tmp_reports)
        chart_c_roll_cost(n_episodes=5, out=tmp_reports)
        chart_e_tick_regimes(contract, n_episodes=5, out=tmp_reports)
        build_summary(contract, out=tmp_reports, n_episodes=5)

        out_dir = Path(tmp_reports)
        assert (out_dir / "a_queue_lottery.png").exists()
        assert (out_dir / "b_markouts.png").exists()
        assert (out_dir / "c_roll_cost.png").exists()
        assert (out_dir / "e_tick_regimes.png").exists()
        assert (out_dir / "summary.csv").exists()

    def test_summary_csv_keys(self, tmp_reports):
        """Every key named in the README table exists in summary.csv."""
        contract = ContractSpec()
        df = build_summary(contract, out=tmp_reports, n_episodes=5)

        expected_cols = [
            "total_ticks",
            "spread_ticks",
            "selection_ticks",
            "opportunity_ticks",
            "fees_ticks",
            "total_perf_bps",
            "regime",
            "policy",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"


class TestReproducibility:
    def test_same_seed_same_output(self, tmp_path):
        """Two fast runs with the same seed give identical summary.csv."""
        contract = ContractSpec()

        out1 = str(tmp_path / "reports1")
        out2 = str(tmp_path / "reports2")

        df1 = build_summary(contract, out=out1, n_episodes=5)
        df2 = build_summary(contract, out=out2, n_episodes=5)

        pd.testing.assert_frame_equal(df1, df2)


class TestReportContent:
    def test_summary_uses_regime_tick_for_bps(self, tmp_reports):
        """bps columns use each regime's own tick size (1.4 / 0.5 / 0.15 bps)."""
        df = build_summary(ContractSpec(), out=tmp_reports, n_episodes=5)
        for _, row in df.iterrows():
            tick_bps = REGIMES[row["regime"]].tick_bps
            assert row["total_perf_bps"] == pytest.approx(-row["total_ticks"] * tick_bps)
        assert set(df["regime"]) == {"large", "medium", "small"}

    def test_components_sum_in_summary(self, tmp_reports):
        df = build_summary(ContractSpec(), out=tmp_reports, n_episodes=5)
        parts = df["spread_ticks"] + df["selection_ticks"] + df["opportunity_ticks"]
        assert (parts + df["fees_ticks"]).to_numpy() == pytest.approx(df["total_ticks"].to_numpy())

    def test_chart_e_reports_r2_for_all_regimes(self, tmp_reports):
        res = chart_e_tick_regimes(ContractSpec(), n_episodes=8, out=tmp_reports)["chart_e"]
        assert set(res) == {"large", "medium", "small"}
        for v in res.values():
            assert 0.0 <= v["r2_queue_ahead"] <= 1.0
            assert 0.0 <= v["r2_spread_imbalance"] <= 1.0

    def test_chart_a_returns_break_even_key(self, tmp_reports):
        res = chart_a_queue_lottery(ContractSpec(), n_episodes=5, out=tmp_reports)
        assert "break_even_ahead" in res and "agg_cost" in res

    def test_chart_c_saving_keys(self, tmp_reports):
        res = chart_c_roll_cost(n_episodes=20, out=tmp_reports)
        assert set(res["thresh_saving"]) == {0.0, 0.25, 0.5, 1.0}
