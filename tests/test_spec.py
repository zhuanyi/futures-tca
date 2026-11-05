"""Tests for spec.py: ContractSpec, Regime, REGIMES, conversions, validation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from futures_tca.spec import (
    REGIMES,
    ContractSpec,
    Regime,
)

# ---------------------------------------------------------------------------
# ContractSpec: frozen dataclass + validation
# ---------------------------------------------------------------------------


class TestContractSpec:
    def test_default_values(self):
        c = ContractSpec()
        assert c.name == "10y-note-like"
        assert c.tick_bps == 1.4
        assert c.tick_value == 15.6
        assert c.dv01_per_lot == 60.0
        assert c.fee_per_lot == 1.0

    def test_frozen(self):
        c = ContractSpec()
        with pytest.raises(AttributeError):
            c.tick_bps = 2.0  # type: ignore[misc]

    def test_validation_negative_tick_bps(self):
        with pytest.raises(ValueError, match="tick_bps"):
            ContractSpec(tick_bps=-1.0)

    def test_validation_zero_tick_value(self):
        with pytest.raises(ValueError, match="tick_value"):
            ContractSpec(tick_value=0.0)

    def test_validation_negative_fee(self):
        with pytest.raises(ValueError, match="fee_per_lot"):
            ContractSpec(fee_per_lot=-0.1)

    def test_validation_nan(self):
        with pytest.raises(ValueError):
            ContractSpec(tick_bps=float("nan"))

    def test_validation_empty_name(self):
        with pytest.raises(ValueError, match="name"):
            ContractSpec(name="")

    def test_zero_fee_allowed(self):
        c = ContractSpec(fee_per_lot=0.0)
        assert c.fee_per_lot == 0.0


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


class TestConversions:
    def setup_method(self):
        self.c = ContractSpec()  # tick_bps=1.4, tick_value=15.6

    def test_ticks_to_bps(self):
        assert self.c.ticks_to_bps(1.0) == 1.4
        assert self.c.ticks_to_bps(0.0) == 0.0
        assert self.c.ticks_to_bps(2.5) == pytest.approx(3.5)

    def test_bps_to_ticks(self):
        assert self.c.bps_to_ticks(1.4) == 1.0
        assert self.c.bps_to_ticks(0.0) == 0.0

    def test_ticks_to_currency(self):
        assert self.c.ticks_to_currency(1.0) == 15.6
        assert self.c.ticks_to_currency(2.0) == 31.2

    def test_currency_to_ticks(self):
        assert self.c.currency_to_ticks(15.6) == 1.0
        assert self.c.currency_to_ticks(31.2) == 2.0

    def test_bps_to_currency(self):
        # 1.4 bps = 1 tick = 15.6 currency
        assert self.c.bps_to_currency(1.4) == 15.6

    def test_currency_to_bps(self):
        assert self.c.currency_to_bps(15.6) == 1.4

    def test_roundtrip_ticks_bps(self):
        for ticks in [0.0, 0.5, 1.0, 3.7, 100.0]:
            assert self.c.bps_to_ticks(self.c.ticks_to_bps(ticks)) == pytest.approx(ticks)

    def test_roundtrip_ticks_currency(self):
        for ticks in [0.0, 0.5, 1.0, 3.7, 100.0]:
            assert self.c.currency_to_ticks(self.c.ticks_to_currency(ticks)) == pytest.approx(ticks)

    def test_roundtrip_bps_currency(self):
        for bps in [0.0, 0.5, 1.4, 10.0]:
            assert self.c.currency_to_bps(self.c.bps_to_currency(bps)) == pytest.approx(bps)

    def test_fee_ticks(self):
        # fee_per_lot=1.0, tick_value=15.6 -> fee_ticks = 1/15.6
        assert self.c.fee_ticks() == pytest.approx(1.0 / 15.6)

    def test_yield_bp_to_ticks(self):
        # 1 bp of yield = 60 currency = 60/15.6 ticks
        assert self.c.yield_bp_to_ticks(1.0) == pytest.approx(60.0 / 15.6)

    def test_ticks_to_yield_bp_roundtrip(self):
        assert self.c.ticks_to_yield_bp(1.0) == pytest.approx(15.6 / 60.0)
        assert self.c.ticks_to_yield_bp(self.c.yield_bp_to_ticks(2.5)) == pytest.approx(2.5)

    def test_for_regime_rescales_tick(self):
        r = REGIMES["small"]
        c = self.c.for_regime(r)
        assert c.tick_bps == r.tick_bps
        assert c.tick_value == pytest.approx(15.6 * r.tick_bps / 1.4)
        assert c.fee_ticks() > self.c.fee_ticks()  # smaller tick -> fee is more ticks


# ---------------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------------


class TestRegime:
    def test_large_regime(self):
        r = REGIMES["large"]
        assert r.label == "large"
        assert r.tick_bps == 1.4
        assert r.q_bar == 300.0
        assert r.s_max == 3
        assert r.kappa == 0.5

    def test_medium_regime(self):
        r = REGIMES["medium"]
        assert r.label == "medium"
        assert r.tick_bps == 0.5
        assert r.q_bar == 60.0

    def test_small_regime(self):
        r = REGIMES["small"]
        assert r.label == "small"
        assert r.tick_bps == 0.15
        assert r.q_bar == 10.0

    def test_frozen(self):
        r = REGIMES["large"]
        with pytest.raises(AttributeError):
            r.q_bar = 100.0  # type: ignore[misc]

    def test_invalid_label(self):
        with pytest.raises(ValueError, match="label"):
            Regime(
                label="huge",
                tick_bps=1.0,
                q_bar=100.0,
                mo_rate=0.5,
                mo_size=10.0,
                cancel_rate=0.01,
                inside_rate=1.0,
                s_max=3,
                kappa=0.5,
            )

    def test_invalid_kappa(self):
        with pytest.raises(ValueError, match="kappa"):
            Regime(
                label="large",
                tick_bps=1.0,
                q_bar=100.0,
                mo_rate=0.5,
                mo_size=10.0,
                cancel_rate=0.01,
                inside_rate=1.0,
                s_max=3,
                kappa=1.5,
            )

    def test_invalid_s_max(self):
        with pytest.raises(ValueError, match="s_max"):
            Regime(
                label="large",
                tick_bps=1.0,
                q_bar=100.0,
                mo_rate=0.5,
                mo_size=10.0,
                cancel_rate=0.01,
                inside_rate=1.0,
                s_max=0,
                kappa=0.5,
            )

    def test_negative_q_bar(self):
        with pytest.raises(ValueError, match="q_bar"):
            Regime(
                label="large",
                tick_bps=1.0,
                q_bar=-10.0,
                mo_rate=0.5,
                mo_size=10.0,
                cancel_rate=0.01,
                inside_rate=1.0,
                s_max=3,
                kappa=0.5,
            )

    def test_add_rate(self):
        r = REGIMES["large"]
        # add_rate = cancel_rate * q_bar + mo_rate * mo_size
        expected = 0.01 * 300.0 + 0.5 * 15.0
        assert r.add_rate == pytest.approx(expected)

    def test_add_rate_medium(self):
        r = REGIMES["medium"]
        expected = r.add_scale * (0.03 * 60.0 + 0.5 * 8.0)
        assert r.add_rate == pytest.approx(expected)

    def test_q_bar_below_one_rejected(self):
        with pytest.raises(ValueError, match="q_bar"):
            replace(REGIMES["large"], q_bar=0.5)

    def test_add_scale_validated(self):
        with pytest.raises(ValueError, match="add_scale"):
            replace(REGIMES["large"], add_scale=0.0)

    def test_all_regimes_present(self):
        assert set(REGIMES.keys()) == {"large", "medium", "small"}

    def test_kappa_zero_allowed(self):
        r = Regime(
            label="large",
            tick_bps=1.0,
            q_bar=100.0,
            mo_rate=0.5,
            mo_size=10.0,
            cancel_rate=0.01,
            inside_rate=1.0,
            s_max=3,
            kappa=0.0,
        )
        assert r.kappa == 0.0

    def test_inside_rate_zero_allowed(self):
        r = Regime(
            label="large",
            tick_bps=1.0,
            q_bar=100.0,
            mo_rate=0.5,
            mo_size=10.0,
            cancel_rate=0.01,
            inside_rate=0.0,
            s_max=3,
            kappa=0.5,
        )
        assert r.inside_rate == 0.0

    def test_negative_inside_rate(self):
        with pytest.raises(ValueError, match="inside_rate"):
            Regime(
                label="large",
                tick_bps=1.0,
                q_bar=100.0,
                mo_rate=0.5,
                mo_size=10.0,
                cancel_rate=0.01,
                inside_rate=-0.1,
                s_max=3,
                kappa=0.5,
            )
