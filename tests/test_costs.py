"""Tests for costs.py: cost decomposition, markouts, perf view.

Test numbers: 16-19.
"""

from __future__ import annotations

import numpy as np
import pytest

from futures_tca.book import (
    EVENT_DTYPE,
    EVT_LIMIT_ADD,
    EVT_MARKET_ORDER,
    SIDE_ASK,
    SIDE_BID,
    EventLog,
)
from futures_tca.costs import (
    add_perf_columns,
    compute_cost,
    compute_markout,
    compute_markouts,
    to_perf,
)
from futures_tca.policies import Fill, replay_agg, replay_pass_at
from futures_tca.spec import REGIMES, ContractSpec


def _make_event(t, kind, side, size, u, b, a, Qb, Qa):
    return (t, kind, side, size, u, b, a, Qb, Qa)


def _make_log(events, regime_name="large"):
    if events:
        arr = np.array(events, dtype=EVENT_DTYPE)
    else:
        arr = np.empty(0, dtype=EVENT_DTYPE)
    regime = REGIMES[regime_name]
    return EventLog(events=arr, warmup=0.0, horizon=1000.0, regime=regime, seed=0)


# ---------------------------------------------------------------------------
# Test 17: Perf columns equal minus cost columns
# ---------------------------------------------------------------------------


class TestPerfColumns:
    def test_perf_equals_minus_cost(self):
        """Every *_perf_* column equals minus its cost column exactly."""
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 100, 100),
        ]
        log = _make_log(events)
        contract = ContractSpec()

        result = replay_agg(log, side=+1, qty=5, t0=0.0)
        cost = compute_cost(result, log, contract)
        perf = add_perf_columns(cost, contract.tick_bps)

        assert perf["total_perf_ticks"] == -cost.total
        assert perf["spread_perf_ticks"] == -cost.spread
        assert perf["selection_perf_ticks"] == -cost.selection
        assert perf["opportunity_perf_ticks"] == -cost.opportunity
        assert perf["fees_perf_ticks"] == -cost.fees
        assert perf["total_perf_bps"] == -cost.total * contract.tick_bps

    def test_to_perf(self):
        assert to_perf(1.5) == -1.5
        assert to_perf(-2.0) == 2.0
        assert to_perf(0.0) == 0.0


# ---------------------------------------------------------------------------
# Test 18: Sign symmetry
# ---------------------------------------------------------------------------


class TestSignSymmetry:
    def test_buyer_seller_symmetry(self):
        """Buyer and seller on mirrored paths have identical costs."""
        # Symmetric setup: b=0, a=1
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 100, 100),
        ]
        log = _make_log(events)
        contract = ContractSpec()

        # Buyer lifts ask at 1
        result_buy = replay_agg(log, side=+1, qty=5, t0=0.0)
        cost_buy = compute_cost(result_buy, log, contract)

        # Seller hits bid at 0
        result_sell = replay_agg(log, side=-1, qty=5, t0=0.0)
        cost_sell = compute_cost(result_sell, log, contract)

        # Costs should be identical (symmetric)
        assert abs(cost_buy.total - cost_sell.total) < 1e-9

    def test_favourable_markout_negative(self):
        """Favourable post-fill move makes markout_cost < 0."""
        # For favourable: mid should rise after a buy (you bought low)
        events_fav = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 100, 100),
            # After fill at t=0, mid rises: a goes from 1 to 2
            _make_event(10.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 2, 100, 100),
        ]
        log_fav = _make_log(events_fav)
        fill_fav = Fill(t=0.0, price=1, qty=5, fill_type="aggressive")
        m_fav = compute_markout(fill_fav, side=+1, log=log_fav, horizon=10.0)
        # m_i = 0.5, m_h = 1.0, markout = +1 * (0.5 - 1.0) = -0.5 < 0
        assert m_fav < 0


# ---------------------------------------------------------------------------
# Test 19: Zero-information check
# ---------------------------------------------------------------------------


class TestZeroInformation:
    def test_markout_is_zero_without_order_flow_information(self):
        """kappa=0, prices free to move: the 60 s markout of an arrival is 0 within 3 SE."""
        from dataclasses import replace

        from futures_tca.book import generate_episode
        from futures_tca.costs import compute_markout

        regime = replace(REGIMES["medium"], kappa=0.0)
        marks = []
        for ep in range(3000):
            log = generate_episode(regime, seed=900_000 + ep, warmup=60.0, horizon=60.0)
            fill = replay_agg(log, +1, 1, 60.0).fills[0]
            marks.append(compute_markout(fill, +1, log, 60.0))
        marks = np.array(marks)
        assert marks.std() > 0.5  # prices did move
        assert abs(marks.mean()) <= 3 * marks.std() / np.sqrt(len(marks))


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------


class TestCostDecomposition:
    def test_basic_cost(self):
        """AGG buy at ask=1 with mid=0.5: spread = +0.5 ticks."""
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 100, 100),
        ]
        log = _make_log(events)
        contract = ContractSpec()

        result = replay_agg(log, side=+1, qty=5, t0=0.0)
        cost = compute_cost(result, log, contract)

        # Buyer lifts ask at 1, mid at fill = 0.5
        # spread = side * (p - m) = +1 * (1 - 0.5) = 0.5
        assert abs(cost.spread - 0.5) < 1e-9
        # selection = side * (m - P_0) = +1 * (0.5 - 0.5) = 0
        assert abs(cost.selection) < 1e-9

    def test_passive_earns_spread(self):
        """PASS with ahead=0 and immediate MO: spread = -0.5 (earns half spread)."""
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 1, 100),
            _make_event(1.0, EVT_MARKET_ORDER, SIDE_BID, 10, 0.0, 0, 1, 0, 100),
        ]
        log = _make_log(events)
        contract = ContractSpec()

        result = replay_pass_at(log, side=+1, qty=5, t0=0.0, ahead_init=0, timeout=10.0)
        cost = compute_cost(result, log, contract)

        # Filled at bid=0, mid at fill = 0.5
        # spread = side * (p - m) = +1 * (0 - 0.5) = -0.5 (earns spread)
        assert [f.fill_type for f in result.fills] == ["passive"]
        assert cost.spread == pytest.approx(-0.5)
        assert cost.selection == pytest.approx(0.0)


class TestMarkouts:
    def test_compute_markouts(self):
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 1, 100),
            _make_event(1.0, EVT_MARKET_ORDER, SIDE_BID, 10, 0.0, 0, 1, 0, 100),
        ]
        log = _make_log(events)

        result = replay_pass_at(log, side=+1, qty=5, t0=0.0, ahead_init=0, timeout=10.0)
        markouts = compute_markouts(result, log, horizons=[1.0, 5.0])

        assert 1.0 in markouts
        assert 5.0 in markouts


class TestMarkoutAvailability:
    def test_zero_markout_is_kept_and_unavailable_is_nan(self):
        from futures_tca.costs import markout_samples

        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 1, 100),
            _make_event(1.0, EVT_MARKET_ORDER, SIDE_BID, 10, 0.0, 0, 1, 1, 100),
        ]
        log = _make_log(events)  # warmup 0, horizon 1000 -> end_time 1000
        result = replay_pass_at(log, side=+1, qty=5, t0=0.0, ahead_init=0, timeout=10.0)
        samples = markout_samples(result, log, [1.0, 5000.0])
        assert samples[1.0] == [0.0]  # mid never moved: a genuine zero is retained
        assert samples[5000.0] == []  # beyond the episode end
        marks = compute_markouts(result, log, horizons=[1.0, 5000.0])
        assert marks[1.0] == 0.0 and np.isnan(marks[5000.0])


class TestRegimeFees:
    def test_fee_uses_regime_tick(self):
        """The fee in ticks follows the regime's tick size (smaller tick, more ticks of fee)."""
        events = [_make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 100, 100)]
        contract = ContractSpec()
        large = compute_cost(replay_agg(_make_log(events, "large"), +1, 1, 0.0),
                             _make_log(events, "large"), contract)
        small = compute_cost(replay_agg(_make_log(events, "small"), +1, 1, 0.0),
                             _make_log(events, "small"), contract)
        assert large.fees == pytest.approx(1.0 / 15.6)
        assert small.fees == pytest.approx(1.0 / (15.6 * 0.15 / 1.4))
