"""Reference mid of passive fills and the weighting of partially filled orders."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from futures_tca.book import (
    EVENT_DTYPE,
    EVT_EXHAUSTION,
    EVT_LIMIT_ADD,
    EVT_MARKET_ORDER,
    SIDE_ASK,
    SIDE_BID,
    EventLog,
)
from futures_tca.costs import compute_cost, compute_markout
from futures_tca.policies import Fill, PolicyResult, replay_pass
from futures_tca.spec import REGIMES, ContractSpec

FEE = 1.0 / 15.6


def _log(events):
    arr = np.array(events, dtype=EVENT_DTYPE)
    return EventLog(events=arr, warmup=0.0, horizon=1000.0, regime=REGIMES["large"], seed=0)


def _clearing_log():
    return _log([
        (0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 10, 100),  # mid 0.5
        (1.0, EVT_MARKET_ORDER, SIDE_BID, 12, 0.0, -1, 1, 80, 100),  # clears, post mid 0.0
        (1.0 + 1e-9, EVT_EXHAUSTION, SIDE_BID, 0, 0.0, -1, 1, 80, 100),
    ])


class TestClearingFill:
    def test_spread_selection_total(self):
        log = _clearing_log()
        cost = compute_cost(replay_pass(log, +1, 1, 0.5, 10.0), log, ContractSpec())
        assert cost.spread == pytest.approx(-0.5)
        assert cost.selection == pytest.approx(0.0)
        assert cost.total == pytest.approx(-0.5 + FEE)

    def test_markout_includes_the_move_of_the_clearing_event(self):
        log = _clearing_log()
        fill = replay_pass(log, +1, 1, 0.5, 10.0).fills[0]
        assert fill.fill_type == "passive"
        assert fill.mid_before == pytest.approx(0.5)
        assert compute_markout(fill, +1, log, 1.0) == pytest.approx(0.5)

    def test_sell_side_mirror(self):
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 100, 10),
            (1.0, EVT_MARKET_ORDER, SIDE_ASK, 12, 0.0, 0, 2, 100, 80),
            (1.0 + 1e-9, EVT_EXHAUSTION, SIDE_ASK, 0, 0.0, 0, 2, 100, 80),
        ])
        result = replay_pass(log, -1, 1, 0.5, 10.0)
        cost = compute_cost(result, log, ContractSpec())
        assert cost.spread == pytest.approx(-0.5)
        assert cost.selection == pytest.approx(0.0)
        assert compute_markout(result.fills[0], -1, log, 1.0) == pytest.approx(0.5)


class TestOtherFillsUnchanged:
    def test_fill_without_mid_before_uses_the_mid_at_fill_time(self):
        log = _log([(0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 10, 100)])
        fill = Fill(t=0.0, price=1, qty=1, fill_type="aggressive")
        assert fill.mid_before is None
        assert compute_markout(fill, +1, log, 1.0) == 0.0

    def test_timeout_cross_uses_the_mid_at_the_cross(self):
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 10, 100),
            (1.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 2, 3, 10, 100),  # mid jumps to 2.5
        ])
        cost = compute_cost(replay_pass(log, +1, 1, 0.5, 5.0), log, ContractSpec())
        assert cost.spread == pytest.approx(0.5)
        assert cost.selection == pytest.approx(2.0)


class TestPartialFill:
    def test_shortfall_per_intended_lot(self):
        """1 of 2 filled at bid 0 (mid 0.5), close mid 1.5: (-0.5 + 1.0) / 2 plus half the fee."""
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 1, 100),
            (1.0, EVT_MARKET_ORDER, SIDE_BID, 1, 0.0, 0, 1, 1, 100),
            (10.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 1, 2, 1, 100),
        ])
        result = PolicyResult(
            side=+1, qty_total=2, qty_filled=1, arrival_t=0.5, arrival_mid=0.5,
            fills=[Fill(t=1.0, price=0, qty=1, fill_type="passive", mid_before=0.5)],
        )
        cost = compute_cost(result, log, ContractSpec())
        assert cost.spread == pytest.approx(-0.5)  # per filled lot
        assert cost.opportunity == pytest.approx(0.5)
        assert cost.fees == pytest.approx(FEE / 2)
        assert cost.total == pytest.approx(0.25 + FEE / 2)

    @given(
        side=st.sampled_from([+1, -1]),
        qty=st.integers(1, 100),
        filled_share=st.floats(0.0, 1.0),
        price=st.integers(-5, 5),
        close=st.integers(-5, 5),
    )
    @settings(max_examples=50, deadline=5000)
    def test_total_is_implementation_shortfall(self, side, qty, filled_share, price, close):
        """Total equals shortfall computed from cash flows, independent of the components."""
        filled = int(qty * filled_share)
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 100, 100),
            (10.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, close, close + 1, 100, 100),
        ])
        fills = [Fill(t=1.0, price=price, qty=filled, fill_type="passive", mid_before=0.5)]
        result = PolicyResult(
            side=side, qty_total=qty, qty_filled=filled, arrival_t=0.0, arrival_mid=0.5,
            fills=fills if filled else [],
        )
        cost = compute_cost(result, log, ContractSpec())
        close_mid = close + 0.5
        paid = filled * side * (price - 0.5) + (qty - filled) * side * (close_mid - 0.5)
        assert cost.total == pytest.approx(paid / qty + FEE * filled / qty)
