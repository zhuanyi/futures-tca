"""Tests for policies.py: replay engine, AGG, PASS, PASS_AT, PASS_NOCOMP, PASS_IMB.

Test numbers: 8-15.
Uses hand-built EventLogs for deterministic testing.
"""

from __future__ import annotations

import numpy as np

from futures_tca.book import (
    EVENT_DTYPE,
    EVT_CANCEL,
    EVT_EXHAUSTION,
    EVT_LIMIT_ADD,
    EVT_MARKET_ORDER,
    SIDE_ASK,
    SIDE_BID,
    EventLog,
)
from futures_tca.policies import (
    replay_agg,
    replay_pass,
    replay_pass_at,
    replay_pass_imb,
)
from futures_tca.spec import REGIMES


def _make_event(
    t: float,
    kind: int,
    side: int,
    size: int,
    u: float,
    b: int,
    a: int,
    Qb: int,
    Qa: int,
) -> tuple:
    """Helper to create a single event tuple."""
    return (t, kind, side, size, u, b, a, Qb, Qa)


def _make_log(events: list[tuple], regime_name: str = "large") -> EventLog:
    """Helper to create an EventLog from a list of event tuples."""
    if events:
        arr = np.array(events, dtype=EVENT_DTYPE)
    else:
        arr = np.empty(0, dtype=EVENT_DTYPE)
    regime = REGIMES[regime_name]
    return EventLog(events=arr, warmup=0.0, horizon=1000.0, regime=regime, seed=0)


# ---------------------------------------------------------------------------
# Test 8: Scripted queue
# ---------------------------------------------------------------------------


class TestScriptedQueue:
    def test_ahead_100_mo_60_60(self):
        """ahead=100, MO 60 then 60: order of size 5 fills 5 at second MO."""
        # Initial state: b=0, a=1, Qb=200, Qa=200
        # Event at t=1: MO hit-bid (seller) size 60, eats from bid queue
        # After: Qb=140 (200-60), but our order has ahead=100, so ahead=100-60=40
        # Event at t=2: MO hit-bid size 60, eats from bid queue
        # consumed = min(40, 60) = 40, fill = min(5, 60-40) = 5
        events = [
            _make_event(1.0, EVT_MARKET_ORDER, SIDE_BID, 60, 0.0, 0, 1, 140, 200),
            _make_event(2.0, EVT_MARKET_ORDER, SIDE_BID, 60, 0.0, 0, 1, 80, 200),
        ]
        log = _make_log(events)

        # Use PASS_AT with ahead=100, timeout=10
        result = replay_pass_at(log, side=+1, qty=5, t0=0.0, ahead_init=100, timeout=10.0)

        # Should fill 5 lots at the second MO
        assert result.qty_filled == 5
        assert len(result.fills) >= 1
        # Find the passive fill
        passive_fills = [f for f in result.fills if f.fill_type == "passive"]
        assert len(passive_fills) == 1
        assert passive_fills[0].qty == 5
        assert passive_fills[0].price == 0  # bid price
        assert passive_fills[0].t == 2.0  # second MO


# ---------------------------------------------------------------------------
# Test 9: Cancel ahead/behind
# ---------------------------------------------------------------------------


class TestCancelAheadBehind:
    def test_cancel_ahead_reduces_ahead(self):
        """Cancel with u*Q < ahead reduces ahead."""
        # Qb=100, ahead=50, cancel at u=0.3 -> u*Q = 30 < 50, so ahead -= 10
        events = [
            _make_event(1.0, EVT_CANCEL, SIDE_BID, 10, 0.3, 0, 1, 90, 100),
        ]
        log = _make_log(events)

        result = replay_pass_at(log, side=+1, qty=5, t0=0.0, ahead_init=50, timeout=10.0)
        # After cancel: ahead = 50 - 10 = 40
        # No MO, so no fill, timeout crosses
        assert result.qty_filled == 5
        cross_fills = [f for f in result.fills if f.fill_type == "cross_timeout"]
        assert len(cross_fills) == 1

    def test_cancel_behind_no_effect(self):
        """Cancel with u*Q >= ahead leaves ahead unchanged."""
        # Qb=100, ahead=50, cancel at u=0.7 -> u*Q = 70 >= 50, no effect
        events = [
            _make_event(1.0, EVT_CANCEL, SIDE_BID, 10, 0.7, 0, 1, 90, 100),
            _make_event(2.0, EVT_MARKET_ORDER, SIDE_BID, 60, 0.0, 0, 1, 30, 100),
        ]
        log = _make_log(events)

        result = replay_pass_at(log, side=+1, qty=5, t0=0.0, ahead_init=50, timeout=10.0)
        # After cancel: ahead still 50
        # MO size 60: consumed = min(50, 60) = 50, fill = min(5, 10) = 5
        passive_fills = [f for f in result.fills if f.fill_type == "passive"]
        assert len(passive_fills) == 1
        assert passive_fills[0].qty == 5


# ---------------------------------------------------------------------------
# Test 10: AGG cost
# ---------------------------------------------------------------------------


class TestAGGCost:
    def test_agg_pays_half_spread_plus_fees(self):
        """AGG on 1-tick spread pays exactly half spread plus fees."""
        # b=0, a=1, spread=1 tick
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 100, 100),
        ]
        log = _make_log(events)

        result = replay_agg(log, side=+1, qty=5, t0=0.0)
        assert result.qty_filled == 5
        assert result.fills[0].price == 1  # lifted the ask
        assert result.fills[0].fill_type == "aggressive"
        # Cost = 0.5 ticks (half spread) + fee_ticks
        # This will be computed in costs.py, but we verify the fill price here


# ---------------------------------------------------------------------------
# Test 11: PASS earns spread
# ---------------------------------------------------------------------------


class TestPASSearnsSpread:
    def test_pass_with_ahead_0_immediate_mo(self):
        """PASS(T) with ahead=0 and immediate MO earns spread."""
        # b=0, a=1, Qb=0 (so ahead=0 for our buy order at bid)
        # MO at t=1 lifts bid (seller hits bid), but Qb=0 so...
        # Actually, if Qb=0, that's an exhaustion state. Let me set Qb=1.
        # Our order posts at bid with ahead=Qb=1.
        # Wait, the test says ahead=0. Let me re-read.
        # "PASS(T) with ahead=0 and an immediate MO at the touch earns the spread"
        # So ahead=0 means we're at the front of the queue.
        # MO hits bid (seller): consumes from front, our order fills.
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 1, 100),
            _make_event(1.0, EVT_MARKET_ORDER, SIDE_BID, 10, 0.0, 0, 1, 0, 100),
        ]
        log = _make_log(events)

        # Post at bid with ahead=0 (we're at the front)
        result = replay_pass_at(log, side=+1, qty=5, t0=0.0, ahead_init=0, timeout=10.0)

        # MO size 10: consumed = min(0, 10) = 0, fill = min(5, 10) = 5
        passive_fills = [f for f in result.fills if f.fill_type == "passive"]
        assert len(passive_fills) == 1
        assert passive_fills[0].qty == 5
        assert passive_fills[0].price == 0  # filled at bid
        # spread_i = s * (p_i - m_i) = +1 * (0 - 0.5) = -0.5 (earns half spread)
        # This will be verified in costs.py


# ---------------------------------------------------------------------------
# Test 12: Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_unfilled_pass_crosses_at_timeout(self):
        """Unfilled PASS(T) crosses at ask at t0+T."""
        # No MO events, so order doesn't fill passively
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 100, 100),
        ]
        log = _make_log(events)

        result = replay_pass(log, side=+1, qty=5, t0=0.0, timeout=10.0, complete=True)

        # Should cross at timeout
        assert result.qty_filled == 5
        cross_fills = [f for f in result.fills if f.fill_type == "cross_timeout"]
        assert len(cross_fills) == 1
        assert cross_fills[0].t == 10.0
        assert cross_fills[0].price == 1  # cross at ask


# ---------------------------------------------------------------------------
# Test 13: Paired replay
# ---------------------------------------------------------------------------


class TestPairedReplay:
    def test_agg_same_alone_or_with_others(self):
        """Replaying other policies leaves the log and the AGG result unchanged."""
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 100, 100),
            _make_event(1.0, EVT_MARKET_ORDER, SIDE_BID, 10, 0.0, 0, 1, 90, 100),
        ]
        log = _make_log(events)
        before = log.events.copy()

        alone = replay_agg(log, side=+1, qty=5, t0=0.5)
        replay_pass(log, side=+1, qty=5, t0=0.5, timeout=5.0)
        replay_pass_at(log, side=+1, qty=5, t0=0.5, ahead_init=3, timeout=5.0)
        replay_pass_imb(log, side=+1, qty=5, t0=0.5, timeout=5.0, i_thr=0.5)
        after = replay_agg(log, side=+1, qty=5, t0=0.5)

        assert np.array_equal(log.events, before)
        assert [(f.t, f.price, f.qty) for f in alone.fills] == [
            (f.t, f.price, f.qty) for f in after.fills
        ]


# ---------------------------------------------------------------------------
# Test 14: Partial fills
# ---------------------------------------------------------------------------


class TestPartialFills:
    def test_fills_sum_to_Q(self):
        """Fills sum to Q for completed policies."""
        events = [
            _make_event(1.0, EVT_MARKET_ORDER, SIDE_BID, 3, 0.0, 0, 1, 97, 100),
            _make_event(2.0, EVT_MARKET_ORDER, SIDE_BID, 3, 0.0, 0, 1, 94, 100),
        ]
        log = _make_log(events)

        result = replay_pass_at(log, side=+1, qty=5, t0=0.0, ahead_init=0, timeout=10.0)

        # Should fill 3 at t=1, 2 at t=2 (total 5)
        assert result.qty_filled == 5
        # May have cross_timeout for remainder
        all_qty = sum(f.qty for f in result.fills)
        assert all_qty == 5

    def test_n_partial_counted(self):
        """n_partial counts the number of passive fills."""
        events = [
            _make_event(1.0, EVT_MARKET_ORDER, SIDE_BID, 3, 0.0, 0, 1, 97, 100),
            _make_event(2.0, EVT_MARKET_ORDER, SIDE_BID, 3, 0.0, 0, 1, 94, 100),
        ]
        log = _make_log(events)

        result = replay_pass_at(log, side=+1, qty=5, t0=0.0, ahead_init=0, timeout=10.0)

        # Should have 2 passive fills (3 lots, then 2 lots)
        assert result.n_partial == 2


# ---------------------------------------------------------------------------
# Test 15: Walk-the-book (AGG)
# ---------------------------------------------------------------------------


class TestWalkTheBook:
    def test_agg_fills_at_touch(self):
        """An order that fits in the touch queue fills entirely at the touch."""
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 100, 100),
        ]
        log = _make_log(events)

        result = replay_agg(log, side=+1, qty=5, t0=0.0)
        assert [(f.price, f.qty) for f in result.fills] == [(1, 5)]

    def test_agg_sell_hits_bid(self):
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 100, 100),
        ]
        log = _make_log(events)

        result = replay_agg(log, side=-1, qty=5, t0=0.0)
        assert [(f.price, f.qty) for f in result.fills] == [(0, 5)]

    def test_agg_larger_than_touch_walks_the_book(self):
        """A buy larger than the touch queue takes the rest one tick higher."""
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 100, 4),
        ]
        log = _make_log(events)

        result = replay_agg(log, side=+1, qty=10, t0=0.0)
        assert result.qty_filled == 10
        assert result.fills[0].price == 1 and result.fills[0].qty == 4
        assert result.fills[1].price == 2
        assert sum(f.qty for f in result.fills) == 10
        assert all(f.fill_type == "aggressive" for f in result.fills)

    def test_agg_sell_walks_down(self):
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 3, 100),
        ]
        log = _make_log(events)
        result = replay_agg(log, side=-1, qty=10, t0=0.0)
        assert result.fills[0].price == 0 and result.fills[0].qty == 3
        assert result.fills[1].price == -1
        assert result.qty_filled == 10

    def test_agg_walk_is_deterministic_and_bounded(self):
        """Same log and t0 give the same walk; sizes <= 3 x mo_size stay within +3 ticks."""
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 100, 1),
        ]
        log = _make_log(events)
        qty = int(3 * log.regime.mo_size)
        r1 = replay_agg(log, side=+1, qty=qty, t0=0.0)
        r2 = replay_agg(log, side=+1, qty=qty, t0=0.0)
        assert [(f.price, f.qty) for f in r1.fills] == [(f.price, f.qty) for f in r2.fills]
        assert r1.qty_filled == qty
        assert max(f.price for f in r1.fills) <= 1 + 3


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------


class TestPolicyResult:
    def test_fill_rate(self):
        result = replay_agg(
            _make_log([_make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 100, 100)]),
            side=+1, qty=5, t0=0.0,
        )
        assert result.fill_rate == 1.0

    def test_fill_time(self):
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 100, 100),
            _make_event(1.0, EVT_MARKET_ORDER, SIDE_BID, 10, 0.0, 0, 1, 90, 100),
        ]
        log = _make_log(events)
        result = replay_pass_at(log, side=+1, qty=5, t0=0.0, ahead_init=0, timeout=10.0)
        assert result.fill_time == 1.0


class TestPASSIMB:
    def test_pass_imb_no_adverse(self):
        """PASS_IMB without adverse imbalance behaves like PASS."""
        events = [
            _make_event(1.0, EVT_MARKET_ORDER, SIDE_BID, 10, 0.0, 0, 1, 90, 100),
        ]
        log = _make_log(events)

        result = replay_pass_imb(log, side=+1, qty=5, t0=0.0, timeout=10.0, i_thr=0.5)
        assert result.qty_filled == 5

    def test_pass_imb_adverse_exit(self):
        """PASS_IMB exits early on adverse imbalance."""
        # Create an event with adverse imbalance (Qa >> Qb for buyer)
        events = [
            _make_event(1.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 10, 200),
        ]
        log = _make_log(events)

        result = replay_pass_imb(log, side=+1, qty=5, t0=0.0, timeout=10.0, i_thr=0.5)
        # Exits at the adverse event and crosses at that event's ask
        assert [(f.fill_type, f.t, f.price, f.qty) for f in result.fills] == [
            ("cross_timeout", 1.0, 1, 5)
        ]


# ---------------------------------------------------------------------------
# Review fixes: pre-event queue, exhaustion replay, populated metadata
# ---------------------------------------------------------------------------


class TestCancelUsesPreEventQueue:
    def test_cancel_position_scaled_by_queue_before_event(self):
        """u * Q is taken against the queue just before the cancel, not the t0 queue."""
        # At t0 the queue is 100; an add makes it 500 before the cancel at u=0.2.
        # u * 500 = 100 >= ahead=60 (behind us); with the stale t0 queue 20 < 60 (ahead).
        events = [
            _make_event(0.5, EVT_LIMIT_ADD, SIDE_BID, 400, 0.0, 0, 1, 500, 100),
            _make_event(1.0, EVT_CANCEL, SIDE_BID, 10, 0.2, 0, 1, 490, 100),
            _make_event(2.0, EVT_MARKET_ORDER, SIDE_BID, 62, 0.0, 0, 1, 428, 100),
        ]
        log = _make_log(events)
        result = replay_pass_at(log, side=+1, qty=5, t0=0.1, ahead_init=60, timeout=10.0)
        # ahead stays 60, so the 62-lot MO leaves 2 lots for us: partial passive fill
        passive = [f for f in result.fills if f.fill_type == "passive"]
        assert [f.qty for f in passive] == [2]


class TestExhaustionReplay:
    def test_clearing_market_order_fills_us_and_continues(self):
        """A clearing MO fills us; the marker row is ignored and nothing breaks."""
        # MO of 12 on a bid queue of 10 clears the level; snapshot is the new level
        events = [
            _make_event(1.0, EVT_MARKET_ORDER, SIDE_BID, 12, 0.0, -1, 1, 80, 100),
            _make_event(1.0 + 1e-9, EVT_EXHAUSTION, SIDE_BID, 0, 0.0, -1, 1, 80, 100),
        ]
        pre = _make_event(0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 10, 100)
        log = _make_log([pre, *events])
        result = replay_pass(log, side=+1, qty=2, t0=0.0, timeout=10.0)
        passive = [f for f in result.fills if f.fill_type == "passive"]
        assert len(passive) == 1
        assert passive[0].qty == 2 and passive[0].price == 0
        # The level cleared, so nothing remains behind us
        assert passive[0].queue_behind_at_fill == 0

    def test_clearing_cancel_removes_queue_ahead(self):
        """After a cancel that clears the queue ahead, a later MO fills us at once."""
        pre = _make_event(0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 10, 100)
        events = [
            _make_event(1.0, EVT_CANCEL, SIDE_BID, 10, 0.0, -1, 1, 70, 100),
            _make_event(1.0 + 1e-9, EVT_EXHAUSTION, SIDE_BID, 0, 0.0, -1, 1, 70, 100),
            _make_event(2.0, EVT_MARKET_ORDER, SIDE_BID, 3, 0.0, -1, 1, 67, 100),
        ]
        log = _make_log([pre, *events])
        result = replay_pass(log, side=+1, qty=2, t0=0.0, timeout=10.0)
        passive = [f for f in result.fills if f.fill_type == "passive"]
        assert len(passive) == 1 and passive[0].t == 2.0


class TestPopulatedMetadata:
    def test_queue_fields_set(self):
        log = _make_log([
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 80, 100),
            _make_event(1.0, EVT_MARKET_ORDER, SIDE_BID, 55, 0.0, 0, 1, 25, 100),
        ])
        result = replay_pass_at(log, side=+1, qty=2, t0=0.0, ahead_init=50, timeout=10.0)
        assert result.queue_ahead_t0 == 50
        f = result.fills[0]
        assert f.fill_type == "passive"
        assert f.queue_ahead_t0 == 50
        # 80 others: 50 ahead, 30 behind; MO 55 eats 50 ahead, 2 of us, 3 of those behind
        assert f.queue_behind_at_fill == 27
        assert result.queue_behind_at_fill == 27

    def test_pass_records_queue_ahead(self):
        log = _make_log([_make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 70, 100)])
        result = replay_pass(log, side=+1, qty=1, t0=0.0, timeout=5.0)
        assert result.queue_ahead_t0 == 70
        assert result.fills[0].queue_ahead_t0 == 70

    def test_fill_type_dominant(self):
        log = _make_log([_make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 70, 100)])
        assert replay_pass(log, +1, 1, 0.0, 5.0).fill_type == "cross_timeout"
        assert replay_pass(log, +1, 1, 0.0, 5.0, complete=False).fill_type == "unfilled"
        assert replay_agg(log, +1, 1, 0.0).fill_type == "aggressive"


class TestPassImbEdgeCases:
    def test_no_events_in_window_does_not_crash(self):
        """PASS_IMB with an empty event window crosses at the timeout."""
        log = _make_log([_make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 100, 100)])
        result = replay_pass_imb(log, side=+1, qty=3, t0=0.0, timeout=5.0, i_thr=0.5)
        assert result.qty_filled == 3
        assert result.fills[0].t == 5.0
        assert result.fills[0].fill_type == "cross_timeout"

    def test_early_exit_time_is_event_time(self):
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 100, 100),
            _make_event(1.5, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 10, 200),
        ]
        result = replay_pass_imb(_make_log(events), +1, 3, 0.0, 10.0, 0.5)
        assert result.fills[-1].t == 1.5


class TestAggZeroTouch:
    def test_empty_touch_skips_zero_fill(self):
        """A zero-size touch level produces no zero-qty fill; the walk continues."""
        log = _make_log([_make_event(0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 10, 0)])
        result = replay_agg(log, side=+1, qty=3, t0=0.0)
        assert result.qty_filled == 3
        assert all(f.qty > 0 for f in result.fills)


class TestPassImbKeepsSameEventFill:
    def test_fill_on_the_triggering_event_is_kept(self):
        """A market order that fills us is kept even if its snapshot is adverse."""
        events = [
            _make_event(0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 1, 5, 100),
            # MO fills our 3 lots; the post-event snapshot is ask-heavy (adverse for a buyer)
            _make_event(1.0, EVT_MARKET_ORDER, SIDE_BID, 8, 0.0, 0, 1, 1, 100),
        ]
        result = replay_pass_imb(_make_log(events), +1, 3, 0.0, 10.0, 0.5)
        assert [f.fill_type for f in result.fills] == ["passive"]
        assert result.fills[0].t == 1.0
