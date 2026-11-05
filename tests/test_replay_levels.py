"""Replay against price levels: orders behind a better level, the queue bound, cancel geometry."""

from __future__ import annotations

import numpy as np
import pytest

from futures_tca.book import (
    EVENT_DTYPE,
    EVT_CANCEL,
    EVT_EXHAUSTION,
    EVT_INSIDE_ADD,
    EVT_LIMIT_ADD,
    EVT_MARKET_ORDER,
    SIDE_ASK,
    SIDE_BID,
    EventLog,
    generate_episode,
)
from futures_tca.policies import replay_pass, replay_pass_at
from futures_tca.spec import REGIMES


def _log(events, regime="large"):
    arr = np.array(events, dtype=EVENT_DTYPE)
    return EventLog(events=arr, warmup=0.0, horizon=1000.0, regime=REGIMES[regime], seed=0)


def _passive(result):
    return [f for f in result.fills if f.fill_type == "passive"]


PRE = (0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 2, 2, 100)  # b=0 a=2 Qb=2: spread 2


class TestBehindBetterLevel:
    def test_market_order_on_better_level_does_not_fill_us(self):
        """Inside add at b+1, then a market order smaller than that level: no passive fill."""
        log = _log([
            PRE,
            (1.0, EVT_INSIDE_ADD, SIDE_BID, 1, 0.0, 1, 2, 1, 100),
            (2.0, EVT_LIMIT_ADD, SIDE_BID, 20, 0.0, 1, 2, 21, 100),
            (3.0, EVT_MARKET_ORDER, SIDE_BID, 5, 0.0, 1, 2, 16, 100),
        ])
        result = replay_pass(log, +1, 1, 0.5, 10.0)
        assert _passive(result) == []
        assert [(f.fill_type, f.price, f.t) for f in result.fills] == [("cross_timeout", 2, 10.5)]

    def test_sell_side_mirror(self):
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 2, 100, 2),
            (1.0, EVT_INSIDE_ADD, SIDE_ASK, 1, 0.0, 0, 1, 100, 1),
            (2.0, EVT_LIMIT_ADD, SIDE_ASK, 20, 0.0, 0, 1, 100, 21),
            (3.0, EVT_MARKET_ORDER, SIDE_ASK, 5, 0.0, 0, 1, 100, 16),
        ])
        result = replay_pass(log, -1, 1, 0.5, 10.0)
        assert _passive(result) == []
        assert [(f.fill_type, f.price) for f in result.fills] == [("cross_timeout", 0)]

    def test_excess_of_clearing_market_order_is_dropped(self):
        """Lots beyond the cleared better level do not reach our level (the engine drops them)."""
        log = _log([
            PRE,
            (1.0, EVT_INSIDE_ADD, SIDE_BID, 1, 0.0, 1, 2, 1, 100),
            (2.0, EVT_LIMIT_ADD, SIDE_BID, 15, 0.0, 1, 2, 16, 100),
            (3.0, EVT_MARKET_ORDER, SIDE_BID, 50, 0.0, 0, 2, 40, 100),  # clears, back to b=0
            (3.0 + 1e-9, EVT_EXHAUSTION, SIDE_BID, 0, 0.0, 0, 2, 40, 100),
            (5.0, EVT_MARKET_ORDER, SIDE_BID, 3, 0.0, 0, 2, 37, 100),  # 2 ahead, then us
        ])
        passive = _passive(replay_pass(log, +1, 1, 0.5, 10.0))
        assert [(f.t, f.price, f.qty) for f in passive] == [(5.0, 0, 1)]

    def test_touch_returning_to_our_level_bounds_ahead_by_the_new_queue(self):
        """30 ahead, the touch comes back with a fresh queue of 4: at most 4 can be ahead."""
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 2, 30, 100),
            (1.0, EVT_INSIDE_ADD, SIDE_BID, 1, 0.0, 1, 2, 1, 100),
            (2.0, EVT_MARKET_ORDER, SIDE_BID, 1, 0.0, 0, 2, 4, 100),  # clears b=1
            (2.0 + 1e-9, EVT_EXHAUSTION, SIDE_BID, 0, 0.0, 0, 2, 4, 100),
            (3.0, EVT_MARKET_ORDER, SIDE_BID, 3, 0.0, 0, 2, 1, 100),  # 4 -> 1 ahead, no fill
            (4.0, EVT_MARKET_ORDER, SIDE_BID, 2, 0.0, -1, 2, 50, 100),  # 1 ahead, then us
        ])
        passive = _passive(replay_pass(log, +1, 1, 0.5, 10.0))
        assert [f.t for f in passive] == [4.0]

    def test_ahead_is_frozen_behind_a_better_level(self):
        """Cancels and market orders at the better level leave our position alone."""
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 2, 10, 100),
            (1.0, EVT_INSIDE_ADD, SIDE_BID, 1, 0.0, 1, 2, 1, 100),
            (2.0, EVT_LIMIT_ADD, SIDE_BID, 5, 0.0, 1, 2, 6, 100),
            (3.0, EVT_CANCEL, SIDE_BID, 1, 0.0, 1, 2, 5, 100),
            (4.0, EVT_MARKET_ORDER, SIDE_BID, 5, 0.0, 0, 2, 50, 100),  # clears b=1
            (4.0 + 1e-9, EVT_EXHAUSTION, SIDE_BID, 0, 0.0, 0, 2, 50, 100),
            (5.0, EVT_MARKET_ORDER, SIDE_BID, 10, 0.0, 0, 2, 40, 100),  # exactly the 10 ahead
            (6.0, EVT_MARKET_ORDER, SIDE_BID, 1, 0.0, 0, 2, 39, 100),  # us
        ])
        passive = _passive(replay_pass(log, +1, 1, 0.5, 20.0))
        assert [f.t for f in passive] == [6.0]


class TestLeadingTheTouch:
    def test_order_left_alone_at_a_cleared_level_fills_on_next_market_order(self):
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 3, 100),
            (1.0, EVT_MARKET_ORDER, SIDE_BID, 3, 0.0, -1, 1, 80, 100),  # exactly the 3 ahead
            (1.0 + 1e-9, EVT_EXHAUSTION, SIDE_BID, 0, 0.0, -1, 1, 80, 100),
            (2.0, EVT_MARKET_ORDER, SIDE_BID, 1, 0.0, -1, 1, 79, 100),
        ])
        passive = _passive(replay_pass(log, +1, 1, 0.5, 10.0))
        assert [(f.t, f.price, f.queue_behind_at_fill) for f in passive] == [(2.0, 0, 0)]

    def test_inside_add_at_our_price_queues_behind_us(self):
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 2, 3, 100),
            (1.0, EVT_MARKET_ORDER, SIDE_BID, 3, 0.0, -1, 2, 80, 100),
            (1.0 + 1e-9, EVT_EXHAUSTION, SIDE_BID, 0, 0.0, -1, 2, 80, 100),
            (2.0, EVT_INSIDE_ADD, SIDE_BID, 1, 0.0, 0, 2, 1, 100),
            (3.0, EVT_LIMIT_ADD, SIDE_BID, 9, 0.0, 0, 2, 10, 100),
            (4.0, EVT_MARKET_ORDER, SIDE_BID, 1, 0.0, 0, 2, 9, 100),
        ])
        passive = _passive(replay_pass(log, +1, 1, 0.5, 10.0))
        assert [f.t for f in passive] == [4.0]


@pytest.mark.parametrize("regime", ["large", "medium", "small"])
def test_no_passive_fill_while_a_better_level_is_in_front(regime):
    """On engine paths the touch on our side is never better than a passive fill's price."""
    checked = 0
    for ep in range(60):
        log = generate_episode(REGIMES[regime], seed=31_000 + ep, warmup=60.0, horizon=60.0)
        result = replay_pass(log, +1, 1, 60.0, 30.0)
        for f in _passive(result):
            pre = log.events[int(np.searchsorted(log.events["t"], f.t)) - 1]
            assert int(pre["b"]) <= f.price
            checked += 1
    assert checked > 10


class TestAheadBound:
    def test_ahead_init_beyond_the_queue_raises(self):
        log = _log([(0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 80, 100)])
        with pytest.raises(ValueError, match="ahead_init"):
            replay_pass_at(log, +1, 1, 0.0, ahead_init=81, timeout=5.0)
        with pytest.raises(ValueError, match="ahead_init"):
            replay_pass_at(log, +1, 1, 0.0, ahead_init=-1, timeout=5.0)
        assert replay_pass_at(log, +1, 1, 0.0, ahead_init=80, timeout=5.0).queue_ahead_t0 == 80

    def test_queue_shrinking_below_ahead_bounds_ahead(self):
        """A cancel past our position that shrinks the queue below ahead pulls ahead down."""
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 6, 100),
            (1.0, EVT_CANCEL, SIDE_BID, 3, 0.9, 0, 1, 3, 100),  # u*Q=5.4 >= ahead 5, Q falls to 3
            (2.0, EVT_MARKET_ORDER, SIDE_BID, 4, 0.0, -1, 1, 70, 100),  # 3 ahead, then us
        ])
        passive = _passive(replay_pass_at(log, +1, 1, 0.5, ahead_init=5, timeout=10.0))
        assert [f.t for f in passive] == [2.0]


class TestCancelStraddle:
    def test_only_lots_in_front_of_us_leave_ahead(self):
        """Cancel of 20 at u=0.49 of Q=100 with ahead=50 removes 1 lot ahead, not 20."""
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 100, 100),
            (1.0, EVT_CANCEL, SIDE_BID, 20, 0.49, 0, 1, 80, 100),
            (2.0, EVT_MARKET_ORDER, SIDE_BID, 49, 0.0, 0, 1, 31, 100),  # exactly the 49 ahead
            (3.0, EVT_MARKET_ORDER, SIDE_BID, 1, 0.0, 0, 1, 30, 100),
        ])
        passive = _passive(replay_pass_at(log, +1, 1, 0.5, ahead_init=50, timeout=10.0))
        assert [f.t for f in passive] == [3.0]

    def test_chunk_wholly_in_front_removes_its_full_size(self):
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 100, 100),
            (1.0, EVT_CANCEL, SIDE_BID, 10, 0.1, 0, 1, 90, 100),  # 10 lots from position 10
            (2.0, EVT_MARKET_ORDER, SIDE_BID, 40, 0.0, 0, 1, 50, 100),  # exactly the 40 left
            (3.0, EVT_MARKET_ORDER, SIDE_BID, 1, 0.0, 0, 1, 49, 100),
        ])
        passive = _passive(replay_pass_at(log, +1, 1, 0.5, ahead_init=50, timeout=10.0))
        assert [f.t for f in passive] == [3.0]


# ---------------------------------------------------------------------------
# Extra tests from review-futures-tca-extra-tests.py (amendments F5)
# ---------------------------------------------------------------------------


class TestMidBeforeTracking:
    def test_mid_before_is_the_mid_after_the_previous_event(self):
        """Mid moves from 1.0 to 0.5 at t=1; the clearing fill at t=2 is measured from 0.5."""
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 2, 10, 100),
            (1.0, EVT_INSIDE_ADD, SIDE_ASK, 1, 0.0, 0, 1, 10, 1),
            (2.0, EVT_MARKET_ORDER, SIDE_BID, 12, 0.0, -1, 1, 80, 1),
            (2.0 + 1e-9, EVT_EXHAUSTION, SIDE_BID, 0, 0.0, -1, 1, 80, 1),
        ])
        fill = _passive(replay_pass(log, +1, 1, 0.5, 10.0))[0]
        assert fill.t == 2.0
        assert fill.mid_before == pytest.approx(0.5)  # not 1.0 (mid at t0), not 0.0 (post-event)


class TestOppositeSideShift:
    def test_opposite_side_shift_puts_a_better_level_in_front(self):
        """An ask exhaustion shifts the whole market: our bid touch goes b 0 -> 1."""
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_BID, 1, 0.0, 0, 3, 10, 100),
            (1.0, EVT_CANCEL, SIDE_ASK, 100, 0.0, 1, 4, 10, 5),
            (1.0 + 1e-9, EVT_EXHAUSTION, SIDE_ASK, 0, 0.0, 1, 4, 10, 5),
            (2.0, EVT_MARKET_ORDER, SIDE_BID, 12, 0.0, 0, 4, 5, 5),  # clears b=1, excess dropped
            (2.0 + 1e-9, EVT_EXHAUSTION, SIDE_BID, 0, 0.0, 0, 4, 5, 5),
        ])
        assert _passive(replay_pass(log, +1, 1, 0.5, 2.5)) == []


class TestCancelPositionFloored:
    def test_cancel_position_is_floored_not_rounded(self):
        """u*Q = 49.6, ahead = 50: the chunk starts at lot 49, so one lot of it is ahead of us."""
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 1, 100, 100),
            (1.0, EVT_CANCEL, SIDE_BID, 20, 0.496, 0, 1, 80, 100),
            (2.0, EVT_MARKET_ORDER, SIDE_BID, 49, 0.0, 0, 1, 31, 100),  # the 49 ahead
            (3.0, EVT_MARKET_ORDER, SIDE_BID, 1, 0.0, 0, 1, 30, 100),
        ])
        passive = _passive(replay_pass_at(log, +1, 1, 0.5, ahead_init=50, timeout=10.0))
        assert [f.t for f in passive] == [3.0]


class TestReturningTouchBounds:
    def test_returning_touch_bounds_ahead_by_exactly_the_new_queue(self):
        """Fresh queue of 6 on return: a market order of exactly 6 is absorbed, 7 reaches us."""
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 2, 30, 100),
            (1.0, EVT_INSIDE_ADD, SIDE_BID, 1, 0.0, 1, 2, 1, 100),
            (2.0, EVT_MARKET_ORDER, SIDE_BID, 1, 0.0, 0, 2, 6, 100),
            (2.0 + 1e-9, EVT_EXHAUSTION, SIDE_BID, 0, 0.0, 0, 2, 6, 100),
            (3.0, EVT_MARKET_ORDER, SIDE_BID, 6, 0.0, -1, 2, 50, 100),  # takes the 6
            (3.0 + 1e-9, EVT_EXHAUSTION, SIDE_BID, 0, 0.0, -1, 2, 50, 100),
            (4.0, EVT_MARKET_ORDER, SIDE_BID, 1, 0.0, -1, 2, 49, 100),
        ])
        passive = _passive(replay_pass_at(log, +1, 1, 0.5, ahead_init=30, timeout=10.0))
        assert [f.t for f in passive] == [4.0]


class TestClearingCancel:
    def test_clearing_cancel_resets_ahead_to_zero(self):
        """A cancel that empties the level removes everything ahead of us."""
        log = _log([
            (0.0, EVT_LIMIT_ADD, SIDE_ASK, 1, 0.0, 0, 2, 10, 100),
            (1.0, EVT_CANCEL, SIDE_BID, 10, 0.5, -1, 2, 40, 100),
            (1.0 + 1e-9, EVT_EXHAUSTION, SIDE_BID, 0, 0.0, -1, 2, 40, 100),
            (2.0, EVT_INSIDE_ADD, SIDE_BID, 1, 0.0, 0, 2, 1, 100),  # new lot behind us
            (3.0, EVT_MARKET_ORDER, SIDE_BID, 1, 0.0, -1, 2, 30, 100),
            (3.0 + 1e-9, EVT_EXHAUSTION, SIDE_BID, 0, 0.0, -1, 2, 30, 100),
        ])
        passive = _passive(replay_pass_at(log, +1, 1, 0.5, ahead_init=5, timeout=10.0))
        assert [f.t for f in passive] == [3.0]
