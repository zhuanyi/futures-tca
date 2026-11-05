"""Execution policies replayed on an EventLog.

Price-taker replay: AGG (cross immediately), PASS(T) (post
then timeout), PASS_AT(a, T) (controlled queue-ahead experiment), PASS_NOCOMP
(no completion), PASS_IMB (adverse-selection avoider). All policies are
deterministic given the EventLog, and none changes the log (no interaction).

Replay rules for a resting order on one side:

* The book engine's ``Qb`` / ``Qa`` count *others'* lots only. ``ahead`` is the
  number of others' lots in front of us.
* A cancel at position ``u`` hits our ``ahead`` when ``u * Q_before < ahead``,
  where ``Q_before`` is the queue size just before that event (the previous
  event's snapshot, tracked as the replay advances).
* A market order on our side eats ``ahead`` first, then us, then whatever is
  behind. The engine logs a queue-clearing cancel / market order with its true
  kind and size, so replay sees it like any other event; the zero-size
  ``EVT_EXHAUSTION`` marker that follows it is ignored. After such an event the
  lots ahead of us are gone, so ``ahead`` falls to 0 (a clearing market order
  already filled us if any lots reached us).
* Events apply according to where the engine's touch on our side is relative to
  our price just before the event:

  - Touch at our price: the rules above.
  - Touch better than our price (an inside add or a whole-market shift): the event
    is at another level and does not touch our order. Lots of a market order beyond
    a cleared level are dropped, as in the engine.
  - Touch worse than our price (our level cleared and we stayed): we are the best
    price, so any market order on our side fills us. Cancels do not concern us.

* ``ahead`` never exceeds the engine's queue while the touch is at our price. This
  also sets our position when the touch returns to our level with a fresh queue.
* Passive fills record the mid just before the filling event in ``mid_before``.

Covered by tests 8-15.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from futures_tca.book import (
    EVT_CANCEL,
    EVT_INSIDE_ADD,
    EVT_LIMIT_ADD,
    EVT_MARKET_ORDER,
    SIDE_ASK,
    SIDE_BID,
    EventLog,
)

_WALK_MAX_LEVELS = 50


@dataclass
class Fill:
    """A single fill event."""

    t: float  # fill time
    price: int  # fill price in ticks
    qty: int  # fill quantity (lots)
    fill_type: str  # "passive", "cross_timeout", "aggressive"
    queue_ahead_t0: int = 0  # others' lots ahead of us when the order was posted
    queue_behind_at_fill: int = 0  # others' lots behind us right after a passive fill
    mid_before: float | None = None  # mid just before the filling event (passive fills)


@dataclass
class PolicyResult:
    """Result of replaying a policy on an EventLog."""

    fills: list[Fill] = field(default_factory=list)
    side: int = 0  # +1 buy, -1 sell
    qty_total: int = 0  # intended quantity
    qty_filled: int = 0  # executed quantity
    arrival_t: float = 0.0  # arrival time
    arrival_mid: float = 0.0  # arrival mid price
    wmid_arrival: float = 0.0  # weighted mid at arrival
    fill_type: str = ""  # dominant fill type
    n_partial: int = 0  # number of passive fills
    queue_ahead_t0: int = 0  # others' lots ahead at arrival (0 for AGG)

    @property
    def fill_rate(self) -> float:
        """Fraction of intended quantity filled."""
        return self.qty_filled / self.qty_total if self.qty_total > 0 else 0.0

    @property
    def fill_time(self) -> float:
        """Time of last fill (or arrival_t if no fills)."""
        if self.fills:
            return self.fills[-1].t
        return self.arrival_t

    @property
    def queue_behind_at_fill(self) -> int:
        """Lots behind us at the last passive fill (0 if there was none)."""
        for f in reversed(self.fills):
            if f.fill_type == "passive":
                return f.queue_behind_at_fill
        return 0


def _new_result(log: EventLog, side: int, qty: int, t0: float) -> PolicyResult:
    result = PolicyResult(side=side, qty_total=qty, arrival_t=t0)
    result.arrival_mid = log.mid_at(t0)
    result.wmid_arrival = log.wmid_at(t0)
    return result


def _finish(result: PolicyResult, fills: list[Fill]) -> PolicyResult:
    result.fills = fills
    result.qty_filled = sum(f.qty for f in fills)
    result.n_partial = sum(1 for f in fills if f.fill_type == "passive")
    if not fills:
        result.fill_type = "unfilled"
    elif any(f.fill_type == "cross_timeout" for f in fills):
        result.fill_type = "cross_timeout"
    else:
        result.fill_type = fills[-1].fill_type
    return result


def _fresh_depth(rng: np.random.Generator, q_bar: float) -> int:
    """Depth of an untracked level: geometric on {1, 2, ...} with mean ``q_bar``."""
    if q_bar <= 1.0:
        return 1
    return int(rng.geometric(1.0 / q_bar))


def replay_agg(
    log: EventLog,
    side: int,
    qty: int,
    t0: float,
) -> PolicyResult:
    """Aggressive policy: cross immediately at t0, walking the book.

    Takes ``min(qty, Q_far)`` at the touch and the remainder at the next levels
    (one tick further each) whose depths are fresh draws from the queue-size law
    (geometric, mean ``q_bar``). The draws come from a generator seeded by the
    log seed and ``t0``, so the result is a pure function of ``(log, t0)``.
    """
    result = _new_result(log, side, qty, t0)
    b, a, Qb, Qa = log.state_at(t0)
    price, available = (a, Qa) if side > 0 else (b, Qb)
    step = 1 if side > 0 else -1

    rng = np.random.default_rng([abs(int(log.seed)), int(round(t0 * 1_000_000))])
    left = qty
    fills: list[Fill] = []
    for level in range(_WALK_MAX_LEVELS):
        take = min(left, available)
        if take > 0:
            fills.append(
                Fill(t=t0, price=price + step * level, qty=take, fill_type="aggressive")
            )
        left -= take
        if left <= 0:
            break
        available = _fresh_depth(rng, log.regime.q_bar)
    else:  # pragma: no cover - needs a size > 50 levels deep
        raise RuntimeError("aggressive order exhausted the modelled depth")

    return _finish(result, fills)


def _queue_before(log: EventLog, idx: int, side_q: str) -> int:
    """Queue size on a side just before event ``idx`` (previous snapshot).

    For ``idx == 0`` (hand-built logs, or a replay starting before the first
    engine event) the pre-event queue is reconstructed from the event itself,
    which is exact except when that first event cleared its level.
    """
    evts = log.events
    if idx > 0:
        return int(evts[idx - 1][side_q])
    # No earlier snapshot: reconstruct from this event's own post-event state.
    e = evts[idx]
    q_post = int(e[side_q])
    if e["kind"] in (EVT_LIMIT_ADD, EVT_INSIDE_ADD):
        return max(0, q_post - int(e["size"]))
    return q_post + int(e["size"])


def _replay_resting(
    log: EventLog,
    side: int,
    qty: int,
    t0: float,
    ahead: int,
    timeout: float,
    post_price: int,
    i_thr: float | None = None,
) -> tuple[list[Fill], int, float, bool]:
    """Follow a resting order through the events in ``(t0, t0 + timeout]``.

    Returns ``(passive_fills, qty_left, exit_t, early_exit)`` where ``exit_t`` is
    the time the order stops resting (the timeout, or the event time of an
    imbalance-triggered early exit).

    For a fill while the order leads the engine's touch, ``mid_before`` is the
    engine's mid, which does not include the order's own quote.
    """
    evts = log.events
    lo = int(np.searchsorted(evts["t"], t0, side="right"))
    hi = int(np.searchsorted(evts["t"], t0 + timeout, side="right"))

    our_side = SIDE_BID if side > 0 else SIDE_ASK
    side_q, best_col = ("Qb", "b") if side > 0 else ("Qa", "a")
    fills: list[Fill] = []
    qty_left = qty
    ahead0 = ahead
    best_pre = post_price  # engine touch on our side just before the event
    mid_pre = log.mid_at(t0)  # mid just before the event

    for idx in range(lo, hi):
        evt = evts[idx]
        kind = evt["kind"]
        size = int(evt["size"])
        level = side * (best_pre - post_price)  # > 0 better level in front, 0 at touch, < 0 we lead

        if evt["side"] == our_side and kind in (EVT_CANCEL, EVT_MARKET_ORDER) and level <= 0:
            q_pre = _queue_before(log, idx, side_q)
            cleared = size >= q_pre  # the event emptied the others' queue

            if level < 0:
                # Nothing in the engine's queue is at or in front of our price.
                fill_qty = min(qty_left, size) if kind == EVT_MARKET_ORDER else 0
                behind = 0
            elif kind == EVT_CANCEL:
                first_hit = int(float(evt["u"]) * q_pre)
                if first_hit < ahead:
                    ahead -= min(size, ahead - first_hit)
                fill_qty = 0
                behind = 0
            else:
                consumed = min(ahead, size)
                ahead -= consumed
                fill_qty = min(qty_left, size - consumed)
                behind_pre = max(0, q_pre - ahead - consumed)
                behind = 0 if cleared else max(0, behind_pre - (size - consumed - fill_qty))

            if fill_qty > 0:
                fills.append(Fill(
                    t=float(evt["t"]),
                    price=post_price,
                    qty=fill_qty,
                    fill_type="passive",
                    queue_ahead_t0=ahead0,
                    queue_behind_at_fill=behind,
                    mid_before=mid_pre,
                ))
                qty_left -= fill_qty
                if qty_left <= 0:
                    break
            if cleared:
                ahead = 0

        best_pre = int(evt[best_col])
        mid_pre = (int(evt["a"]) + int(evt["b"])) / 2.0
        if best_pre == post_price:
            # The touch is at our level: the engine's queue bounds what can be ahead of
            # us. This also covers the touch returning to our level with a fresh queue.
            ahead = min(ahead, int(evt[side_q]))

        # Imbalance exit is tested after the event's own effect (a fill on this
        # very event is kept), on the event's post-event snapshot.
        if i_thr is not None:
            q_sum = int(evt["Qb"]) + int(evt["Qa"])
            imb = (int(evt["Qb"]) - int(evt["Qa"])) / q_sum if q_sum > 0 else 0.0
            # Buyer is hurt by an ask-heavy book, seller by a bid-heavy one.
            if (side > 0 and imb < -i_thr) or (side < 0 and imb > i_thr):
                return fills, qty_left, float(evt["t"]), True

    return fills, qty_left, t0 + timeout, False


def _cross_residual(
    log: EventLog, side: int, qty_left: int, exit_t: float, ahead0: int
) -> Fill:
    b_t, a_t, _, _ = log.state_at(exit_t)
    return Fill(
        t=exit_t,
        price=a_t if side > 0 else b_t,
        qty=qty_left,
        fill_type="cross_timeout",
        queue_ahead_t0=ahead0,
    )


def _post_state(log: EventLog, side: int, t0: float) -> tuple[int, int]:
    """(post price, queue ahead if joining the back) for a passive order at t0."""
    b, a, Qb, Qa = log.state_at(t0)
    return (b, Qb) if side > 0 else (a, Qa)


def replay_pass(
    log: EventLog,
    side: int,
    qty: int,
    t0: float,
    timeout: float,
    complete: bool = True,
) -> PolicyResult:
    """Passive policy: post at the touch at t0, cancel the residual at t0+timeout.

    If ``complete`` the residual is crossed at the timeout (PASS_T); otherwise it
    is just cancelled (PASS_NOCOMP).
    """
    result = _new_result(log, side, qty, t0)
    post_price, ahead = _post_state(log, side, t0)
    result.queue_ahead_t0 = ahead

    fills, qty_left, exit_t, _ = _replay_resting(
        log, side, qty, t0, ahead, timeout, post_price
    )
    if qty_left > 0 and complete:
        fills.append(_cross_residual(log, side, qty_left, exit_t, ahead))
    return _finish(result, fills)


def replay_pass_at(
    log: EventLog,
    side: int,
    qty: int,
    t0: float,
    ahead_init: int,
    timeout: float,
) -> PolicyResult:
    """Controlled experiment: assume the order already rests with ``ahead_init`` ahead at t0.

    Raises ValueError unless ``0 <= ahead_init <= queue at t0``: lots ahead of us must exist.
    """
    result = _new_result(log, side, qty, t0)
    post_price, queue = _post_state(log, side, t0)
    if not 0 <= ahead_init <= queue:
        raise ValueError(f"ahead_init must be in [0, {queue}] at t0={t0}, got {ahead_init}")
    result.queue_ahead_t0 = ahead_init

    fills, qty_left, exit_t, _ = _replay_resting(
        log, side, qty, t0, ahead_init, timeout, post_price
    )
    if qty_left > 0:
        fills.append(_cross_residual(log, side, qty_left, exit_t, ahead_init))
    return _finish(result, fills)


def replay_pass_imb(
    log: EventLog,
    side: int,
    qty: int,
    t0: float,
    timeout: float,
    i_thr: float,
) -> PolicyResult:
    """Passive with adverse-selection avoidance: cancel and cross early on adverse imbalance."""
    result = _new_result(log, side, qty, t0)
    post_price, ahead = _post_state(log, side, t0)
    result.queue_ahead_t0 = ahead

    fills, qty_left, exit_t, _ = _replay_resting(
        log, side, qty, t0, ahead, timeout, post_price, i_thr=i_thr
    )
    if qty_left > 0:
        fills.append(_cross_residual(log, side, qty_left, exit_t, ahead))
    return _finish(result, fills)
