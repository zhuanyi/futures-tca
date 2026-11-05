"""Queue-reactive limit order book engine.

Generates an EventLog (numpy structured array) via Gillespie sampling of
competing exponential clocks. No strategy logic: policies are replayed
separately on a recorded EventLog (common random numbers).

State machine
--------------------------------
State: (b, a, Qb, Qa) -- best bid/ask in integer ticks, others' queue sizes in lots.
Events, each with its own exponential clock:

1. Limit add at bid/ask: event rate ``add_rate / mo_size`` per side, size
   geometric(mean ``mo_size``), joins the back. ``add_rate`` (lots/s) is chosen
   in ``Regime.add_rate`` so the stationary mean best-queue size is ``q_bar``.
2. Cancel at bid/ask: lot hazard ``cancel_rate`` so the event rate is
   ``cancel_rate * Q / mo_size``; removes a geometric(mean ``mo_size``) chunk
   (assumption: same mean as other order sizes), clipped to the queue, at a
   uniform position ``u`` stored in the log.
3. Market order hit-bid / lift-ask: rate ``mo_rate * (1 -/+ kappa * I)``, size
   geometric(mean ``mo_size``), eats from the front.
4. Inside add when ``s > 1``: rate ``inside_rate * (s - 1)``, side chosen with
   equal probability, new best queue of size 1 one tick inside.
5. Level exhaustion: when a cancel or market order empties the best queue the
   best price moves one tick away and the new queue is a fresh draw from the
   queue-size law (geometric, mean ``q_bar``, >= 1). If the spread would exceed
   ``s_max`` the opposite quote shifts too (the whole market moves). Lots of a
   market order beyond the cleared level are dropped (depth beyond the touch
   is not tracked).

Exhaustion record format
------------------------
The event that empties a queue is always logged with its true kind (CANCEL or
MARKET_ORDER), its true size and ``u``, so replays see the cancel / fill. Its
snapshot is the resolved post-event state (new price, fresh queue, so
``Q >= 1`` holds after every event). It is immediately followed by a zero-size
``EVT_EXHAUSTION`` marker on the same side at the next representable float
time, so that times stay strictly increasing and observers can detect the
price move.

All parameter values are stated assumptions (see README).
"""

from __future__ import annotations

import math

import numpy as np

from futures_tca.spec import Regime

# Event kind codes
EVT_LIMIT_ADD = 0
EVT_CANCEL = 1
EVT_MARKET_ORDER = 2
EVT_INSIDE_ADD = 3
EVT_EXHAUSTION = 4

# Side codes
SIDE_BID = 0
SIDE_ASK = 1

EVENT_DTYPE = np.dtype(
    [
        ("t", "f8"),
        ("kind", "i4"),
        ("side", "i4"),
        ("size", "i4"),
        ("u", "f8"),
        ("b", "i4"),
        ("a", "i4"),
        ("Qb", "i4"),
        ("Qa", "i4"),
    ]
)

_BLOCK = 8192  # random numbers drawn per refill


class EventLog:
    """Container for a simulated episode's event stream.

    Attributes
    ----------
    events : np.ndarray
        Structured array with dtype EVENT_DTYPE, times strictly increasing.
    warmup : float
        Warmup duration in seconds (events before this are discarded for stats).
    horizon : float
        Measurement horizon in seconds after the warmup.
    regime : Regime
        The regime used for this episode.
    seed : int
        The random seed used.
    """

    def __init__(
        self,
        events: np.ndarray,
        warmup: float,
        horizon: float,
        regime: Regime,
        seed: int,
    ) -> None:
        self.events = events
        self.warmup = warmup
        self.horizon = horizon
        self.regime = regime
        self.seed = seed

    def __len__(self) -> int:
        return len(self.events)

    @property
    def end_time(self) -> float:
        """Episode end time: warmup plus measurement horizon."""
        return self.warmup + self.horizon

    @property
    def post_warmup(self) -> np.ndarray:
        """Events after the warmup period."""
        mask = self.events["t"] > self.warmup
        return self.events[mask]

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def n_post_warmup(self) -> int:
        return len(self.post_warmup)

    # -- state lookup ------------------------------------------------------

    def index_at(self, t: float) -> int:
        """Index of the last event at or before ``t`` (clamped to 0)."""
        idx = int(np.searchsorted(self.events["t"], t, side="right")) - 1
        return max(idx, 0)

    def state_at(self, t: float) -> tuple[int, int, int, int]:
        """(b, a, Qb, Qa) after the last event at or before ``t``.

        With an empty log or before the first event the state is the engine's
        initial state (b=0, a=1) or, for a non-empty log, the first snapshot.
        """
        if len(self.events) == 0:
            return (0, 1, 1, 1)
        e = self.events[self.index_at(t)]
        return (int(e["b"]), int(e["a"]), int(e["Qb"]), int(e["Qa"]))

    def mid_at(self, t: float) -> float:
        """Mid price in ticks after the last event at or before ``t``."""
        b, a, _, _ = self.state_at(t)
        return (a + b) / 2.0

    def wmid_at(self, t: float) -> float:
        """Size-weighted mid ``(b*Qa + a*Qb) / (Qa + Qb)`` at ``t``."""
        b, a, Qb, Qa = self.state_at(t)
        q_sum = Qb + Qa
        if q_sum == 0:
            return (a + b) / 2.0
        return (b * Qa + a * Qb) / q_sum


class _Draws:
    """Block-sampled uniforms and exponentials from one generator.

    Drawing in blocks keeps the per-event cost low; the stream is still a
    deterministic function of the seed.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._u = rng.random(_BLOCK)
        self._e = rng.standard_exponential(_BLOCK)
        self._iu = 0
        self._ie = 0

    def uniform(self) -> float:
        if self._iu >= _BLOCK:
            self._u = self._rng.random(_BLOCK)
            self._iu = 0
        v = self._u[self._iu]
        self._iu += 1
        return float(v)

    def exponential(self) -> float:
        if self._ie >= _BLOCK:
            self._e = self._rng.standard_exponential(_BLOCK)
            self._ie = 0
        v = self._e[self._ie]
        self._ie += 1
        return float(v)

    def geometric(self, log1mp: float) -> int:
        """Geometric on {1, 2, ...} given ``log(1 - p)``; ``-inf`` means p = 1."""
        if log1mp == -math.inf:
            return 1
        return 1 + int(math.log(1.0 - self.uniform()) / log1mp)


def _log1mp(mean: float) -> float:
    """``log(1 - p)`` for a geometric law on {1, 2, ...} with the given mean."""
    if mean <= 1.0:
        return -math.inf
    return math.log1p(-1.0 / mean)


def generate_episode(
    regime: Regime,
    seed: int,
    warmup: float = 120.0,
    horizon: float = 900.0,
    exhaustion_moves_price: bool = True,
) -> EventLog:
    """Generate one episode of the queue-reactive book.

    Parameters
    ----------
    regime : Regime
        Tick-regime parameters.
    seed : int
        Random seed for reproducibility.
    warmup : float
        Warmup duration in seconds (default 120).
    horizon : float
        Measurement horizon in seconds after the warmup (default 900); the
        simulation runs for ``warmup + horizon`` seconds.
    exhaustion_moves_price : bool
        Test flag (spec test 19). When False an emptied queue is re-seeded from
        the queue-size law but the best price stays put, so the mid never moves
        and any measured markout is pure replay bias.

    Returns
    -------
    EventLog
        Structured array of events with post-event state snapshots.
    """
    draws = _Draws(np.random.default_rng(seed))
    end_time = warmup + horizon

    q_geom = _log1mp(regime.q_bar)  # fresh-queue law
    sz_geom = _log1mp(regime.mo_size)  # add / cancel / market-order size law
    r_add = regime.add_rate / regime.mo_size  # events per second, per side
    cancel_per_lot = regime.cancel_rate / regime.mo_size
    mo_rate = regime.mo_rate
    kappa = regime.kappa
    s_max = regime.s_max

    # Initial state: spread = 1 tick, queues drawn from the queue-size law
    b, a = 0, 1
    Qb = draws.geometric(q_geom)
    Qa = draws.geometric(q_geom)

    t = 0.0
    rows: list[tuple] = []
    append = rows.append

    while True:
        s = a - b
        imb = (Qb - Qa) / (Qb + Qa)

        r_cancel_bid = cancel_per_lot * Qb
        r_cancel_ask = cancel_per_lot * Qa
        r_hit = mo_rate * max(0.0, 1.0 - kappa * imb)
        r_lift = mo_rate * max(0.0, 1.0 + kappa * imb)
        r_inside = regime.inside_rate * (s - 1)

        total_rate = 2.0 * r_add + r_cancel_bid + r_cancel_ask + r_hit + r_lift + r_inside
        t += draws.exponential() / total_rate
        if t >= end_time:
            break

        x = draws.uniform() * total_rate

        if x < r_add:  # 1a. limit add at bid
            size = draws.geometric(sz_geom)
            Qb += size
            append((t, EVT_LIMIT_ADD, SIDE_BID, size, 0.0, b, a, Qb, Qa))
            continue
        x -= r_add
        if x < r_add:  # 1b. limit add at ask
            size = draws.geometric(sz_geom)
            Qa += size
            append((t, EVT_LIMIT_ADD, SIDE_ASK, size, 0.0, b, a, Qb, Qa))
            continue
        x -= r_add

        if x < r_cancel_bid:  # 2a. cancel at bid
            kind, side = EVT_CANCEL, SIDE_BID
            u = draws.uniform()
            size = min(draws.geometric(sz_geom), Qb)
        elif x < r_cancel_bid + r_cancel_ask:  # 2b. cancel at ask
            kind, side = EVT_CANCEL, SIDE_ASK
            u = draws.uniform()
            size = min(draws.geometric(sz_geom), Qa)
        else:
            x -= r_cancel_bid + r_cancel_ask
            if x < r_hit:  # 3a. market order hits the bid (seller)
                kind, side, u = EVT_MARKET_ORDER, SIDE_BID, 0.0
                size = draws.geometric(sz_geom)
            elif x < r_hit + r_lift:  # 3b. market order lifts the ask (buyer)
                kind, side, u = EVT_MARKET_ORDER, SIDE_ASK, 0.0
                size = draws.geometric(sz_geom)
            else:  # 4. inside add
                if draws.uniform() < 0.5:
                    b += 1
                    Qb = 1
                    append((t, EVT_INSIDE_ADD, SIDE_BID, 1, 0.0, b, a, Qb, Qa))
                else:
                    a -= 1
                    Qa = 1
                    append((t, EVT_INSIDE_ADD, SIDE_ASK, 1, 0.0, b, a, Qb, Qa))
                continue

        # Cancel or market order on `side`: remove lots from the best queue.
        exhausted = False
        if side == SIDE_BID:
            Qb -= min(size, Qb)
            exhausted = Qb == 0
            if exhausted:
                if exhaustion_moves_price:
                    b -= 1
                Qb = draws.geometric(q_geom)
                if a - b > s_max:
                    a -= 1  # whole market moves
        else:
            Qa -= min(size, Qa)
            exhausted = Qa == 0
            if exhausted:
                if exhaustion_moves_price:
                    a += 1
                Qa = draws.geometric(q_geom)
                if a - b > s_max:
                    b += 1
        append((t, kind, side, size, u, b, a, Qb, Qa))
        if exhausted:
            t = math.nextafter(t, math.inf)
            append((t, EVT_EXHAUSTION, side, 0, 0.0, b, a, Qb, Qa))

    if rows:
        events = np.array(rows, dtype=EVENT_DTYPE)
    else:
        events = np.empty(0, dtype=EVENT_DTYPE)

    return EventLog(events=events, warmup=warmup, horizon=horizon, regime=regime, seed=seed)
