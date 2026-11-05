"""Tests for book.py: EventLog, generate_episode, invariants, stationarity.

Test numbers: 1, 2, 3, 4, 5, 6, 7.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from futures_tca.book import (
    EVENT_DTYPE,
    EVT_CANCEL,
    EVT_EXHAUSTION,
    EVT_INSIDE_ADD,
    EVT_MARKET_ORDER,
    EventLog,
    generate_episode,
)
from futures_tca.spec import REGIMES, Regime

# ---------------------------------------------------------------------------
# Test 1: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_identical(self):
        """Same seed -> identical EventLog (all arrays equal)."""
        log1 = generate_episode(REGIMES["large"], seed=42)
        log2 = generate_episode(REGIMES["large"], seed=42)
        assert len(log1) == len(log2)
        for field in EVENT_DTYPE.names:
            np.testing.assert_array_equal(
                log1.events[field], log2.events[field],
                err_msg=f"field {field} differs",
            )

    def test_different_seed_different(self):
        """Different seed -> different EventLog."""
        log1 = generate_episode(REGIMES["large"], seed=42)
        log2 = generate_episode(REGIMES["large"], seed=43)
        # At least one field should differ
        any_different = False
        for field in EVENT_DTYPE.names:
            if not np.array_equal(log1.events[field], log2.events[field]):
                any_different = True
                break
        assert any_different, "Different seeds produced identical logs"


# ---------------------------------------------------------------------------
# Test 2: Invariants (hypothesis)
# ---------------------------------------------------------------------------


@st.composite
def regime_and_seed(draw):
    """Generate a (regime_label, seed) pair."""
    label = draw(st.sampled_from(["large", "medium", "small"]))
    seed = draw(st.integers(0, 100_000))
    return label, seed


class TestInvariants:
    @given(data=regime_and_seed())
    @settings(max_examples=50, deadline=10_000)
    def test_book_invariants(self, data):
        """b < a, a-b in [1, s_max], prices integers, Qb,Qa >= 1, times increasing."""
        label, seed = data
        regime = REGIMES[label]
        log = generate_episode(regime, seed=seed, warmup=10.0, horizon=50.0)

        if len(log) == 0:
            return  # trivial case

        evts = log.events

        # Times strictly increasing
        if len(evts) > 1:
            dt = np.diff(evts["t"])
            assert np.all(dt > 0), "Times not strictly increasing"

        # All prices are integers (they're stored as i4, so this is guaranteed
        # by the dtype, but check the spread constraints)
        b = evts["b"]
        a = evts["a"]
        Qb = evts["Qb"]
        Qa = evts["Qa"]
        spread = a - b

        # b < a (spread >= 1)
        assert np.all(spread >= 1), f"Spread < 1: min spread = {spread.min()}"

        # spread <= s_max
        assert np.all(spread <= regime.s_max), (
            f"Spread > s_max: max spread = {spread.max()}, s_max = {regime.s_max}"
        )

        # Qb, Qa >= 1 after every event
        assert np.all(Qb >= 1), f"Qb < 1: min Qb = {Qb.min()}"
        assert np.all(Qa >= 1), f"Qa < 1: min Qa = {Qa.min()}"


# ---------------------------------------------------------------------------
# Test 3: Stationarity
# ---------------------------------------------------------------------------


class TestStationarity:
    @pytest.mark.parametrize("label", ["large", "medium"])
    def test_mean_queue_size(self, label):
        """After warmup, mean best-queue size within +-15% of q_bar over 200 episodes."""
        regime = REGIMES[label]
        n_episodes = 200
        mean_qbs = []
        mean_qas = []

        for i in range(n_episodes):
            log = generate_episode(regime, seed=20261101 + i, warmup=120.0, horizon=120.0)
            pw = log.post_warmup
            if len(pw) == 0:
                continue
            mean_qbs.append(float(pw["Qb"].mean()))
            mean_qas.append(float(pw["Qa"].mean()))

        if not mean_qbs:
            pytest.skip("No post-warmup events")

        overall_mean_Qb = np.mean(mean_qbs)
        overall_mean_Qa = np.mean(mean_qas)

        q_bar = regime.q_bar
        tol = 0.15 * q_bar

        assert abs(overall_mean_Qb - q_bar) < tol, (
            f"{label}: mean Qb = {overall_mean_Qb:.1f}, expected {q_bar} +- {tol}"
        )
        assert abs(overall_mean_Qa - q_bar) < tol, (
            f"{label}: mean Qa = {overall_mean_Qa:.1f}, expected {q_bar} +- {tol}"
        )


# ---------------------------------------------------------------------------
# Test 6: Imbalance effect
# ---------------------------------------------------------------------------


def _next_mid_move(ev: np.ndarray) -> np.ndarray:
    """Sign (+1/-1/0 if none) of the next mid change strictly after each event."""
    mid2 = ev["a"].astype(np.int64) + ev["b"].astype(np.int64)
    changes = np.nonzero(np.diff(mid2) != 0)[0] + 1  # indices where the mid changed
    k = np.searchsorted(changes, np.arange(len(ev)), side="right")
    ok = k < len(changes)
    out = np.zeros(len(ev), dtype=int)
    idx = changes[k[ok]]
    out[ok] = np.sign(mid2[idx] - mid2[idx - 1])
    return out


def _mid_move_counts(regime: Regime, n_episodes: int, seed0: int) -> tuple[int, int, int, int]:
    """(ups, downs, next-up | I>0.5, next-down | I>0.5) over post-warmup events."""
    up = down = up_imb = down_imb = 0
    for i in range(n_episodes):
        ev = generate_episode(regime, seed=seed0 + i, warmup=60.0, horizon=300.0).post_warmup
        d_mid = np.diff(ev["a"].astype(np.int64) + ev["b"].astype(np.int64))
        up += int((d_mid > 0).sum())
        down += int((d_mid < 0).sum())
        nxt = _next_mid_move(ev)
        imb = (ev["Qb"] - ev["Qa"]) / (ev["Qb"] + ev["Qa"])
        up_imb += int(((nxt == 1) & (imb > 0.5)).sum())
        down_imb += int(((nxt == -1) & (imb > 0.5)).sum())
    return up, down, up_imb, down_imb


class TestImbalance:
    def test_kappa_zero_fifty_fifty(self):
        """With kappa=0 the next mid move is up with probability 50% +- 1.5%."""
        regime = replace(REGIMES["large"], kappa=0.0)
        up, down, _, _ = _mid_move_counts(regime, n_episodes=1500, seed0=100_000)
        total = up + down
        assert total > 10_000, f"too few mid moves for a 1.5% tolerance: {total}"
        p_up = up / total
        assert abs(p_up - 0.5) <= 0.015, f"kappa=0: P(up) = {p_up:.3f}, expected 0.5 +- 0.015"

    def test_kappa_positive_imbalance_effect(self):
        """With kappa=0.5 and I > 0.5 the next mid move is up with probability >= 55%."""
        regime = REGIMES["large"]  # kappa = 0.5
        _, _, up_imb, down_imb = _mid_move_counts(regime, n_episodes=300, seed0=200_000)
        total = up_imb + down_imb
        assert total >= 1000, f"too few imbalanced observations: {total}"
        assert up_imb / total >= 0.55, f"P(up | I>0.5) = {up_imb / total:.3f}"

    def test_kappa_raises_imbalance_signal(self):
        """The imbalance signal is stronger with kappa=0.5 than with kappa=0."""
        base = REGIMES["large"]
        p = {}
        for kappa in (0.0, 0.5):
            _, _, u, d = _mid_move_counts(replace(base, kappa=kappa), 300, 300_000)
            p[kappa] = u / (u + d)
        assert p[0.5] > p[0.0]


# ---------------------------------------------------------------------------
# Exhaustion records, inside-add guard, test flag
# ---------------------------------------------------------------------------


class TestExhaustionRecords:
    @pytest.mark.parametrize("label", ["large", "medium", "small"])
    def test_clearing_event_keeps_its_true_kind(self, label):
        """A queue-clearing cancel / market order is logged as such, then a marker."""
        log = generate_episode(REGIMES[label], seed=7, warmup=0.0, horizon=600.0)
        ev = log.events
        marks = np.nonzero(ev["kind"] == EVT_EXHAUSTION)[0]
        assert len(marks) > 0
        assert marks[0] > 0
        prev = ev[marks - 1]
        # The row before every marker is the cancel / market order that cleared the level
        assert np.all(np.isin(prev["kind"], [EVT_CANCEL, EVT_MARKET_ORDER]))
        assert np.all(prev["side"] == ev[marks]["side"])
        assert np.all(prev["size"] > 0)
        assert np.all(ev[marks]["size"] == 0)
        # Same snapshot on both rows (the marker only flags the price move)
        for f in ("b", "a", "Qb", "Qa"):
            np.testing.assert_array_equal(prev[f], ev[marks][f])

    def test_clearing_cancel_carries_u(self):
        log = generate_episode(REGIMES["small"], seed=11, warmup=0.0, horizon=600.0)
        ev = log.events
        marks = np.nonzero(ev["kind"] == EVT_EXHAUSTION)[0]
        prev = ev[marks - 1]
        cancels = prev[prev["kind"] == EVT_CANCEL]
        assert len(cancels) > 0
        assert np.all((cancels["u"] >= 0.0) & (cancels["u"] < 1.0))

    def test_marker_moves_price(self):
        """A bid marker lowers the bid (or lowers the ask when the market shifts)."""
        log = generate_episode(REGIMES["medium"], seed=3, warmup=0.0, horizon=600.0)
        ev = log.events
        marks = np.nonzero(ev["kind"] == EVT_EXHAUSTION)[0]
        before = ev[marks - 2]  # row before the clearing event
        after = ev[marks]
        moved = (before["b"] != after["b"]) | (before["a"] != after["a"])
        assert np.all(moved)


class TestInsideAddGuard:
    def test_inside_add_only_when_spread_wide(self):
        """Inside adds (model rule) happen only when s > 1 before the event."""
        log = generate_episode(REGIMES["small"], seed=5, warmup=0.0, horizon=600.0)
        ev = log.events
        inside = np.nonzero(ev["kind"] == EVT_INSIDE_ADD)[0]
        assert len(inside) > 0
        inside = inside[inside > 0]
        spread_before = (ev["a"] - ev["b"])[inside - 1]
        spread_after = (ev["a"] - ev["b"])[inside]
        assert np.all(spread_before > 1)
        assert np.all(spread_after == spread_before - 1)

    def test_no_inside_add_at_one_tick(self):
        """With inside_rate huge but s_max=1 the spread stays 1 and no inside add occurs."""
        regime = replace(REGIMES["large"], s_max=1, inside_rate=1e6)
        log = generate_episode(regime, seed=1, warmup=0.0, horizon=300.0)
        assert not np.any(log.events["kind"] == EVT_INSIDE_ADD)
        assert np.all(log.events["a"] - log.events["b"] == 1)


class TestNoPriceMoveFlag:
    def test_mid_constant_when_exhaustion_does_not_move_price(self):
        log = generate_episode(
            REGIMES["large"], seed=2, warmup=0.0, horizon=600.0, exhaustion_moves_price=False
        )
        ev = log.events
        assert np.any(ev["kind"] == EVT_EXHAUSTION)
        assert np.all(ev["b"] == ev["b"][0])
        assert np.all(ev["a"] == ev["a"][0])
        assert np.all(ev["Qb"] >= 1) and np.all(ev["Qa"] >= 1)


# ---------------------------------------------------------------------------
# Test 7: Throughput
# ---------------------------------------------------------------------------


class TestThroughput:
    def test_events_per_second(self):
        """>= 50k events/s on the CI runner."""
        regime = REGIMES["large"]
        n_episodes = 20
        total_events = 0

        start = time.monotonic()
        for i in range(n_episodes):
            log = generate_episode(regime, seed=300000 + i)
            total_events += len(log)
        elapsed = time.monotonic() - start

        rate = total_events / elapsed
        assert rate >= 50_000, (
            f"Throughput {rate:.0f} events/s < 50,000 events/s"
        )


# ---------------------------------------------------------------------------
# Additional basic tests
# ---------------------------------------------------------------------------


class TestEventLog:
    def test_eventlog_properties(self):
        log = generate_episode(REGIMES["large"], seed=42)
        assert isinstance(log, EventLog)
        assert log.n_events > 0
        assert log.n_post_warmup > 0
        assert log.n_post_warmup <= log.n_events
        assert log.regime is REGIMES["large"]
        assert log.seed == 42

    def test_post_warmup_mask(self):
        log = generate_episode(REGIMES["large"], seed=42, warmup=10.0, horizon=50.0)
        pw = log.post_warmup
        assert np.all(pw["t"] > 10.0)

    def test_event_dtype(self):
        log = generate_episode(REGIMES["large"], seed=42)
        assert log.events.dtype == EVENT_DTYPE


# ---------------------------------------------------------------------------
# Tests 4 and 5: calibration targets
# Targets carry margin over the stated bounds; they were fixed before the
# calibration run (see scripts/calibrate_add_scale.py).
# ---------------------------------------------------------------------------

_CAL_SEED0 = 7_100_000


def _spread_fractions(regime: Regime, n_episodes: int = 40) -> tuple[float, float, float, float]:
    """Time-weighted P(s == 1), P(s <= 2), P(s >= 3) and the median spread."""
    spreads, dts = [], []
    for i in range(n_episodes):
        log = generate_episode(regime, seed=_CAL_SEED0 + i, warmup=120.0, horizon=300.0)
        ev = log.post_warmup
        spreads.append((ev["a"] - ev["b"]).astype(int))
        dts.append(np.diff(ev["t"], append=log.end_time))
    s, dt = np.concatenate(spreads), np.concatenate(dts)
    total = dt.sum()
    order = np.argsort(s, kind="stable")
    cum = np.cumsum(dt[order]) / total
    median = int(s[order][np.searchsorted(cum, 0.5)])
    return (
        float(dt[s == 1].sum() / total),
        float(dt[s <= 2].sum() / total),
        float(dt[s >= 3].sum() / total),
        float(median),
    )


class TestSpreadTargets:
    """Test 4: spread targets as a fraction of time."""

    def test_large_one_tick(self):
        p1, _, _, _ = _spread_fractions(REGIMES["large"])
        assert p1 >= 0.92, f"LARGE P(s==1) = {p1:.3f}"  # spec: >= 0.90

    def test_medium_at_most_two_ticks(self):
        _, p2, _, _ = _spread_fractions(REGIMES["medium"])
        assert p2 >= 0.75, f"MEDIUM P(s<=2) = {p2:.3f}"  # spec: >= 0.70

    def test_small_wide(self):
        _, _, p3, median = _spread_fractions(REGIMES["small"])
        assert median >= 3, f"SMALL median spread = {median}"  # spec: >= 3
        assert p3 >= 0.55, f"SMALL P(s>=3) = {p3:.3f}"


class TestCrossRegimeComparability:
    """Test 5: 1-minute mid-change sd in bps within a factor of 2 (target 1.7)."""

    def test_one_minute_mid_sd_ratio(self):
        sds = {}
        for label, regime in REGIMES.items():
            moves = []
            for i in range(40):
                log = generate_episode(regime, seed=_CAL_SEED0 + i, warmup=120.0, horizon=600.0)
                t0 = log.post_warmup["t"][0]
                mids = np.array([log.mid_at(t0 + 60.0 * k) for k in range(9)])
                moves.append(np.diff(mids) * regime.tick_bps)
            sds[label] = float(np.concatenate(moves).std())
        ratio = max(sds.values()) / min(sds.values())
        assert ratio <= 1.7, f"1-min mid sd (bps) by regime = {sds}, ratio {ratio:.2f}"
