"""Find the Regime.add_scale that restores the stationary mean best-queue size.

Bisection on add_scale; each trial averages the post-warmup event-mean of Qb/Qa
(over all logged rows, the same convention as the stationarity test)
over fixed seeds (disjoint from the test seeds). Prints the factor per regime.

Usage: python scripts/calibrate_add_scale.py [n_episodes]
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np

from futures_tca.book import generate_episode
from futures_tca.spec import REGIMES

SEED0 = 7_000_000


def mean_queue(regime, n_episodes: int) -> float:
    m = []
    for i in range(n_episodes):
        log = generate_episode(regime, seed=SEED0 + i, warmup=120.0, horizon=120.0)
        pw = log.post_warmup
        if pw.size == 0:
            continue
        m.append(0.5 * (pw["Qb"].mean() + pw["Qa"].mean()))
    if not m:
        raise RuntimeError("no post-warmup events in any calibration episode")
    return float(np.mean(m))


def calibrate(label: str, n_episodes: int) -> float:
    base = REGIMES[label]
    lo, hi = 0.3, 1.2
    if mean_queue(replace(base, add_scale=lo), n_episodes) > base.q_bar or (
        mean_queue(replace(base, add_scale=hi), n_episodes) < base.q_bar
    ):
        raise RuntimeError(f"{label}: root of the mean queue not bracketed by [{lo}, {hi}]")
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        q = mean_queue(replace(base, add_scale=mid), n_episodes)
        if q > base.q_bar:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    for label in REGIMES:
        scale = calibrate(label, n)
        q = mean_queue(replace(REGIMES[label], add_scale=scale), n)
        print(f"{label}: add_scale={scale:.3f} mean_queue={q:.1f} (q_bar={REGIMES[label].q_bar})")
