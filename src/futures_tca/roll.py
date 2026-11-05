"""Roll-cost methodology.

OU richness process, calendar/leg quote model, liquidity migration profile.
Roll methods: CAL_CROSS, LEG_SIMUL, LEG_SEQ, FIXED_DAY, THRESH. All aggressive
in v1 (quote-based, no passive calendar queue).

Truth model:
- Fair calendar spread s* = c + z_t, c = carry, z_t = OU richness
- Market calendar mid m_t = s*_t + e_t, e_t = OU dealer skew
- Liquidity migration rho_d = logistic((d - d_roll) / w)
- Outright mid: front follows random walk, next = front + m_t

Covered by tests 20-23.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# OU process
# ---------------------------------------------------------------------------


def simulate_ou(
    n_steps: int,
    dt: float,
    mean: float,
    half_life: float,
    stationary_sd: float,
    rng: np.random.Generator,
    x0: float | None = None,
) -> np.ndarray:
    """Simulate an Ornstein-Uhlenbeck process using exact discretization.

    x[t] = mu + (x[t-1] - mu) * exp(-theta * dt) + sigma_e * Z
    where theta = log(2) / half_life,
          sigma_e = stationary_sd * sqrt(1 - exp(-2 * theta * dt)).

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    dt : float
        Time step size in seconds.
    mean : float
        Long-run mean (mu).
    half_life : float
        Half-life in seconds.
    stationary_sd : float
        Stationary standard deviation.
    rng : np.random.Generator
        Random number generator.
    x0 : float, optional
        Initial value (default: mean).

    Returns
    -------
    np.ndarray
        Array of shape (n_steps,) with the process path.
    """
    theta = np.log(2) / half_life
    # Exact discretization: sigma_e is the std of the innovation
    sigma_e = stationary_sd * np.sqrt(1 - np.exp(-2 * theta * dt))
    decay = np.exp(-theta * dt)

    if x0 is None:
        x0 = mean

    x = np.empty(n_steps)
    x[0] = x0

    Z = rng.standard_normal(n_steps - 1)
    for t in range(1, n_steps):
        x[t] = mean + (x[t - 1] - mean) * decay + sigma_e * Z[t - 1]

    return x


# ---------------------------------------------------------------------------
# Liquidity migration
# ---------------------------------------------------------------------------


# d_roll and w defaults are assumptions (README, "Roll assumptions").
def liquidity_migration(d: float, d_roll: float = -5.0, w: float = 2.0) -> float:
    """Liquidity migration profile: fraction of volume in next contract.

    rho_d = logistic((d - d_roll) / w)
    where d is days to expiry (negative = before expiry).

    Parameters
    ----------
    d : float or array
        Days relative to expiry (e.g. -10 = 10 days before expiry).
    d_roll : float
        Midpoint of migration (default: -5, i.e. 5 days before expiry).
    w : float
        Width parameter (default: 2 days).

    Returns
    -------
    float or array
        rho_d in [0, 1].
    """
    return 1.0 / (1.0 + np.exp(-(d - d_roll) / w))


# ---------------------------------------------------------------------------
# Quote widths (assumptions, marked as such)
# ---------------------------------------------------------------------------


# Calendar book width probabilities by rho_d (assumption)
# When rho_d is high (next contract is liquid), calendar spread is tight
_CAL_WIDTH_PROBS_LOW_RHO = {1: 0.2, 2: 0.5, 3: 0.3}  # illiquid next
_CAL_WIDTH_PROBS_HIGH_RHO = {1: 0.6, 2: 0.3, 3: 0.1}  # liquid next


def calendar_quote_width(rho: float, rng: np.random.Generator) -> int:
    """Draw calendar spread quote width based on liquidity migration.

    Interpolates between low-rho and high-rho probability tables.
    """
    # Interpolate probabilities
    probs = {}
    for w in [1, 2, 3]:
        p_low = _CAL_WIDTH_PROBS_LOW_RHO[w]
        p_high = _CAL_WIDTH_PROBS_HIGH_RHO[w]
        probs[w] = p_low * (1 - rho) + p_high * rho

    # Normalize
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}

    # Draw
    u = rng.uniform()
    cum = 0.0
    for w, p in sorted(probs.items()):
        cum += p
        if u < cum:
            return w
    return 3


def front_leg_width() -> int:
    """Front leg quote width: always 1 tick (large-tick assumption)."""
    return 1


def next_leg_width(rho: float, rng: np.random.Generator) -> int:
    """Next leg quote width based on liquidity migration (assumption)."""
    # When rho is high, next contract is liquid -> width 1
    # When rho is low, next contract is illiquid -> width 1-3
    if rho > 0.7:
        return 1
    elif rho > 0.4:
        return int(rng.choice([1, 2], p=[0.7, 0.3]))
    else:
        return int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))


# ---------------------------------------------------------------------------
# Roll methods
# ---------------------------------------------------------------------------


@dataclass
class RollResult:
    """Result of a roll execution."""

    method: str
    day: float  # day relative to expiry
    cost_ticks: float  # cost vs fair value (positive = paid)
    fair_spread: float  # fair calendar spread at roll time
    market_spread: float  # market calendar mid at roll time
    richness: float  # market - fair (positive = rich)


def roll_cal_cross(
    fair_spread: float,
    market_spread: float,
    cal_width: int,
    side: int,
) -> float:
    """Cross the calendar book.

    Cost = cal_width / 2 + side * (market_spread - fair_spread)
    For buyer of next (side=+1): cost = w/2 + (m - s*)
    For seller of next (side=-1): cost = w/2 - (m - s*)
    """
    return cal_width / 2.0 + side * (market_spread - fair_spread)


def roll_leg_simul(
    fair_spread: float,
    market_spread: float,
    front_width: int,
    next_width: int,
    side: int,
) -> float:
    """Leg both outright contracts simultaneously.

    Cost = front_width/2 + next_width/2 + side * (market_spread - fair_spread)
    """
    return (front_width + next_width) / 2.0 + side * (market_spread - fair_spread)


def roll_leg_seq(
    fair_spread: float,
    market_spread: float,
    front_width: int,
    next_width: int,
    side: int,
    delta: float,
    sigma_leg: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Leg sequentially with latency delta.

    Cost = leg_simul cost + legging error
    legging error ~ N(0, sigma_leg^2 * delta)
    Returns (cost, legging_error).
    """
    base_cost = roll_leg_simul(
        fair_spread, market_spread, front_width, next_width, side
    )
    legging_error = rng.normal(0, sigma_leg * np.sqrt(delta))
    return base_cost + legging_error, legging_error


SECONDS_PER_DAY = 24 * 3600.0


def simulate_roll_episode(
    n_days: int = 11,
    day_start: float = -10.0,
    day_end: float = 0.0,
    carry: float = 2.0,  # assumption
    richness_half_life: float = 4 * 3600,  # assumption: 4 hours in seconds
    richness_sd: float = 0.4,  # assumption
    skew_half_life: float = 20 * 60,  # assumption: 20 minutes in seconds
    skew_sd: float = 0.25,  # assumption
    seed: int = 20261101,
) -> dict[str, np.ndarray]:
    """Simulate one roll episode sampled once per trading day.

    Both OU processes use the exact transition of ``simulate_ou`` over a one-day
    step and start from their stationary law. With half-lives of 4 h and 20 min
    the one-day autocorrelation is ~0.016 and ~0, so the daily samples are
    effectively independent draws with the stated stationary sd (a consequence of
    the spec parameters, not of the discretisation). The random draws are made
    in a fixed order (z, e, calendar widths, next-leg widths); reordering them
    changes every episode. Returns arrays per day:

    - days: day relative to expiry
    - fair_spread: fair calendar spread ``carry + z``
    - market_spread: market calendar mid ``fair + e``
    - richness: market - fair (= the dealer skew ``e``; positive = rich)
    - rho: liquidity migration
    - cal_width, front_width, next_width: quote widths in ticks drawn from the
      rho-dependent tables
    """
    rng = np.random.default_rng(seed)
    days = np.linspace(day_start, day_end, n_days)

    step = SECONDS_PER_DAY * (days[1] - days[0]) if n_days > 1 else SECONDS_PER_DAY
    z = simulate_ou(
        n_days, step, 0.0, richness_half_life, richness_sd, rng,
        x0=rng.normal(0.0, richness_sd),
    )
    e = simulate_ou(
        n_days, step, 0.0, skew_half_life, skew_sd, rng,
        x0=rng.normal(0.0, skew_sd),
    )

    fair_spread = carry + z
    market_spread = fair_spread + e
    rho = liquidity_migration(days)

    cal_width = np.array([calendar_quote_width(r, rng) for r in rho])
    front_width = np.full(n_days, front_leg_width())
    next_width = np.array([next_leg_width(r, rng) for r in rho])

    return {
        "days": days,
        "fair_spread": fair_spread,
        "market_spread": market_spread,
        "richness": market_spread - fair_spread,
        "rho": rho,
        "cal_width": cal_width,
        "front_width": front_width,
        "next_width": next_width,
    }


def _day_index(episode: dict[str, np.ndarray], day: float) -> int:
    return int(np.argmin(np.abs(episode["days"] - day)))


def roll_fixed_day(episode: dict[str, np.ndarray], day: float, side: int = 1) -> float:
    """FIXED_DAY(d): cross the calendar book on the day closest to ``day``."""
    i = _day_index(episode, day)
    return roll_cal_cross(
        episode["fair_spread"][i], episode["market_spread"][i],
        int(episode["cal_width"][i]), side,
    )


def roll_thresh(
    episode: dict[str, np.ndarray],
    k: float,
    sigma_eps: float,
    rng: np.random.Generator,
    side: int = 1,
    forced_day: float = -2.0,  # assumption
) -> tuple[float, float]:
    """THRESH(k, eps): cross the calendar book on the first "cheap" day.

    The desk sees ``m - (s* + eps)`` with ``eps ~ N(0, sigma_eps^2)`` drawn
    independently each day. For a buyer (``side=+1``) the day is cheap when the
    estimated richness is at most ``-k``; for a seller when it is at least
    ``+k``. If no day qualifies the roll is forced on ``forced_day``.

    Returns ``(cost_ticks, day_rolled)``; the cost uses the true fair spread.
    """
    days = episode["days"]
    fair = episode["fair_spread"]
    market = episode["market_spread"]
    eps = rng.normal(0.0, sigma_eps, size=len(days)) if sigma_eps > 0 else np.zeros(len(days))
    est_richness = market - (fair + eps)

    forced_i = _day_index(episode, forced_day)
    chosen = forced_i
    for i in range(forced_i):
        if side * est_richness[i] <= -k:
            chosen = i
            break
    cost = roll_cal_cross(
        fair[chosen], market[chosen], int(episode["cal_width"][chosen]), side
    )
    return cost, float(days[chosen])


def thresh_saving(
    k: float,
    sigma_eps: float,
    n_episodes: int,
    seed: int = 20261101,
    fixed_day: float = -5.0,
    side: int = 1,
) -> np.ndarray:
    """Per-episode saving of THRESH over FIXED_DAY (positive = THRESH cheaper)."""
    savings = np.empty(n_episodes)
    for i in range(n_episodes):
        ep = simulate_roll_episode(seed=seed + i)
        rng = np.random.default_rng([seed, i, 1])
        cost_t, _ = roll_thresh(ep, k, sigma_eps, rng, side=side)
        savings[i] = roll_fixed_day(ep, fixed_day, side) - cost_t
    return savings
