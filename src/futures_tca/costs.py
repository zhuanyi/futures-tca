"""Cost decomposition, markouts and performance view.

Fills -> cost components (spread, selection, opportunity, fees) in ticks,
positive = paid. Performance view (*_perf_*) = minus cost. Markouts for
chart (b). TCA row assembly for summary.csv.

Sign convention: positive = cost (paid), as in P1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from futures_tca.book import EventLog
from futures_tca.policies import Fill, PolicyResult
from futures_tca.spec import ContractSpec

DEFAULT_MARKOUT_HORIZONS = [1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]


@dataclass
class CostDecomposition:
    """Cost decomposition for a single order.

    All costs in ticks, positive = paid. ``total``, ``opportunity`` and ``fees`` are
    per intended lot; ``spread``, ``selection`` and ``slip_arrival`` are per filled lot.
    """

    spread: float = 0.0  # s * (p_i - m_i), per filled lot
    selection: float = 0.0  # s * (m_i - P_0), per filled lot
    slip_arrival: float = 0.0  # spread + selection, per filled lot
    opportunity: float = 0.0  # s * (Q - q) * (P_c - P_0) / Q
    fees: float = 0.0  # fee_ticks * q / Q
    total: float = 0.0  # (q / Q) * slip_arrival + opportunity + fees

    # Per-fill markouts at various horizons
    markout_cost: dict[float, float] = field(default_factory=dict)

    # Metadata
    side: int = 0  # +1 buy, -1 sell
    qty_total: int = 0
    qty_filled: int = 0
    fill_rate: float = 0.0
    time_to_fill: float = 0.0  # seconds from arrival to the last fill
    n_partial: int = 0
    queue_ahead_t0: int = 0
    queue_behind_at_fill: int = 0
    regime: str = ""
    policy: str = ""
    fill_type: str = ""


def _reference_mid(fill: Fill, log: EventLog) -> float:
    """Mid a fill is measured against: just before the filling event when recorded."""
    return fill.mid_before if fill.mid_before is not None else log.mid_at(fill.t)


def compute_cost(
    result: PolicyResult,
    log: EventLog,
    contract: ContractSpec,
    horizon: float | None = None,
) -> CostDecomposition:
    """Compute cost decomposition for a policy result.

    Parameters
    ----------
    result : PolicyResult
        The policy replay result.
    log : EventLog
        The event log used for replay.
    contract : ContractSpec
        Contract specification for fee conversion.
    horizon : float, optional
        Horizon for close price (default: end of episode).

    Returns
    -------
    CostDecomposition
        Cost components in ticks per lot.
    """
    side = result.side
    Q = result.qty_total
    q = result.qty_filled
    P_0 = result.arrival_mid  # arrival mid

    if horizon is None:
        horizon = log.end_time

    # Close price (mid at end of horizon)
    P_c = log.mid_at(horizon)

    # Fee in ticks of this regime's tick size
    fee_ticks = contract.for_regime(log.regime).fee_ticks()

    # Compute per-fill costs
    total_spread = 0.0
    total_selection = 0.0
    total_qty = 0

    for fill in result.fills:
        p_i = fill.price
        qty_i = fill.qty
        m_i = _reference_mid(fill, log)

        # Spread cost: s * (p_i - m_i)
        spread_i = side * (p_i - m_i)
        # Selection cost: s * (m_i - P_0)
        selection_i = side * (m_i - P_0)

        total_spread += spread_i * qty_i
        total_selection += selection_i * qty_i
        total_qty += qty_i

    # Average over filled quantity
    if total_qty > 0:
        avg_spread = total_spread / total_qty
        avg_selection = total_selection / total_qty
    else:
        avg_spread = 0.0
        avg_selection = 0.0

    slip_arrival = avg_spread + avg_selection

    # Opportunity cost: only policies that leave a residual unfilled (PASS_NOCOMP)
    # have Q - q > 0; completing policies cross the residual, which is a fill.
    if Q > 0:
        opportunity = side * (Q - q) * (P_c - P_0) / Q
    else:
        opportunity = 0.0

    # Fees
    fees = fee_ticks * q / Q if Q > 0 else 0.0

    # Total
    total = (q / Q if Q > 0 else 0.0) * slip_arrival + opportunity + fees

    return CostDecomposition(
        spread=avg_spread,
        selection=avg_selection,
        slip_arrival=slip_arrival,
        opportunity=opportunity,
        fees=fees,
        total=total,
        side=side,
        qty_total=Q,
        qty_filled=q,
        fill_rate=q / Q if Q > 0 else 0.0,
        time_to_fill=result.fill_time - result.arrival_t,
        n_partial=result.n_partial,
        queue_ahead_t0=result.queue_ahead_t0,
        queue_behind_at_fill=result.queue_behind_at_fill,
        regime=log.regime.label,
        policy="",  # set by caller
        fill_type=result.fill_type,
    )


def compute_markout(
    fill: Fill,
    side: int,
    log: EventLog,
    horizon: float,
) -> float:
    """Compute markout cost at a given horizon after a fill.

    markout_cost_h = s * (m_i - m_{t_i + h}) ticks.
    Callers must keep ``fill.t + horizon <= log.end_time`` (``markout_samples``
    does); past the end of the log the last snapshot is used.
    Positive = mid moved against the fill (cost).
    """
    m_i = _reference_mid(fill, log)
    m_h = log.mid_at(fill.t + horizon)
    return side * (m_i - m_h)


def markout_samples(
    result: PolicyResult,
    log: EventLog,
    horizons: list[float] | None = None,
) -> dict[float, list[float]]:
    """Per-fill markout costs of the passive fills, by horizon.

    A fill contributes to a horizon only when ``fill.t + h`` lies inside the
    episode; a legitimate zero markout is kept.
    """
    if horizons is None:
        horizons = DEFAULT_MARKOUT_HORIZONS

    samples: dict[float, list[float]] = {h: [] for h in horizons}
    for fill in result.fills:
        if fill.fill_type != "passive":
            continue
        for h in horizons:
            if fill.t + h <= log.end_time:
                samples[h].append(compute_markout(fill, result.side, log, h))
    return samples


def compute_markouts(
    result: PolicyResult,
    log: EventLog,
    horizons: list[float] | None = None,
) -> dict[float, float]:
    """Mean passive-fill markout cost per horizon; NaN where no fill has data."""
    samples = markout_samples(result, log, horizons)
    return {h: float(np.mean(ms)) if ms else float("nan") for h, ms in samples.items()}


def to_perf(cost: float) -> float:
    """Convert cost to performance view (positive = beat benchmark)."""
    return -cost


def add_perf_columns(cost: CostDecomposition, tick_bps: float) -> dict[str, float]:
    """Add performance view columns to cost decomposition.

    Returns dict with both cost and perf columns.
    """
    result = {
        # Cost columns
        "spread_ticks": cost.spread,
        "selection_ticks": cost.selection,
        "slip_arrival_ticks": cost.slip_arrival,
        "opportunity_ticks": cost.opportunity,
        "fees_ticks": cost.fees,
        "total_ticks": cost.total,
        # Performance columns (negative of cost)
        "spread_perf_ticks": to_perf(cost.spread),
        "selection_perf_ticks": to_perf(cost.selection),
        "slip_arrival_perf_ticks": to_perf(cost.slip_arrival),
        "opportunity_perf_ticks": to_perf(cost.opportunity),
        "fees_perf_ticks": to_perf(cost.fees),
        "total_perf_ticks": to_perf(cost.total),
        # BPS columns
        "total_perf_bps": to_perf(cost.total) * tick_bps,
        "spread_perf_bps": to_perf(cost.spread) * tick_bps,
        "selection_perf_bps": to_perf(cost.selection) * tick_bps,
        "opportunity_perf_bps": to_perf(cost.opportunity) * tick_bps,
        "fees_perf_bps": to_perf(cost.fees) * tick_bps,
        "slip_arrival_perf_bps": to_perf(cost.slip_arrival) * tick_bps,
        # Metadata
        "side": cost.side,
        "qty_total": cost.qty_total,
        "qty_filled": cost.qty_filled,
        "fill_rate": cost.fill_rate,
        "time_to_fill_s": cost.time_to_fill,
        "passive_fills": cost.n_partial,
        "regime": cost.regime,
        "policy": cost.policy,
        "queue_ahead_t0": cost.queue_ahead_t0,
        "queue_behind_at_fill": cost.queue_behind_at_fill,
        "fill_type": cost.fill_type,
    }
    return result
