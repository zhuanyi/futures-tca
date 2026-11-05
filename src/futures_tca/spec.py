"""Contract specification and tick-regime parameters.

Defines ContractSpec (frozen dataclass) for a generic 10y-note-like futures
contract and Regime (LARGE, MEDIUM, SMALL) with queue-reactive book parameters.
Conversions between ticks, bps, currency and DV01 live here.

All parameter values are stated assumptions, not exchange specifications
(strictly synthetic, no vendor data).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# ---------------------------------------------------------------------------
# Contract specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractSpec:
    """Generic futures contract specification.

    All numeric parameters are stated assumptions for a synthetic 10y-note-like
    contract; they are not exchange specifications.

    Attributes
    ----------
    name : str
        Generic label, e.g. "10y-note-like".
    tick_bps : float
        Tick size in basis points of price.
    tick_value : float
        Currency value of one tick move per lot (assumption).
    dv01_per_lot : float
        Currency value of a 1 bp yield move per lot (assumption).
    fee_per_lot : float
        One-side fee in currency (assumption).
    """

    name: str = "10y-note-like"
    tick_bps: float = 1.4
    tick_value: float = 15.6
    dv01_per_lot: float = 60.0
    fee_per_lot: float = 1.0

    def __post_init__(self) -> None:
        for field in ("tick_bps", "tick_value", "dv01_per_lot"):
            val = getattr(self, field)
            if not np.isfinite(val) or val <= 0:
                raise ValueError(f"{field} must be positive and finite, got {val}")
        if not np.isfinite(self.fee_per_lot) or self.fee_per_lot < 0:
            raise ValueError(f"fee_per_lot must be non-negative, got {self.fee_per_lot}")
        if not self.name:
            raise ValueError("name must be a non-empty string")

    # -- conversions -------------------------------------------------------

    def ticks_to_bps(self, ticks: float) -> float:
        """Convert a price distance in ticks to basis points."""
        return ticks * self.tick_bps

    def bps_to_ticks(self, bps: float) -> float:
        """Convert a price distance in basis points to ticks."""
        return bps / self.tick_bps

    def ticks_to_currency(self, ticks: float) -> float:
        """Convert a price distance in ticks to currency per lot."""
        return ticks * self.tick_value

    def currency_to_ticks(self, currency: float) -> float:
        """Convert a currency amount per lot to ticks."""
        return currency / self.tick_value

    def bps_to_currency(self, bps: float) -> float:
        """Convert a price distance in bps to currency per lot."""
        return self.ticks_to_currency(self.bps_to_ticks(bps))

    def currency_to_bps(self, currency: float) -> float:
        """Convert a currency amount per lot to bps."""
        return self.ticks_to_bps(self.currency_to_ticks(currency))

    def fee_ticks(self) -> float:
        """Fee expressed in ticks per lot."""
        return self.fee_per_lot / self.tick_value

    def yield_bp_to_ticks(self, yield_bp: float) -> float:
        """Convert a yield move in bp to price ticks.

        One bp of yield moves the position by ``dv01_per_lot`` currency and one
        tick is worth ``tick_value`` currency, so 1 bp of yield is
        ``dv01_per_lot / tick_value`` ticks.
        """
        return yield_bp * self.dv01_per_lot / self.tick_value

    def ticks_to_yield_bp(self, ticks: float) -> float:
        """Convert price ticks to a yield move in bp (inverse of ``yield_bp_to_ticks``)."""
        return ticks * self.tick_value / self.dv01_per_lot

    def for_regime(self, regime: Regime) -> ContractSpec:
        """Contract rescaled to a regime's tick size.

        The tick is ``regime.tick_bps`` of price, so the currency value of a tick
        scales by ``regime.tick_bps / tick_bps`` (fees in ticks follow).
        """
        scale = regime.tick_bps / self.tick_bps
        return replace(self, tick_bps=regime.tick_bps, tick_value=self.tick_value * scale)


# Default contract spec used throughout the project.
DEFAULT_CONTRACT = ContractSpec()


# ---------------------------------------------------------------------------
# Tick-regime parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Regime:
    """Queue-reactive book regime parameters.

    All values are stated assumptions for three synthetic tick regimes
    (large, medium, small). Times in seconds, sizes in lots.

    Attributes
    ----------
    label : str
        One of "large", "medium", "small".
    tick_bps : float
        Tick size as bps of price.
    q_bar : float
        Mean best-queue size (lots).
    mo_rate : float
        Market-order arrival rate per second, per side.
    mo_size : float
        Mean market-order size (geometric distribution mean).
    cancel_rate : float
        Cancel hazard per resting lot per second.
    inside_rate : float
        Rate of limit orders posted inside a gap of >1 tick, per side.
    s_max : int
        Maximum spread in ticks.
    kappa : float
        Imbalance sensitivity of market-order flow.
    add_scale : float
        Calibration factor on the limit-add rate (tuning knob). Level
        exhaustion re-seeds the best queue from the queue-size law, which adds
        lots on top of the ``add_rate`` balance, so the unscaled formula
        overshoots ``q_bar``; ``scripts/calibrate_add_scale.py`` finds the
        factor that restores the stationary mean.
    """

    label: str
    tick_bps: float
    q_bar: float
    mo_rate: float
    mo_size: float
    cancel_rate: float
    inside_rate: float
    s_max: int
    kappa: float
    add_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.label not in ("large", "medium", "small"):
            raise ValueError(f"label must be large/medium/small, got {self.label!r}")
        for field in ("tick_bps", "q_bar", "mo_rate", "mo_size", "cancel_rate"):
            val = getattr(self, field)
            if not np.isfinite(val) or val <= 0:
                raise ValueError(f"{field} must be positive and finite, got {val}")
        if not np.isfinite(self.inside_rate) or self.inside_rate < 0:
            raise ValueError(f"inside_rate must be non-negative, got {self.inside_rate}")
        if not isinstance(self.s_max, int) or self.s_max < 1:
            raise ValueError(f"s_max must be a positive integer, got {self.s_max}")
        if self.q_bar < 1.0:
            raise ValueError(f"q_bar must be >= 1 lot, got {self.q_bar}")
        if not np.isfinite(self.kappa) or not 0.0 <= self.kappa <= 1.0:
            raise ValueError(f"kappa must be in [0, 1], got {self.kappa}")
        if not np.isfinite(self.add_scale) or self.add_scale <= 0:
            raise ValueError(f"add_scale must be positive and finite, got {self.add_scale}")

    @property
    def add_rate(self) -> float:
        """Limit-order add rate (per side) that gives stationary mean q_bar.

        add_rate = add_scale * (cancel_rate * q_bar + mo_rate * mo_size)
        (balances cancels and market-order consumption in steady state;
        ``add_scale`` corrects for the lots re-seeded at level exhaustion).
        """
        return self.add_scale * (self.cancel_rate * self.q_bar + self.mo_rate * self.mo_size)


# ---------------------------------------------------------------------------
# Pre-defined regimes (stated assumptions, not calibrated to an asset)
# ---------------------------------------------------------------------------

REGIMES: dict[str, Regime] = {
    "large": Regime(
        label="large",
        tick_bps=1.4,
        q_bar=300.0,
        mo_rate=0.5,
        mo_size=15.0,
        cancel_rate=0.01,
        inside_rate=5.0,
        s_max=3,
        kappa=0.5,
    ),
    "medium": Regime(
        label="medium",
        tick_bps=0.5,
        q_bar=60.0,
        mo_rate=0.5,
        mo_size=8.0,
        cancel_rate=0.03,
        inside_rate=0.8,
        s_max=5,
        kappa=0.5,
        add_scale=0.88,  # calibrated: scripts/calibrate_add_scale.py
    ),
    "small": Regime(
        label="small",
        tick_bps=0.15,
        q_bar=10.0,
        mo_rate=1.2,  # raised from 0.8 so 1-min mid sd is within 1.7x of LARGE
        mo_size=3.0,
        cancel_rate=0.10,
        inside_rate=0.15,
        s_max=12,
        kappa=0.5,
        add_scale=0.62,  # calibrated: scripts/calibrate_add_scale.py
    ),
}
