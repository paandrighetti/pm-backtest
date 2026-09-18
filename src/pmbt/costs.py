"""Cost model and closed-form fair values for crypto binaries.

Costs. Price history is a mid or last-trade series, so every entry crosses a spread:
    fill = p + half_spread + slippage, plus a venue fee.
Polymarket: no fee on most event markets; the 15-minute crypto markets have a taker curve
handled in updown-desk, not here. Kalshi: taker fee 0.07 * C * P * (1 - P), rounded up to
the cent per contract, per Kalshi's published fee schedule.

Fair values (driftless GBM, s = sigma * sqrt(tau)):
    digital_above  P(S_T >= K)            = Phi( ln(S/K)/s - s/2 )
    digital_below  P(S_T <  K)            = 1 - digital_above
    touch (K > S)  P(max S_t >= K)        = 2 Phi( -ln(K/S)/s )   reflection principle,
    touch (K < S)  P(min S_t <= K)        = 2 Phi( -ln(S/K)/s )   driftless log-price approx.
Deltas are analytic derivatives in spot units per share.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SECONDS_PER_YEAR = 365 * 24 * 3600


@dataclass(frozen=True)
class CostModel:
    half_spread: float = 0.01
    slippage: float = 0.0
    venue: str = "polymarket"
    spot_fee_bps: float = 5.0  # taker fee on the spot hedge leg

    def buy_price(self, p: float) -> float:
        return min(0.999, p + self.half_spread + self.slippage)

    def fee(self, price: float, shares: float) -> float:
        if self.venue == "kalshi":
            return math.ceil(round(0.07 * shares * price * (1.0 - price) * 100, 6)) / 100
        return 0.0

    def entry_cost(self, p: float, shares: float) -> tuple[float, float]:
        """(fill price, total cost including fee) for buying `shares` at mid p."""
        price = self.buy_price(p)
        return price, price * shares + self.fee(price, shares)


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _Phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def fair_value(
    market_type: str, spot: float, strike: float, sigma: float, tau_s: float
) -> tuple[float, float]:
    """Return (probability of YES, delta in spot units per YES share)."""
    if tau_s <= 0 or sigma <= 0:
        if market_type == "digital_above":
            return (float(spot >= strike), 0.0)
        if market_type == "digital_below":
            return (float(spot < strike), 0.0)
        return (1.0, 0.0) if spot == strike else (0.0, 0.0)
    s = sigma * math.sqrt(tau_s / SECONDS_PER_YEAR)
    if market_type in ("digital_above", "digital_below"):
        d = math.log(spot / strike) / s - s / 2.0
        p, delta = _Phi(d), _phi(d) / (spot * s)
        return (p, delta) if market_type == "digital_above" else (1.0 - p, -delta)
    if market_type == "touch":
        if strike > spot:
            x = -math.log(strike / spot) / s
            return 2.0 * _Phi(x), 2.0 * _phi(x) / (spot * s)
        if strike < spot:
            x = -math.log(spot / strike) / s
            return 2.0 * _Phi(x), -2.0 * _phi(x) / (spot * s)
        return 1.0, 0.0
    raise ValueError(f"unknown market_type {market_type}")


def realized_vol_annualized(ts_s: np.ndarray, px: np.ndarray, min_ticks: int = 24) -> float | None:
    """sum(r^2) / sum(dt), annualized; handles irregular spacing."""
    ts_s = np.asarray(ts_s, dtype=float)
    px = np.asarray(px, dtype=float)
    keep = np.concatenate(([True], np.diff(ts_s) > 0))
    ts_s, px = ts_s[keep], px[keep]
    if len(px) < min_ticks:
        return None
    var = (np.diff(np.log(px)) ** 2).sum() / np.diff(ts_s).sum()
    return float(math.sqrt(var * SECONDS_PER_YEAR)) if var > 0 else None
