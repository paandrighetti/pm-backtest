"""Strategies. Each one is a pure signal generator over point-in-time data.

favorite_carry   buy the side priced at or above `threshold` once time to resolution is at
                 most `max_days`; hold to resolution. The favorite-longshot bias literature
                 (Thaler and Ziemba 1988; Snowberg and Wolfers 2010) says favorites are
                 slightly underpriced; the test is whether that survives the spread.
dutch_book       multi-outcome events whose YES prices sum below 1 minus costs (buy all) or
                 above 1 plus costs (buy all NO). Hourly data makes quotes quasi-synchronous
                 at best, so this is an upper bound on what a bot could have captured.
binary_hedge     crypto price markets priced against a driftless-GBM fair value from spot and
                 trailing realized volatility; enter when mispricing exceeds `threshold`,
                 then delta-hedge with spot at a fixed rebalance interval. Reports hedged and
                 unhedged PnL side by side, so the hedge is tested, not assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .costs import fair_value, realized_vol_annualized
from .data import MarketView, PriceSeries, Store
from .engine import Engine, Trade


def params_id(**kw) -> str:
    return json.dumps(kw, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class FavoriteCarry:
    threshold: float = 0.95
    max_days: float = 7.0
    min_hours: float = 1.0
    side: str = "best"  # yes | no | best (whichever side is at or above threshold first)
    name: str = "favorite_carry"

    @property
    def pid(self) -> str:
        return params_id(
            threshold=self.threshold,
            max_days=self.max_days,
            min_hours=self.min_hours,
            side=self.side,
        )

    def run(self, engine: Engine, view: MarketView, series: PriceSeries) -> list[Trade]:
        tau = view.end_ts - series.ts
        eligible = (tau <= self.max_days * 86400) & (tau >= self.min_hours * 3600)
        yes_ok = (
            eligible & (series.px >= self.threshold)
            if self.side in ("yes", "best")
            else np.zeros_like(eligible)
        )
        no_ok = (
            eligible & (1.0 - series.px >= self.threshold)
            if self.side in ("no", "best")
            else np.zeros_like(eligible)
        )
        hits = np.flatnonzero(yes_ok | no_ok)
        if len(hits) == 0:
            return []
        i = int(hits[0])
        side = "yes" if yes_ok[i] else "no"
        t = engine.buy(self.name, self.pid, view, side, int(series.ts[i]))
        return [t] if t else []


@dataclass(frozen=True)
class DutchBook:
    threshold: float = 0.01
    max_stale_h: float = 2.0
    grid_s: int = 3600
    name: str = "dutch_book"

    @property
    def pid(self) -> str:
        return params_id(threshold=self.threshold, max_stale_h=self.max_stale_h)

    def run(
        self, engine: Engine, views: list[MarketView], series: dict[str, PriceSeries]
    ) -> list[Trade]:
        legs = [
            (v, series[v.market_id])
            for v in views
            if v.market_id in series and len(series[v.market_id])
        ]
        if len(legs) < 3:
            return []
        n = len(legs)
        t0 = max(int(s.ts[0]) for _, s in legs)
        t1 = min(v.end_ts for v, _ in legs)
        t0 = t0 - t0 % self.grid_s + self.grid_s
        stale = int(self.max_stale_h * 3600)
        cost_n = n * engine.costs.half_spread
        for t in range(t0, t1, self.grid_s):
            ps = [s.at(t, stale) for _, s in legs]
            if any(p is None for p in ps):
                continue
            total = float(sum(ps))
            if total < 1.0 - cost_n - self.threshold:
                side = "yes"
            elif total > 1.0 + cost_n + self.threshold:
                side = "no"
            else:
                continue
            trades = [
                engine.buy(self.name, self.pid, v, side, t, note=f"sum={total:.4f}")
                for v, _ in legs
            ]
            if all(trades):
                return trades  # one entry per event
            return []  # a leg could not fill: the book cannot be completed, skip the event
        return []


@dataclass(frozen=True)
class BinaryHedge:
    threshold: float = 0.05
    rebalance_h: float = 6.0
    cap_dollar_delta: float = 5.0  # per share, guards the gamma blow-up near the strike
    vol_lookback_d: float = 30.0
    min_tau_h: float = 6.0
    max_abs_mispricing: float = 0.4  # larger gaps are treated as a parsing or rules mismatch
    name: str = "binary_hedge"

    @property
    def pid(self) -> str:
        return params_id(
            threshold=self.threshold,
            rebalance_h=self.rebalance_h,
            cap_dollar_delta=self.cap_dollar_delta,
            vol_lookback_d=self.vol_lookback_d,
        )

    def _sigma(self, spot: PriceSeries, ts: int) -> float | None:
        t, p = spot.window(ts - int(self.vol_lookback_d * 86400), ts)
        return realized_vol_annualized(t, p)

    def run(
        self, engine: Engine, view: MarketView, series: PriceSeries, store: Store
    ) -> list[Trade]:
        if view.market_type is None or view.underlying is None or view.strike is None:
            return []
        spot = store.spot.get(f"{view.underlying}USDT")
        if spot is None:
            return []
        for ts, p in zip(series.ts, series.px, strict=True):
            ts = int(ts)
            tau = view.end_ts - ts
            if tau < self.min_tau_h * 3600:
                break
            s = spot.at(ts, max_stale_s=2 * 3600)
            sigma = self._sigma(spot, ts)
            if s is None or sigma is None:
                continue
            fv, _ = fair_value(view.market_type, s, view.strike, sigma, tau)
            gap = fv - float(p)
            if abs(gap) > self.max_abs_mispricing:
                continue
            if gap >= self.threshold:
                side = "yes"
            elif -gap >= self.threshold:
                side = "no"
            else:
                continue
            trade = engine.buy(
                self.name, self.pid, view, side, ts, note=f"fv={fv:.3f} sigma={sigma:.2f}"
            )
            if trade is None:
                return []

            def delta_fn(t: int, spot_px: float, _sig: float = sigma) -> float:
                sig = self._sigma(spot, t) or _sig
                return fair_value(view.market_type, spot_px, view.strike, sig, view.end_ts - t)[1]

            engine.hedge(trade, spot, delta_fn, int(self.rebalance_h * 3600), self.cap_dollar_delta)
            return [trade]
        return []


def run_market_strategy(store: Store, engine: Engine, strategy, universe) -> list[Trade]:
    out: list[Trade] = []
    for row in universe.itertuples(index=False):
        view = store.view(row)
        series = store.series[view.market_id]
        if isinstance(strategy, BinaryHedge):
            out.extend(strategy.run(engine, view, series, store))
        else:
            out.extend(strategy.run(engine, view, series))
    return out


def run_event_strategy(store: Store, engine: Engine, strategy: DutchBook, universe) -> list[Trade]:
    out: list[Trade] = []
    for _eid, g in store.events(universe).items():
        views = [store.view(r) for r in g.itertuples(index=False)]
        out.extend(strategy.run(engine, views, store.series))
    return out
