"""Execution and settlement. Strategies emit signals; only the engine sees outcomes.

Rules enforced here, not in strategies:
  * a signal at ts fills at the first price observation strictly after ts, never later
    than the market end;
  * the fill crosses the spread and pays the venue fee (CostModel);
  * positions are held to resolution and paid 1 per winning share.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

from .costs import CostModel
from .data import MarketView, PriceSeries, Store


@dataclass
class Trade:
    strategy: str
    params: str
    market_id: str
    event_id: str
    side: str
    ts_signal: int
    ts_fill: int
    ts_settle: int
    price: float
    shares: float
    cost: float
    payoff: float
    pnl: float
    days_locked: float
    note: str = ""
    mid: float = math.nan
    unhedged_pnl: float = math.nan
    hedge_pnl: float = 0.0
    hedge_cost: float = 0.0


class Engine:
    def __init__(self, store: Store, costs: CostModel) -> None:
        self.store = store
        self.costs = costs

    def buy(
        self,
        strategy: str,
        params: str,
        view: MarketView,
        side: str,
        ts_signal: int,
        shares: float = 1.0,
        note: str = "",
    ) -> Trade | None:
        series = self.store.series[view.market_id]
        nxt = series.next_after(ts_signal)
        if nxt is None or nxt[0] > view.end_ts:
            return None
        ts_fill, p_yes = nxt
        mid = p_yes if side == "yes" else 1.0 - p_yes
        price, cost = self.costs.entry_cost(mid, shares)
        outcome = self.store.outcome(view.market_id)
        won = (outcome == 1.0) if side == "yes" else (outcome == 0.0)
        payoff = shares if won else 0.0
        return Trade(
            mid=mid,
            strategy=strategy,
            params=params,
            market_id=view.market_id,
            event_id=view.event_id,
            side=side,
            ts_signal=ts_signal,
            ts_fill=ts_fill,
            ts_settle=view.end_ts,
            price=price,
            shares=shares,
            cost=cost,
            payoff=payoff,
            pnl=payoff - cost,
            days_locked=max(0.0, (view.end_ts - ts_fill) / 86400.0),
            note=note,
        )

    def hedge(
        self, trade: Trade, spot: PriceSeries, delta_fn, rebalance_s: int, cap_dollar_delta: float
    ) -> Trade:
        """Simulate a discrete spot hedge from fill to settlement and fold it into the trade.

        delta_fn(ts, spot) returns the YES-share delta in spot units at ts. A long YES position
        is hedged with -delta spot; a long NO position (short YES) with +delta.
        """
        sign = -1.0 if trade.side == "yes" else 1.0
        pos, prev_s, hedge_pnl, hedge_cost = 0.0, None, 0.0, 0.0
        fee = self.costs.spot_fee_bps / 1e4
        t = trade.ts_fill
        while t < trade.ts_settle:
            s = spot.at(t, max_stale_s=6 * 3600)
            if s is None:
                t += rebalance_s
                continue
            if prev_s is not None:
                hedge_pnl += pos * (s - prev_s)
            d = delta_fn(t, s)
            d = math.copysign(min(abs(d), cap_dollar_delta / s), d)
            target = sign * d * trade.shares
            hedge_cost += abs(target - pos) * s * fee
            pos, prev_s = target, s
            t += rebalance_s
        s_end = spot.at(trade.ts_settle, max_stale_s=6 * 3600)
        if prev_s is not None and s_end is not None:
            hedge_pnl += pos * (s_end - prev_s)
            hedge_cost += abs(pos) * s_end * fee
        trade.unhedged_pnl = trade.pnl
        trade.hedge_pnl = hedge_pnl
        trade.hedge_cost = hedge_cost
        trade.pnl = trade.unhedged_pnl + hedge_pnl - hedge_cost
        return trade


def to_frame(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in trades]) if trades else pd.DataFrame()
