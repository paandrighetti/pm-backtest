"""Load the canonical tables and expose point-in-time views.

PriceSeries.at(ts) returns the last observation at or before ts. Strategies receive series
and metadata; they never receive outcomes. Fills are resolved by the engine strictly after
the signal time.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from .schema import MARKET_COLUMNS


@dataclass(frozen=True)
class MarketView:
    """What a strategy is allowed to know about a market: no outcome."""

    market_id: str
    event_id: str
    question: str
    end_ts: int
    volume: float
    n_in_event: int
    underlying: str | None
    strike: float | None
    market_type: str | None


class PriceSeries:
    def __init__(self, ts: np.ndarray, px: np.ndarray) -> None:
        order = np.argsort(ts, kind="stable")
        self.ts = ts[order].astype(np.int64)
        self.px = px[order].astype(float)

    def __len__(self) -> int:
        return len(self.ts)

    def at(self, ts: int, max_stale_s: int | None = None) -> float | None:
        i = int(np.searchsorted(self.ts, ts, side="right")) - 1
        if i < 0:
            return None
        if max_stale_s is not None and ts - self.ts[i] > max_stale_s:
            return None
        return float(self.px[i])

    def next_after(self, ts: int) -> tuple[int, float] | None:
        """First observation strictly after ts; this is where a signal at ts can fill."""
        i = int(np.searchsorted(self.ts, ts, side="right"))
        if i >= len(self.ts):
            return None
        return int(self.ts[i]), float(self.px[i])

    def window(self, t0: int, t1: int) -> tuple[np.ndarray, np.ndarray]:
        lo = int(np.searchsorted(self.ts, t0, side="left"))
        hi = int(np.searchsorted(self.ts, t1, side="right"))
        return self.ts[lo:hi], self.px[lo:hi]


class Store:
    def __init__(
        self,
        markets: pd.DataFrame,
        prices: pd.DataFrame | None = None,
        spot: pd.DataFrame | None = None,
        series: dict[str, PriceSeries] | None = None,
    ):
        self.markets = markets.reset_index(drop=True)
        self._outcomes = dict(zip(markets["market_id"], markets["outcome"], strict=True))
        if series is not None:
            self.series = series
        else:
            self.series = {
                mid: PriceSeries(g["ts"].to_numpy(), g["p_yes"].to_numpy())
                for mid, g in prices.groupby("market_id", sort=False)
            }
        self.spot: dict[str, PriceSeries] = {}
        if spot is not None and not spot.empty:
            for sym, g in spot.groupby("symbol", sort=False):
                self.spot[sym] = PriceSeries(g["ts"].to_numpy(), g["close"].to_numpy())

    @classmethod
    def load(cls, data_dir: str) -> Store:
        """Group prices per market inside DuckDB (out of core) instead of one giant frame."""
        markets = pd.read_parquet(os.path.join(data_dir, "markets.parquet"))[MARKET_COLUMNS]
        pfiles = sorted(glob.glob(os.path.join(data_dir, "prices", "*.parquet")))
        series: dict[str, PriceSeries] = {}
        if pfiles:
            grouped = duckdb.sql(
                f"SELECT market_id, list(ts ORDER BY ts) AS ts, list(p_yes ORDER BY ts) AS px "
                f"FROM read_parquet({pfiles!r}) GROUP BY market_id"
            )
            for market_id, ts, px in grouped.fetchall():
                series[market_id] = PriceSeries(
                    np.asarray(ts, dtype=np.int64), np.asarray(px, dtype=float)
                )
        sfiles = sorted(glob.glob(os.path.join(data_dir, "spot", "*.parquet")))
        spot = duckdb.sql(f"SELECT * FROM read_parquet({sfiles!r})").df() if sfiles else None
        return cls(markets, spot=spot, series=series)

    def outcome(self, market_id: str) -> float:
        return self._outcomes[market_id]

    def view(self, row) -> MarketView:
        return MarketView(
            market_id=row.market_id,
            event_id=row.event_id,
            question=row.question,
            end_ts=int(row.end_ts),
            volume=float(row.volume),
            n_in_event=int(row.n_in_event),
            underlying=row.underlying if isinstance(row.underlying, str) else None,
            strike=None if pd.isna(row.strike) else float(row.strike),
            market_type=row.market_type if isinstance(row.market_type, str) else None,
        )

    def universe(self, min_volume: float = 0.0, exclude_voided: bool = True) -> pd.DataFrame:
        df = self.markets
        df = df[df["end_ts"].notna() & df["closed"].fillna(False).astype(bool)]
        df = df[df["volume"] >= min_volume]
        if exclude_voided:
            df = df[df["outcome"].notna()]
        return df[df["market_id"].isin(self.series.keys())].reset_index(drop=True)

    def events(self, universe: pd.DataFrame, min_markets: int = 3) -> dict[str, pd.DataFrame]:
        groups = {eid: g for eid, g in universe.groupby("event_id") if len(g) >= min_markets}
        return groups
