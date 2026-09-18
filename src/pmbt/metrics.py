"""Metrics on a trade blotter. Every number here is computable from the blotter alone."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown(pnl_in_time_order: np.ndarray) -> float:
    cum = np.cumsum(pnl_in_time_order)
    peak = np.maximum.accumulate(np.concatenate(([0.0], cum)))[1:]
    return float((cum - peak).min()) if len(cum) else 0.0


def summarize(trades: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Per-trade return statistics, capital-locked annualization and a daily Sharpe."""
    if trades.empty:
        return {"n": 0}
    t = trades.sort_values("ts_settle")
    ret = (t["pnl"] / t["cost"]).to_numpy()
    n = len(ret)
    mean = float(ret.mean())
    std = float(ret.std(ddof=1)) if n > 1 else float("nan")
    rng = np.random.default_rng(seed)
    boots = (
        np.array([ret[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
        if n > 1
        else np.array([mean])
    )
    capital_days = float((t["cost"] * t["days_locked"]).sum())
    daily = t.groupby(pd.to_datetime(t["ts_settle"], unit="s").dt.date)["pnl"].sum()
    sharpe = (
        float(daily.mean() / daily.std(ddof=1) * np.sqrt(365))
        if len(daily) > 2 and daily.std(ddof=1) > 0
        else float("nan")
    )
    out = {
        "n": n,
        "hit_rate": float((t["pnl"] > 0).mean()),
        "pnl_total": float(t["pnl"].sum()),
        "cost_total": float(t["cost"].sum()),
        "ret_per_trade": mean,
        "return_on_capital": float(t["pnl"].sum() / t["cost"].sum())
        if t["cost"].sum()
        else float("nan"),
        "ret_ci_low": float(np.percentile(boots, 2.5)),
        "ret_ci_high": float(np.percentile(boots, 97.5)),
        "t_stat": float(mean / (std / np.sqrt(n))) if std and std > 0 else float("nan"),
        "ann_return_on_locked": float(t["pnl"].sum() / capital_days * 365)
        if capital_days > 0
        else float("nan"),
        "sharpe_daily": sharpe,
        "max_drawdown": _max_drawdown(t["pnl"].to_numpy()),
        "mean_days_locked": float(t["days_locked"].mean()),
    }
    if "unhedged_pnl" in t and t["unhedged_pnl"].notna().any():
        u = t["unhedged_pnl"].to_numpy()
        h = t["pnl"].to_numpy()
        out["unhedged_pnl_total"] = float(np.nansum(u))
        out["var_ratio_hedged_over_unhedged"] = (
            float(np.var(h) / np.var(u)) if np.var(u) > 0 else float("nan")
        )
        out["hedge_cost_total"] = float(t["hedge_cost"].sum())
    return out


def grid_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for (strategy, pid), g in trades.groupby(["strategy", "params"]):
        rows.append({"strategy": strategy, "params": pid, **summarize(g, n_boot=500)})
    return pd.DataFrame(rows)


def calibration_by_price(
    trades: pd.DataFrame, edges=(0.5, 0.8, 0.9, 0.95, 0.98, 1.0)
) -> pd.DataFrame:
    """Realized win rate against the fill price, the direct test of favorite mispricing."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["bucket"] = pd.cut(t["price"], list(edges), include_lowest=True)
    has_mid = "mid" in t and t["mid"].notna().any()
    aggs = dict(
        n=("pnl", "count"),
        avg_price=("price", "mean"),
        win_rate=("payoff", lambda s: float((s > 0).mean())),
        ret_per_trade=("pnl", lambda s: float((s / t.loc[s.index, "cost"]).mean())),
    )
    if has_mid:
        aggs["avg_mid"] = ("mid", "mean")
    out = t.groupby("bucket", observed=True).agg(**aggs).reset_index()
    out["bucket"] = out["bucket"].astype(str)
    out["gap_at_fill"] = out["win_rate"] - out["avg_price"]
    if has_mid:  # what the bias is worth before any spread
        out["gap_at_mid"] = out["win_rate"] - out["avg_mid"]
    return out


def aggregate_books(trades: pd.DataFrame) -> pd.DataFrame:
    """Collapse the legs of each Dutch book into one row per event.

    Per-leg returns mislead: a cheap winning leg returns many times its cost while the other
    legs each lose a little, so the per-leg mean is positive even when the book as a whole
    loses. The unit of decision is the book, so the unit of measurement must be too.
    """
    if trades.empty or "event_id" not in trades:
        return trades
    agg = trades.groupby(["strategy", "params", "event_id"], sort=False).agg(
        market_id=("market_id", "first"),
        side=("side", "first"),
        ts_signal=("ts_signal", "min"),
        ts_fill=("ts_fill", "max"),
        ts_settle=("ts_settle", "max"),
        price=("price", "sum"),
        shares=("shares", "sum"),
        cost=("cost", "sum"),
        payoff=("payoff", "sum"),
        pnl=("pnl", "sum"),
        days_locked=("days_locked", "max"),
        legs=("market_id", "count"),
    )
    return agg.reset_index()


def spread_sensitivity(
    trades: pd.DataFrame, half_spreads=(0.0, 0.002, 0.005, 0.01)
) -> pd.DataFrame:
    """Return per unit of capital under alternative half spreads, recomputed from the mid.

    Hourly price history has no depth, so the spread is an assumption; this table shows how
    much of the verdict depends on it. Venue fees are not re-applied (zero on Polymarket).
    """
    if trades.empty or "mid" not in trades or trades["mid"].isna().all():
        return pd.DataFrame()
    rows = []
    for hs in half_spreads:
        price = np.minimum(0.999, trades["mid"].to_numpy() + hs)
        cost = price * trades["shares"].to_numpy()
        pnl = trades["payoff"].to_numpy() - cost
        ret = pnl / cost
        n = len(ret)
        std = ret.std(ddof=1) if n > 1 else float("nan")
        rows.append(
            {
                "half_spread": hs,
                "n": n,
                "ret_per_trade": float(ret.mean()),
                "t_stat": float(ret.mean() / (std / np.sqrt(n)))
                if std and std > 0
                else float("nan"),
                "pnl_total": float(pnl.sum()),
                "return_on_capital": float(pnl.sum() / cost.sum()) if cost.sum() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def by_side(trades: pd.DataFrame) -> pd.DataFrame:
    """Books bought on the YES side (prices summed below 1) against the NO side (above 1).

    YES books are cheap and NO books expensive, so an equal-weighted mean that beats the
    capital-weighted one says the residual sits in the cheap books; this table shows it.
    """
    if trades.empty or "side" not in trades:
        return pd.DataFrame()
    rows = []
    for side, g in trades.groupby("side"):
        r = summarize(g, n_boot=500)
        rows.append(
            {
                "side": side,
                "n": r["n"],
                "hit_rate": r["hit_rate"],
                "cost_total": r["cost_total"],
                "pnl_total": r["pnl_total"],
                "ret_per_trade": r["ret_per_trade"],
                "return_on_capital": r["return_on_capital"],
                "t_stat": r["t_stat"],
            }
        )
    return pd.DataFrame(rows)


def book_validity(books: pd.DataFrame) -> pd.DataFrame:
    """How many listed outcomes actually won in each book.

    A book on a complete, mutually exclusive outcome set has exactly one winner, so a YES
    book can never lose and a NO book always pays legs - 1. Books with no winner mean the
    winning outcome was not in the set (missing market, or filtered out by the volume floor);
    books with several winners mean the markets were not mutually exclusive. Either way the
    price sum said nothing about arbitrage.
    """
    if books.empty or "legs" not in books:
        return pd.DataFrame()
    b = books.copy()
    b["winners"] = np.where(b["side"] == "yes", b["payoff"], b["legs"] - b["payoff"]).round()
    out = (
        b.groupby("side")
        .agg(
            n=("winners", "count"),
            share_no_winner=("winners", lambda w: float((w == 0).mean())),
            share_one_winner=("winners", lambda w: float((w == 1).mean())),
            share_several_winners=("winners", lambda w: float((w >= 2).mean())),
        )
        .reset_index()
    )
    return out
