import numpy as np
import pandas as pd

from pmbt.costs import CostModel
from pmbt.data import Store
from pmbt.engine import Engine, to_frame
from pmbt.metrics import calibration_by_price, grid_summary, summarize
from pmbt.report import build
from pmbt.strategies import (
    BinaryHedge,
    DutchBook,
    FavoriteCarry,
    run_event_strategy,
    run_market_strategy,
)


def _run(store, strat, half_spread=0.01):
    engine = Engine(store, CostModel(half_spread=half_spread))
    if isinstance(strat, DutchBook):
        return to_frame(run_event_strategy(store, engine, strat, store.universe()))
    return to_frame(run_market_strategy(store, engine, strat, store.universe()))


def test_fills_are_strictly_after_signal_and_before_end(store):
    t = _run(store, FavoriteCarry(threshold=0.9, max_days=10))
    assert not t.empty
    assert (t["ts_fill"] > t["ts_signal"]).all()
    assert (t["ts_fill"] <= t["ts_settle"]).all()
    for row in t.itertuples(index=False):  # fill = observation at ts_fill, spread crossed
        obs = store.series[row.market_id].at(row.ts_fill)
        mid = obs if row.side == "yes" else 1.0 - obs
        assert row.price == min(0.999, mid + 0.01)
    assert t.groupby("market_id").size().max() == 1


def test_carry_loses_the_spread_when_calibrated_and_wins_when_biased(store, biased_store):
    calibrated = summarize(_run(store, FavoriteCarry(threshold=0.9, max_days=10)))
    biased = summarize(_run(biased_store, FavoriteCarry(threshold=0.9, max_days=10)))
    assert calibrated["n"] > 50 and biased["n"] > 50
    # calibrated prices: the spread is the only expected loss, so the CI brackets -0.01
    assert calibrated["ret_ci_low"] < -0.01 < calibrated["ret_ci_high"]
    # favorites underpriced by 0.06 and a 0.01 spread: about +0.05 per unit of capital
    assert biased["ret_per_trade"] > calibrated["ret_per_trade"]
    assert biased["ret_ci_low"] > 0.0
    cal = calibration_by_price(_run(biased_store, FavoriteCarry(threshold=0.9, max_days=10)))
    assert "gap_at_fill" in cal


def test_dutch_book_finds_planted_windows_and_nothing_else(store):
    t = _run(store, DutchBook(threshold=0.01))
    assert not t.empty
    events = t["event_id"].unique()
    planted = {f"ev{e}" for e in range(0, 40, 4)}
    assert set(events) <= planted  # normalized prices never trigger, only the planted windows
    assert len(events) >= 8
    per_event = t.groupby("event_id")["pnl"].sum()
    assert (per_event > 0).all()  # sum 0.93 + 4 * 0.01 spread leaves 0.03 per book
    assert (t.groupby("event_id").size() == 4).all()


def test_binary_hedge_reduces_variance(store):
    unhedged = _run(store, BinaryHedge(threshold=0.05, rebalance_h=1.0, cap_dollar_delta=1e9))
    assert not unhedged.empty
    s = summarize(unhedged)
    assert s["n"] >= 10
    assert s["var_ratio_hedged_over_unhedged"] < 1.0
    assert (unhedged["hedge_cost"] >= 0).all()
    assert unhedged["note"].str.startswith("fv=").all()


def test_grid_and_report_build(data_dir, tmp_path):
    store = Store.load(data_dir)
    assert len(store.series) == len(store.markets)
    assert "BTCUSDT" in store.spot
    cfg = {
        "data_dir": data_dir,
        "split_date": pd.to_datetime(store.markets["end_ts"].median(), unit="s").strftime(
            "%Y-%m-%d"
        ),
        "min_trades": 10,
        "costs": {"half_spread": 0.01},
        "universe": {"min_volume": 0},
        "strategies": {
            "favorite_carry": {"grid": {"threshold": [0.9, 0.95], "max_days": [10]}},
            "dutch_book": {"grid": {"threshold": [0.0, 0.01]}},
            "binary_hedge": {"grid": {"threshold": [0.05], "rebalance_h": [6]}},
        },
    }
    report, blotters = build(cfg)
    assert "## favorite_carry" in report and "Out of sample" in report
    assert set(blotters) == {"favorite_carry", "dutch_book", "binary_hedge"}
    g = grid_summary(blotters["favorite_carry"])
    assert {"t_stat", "ret_ci_low", "ann_return_on_locked"} <= set(g.columns)
    assert np.isfinite(g["t_stat"]).any()


def test_books_aggregate_per_event_and_spread_sensitivity(store):
    from pmbt.metrics import aggregate_books, spread_sensitivity

    engine = Engine(store, CostModel(half_spread=0.01))
    legs = to_frame(run_event_strategy(store, engine, DutchBook(threshold=0.01), store.universe()))
    books = aggregate_books(legs)
    assert len(books) == legs["event_id"].nunique()
    assert abs(books["pnl"].sum() - legs["pnl"].sum()) < 1e-9
    assert (books["legs"] == 4).all()
    carry = to_frame(
        run_market_strategy(
            store, engine, FavoriteCarry(threshold=0.9, max_days=10), store.universe()
        )
    )
    sens = spread_sensitivity(carry)
    assert list(sens["half_spread"]) == [0.0, 0.002, 0.005, 0.01]
    assert sens["ret_per_trade"].is_monotonic_decreasing  # wider spread, lower return
    # the 0.01 row must reproduce the engine's own fills
    assert abs(sens.iloc[-1]["pnl_total"] - carry["pnl"].sum()) < 1e-6


def test_calibration_reports_gap_at_mid_and_side_split(store):
    from pmbt.metrics import by_side, calibration_by_price, summarize

    engine = Engine(store, CostModel(half_spread=0.01))
    carry = to_frame(
        run_market_strategy(
            store, engine, FavoriteCarry(threshold=0.9, max_days=10), store.universe()
        )
    )
    cal = calibration_by_price(carry)
    assert {"avg_mid", "gap_at_mid", "gap_at_fill"} <= set(cal.columns)
    assert ((cal["gap_at_mid"] - cal["gap_at_fill"]).round(6) >= 0).all()  # fill is above mid
    assert (
        abs(summarize(carry)["return_on_capital"] - carry["pnl"].sum() / carry["cost"].sum()) < 1e-9
    )
    from pmbt.metrics import aggregate_books

    books = aggregate_books(
        to_frame(run_event_strategy(store, engine, DutchBook(threshold=0.01), store.universe()))
    )
    sides = by_side(books)
    assert sides["n"].sum() == len(books)


def test_book_validity_on_exclusive_synthetic_events(store):
    from pmbt.metrics import aggregate_books, book_validity

    engine = Engine(store, CostModel(half_spread=0.01))
    books = aggregate_books(
        to_frame(run_event_strategy(store, engine, DutchBook(threshold=0.01), store.universe()))
    )
    v = book_validity(books)
    assert (v["share_one_winner"] == 1.0).all()  # synthetic events are complete and exclusive
    assert (v["share_no_winner"] == 0.0).all() and (v["share_several_winners"] == 0.0).all()
