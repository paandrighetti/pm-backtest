"""Walk-forward report driven by config.yaml.

Selection rule, declared once and applied mechanically: on the in-sample period (markets
resolved before `split_date`), keep the parameter set with the highest t-statistic among
those with at least `min_trades`; evaluate only that set out of sample. The full in-sample
grid is printed so the multiple-comparison burden is visible.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import yaml

from .costs import CostModel
from .data import Store
from .engine import Engine, to_frame
from .metrics import (
    aggregate_books,
    book_validity,
    by_side,
    calibration_by_price,
    grid_summary,
    spread_sensitivity,
    summarize,
)
from .strategies import (
    BinaryHedge,
    DutchBook,
    FavoriteCarry,
    run_event_strategy,
    run_market_strategy,
)

log = logging.getLogger("pmbt.report")
STRATEGIES = {"favorite_carry": FavoriteCarry, "dutch_book": DutchBook, "binary_hedge": BinaryHedge}


def _md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_no trades_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(f"{v:{floatfmt}}" if isinstance(v, float) else str(v) for v in row)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _grid(cls, grid: dict) -> list:
    keys = list(grid)
    return [
        cls(**dict(zip(keys, vals, strict=True)))
        for vals in itertools.product(*(grid[k] for k in keys))
    ]


def _run(store: Store, engine: Engine, strategy, universe) -> pd.DataFrame:
    if isinstance(strategy, DutchBook):
        return aggregate_books(to_frame(run_event_strategy(store, engine, strategy, universe)))
    return to_frame(run_market_strategy(store, engine, strategy, universe))


def build(cfg: dict) -> tuple[str, dict[str, pd.DataFrame]]:
    store = Store.load(cfg["data_dir"])
    costs = CostModel(**cfg.get("costs", {}))
    engine = Engine(store, costs)
    uni = store.universe(**cfg.get("universe", {}))
    split = int(
        datetime.strptime(cfg["split_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    )
    ins, oos = uni[uni["end_ts"] < split], uni[uni["end_ts"] >= split]
    min_trades = int(cfg.get("min_trades", 30))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        f"# pm-backtest report ({generated})\n",
        f"Universe: {len(uni)} resolved markets, {len(ins)} in sample (resolved before "
        f"{cfg['split_date']}), {len(oos)} out of sample. Costs: {costs}.\n",
        f"Selection rule: highest in-sample t-stat with n >= {min_trades}; that single "
        "parameter set is then run out of sample.\n",
    ]
    blotters: dict[str, pd.DataFrame] = {}
    for name, spec in cfg["strategies"].items():
        cls = STRATEGIES[name]
        candidates = _grid(cls, spec["grid"])
        log.info("%s: %d parameter sets in sample", name, len(candidates))
        ins_trades = (
            pd.concat([_run(store, engine, s, ins) for s in candidates], ignore_index=True)
            if candidates
            else pd.DataFrame()
        )
        grid = grid_summary(ins_trades)
        parts.append(f"## {name}\n")
        if name == "dutch_book":
            parts.append("One row per book (all legs of one event bought together), not per leg.\n")
        parts.append("### In-sample grid\n")
        parts.append(_md(grid))
        eligible = grid[grid["n"] >= min_trades] if not grid.empty else grid
        if eligible.empty:
            parts.append(
                f"No parameter set reached {min_trades} in-sample trades; "
                "nothing run out of sample.\n"
            )
            blotters[name] = ins_trades
            continue
        best_pid = eligible.sort_values("t_stat", ascending=False).iloc[0]["params"]
        best = next(s for s in candidates if s.pid == best_pid)
        oos_trades = _run(store, engine, best, oos)
        parts.append(f"### Out of sample, params `{best_pid}`\n")
        parts.append(_md(pd.DataFrame([summarize(oos_trades)])))
        if name == "dutch_book":
            parts.append("### Validity check: winners per book (a true book has exactly one)\n")
            parts.append(
                "A book with no winner bought an incomplete outcome set; a book with several "
                "winners bought markets that were not mutually exclusive. Only one-winner books "
                "say anything about arbitrage.\n"
            )
            parts.append("In sample:\n")
            parts.append(_md(book_validity(ins_trades)))
        if name == "dutch_book" and not oos_trades.empty:
            parts.append("Out of sample:\n")
            parts.append(_md(book_validity(oos_trades)))
            parts.append(
                "### Out of sample by side (YES books: prices summed below 1; NO books: above 1)\n"
            )
            parts.append(_md(by_side(oos_trades)))
        if name == "favorite_carry" and not oos_trades.empty:
            parts.append("### Out-of-sample calibration by fill price\n")
            parts.append(_md(calibration_by_price(oos_trades)))
            parts.append("### Out-of-sample sensitivity to the half-spread assumption\n")
            parts.append(_md(spread_sensitivity(oos_trades)))
        blotters[name] = pd.concat([ins_trades, oos_trades.assign(sample="oos")], ignore_index=True)
    return "\n".join(parts), blotters


def main() -> None:
    p = argparse.ArgumentParser(description="pm-backtest walk-forward report")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="reports")
    args = p.parse_args()
    logging.basicConfig(
        level="INFO", format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    report, blotters = build(cfg)
    os.makedirs(args.out, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = os.path.join(args.out, f"{day}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)
    for name, df in blotters.items():
        if not df.empty:
            df.to_csv(os.path.join(args.out, f"trades_{name}.csv"), index=False)
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
