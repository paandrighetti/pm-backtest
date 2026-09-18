"""Ingest resolved Polymarket markets with their price history, and Binance hourly spot.

    pmbt-ingest polymarket --start 2025-09-01 --end 2026-09-01 --min-volume 10000
    pmbt-ingest binance --symbol BTCUSDT --start 2025-08-01 --end 2026-09-01

Both commands are resumable: markets already present in data/prices are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import logging
import os
import re
import sys
from datetime import datetime, timezone

import duckdb
import httpx
import pandas as pd
from polymarket import AsyncPublicClient
from polymarket.errors import RateLimitError, RequestRejectedError, TransportError

from .schema import MARKET_COLUMNS, parse_question

log = logging.getLogger("pmbt.ingest")
BINANCE = "https://api.binance.com/api/v3/klines"


def _ts(d: str) -> int:
    return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _outcome(m) -> float:
    yes = m.outcomes.yes.price
    if yes is None:
        return float("nan")
    yes = float(yes)
    if yes >= 0.99:
        return 1.0
    if yes <= 0.01:
        return 0.0
    return float("nan")  # voided (0.5 / 0.5) or not resolved yet


def market_row(m) -> dict:
    pq = parse_question(m.question or "")
    events = m.events or []
    return {
        "market_id": m.outcomes.yes.token_id,
        "event_id": str(events[0].id) if events else str(m.condition_id),
        "venue": "polymarket",
        "question": m.question,
        "end_ts": int(m.state.end_date.timestamp()) if m.state and m.state.end_date else None,
        "closed": bool(m.state.closed) if m.state else None,
        "outcome": _outcome(m),
        "volume": float(m.metrics.volume_num or 0.0) if m.metrics else 0.0,
        "n_in_event": 1,
        "underlying": pq.underlying,
        "strike": pq.strike,
        "market_type": pq.market_type,
    }


def _done_markets(data_dir: str) -> set[str]:
    files = glob.glob(os.path.join(data_dir, "prices", "*.parquet"))
    if not files:
        return set()
    return set(
        duckdb.sql(f"SELECT DISTINCT market_id FROM read_parquet({files!r})").df()["market_id"]
    )


async def _with_backoff(call, label: str, attempts: int = 8):
    """Retry a coroutine factory on rate limits (honoring retry_after) and transport errors."""
    for attempt in range(attempts):
        try:
            return await call()
        except RateLimitError as exc:
            wait = max(float(exc.retry_after or 0.0), min(120.0, 2.0**attempt))
            log.warning("%s: rate limited, sleeping %.0f s", label, wait)
        except TransportError as exc:
            wait = min(60.0, 2.0**attempt)
            log.warning("%s: %s, sleeping %.0f s", label, exc, wait)
        await asyncio.sleep(wait)
    raise RuntimeError(f"{label}: gave up after {attempts} attempts")


# Short-horizon series (15-minute, hourly and daily Up/Down) are the subject of updown-desk, not
# of this backtester, and they dominate the market count. Excluded at listing time.
EXCLUDE = re.compile(r"updown|up-or-down|up or down", re.I)
CHECKPOINT_EVERY = 50


def _checkpoint_paths(data_dir: str) -> tuple[str, str]:
    return os.path.join(data_dir, "listing.cursor"), os.path.join(
        data_dir, "markets_partial.parquet"
    )


def _save_checkpoint(data_dir: str, cursor: str, rows: list[dict]) -> None:
    cur_path, rows_path = _checkpoint_paths(data_dir)
    pd.DataFrame(rows, columns=MARKET_COLUMNS).to_parquet(rows_path, index=False)
    with open(cur_path, "w", encoding="utf-8") as fh:
        fh.write(cursor)


def _load_checkpoint(data_dir: str) -> tuple[str | None, list[dict]]:
    cur_path, rows_path = _checkpoint_paths(data_dir)
    if not (os.path.exists(cur_path) and os.path.exists(rows_path)):
        return None, []
    with open(cur_path, encoding="utf-8") as fh:
        cursor = fh.read().strip() or None
    rows = pd.read_parquet(rows_path).to_dict("records")
    log.info("resuming listing from checkpoint: %d rows", len(rows))
    return cursor, rows


def _clear_checkpoint(data_dir: str) -> None:
    for path in _checkpoint_paths(data_dir):
        if os.path.exists(path):
            os.remove(path)


async def _list_markets(
    client: AsyncPublicClient,
    data_dir: str,
    start: str,
    end: str,
    min_volume: float,
    pace_s: float,
) -> list[dict]:
    pager = client.list_markets(
        closed=True,
        end_date_min=start,
        end_date_max=end,
        volume_num_min=min_volume,
        order="endDate",
        ascending=True,
        page_size=100,
    )
    cursor, rows = _load_checkpoint(data_dir)
    n_pages = n_excluded = 0
    while True:
        source = pager.from_cursor(cursor) if cursor else pager
        try:
            page = await _with_backoff(source.first_page, f"markets page {n_pages}")
        except Exception:
            if cursor:
                _save_checkpoint(data_dir, cursor, rows)
                log.error("listing failed at cursor %s, checkpoint saved", cursor[:24])
            raise
        for m in page.items:
            if not (m.outcomes and m.outcomes.yes and m.outcomes.yes.token_id):
                continue
            if EXCLUDE.search(m.slug or "") or EXCLUDE.search(m.question or ""):
                n_excluded += 1
                continue
            rows.append(market_row(m))
        n_pages += 1
        if n_pages % 20 == 0:
            log.info(
                "markets: %d kept, %d excluded, after %d pages", len(rows), n_excluded, n_pages
            )
        if not page.has_more or page.next_cursor is None:
            _clear_checkpoint(data_dir)
            return rows
        cursor = page.next_cursor
        if n_pages % CHECKPOINT_EVERY == 0:
            _save_checkpoint(data_dir, cursor, rows)
        await asyncio.sleep(pace_s)


async def _probe_window_days(client: AsyncPublicClient, r: dict, fidelity: int) -> int:
    """Largest start/end window the endpoint accepts, found by halving from 29 days."""
    end = int(r["end_ts"]) + 3600
    for days in (29, 14, 7, 3, 1):
        try:
            await _with_backoff(
                lambda days=days: client.get_price_history(
                    token_id=r["market_id"],
                    start_ts=end - days * 86400,
                    end_ts=end,
                    fidelity=fidelity,
                ),
                f"probe {days}d",
            )
        except RequestRejectedError as exc:
            if "too long" in str(exc).lower():
                continue
            raise
        log.info("price history window accepted by the endpoint: %d days", days)
        return days
    raise RuntimeError("the endpoint rejected even a one-day window")


async def ingest_polymarket(
    data_dir: str,
    start: str,
    end: str,
    min_volume: float,
    fidelity: int,
    concurrency: int = 2,
    pace_s: float = 0.5,
    relist: bool = False,
    lookback_days: int = 7,
) -> None:
    os.makedirs(os.path.join(data_dir, "prices"), exist_ok=True)
    markets_path = os.path.join(data_dir, "markets.parquet")
    async with AsyncPublicClient() as client:
        if os.path.exists(markets_path) and not relist:
            markets = pd.read_parquet(markets_path)
            log.info(
                "markets: reusing %d rows from %s (pass --relist to refresh)",
                len(markets),
                markets_path,
            )
        else:
            rows = await _list_markets(client, data_dir, start, end, min_volume, pace_s)
            markets = pd.DataFrame(rows, columns=MARKET_COLUMNS)
            markets["n_in_event"] = markets.groupby("event_id")["market_id"].transform("count")
            markets.to_parquet(markets_path, index=False)
            log.info("markets: %d rows written", len(markets))
        rows = markets.to_dict("records")

        empty_path = os.path.join(data_dir, "empty_markets.txt")
        empty: set[str] = set()
        if os.path.exists(empty_path):
            with open(empty_path, encoding="utf-8") as fh:
                empty = {line.strip() for line in fh if line.strip()}
        done = _done_markets(data_dir) | empty
        todo = [r for r in rows if r["market_id"] not in done and pd.notna(r["end_ts"])]
        log.info("price history: %d to fetch, %d already present", len(todo), len(done))
        sem = asyncio.Semaphore(concurrency)
        part = len(glob.glob(os.path.join(data_dir, "prices", "*.parquet")))

        # interval=max silently returns only the last month, and explicit windows are rejected
        # beyond an undocumented length. The accepted length is probed once on the first
        # market and every window is then cut into chunks of that size.
        chunk_days = await _probe_window_days(client, todo[0], fidelity) if todo else 1
        chunk_s = chunk_days * 86400

        async def fetch(r: dict) -> pd.DataFrame | None:
            token = r["market_id"]
            end = int(r["end_ts"]) + 3600
            frames: list[pd.DataFrame] = []
            async with sem:
                t0 = end - lookback_days * 86400
                while t0 < end:
                    t1 = min(t0 + chunk_s, end)
                    try:
                        pts = await _with_backoff(
                            lambda t0=t0, t1=t1: client.get_price_history(
                                token_id=token, start_ts=t0, end_ts=t1, fidelity=fidelity
                            ),
                            f"history {token[:12]}",
                        )
                    except (RuntimeError, RequestRejectedError) as exc:
                        log.warning("%s: %s", token[:12], exc)
                        return None
                    if pts:
                        frames.append(
                            pd.DataFrame(
                                {
                                    "market_id": token,
                                    "ts": [int(p.t) for p in pts],
                                    "p_yes": [float(p.p) for p in pts],
                                }
                            )
                        )
                    t0 = t1
                    await asyncio.sleep(pace_s)
            if not frames:
                return pd.DataFrame(columns=["market_id", "ts", "p_yes"])  # empty: recorded
            return pd.concat(frames, ignore_index=True).drop_duplicates("ts")

        for i in range(0, len(todo), 200):
            chunk = await asyncio.gather(*(fetch(r) for r in todo[i : i + 200]))
            frames = [f for f in chunk if f is not None and not f.empty]
            empties = [
                r["market_id"]
                for r, f in zip(todo[i : i + 200], chunk, strict=True)
                if f is not None and f.empty
            ]
            if i == 0:
                log.info("first batch: %d of %d markets returned history", len(frames), len(chunk))
                if not frames:
                    raise RuntimeError(
                        "no history for the oldest markets: the endpoint may not serve them"
                    )
            if empties:
                empty.update(empties)
                with open(empty_path, "a", encoding="utf-8") as fh:
                    fh.write("\n".join(empties) + "\n")
            if frames:
                out = pd.concat(frames, ignore_index=True)
                out.to_parquet(
                    os.path.join(data_dir, "prices", f"part-{part:05d}.parquet"), index=False
                )
                part += 1
            log.info(
                "price history: %d / %d (%d empty so far)",
                min(i + 200, len(todo)),
                len(todo),
                len(empty),
            )


def ingest_binance(data_dir: str, symbol: str, start: str, end: str, interval: str = "1h") -> None:
    os.makedirs(os.path.join(data_dir, "spot"), exist_ok=True)
    t0, t1 = _ts(start) * 1000, _ts(end) * 1000
    rows = []
    with httpx.Client(timeout=30) as http:
        while t0 < t1:
            r = http.get(
                BINANCE,
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": t0,
                    "endTime": t1,
                    "limit": 1000,
                },
            )
            r.raise_for_status()
            kl = r.json()
            if not kl:
                break
            rows.extend(
                {"symbol": symbol, "ts": int(k[0] // 1000), "close": float(k[4])} for k in kl
            )
            t0 = int(kl[-1][0]) + 1
    df = pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts")
    df.to_parquet(os.path.join(data_dir, "spot", f"{symbol}.parquet"), index=False)
    log.info("spot %s: %d rows", symbol, len(df))


def main() -> None:
    p = argparse.ArgumentParser(description="pm-backtest ingestion")
    p.add_argument("--data-dir", default=os.environ.get("PMBT_DATA_DIR", "data"))
    sub = p.add_subparsers(dest="cmd", required=True)
    pm = sub.add_parser("polymarket")
    pm.add_argument("--start", required=True)
    pm.add_argument("--end", required=True)
    pm.add_argument("--min-volume", type=float, default=10_000)
    pm.add_argument("--fidelity", type=int, default=60, help="minutes per price point")
    pm.add_argument("--concurrency", type=int, default=2)
    pm.add_argument("--pace", type=float, default=0.5, help="seconds between requests")
    pm.add_argument("--relist", action="store_true", help="ignore an existing markets.parquet")
    pm.add_argument("--lookback-days", type=int, default=7, help="history before each market end")
    bn = sub.add_parser("binance")
    bn.add_argument("--symbol", default="BTCUSDT")
    bn.add_argument("--start", required=True)
    bn.add_argument("--end", required=True)
    args = p.parse_args()
    logging.basicConfig(
        level="INFO", format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.cmd == "polymarket":
        asyncio.run(
            ingest_polymarket(args.data_dir, args.start, args.end, args.min_volume, args.fidelity)
        )
    else:
        ingest_binance(args.data_dir, args.symbol, args.start, args.end)


if __name__ == "__main__":
    main()
