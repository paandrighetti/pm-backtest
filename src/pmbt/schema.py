"""Canonical tables and the question parser for crypto price markets.

markets.parquet   one row per binary market
    market_id, event_id, venue, question, end_ts, closed, outcome (1 YES, 0 NO, NaN voided or
    unresolved), volume, n_in_event, underlying, strike, market_type
prices/*.parquet  market_id, ts, p_yes           (hourly or finer, one row per observation)
spot/*.parquet    symbol, ts, close              (hourly spot closes for hedging)

market_type for crypto questions:
    digital_above   "Will BTC be above $X on <date>"          pays if S_T >= X
    digital_below   "Will BTC be below $X on <date>"          pays if S_T <  X
    touch           "Will BTC reach / hit / dip to $X by ..."  pays if the barrier trades
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MARKET_COLUMNS = [
    "market_id",
    "event_id",
    "venue",
    "question",
    "end_ts",
    "closed",
    "outcome",
    "volume",
    "n_in_event",
    "underlying",
    "strike",
    "market_type",
]
PRICE_COLUMNS = ["market_id", "ts", "p_yes"]
SPOT_COLUMNS = ["symbol", "ts", "close"]

_UNDERLYING = [
    (re.compile(r"\b(bitcoin|btc)\b", re.I), "BTC"),
    (re.compile(r"\b(ethereum|ether|eth)\b", re.I), "ETH"),
    (re.compile(r"\b(solana|sol)\b", re.I), "SOL"),
    (re.compile(r"\b(xrp|ripple)\b", re.I), "XRP"),
]
_STRIKE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*([kKmM])?")
_ABOVE = re.compile(r"\b(above|higher than|greater than|at or above|over)\b", re.I)
_BELOW = re.compile(r"\b(below|lower than|less than|under)\b", re.I)
_TOUCH = re.compile(r"\b(reach|hit|touch|dip to|drop to|fall to)\b", re.I)


@dataclass(frozen=True)
class ParsedQuestion:
    underlying: str | None
    strike: float | None
    market_type: str | None


def parse_question(question: str) -> ParsedQuestion:
    """Classify a question. Anything ambiguous returns market_type None and is skipped."""
    underlying = next((name for rx, name in _UNDERLYING if rx.search(question)), None)
    m = _STRIKE.search(question)
    if underlying is None or m is None:
        return ParsedQuestion(underlying, None, None)
    strike = float(m.group(1).replace(",", ""))
    suffix = (m.group(2) or "").lower()
    strike *= {"k": 1e3, "m": 1e6}.get(suffix, 1.0)
    if _TOUCH.search(question):
        kind = "touch"
    elif _ABOVE.search(question) and not _BELOW.search(question):
        kind = "digital_above"
    elif _BELOW.search(question) and not _ABOVE.search(question):
        kind = "digital_below"
    else:
        kind = None
    return ParsedQuestion(underlying, strike, kind)
