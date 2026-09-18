"""Synthetic universe with known properties, used by every test."""

import math
import os

import numpy as np
import pandas as pd
import pytest

from pmbt.costs import SECONDS_PER_YEAR, fair_value
from pmbt.data import Store

T0 = 1_760_000_000  # arbitrary epoch second, aligned to the hour below
H = 3600


def synthetic(
    seed: int = 0,
    bias: float = 0.0,
    n_binary: int = 600,
    n_events: int = 40,
    n_crypto: int = 40,
    plant_sum: float = 0.93,
):
    rng = np.random.default_rng(seed)
    t0 = T0 - T0 % H
    markets, prices = [], []

    # binary event markets. True probability p_t = Phi(X_t / sqrt(T - t)) with X a random walk
    # is a martingale that converges to the outcome 1{X_T > 0}: calibrated by construction.
    # `bias` distorts the displayed price of favorites (down) and longshots (up).
    for k in range(n_binary):
        life_h = int(rng.integers(48, 24 * 20))
        end = t0 + int(rng.integers(1, 120)) * 86400
        ts = np.arange(end - life_h * H, end, H)
        x = rng.normal(0, math.sqrt(life_h) / 2) + np.cumsum(rng.normal(0, 1, life_h))
        remaining = np.arange(life_h, 0, -1)
        p_true = 0.5 * np.array([math.erfc(-v / math.sqrt(2)) for v in x / np.sqrt(remaining)])
        outcome = float(x[-1] + rng.normal() > 0)
        p = np.where(p_true >= 0.9, p_true - bias, np.where(p_true <= 0.1, p_true + bias, p_true))
        p = np.clip(p, 0.005, 0.995)
        mid = f"b{k}"
        markets.append(
            dict(
                market_id=mid,
                event_id=f"eb{k}",
                venue="polymarket",
                question=f"Will thing {k} happen?",
                end_ts=end,
                closed=True,
                outcome=outcome,
                volume=50_000.0,
                n_in_event=1,
                underlying=None,
                strike=None,
                market_type=None,
            )
        )
        prices.append(pd.DataFrame({"market_id": mid, "ts": ts, "p_yes": p}))

    # multi-outcome events, some with a planted Dutch book window
    planted = set(range(0, n_events, 4))
    for e in range(n_events):
        n_out = 4
        end = t0 + int(rng.integers(5, 100)) * 86400
        life_h = 24 * 10
        ts = np.arange(end - life_h * H, end, H)
        true = rng.dirichlet(np.ones(n_out) * 2)
        winner = int(rng.choice(n_out, p=true))
        base = np.clip(true[:, None] + rng.normal(0, 0.01, (n_out, life_h)), 0.01, 0.99)
        base = base / base.sum(axis=0)  # sums to exactly 1: no free lunch by construction
        if e in planted:
            base[:, 100:110] *= plant_sum
        for o in range(n_out):
            mid = f"e{e}o{o}"
            markets.append(
                dict(
                    market_id=mid,
                    event_id=f"ev{e}",
                    venue="polymarket",
                    question=f"Outcome {o} of event {e}?",
                    end_ts=end,
                    closed=True,
                    outcome=float(o == winner),
                    volume=80_000.0,
                    n_in_event=n_out,
                    underlying=None,
                    strike=None,
                    market_type=None,
                )
            )
            prices.append(pd.DataFrame({"market_id": mid, "ts": ts, "p_yes": base[o]}))

    # spot path and crypto digital markets priced at model value plus noise
    days = 90
    n_spot = days * 24
    sigma = 0.6
    spot_ts = np.arange(t0 - 30 * 86400, t0 - 30 * 86400 + n_spot * H, H)
    spot = 60_000 * np.exp(
        np.cumsum(rng.normal(0, sigma * math.sqrt(H / SECONDS_PER_YEAR), n_spot))
    )
    spot_df = pd.DataFrame({"symbol": "BTCUSDT", "ts": spot_ts, "close": spot})
    for c in range(n_crypto):
        end_idx = int(rng.integers(24 * 40, n_spot - 24))
        start_idx = end_idx - 24 * int(rng.integers(3, 20))
        end = int(spot_ts[end_idx])
        strike = float(spot[start_idx] * rng.uniform(0.93, 1.07))
        ts = spot_ts[start_idx:end_idx]
        fv = np.array(
            [
                fair_value("digital_above", float(spot[i]), strike, sigma, end - int(spot_ts[i]))[0]
                for i in range(start_idx, end_idx)
            ]
        )
        p = np.clip(fv + rng.normal(0, 0.08, len(fv)), 0.01, 0.99)
        outcome = float(spot[end_idx] >= strike)
        mid = f"c{c}"
        markets.append(
            dict(
                market_id=mid,
                event_id=f"ec{c}",
                venue="polymarket",
                question=f"Will Bitcoin be above ${strike:,.0f} on day {c}?",
                end_ts=end,
                closed=True,
                outcome=outcome,
                volume=100_000.0,
                n_in_event=1,
                underlying="BTC",
                strike=strike,
                market_type="digital_above",
            )
        )
        prices.append(pd.DataFrame({"market_id": mid, "ts": ts, "p_yes": p}))

    return pd.DataFrame(markets), pd.concat(prices, ignore_index=True), spot_df


@pytest.fixture
def store():
    m, p, s = synthetic()
    return Store(m, p, s)


@pytest.fixture
def biased_store():
    m, p, s = synthetic(seed=1, bias=0.06)
    return Store(m, p, s)


@pytest.fixture
def data_dir(tmp_path):
    m, p, s = synthetic()
    root = tmp_path / "data"
    os.makedirs(root / "prices")
    os.makedirs(root / "spot")
    m.to_parquet(root / "markets.parquet", index=False)
    p.iloc[: len(p) // 2].to_parquet(root / "prices" / "part-00000.parquet", index=False)
    p.iloc[len(p) // 2 :].to_parquet(root / "prices" / "part-00001.parquet", index=False)
    s.to_parquet(root / "spot" / "BTCUSDT.parquet", index=False)
    return str(root)
