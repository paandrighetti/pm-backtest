# pm-backtest

Walk-forward backtests of three well-known prediction-market strategy families on resolved
Polymarket markets, with costs, causal fills and a pre-declared selection rule. Independent
from `updown-desk`: this project works on hourly history of resolved markets, not on a live
recording.

## Strategies

| family | what is tested | source of the claim |
| --- | --- | --- |
| `favorite_carry` | buy the side priced at or above a threshold close to resolution, hold to payout | favorite-longshot bias: Thaler and Ziemba (1988); Snowberg and Wolfers (2010); prediction-market calibration: Page and Clemen (2013) |
| `dutch_book` | multi-outcome events whose YES prices sum below 1 minus costs (buy all) or above 1 plus costs (buy all NO) | arbitrage-free pricing; hourly data makes this an upper bound |
| `binary_hedge` | crypto price markets ("above $X on date", "reach $X by date") priced against a driftless-GBM fair value from spot and trailing realized volatility; enter on mispricing, then delta-hedge with spot | digital option pricing; hedging a digital near the strike is known to be unstable (Taleb, Dynamic Hedging, 1997), so hedged and unhedged PnL are reported side by side |

Each strategy is a pure signal generator over point-in-time data. Outcomes live in the
engine only. A signal at `t` fills at the first observation strictly after `t`, crosses a
configurable half spread, pays the venue fee, and is held to resolution.

## First results (September 2025 to September 2026, 84 160 resolved markets, volume >= 50 000)

Split at 2026-06-01: 54 520 markets in sample, 29 640 out of sample. Half spread 0.01.

- **Favorite carry.** Every cell of the grid loses in sample; the selected cell (buy at
  0.95 or above within a day of resolution) loses 0.8 % per trade out of sample on
  10 825 trades, t = -6.5, and still loses 0.5 % (t = -4.1) with the half spread set to
  zero. Measured at the mid, before any spread, favorites are overpriced at every tier:
  by 0.2 points above 0.98 (99.4 % realized against a mid of 0.996), by 2.6 points
  between 0.95 and 0.98 (93.2 % against 0.958) and by 4.6 points between 0.90 and 0.95.
  There is no favorite-longshot premium to collect on this universe; if anything the
  near-certainties are slightly dear. Caveat: a first-crossing entry on hourly last-trade
  prints can catch thin quotes that would not have filled, which would inflate the
  overpricing; only order-book data can settle that.
- **Dutch book.** Measured per book, not per leg. On capital the books returned 5.2 % in
  sample and 0.0 % out of sample; the residual sits entirely in the cheap YES books (19 %
  on 220 units of capital, 289 books) against nothing on the 2 008 NO books. The winners
  per book settle what that means. A true book has exactly one winning outcome. On the NO
  side (prices summed above 1) 54 % of the books had several winners: the markets were
  nested thresholds, not exclusive outcomes, and a sum above 1 is what nested markets look
  like without any mispricing. That side is not a test of arbitrage at all. On the YES
  side (prices summed below 1) about two thirds of the books were structurally valid and
  profit by construction; the rest bought an incomplete outcome set (the volume floor drops
  the small outcomes). What the valid YES books cannot show is whether the hourly prints
  were tradeable at any size: the whole side earned about forty units over three months at
  one share per leg. A valid test needs the negative-risk flag, the complete outcome list
  of each event and order-book depth; the first two are the head of the roadmap.
- **Crypto binary hedge.** The best in-sample cell (t = 4.1, selected among nine) loses
  9 % per trade out of sample (t = -2.4), hedged or not. The spot hedge halves the PnL
  variance as intended and costs about 1 % of capital, but the fair value it protects has
  no edge over the market price.

Same conclusion as the live recording in `updown-desk`: Polymarket prices beat simple
models, and every in-sample edge vanished out of sample.

## Protocol

1. Universe: closed markets with a known outcome and volume above a floor.
2. Split by resolution date. In sample: markets resolved before `split_date`. Out of
   sample: the rest.
3. Grid over parameters in sample. Selection rule, fixed in `config.yaml` before any
   run: highest t-statistic among parameter sets with at least `min_trades` trades.
4. The selected set alone is run out of sample. The full in-sample grid is printed so the
   number of comparisons is visible.
5. Metrics: return per unit of capital with a bootstrap confidence interval, annualized
   return on capital actually locked, daily Sharpe, max drawdown, hit rate, mean days
   locked. For `favorite_carry`, realized win rate by fill-price bucket against the
   break-even line. For `binary_hedge`, hedged over unhedged PnL variance and hedge costs.

## Data

```
pmbt-ingest polymarket --start 2025-09-01 --end 2026-09-01 --min-volume 10000
pmbt-ingest binance    --symbol BTCUSDT --start 2025-08-01 --end 2026-09-01
pmbt-report --config config.yaml
```

Markets and outcomes come from the Gamma API through the official `polymarket-client`
SDK; hourly prices from the CLOB `prices-history` endpoint; spot from Binance public
klines. Ingestion is resumable. Canonical tables are documented in `src/pmbt/schema.py`
and are plain parquet, so another source (Kalshi, a flat-file archive) plugs in by writing
the same columns.

## What this cannot do, on purpose

- **Alternative data with no archive.** Signals such as late-night activity around
  government buildings, aircraft trackers or social-media velocity have no clean history
  aligned with market prices. Any backtest of them would be built from events already
  known, which is selection bias, not evidence. The defensible path is a forward test with
  the hypothesis, data source, measurement and decision rule written down before the
  first observation. That belongs in a recorder such as `updown-desk`, not here.
- **Order-book fills.** Hourly prices are mid or last-trade series; there is no depth. The
  half spread is an assumption and is reported as one.
- **Kalshi cross-venue arbitrage.** Requires event matching across venues; the cost model
  has the Kalshi fee curve, the matcher does not exist yet.

## Known limitations

- Price history is fetched at 60-minute fidelity for the seven days before each
  resolution (the endpoint rejects longer windows), so `max_days` above 7 is capped at 7.
  The Dutch-book test counts quasi-synchronous quotes and overstates capturable arbitrage.
- Crypto question parsing is regex-based; markets with unusual wording are skipped and
  a mispricing above 0.40 is treated as a rules mismatch rather than an opportunity.
- Touch markets that resolve before their nominal end date are settled at the nominal
  date; the timing difference is ignored.
- Returns per trade are treated as independent in the bootstrap; correlated events
  (many markets resolving on one election night) make the intervals too narrow.

## Tests

`pytest -q` runs on a synthetic universe with known properties: a martingale probability
process (calibrated by construction) with an optional favorite bias, planted Dutch-book
windows, and a GBM spot with digital markets priced at model value plus noise. The tests
check causality of fills, that the carry strategy loses exactly the spread when prices are
calibrated and gains when favorites are underpriced, that the Dutch-book detector fires only
on planted windows, and that the hedge reduces PnL variance.

## Roadmap

1. Dutch book on valid sets only: keep the negative-risk flag and fetch the complete
   outcome list of each candidate event, then require exactly one winner per book.
2. Kalshi event matching for cross-venue books.
3. Order-book snapshots for the favorite carry, to separate overpricing from thin prints.
