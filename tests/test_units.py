import math

import pytest

from pmbt.costs import SECONDS_PER_YEAR, CostModel, fair_value
from pmbt.schema import parse_question


def test_parse_question_variants():
    q = parse_question("Will Bitcoin be above $100,000 on September 30?")
    assert (q.underlying, q.strike, q.market_type) == ("BTC", 100_000.0, "digital_above")
    q = parse_question("Will ETH dip to $2k by June 30?")
    assert (q.underlying, q.strike, q.market_type) == ("ETH", 2_000.0, "touch")
    q = parse_question("Will Solana be below $120 on May 1?")
    assert (q.underlying, q.strike, q.market_type) == ("SOL", 120.0, "digital_below")
    assert parse_question("Bitcoin Up or Down - March 21").market_type is None
    assert parse_question("Will Bitcoin be between $90k and $100k?").market_type is None


def test_digital_limits_and_delta_sign():
    p, d = fair_value("digital_above", 101.0, 100.0, 0.5, 0)
    assert (p, d) == (1.0, 0.0)
    p, d = fair_value("digital_above", 100.0, 100.0, 0.5, 7 * 86400)
    assert 0.45 < p < 0.5 and d > 0
    pb, db = fair_value("digital_below", 100.0, 100.0, 0.5, 7 * 86400)
    assert pb == pytest.approx(1.0 - p) and db == pytest.approx(-d)


def test_touch_is_twice_the_digital_tail():
    s = 0.5 * math.sqrt(30 * 86400 / SECONDS_PER_YEAR)
    strike = 100.0 * math.exp(s)  # one sigma above
    p_touch, d_touch = fair_value("touch", 100.0, strike, 0.5, 30 * 86400)
    assert p_touch == pytest.approx(2 * 0.5 * math.erfc(1 / math.sqrt(2)), rel=1e-6)
    assert d_touch > 0
    p_down, d_down = fair_value("touch", 100.0, 100.0 / math.exp(s), 0.5, 30 * 86400)
    assert p_down == pytest.approx(p_touch) and d_down < 0
    assert fair_value("touch", 100.0, 120.0, 0.5, 0) == (0.0, 0.0)


def test_cost_model_presets():
    pm = CostModel(half_spread=0.01)
    price, cost = pm.entry_cost(0.95, 10)
    assert price == pytest.approx(0.96) and cost == pytest.approx(9.6)
    ks = CostModel(half_spread=0.0, venue="kalshi")
    assert ks.fee(0.5, 100) == pytest.approx(1.75)
    assert ks.fee(0.99, 1) == pytest.approx(0.01)  # rounded up to the cent
