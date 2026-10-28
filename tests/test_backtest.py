"""Pricing and metric checks for the backtest engine."""

import math

import numpy as np
import pandas as pd

from vrp.backtest import bs_price_delta, performance_metrics


def test_put_call_parity():
    s, k, t, r, v = 4000.0, 4100.0, 0.1, 0.03, 0.18
    c, dc = bs_price_delta(s, k, t, r, v, "CALL")
    p, dp = bs_price_delta(s, k, t, r, v, "PUT")
    assert math.isclose(c - p, s - k * math.exp(-r * t), rel_tol=1e-10)
    assert math.isclose(dc - dp, 1.0, rel_tol=1e-12)


def test_metrics_on_known_curve():
    eq = pd.Series(np.r_[100.0, 110.0, 99.0, 121.0])
    m = performance_metrics(eq, rf=0.0)
    assert math.isclose(m["Total return (%)"], 21.0)
    assert math.isclose(m["Max drawdown (%)"], -10.0)
