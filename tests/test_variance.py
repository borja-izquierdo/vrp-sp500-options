"""Sanity tests for the volatility estimators and the variance-swap replication."""

import math

import numpy as np
import pandas as pd
import pytest

from vrp.backtest import bs_price_delta
from vrp.variance import TRADING_DAYS, mfiv_from_strip, rv_close_to_close, rv_yang_zhang


def _gbm_ohlc(sigma: float, n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Daily OHLC from a driftless GBM simulated on a fine intraday grid."""
    rng = np.random.default_rng(seed)
    steps = 78
    dt = 1 / (TRADING_DAYS * steps)
    log_s = np.cumsum(rng.normal(-0.5 * sigma**2 * dt, sigma * np.sqrt(dt), n * steps))
    paths = np.exp(log_s).reshape(n, steps)
    return pd.DataFrame({"date": pd.bdate_range("2000-01-03", periods=n),
                         "open": paths[:, 0], "high": paths.max(1),
                         "low": paths.min(1), "close": paths[:, -1]})


@pytest.mark.parametrize("estimator", [rv_close_to_close, rv_yang_zhang])
def test_estimators_recover_gbm_vol(estimator):
    sigma = 0.20
    rv = estimator(_gbm_ohlc(sigma), window=21).dropna()
    assert abs(rv.mean() - sigma) < 0.02


def test_mfiv_recovers_flat_black_scholes_vol():
    spot, r, t, sigma = 100.0, 0.02, 30 / 365, 0.25
    strikes = np.arange(40.0, 200.5, 0.5)
    calls = np.array([bs_price_delta(spot, k, t, r, sigma, "CALL")[0] for k in strikes])
    puts = np.array([bs_price_delta(spot, k, t, r, sigma, "PUT")[0] for k in strikes])
    assert math.isclose(mfiv_from_strip(strikes, calls, puts, r, t), sigma, rel_tol=5e-3)
