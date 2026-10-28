"""Realised and implied variance, and the variance-risk-premium target.

Realised volatility (thesis §3.3.1), annualised with 252 sessions:
close-to-close, Parkinson (1980), Garman-Klass (1980) and Yang-Zhang (2000).

Implied volatility (thesis §2.3): the model-free implied volatility (MFIV) is
the square root of the price of a variance swap, replicated with a strip of
out-of-the-money options (Carr & Wu, 2009; CBOE VIX methodology). The thesis
results use ``VIX / 100`` as the 30-day MFIV of the S&P 500, which is exactly
what the VIX computes. :func:`mfiv_from_strip` implements the replication on a
single expiration; it is unit-tested on synthetic prices but **was not used**
to produce the thesis results.

Target (thesis §3.4.1): ``VRP_t = MFIV_t - RV_YZ(t, t+21]``, the gap between
implied volatility today and Yang-Zhang realised volatility over the next 21
sessions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
LN2 = np.log(2.0)


# ---------------------------------------------------------------------------
# Realised volatility estimators (trailing, annualised)
# ---------------------------------------------------------------------------
def rv_close_to_close(ohlc: pd.DataFrame, window: int) -> pd.Series:
    """Standard deviation of daily log returns."""
    r = np.log(ohlc["close"] / ohlc["close"].shift(1))
    return r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def rv_parkinson(ohlc: pd.DataFrame, window: int) -> pd.Series:
    """Parkinson (1980) high-low range estimator."""
    hl2 = np.log(ohlc["high"] / ohlc["low"]) ** 2
    return np.sqrt(TRADING_DAYS / (4.0 * LN2) * hl2.rolling(window).mean())


def rv_garman_klass(ohlc: pd.DataFrame, window: int) -> pd.Series:
    """Garman-Klass (1980) OHLC estimator."""
    hl2 = np.log(ohlc["high"] / ohlc["low"]) ** 2
    co2 = np.log(ohlc["close"] / ohlc["open"]) ** 2
    term = 0.5 * hl2 - (2.0 * LN2 - 1.0) * co2
    return np.sqrt(TRADING_DAYS * term.rolling(window).mean())


def rv_yang_zhang(ohlc: pd.DataFrame, window: int) -> pd.Series:
    """Yang-Zhang (2000): overnight + open-to-close + Rogers-Satchell, drift-independent."""
    o = np.log(ohlc["open"] / ohlc["close"].shift(1))   # overnight jump
    c = np.log(ohlc["close"] / ohlc["open"])             # open to close
    u = np.log(ohlc["high"] / ohlc["open"])
    d = np.log(ohlc["low"] / ohlc["open"])
    rs = u * (u - c) + d * (d - c)                       # Rogers-Satchell
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var = (o.rolling(window).var(ddof=1) + k * c.rolling(window).var(ddof=1)
           + (1.0 - k) * rs.rolling(window).mean())
    return np.sqrt(TRADING_DAYS * var)


def realised_vol_panel(ohlc: pd.DataFrame, window: int) -> pd.DataFrame:
    """The four estimators aligned by date: rv_cc, rv_parkinson, rv_gk, rv_yz."""
    out = pd.DataFrame({"date": ohlc["date"].values})
    out["rv_cc"] = rv_close_to_close(ohlc, window).values
    out["rv_parkinson"] = rv_parkinson(ohlc, window).values
    out["rv_gk"] = rv_garman_klass(ohlc, window).values
    out["rv_yz"] = rv_yang_zhang(ohlc, window).values
    return out


def forward_yang_zhang(ohlc: pd.DataFrame, horizon: int) -> pd.Series:
    """Yang-Zhang volatility over (t, t+horizon]: the trailing estimate shifted back."""
    return rv_yang_zhang(ohlc, horizon).shift(-horizon)


# ---------------------------------------------------------------------------
# Model-free implied volatility
# ---------------------------------------------------------------------------
def mfiv_from_vix(macro: pd.DataFrame) -> pd.DataFrame:
    """30-day MFIV of the S&P 500 as ``VIX / 100`` (used for all thesis results)."""
    out = macro[["date", "macro_VIX"]].copy()
    out["mfiv"] = out["macro_VIX"] / 100.0
    return out[["date", "mfiv"]]


def mfiv_from_strip(strikes: np.ndarray, call_mid: np.ndarray, put_mid: np.ndarray,
                    r: float, t_years: float) -> float:
    """Variance-swap replication for one expiration (CBOE VIX formula).

    ``sigma^2 = 2/T * sum(dK/K^2 * e^{rT} * Q(K)) - 1/T * (F/K0 - 1)^2``

    where F is the forward implied by put-call parity at the strike with the
    smallest |C - P|, K0 is the first strike at or below F, and Q(K) is the
    OTM option price (put below K0, call above, average at K0).

    Not used for the thesis results; see the module docstring.

    Parameters
    ----------
    strikes, call_mid, put_mid : arrays of equal length, sorted by strike.
    r : continuously compounded risk-free rate.
    t_years : time to expiration in years.

    Returns
    -------
    Annualised implied volatility, or NaN if the strip is too thin.
    """
    k = np.asarray(strikes, float)
    c = np.asarray(call_mid, float)
    p = np.asarray(put_mid, float)
    if len(k) < 3:
        return np.nan
    i_star = int(np.nanargmin(np.abs(c - p)))
    fwd = k[i_star] + np.exp(r * t_years) * (c[i_star] - p[i_star])
    below = np.where(k <= fwd)[0]
    if len(below) == 0:
        return np.nan
    i0 = below[-1]
    q = np.where(k < k[i0], p, c)
    q[i0] = 0.5 * (p[i0] + c[i0])
    dk = np.empty_like(k)
    dk[1:-1] = (k[2:] - k[:-2]) / 2.0
    dk[0], dk[-1] = k[1] - k[0], k[-1] - k[-2]
    var = (2.0 / t_years) * np.sum(dk / k**2 * np.exp(r * t_years) * q) \
        - (1.0 / t_years) * (fwd / k[i0] - 1.0) ** 2
    return float(np.sqrt(var)) if var > 0 else np.nan


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
def vrp_target(mfiv: pd.Series, rv_forward: pd.Series) -> pd.Series:
    """Volatility-points VRP: implied minus subsequently realised volatility."""
    return mfiv - rv_forward
