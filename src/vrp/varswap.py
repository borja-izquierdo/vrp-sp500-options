"""Synthetic one-month variance swaps on the S&P 500.

A variance swap pays ``N_var * (RV^2 - K^2)`` at maturity, with ``RV^2`` the
annualised realised variance of daily log returns and ``K^2`` the variance
strike. The fair strike is the price of the replicating option strip, which is
what the VIX squared measures for a 30-day horizon (Carr & Wu, 2009; CBOE). A
*short* swap entered at ``K^2 = (VIX_t / 100)^2`` and held for 21 sessions
therefore earns the variance risk premium in its purest form, without option
execution, greeks or hedging.

Convention: P&L is reported per unit of **vega notional**, i.e. in volatility
points: ``(K^2 - RV^2) / (2K)`` with K and RV in vol points. A swap with
$100,000 vega notional gains or loses $100,000 per vol point of this number.
Real swaps trade slightly above VIX^2 (convexity, jumps) and usually carry a
cap at 2.5 x K; both are ignored here and stated as limitations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def forward_realised_variance(close: pd.Series, horizon: int = 21) -> pd.Series:
    """Annualised realised variance of log returns over (t, t+horizon], zero-mean convention."""
    r2 = np.log(close / close.shift(1)) ** 2
    return r2.rolling(horizon).sum().shift(-horizon) * TRADING_DAYS / horizon


def swap_panel(underlying: pd.DataFrame, macro: pd.DataFrame, horizon: int = 21,
               cap: Optional[float] = None) -> pd.DataFrame:
    """Daily panel: strike (vol pts), forward realised vol (vol pts) and short-swap P&L.

    ``cap`` (e.g. 2.5) caps realised volatility at ``cap x K``, as in capped
    variance swaps; None = uncapped.

    Columns: date, k_vol, rv_vol, pnl_vega (vol points per unit vega notional),
    pnl_var (variance points, i.e. vol pts squared).
    """
    u = underlying[["date", "close"]].sort_values("date").reset_index(drop=True)
    u["rv_var"] = forward_realised_variance(u["close"], horizon) * 1e4  # vol pts^2
    df = u.merge(macro[["date", "macro_VIX"]], on="date", how="left")
    df["k_vol"] = df["macro_VIX"]
    if cap is not None:
        df["rv_var"] = np.minimum(df["rv_var"], (cap * df["k_vol"]) ** 2)
    df["rv_vol"] = np.sqrt(df["rv_var"])
    df["pnl_var"] = df["k_vol"] ** 2 - df["rv_var"]
    df["pnl_vega"] = df["pnl_var"] / (2.0 * df["k_vol"])
    return df[["date", "k_vol", "rv_vol", "pnl_var", "pnl_vega"]].dropna().reset_index(drop=True)


def monthly_strategy(panel: pd.DataFrame, signal: Optional[pd.DataFrame] = None,
                     threshold: float | pd.Series = 0.0, horizon: int = 21,
                     offset: int = 0, start: Optional[pd.Timestamp] = None,
                     end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Non-overlapping monthly short swaps entered every ``horizon`` sessions.

    With ``signal`` (date, prediction) a swap is entered only when the
    prediction exceeds ``threshold`` (a scalar or a per-date Series); otherwise
    the month is flat (P&L 0). ``offset`` shifts the entry calendar so results
    can be averaged over all 21 possible start days.

    Returns one row per entry date with columns date, traded, pnl_vega, k_vol, rv_vol.
    """
    p = panel
    if start is not None:
        p = p[p["date"] >= start]
    if end is not None:
        p = p[p["date"] <= end]
    p = p.iloc[offset::horizon].copy()
    if signal is None:
        p["traded"] = True
    else:
        s = signal.set_index("date")["prediction"]
        thr = (threshold.reindex(p["date"]).values if isinstance(threshold, pd.Series)
               else threshold)
        p["traded"] = (s.reindex(p["date"]).values > thr)
    p["pnl_vega"] = np.where(p["traded"], p["pnl_vega"], 0.0)
    return p[["date", "traded", "pnl_vega", "k_vol", "rv_vol"]].reset_index(drop=True)


def strategy_stats(trades: pd.DataFrame, horizon: int = 21) -> dict[str, float]:
    """Mean monthly P&L, its volatility, annualised Sharpe of monthly P&L, worst month, hit rate."""
    x = trades["pnl_vega"].astype(float)
    per_year = TRADING_DAYS / horizon
    traded = trades["traded"].astype(bool)
    return {
        "months": int(len(x)), "months_traded": int(traded.sum()),
        "mean_pnl": float(x.mean()), "vol_pnl": float(x.std(ddof=1)),
        "sharpe_ann": float(x.mean() / x.std(ddof=1) * np.sqrt(per_year)) if x.std(ddof=1) > 0 else np.nan,
        "worst_month": float(x.min()),
        "hit_rate": float((x[traded] > 0).mean()) if traded.any() else np.nan,
        "skew": float(x[traded].skew()) if traded.sum() > 2 else np.nan,
    }


def offset_averaged_stats(panel: pd.DataFrame, horizon: int = 21, **kwargs) -> pd.DataFrame:
    """Statistics of :func:`monthly_strategy` for every start offset 0..horizon-1."""
    rows = [{"offset": o, **strategy_stats(monthly_strategy(panel, horizon=horizon, offset=o, **kwargs),
                                           horizon)} for o in range(horizon)]
    return pd.DataFrame(rows)
