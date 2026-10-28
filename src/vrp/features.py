"""Feature matrix and target (thesis §3.3-3.4).

Inputs: SPX OHLC, the cleaned option chain (§3.2) and the macro panel.
Output: one row per session with 22 predictors, ``rv_yz_forward`` and ``target``.

Predictor families
------------------
* Realised volatility (21d): rv_cc, rv_parkinson, rv_gk, rv_yz.
* Option microstructure (expiration closest to 30 DTE): atm_iv, skew_slope
  (25-delta put IV minus 25-delta call IV), pcr_volume, pcr_oi, iv_rank,
  iv_percentile (252-session look-back).
* Macro and momentum: ret_5d/10d/21d/63d, vix, vix_chg, skew, skew_chg,
  ust_3m, ust_10y, term_slope.
* Implied level: mfiv (= VIX/100).

All predictors use information available at the close of day t; only
``rv_yz_forward`` (and hence ``target``) looks forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from vrp.variance import forward_yang_zhang, mfiv_from_vix, realised_vol_panel, vrp_target

NON_FEATURE_COLUMNS = ("date", "target", "rv_yz_forward")


@dataclass
class FeatureConfig:
    """Windows and conventions for the feature matrix."""

    rv_window: int = 21
    target_horizon: int = 21
    rank_lookback: int = 252
    iv_target_dte: int = 30
    momentum_windows: tuple[int, ...] = (5, 10, 21, 63)
    skew_put_delta: float = -0.25
    skew_call_delta: float = 0.25


# ---------------------------------------------------------------------------
# Option microstructure
# ---------------------------------------------------------------------------
def _spot(g: pd.DataFrame) -> float:
    for col in ("underlying_price", "spot"):
        if col in g.columns and g[col].notna().any():
            return float(g[col].dropna().iloc[0])
    return np.nan


def _rolling_rank(s: pd.Series, lookback: int) -> pd.Series:
    lo = s.rolling(lookback, min_periods=lookback // 2).min()
    hi = s.rolling(lookback, min_periods=lookback // 2).max()
    return (s - lo) / (hi - lo) * 100.0


def _rolling_percentile(s: pd.Series, lookback: int) -> pd.Series:
    def pct(window: np.ndarray) -> float:
        cur, past = window[-1], window[:-1]
        past = past[~np.isnan(past)]
        if len(past) == 0 or np.isnan(cur):
            return np.nan
        return 100.0 * np.mean(past < cur)
    return s.rolling(lookback, min_periods=lookback // 2).apply(pct, raw=True)


def option_microstructure(chain: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Daily ATM IV, skew slope, put-call ratios, IV rank and IV percentile."""
    cols = ["date", "atm_iv", "skew_slope", "pcr_volume", "pcr_oi"]
    if chain is None or chain.empty:
        return pd.DataFrame(columns=cols + ["iv_rank", "iv_percentile"])

    df = chain.copy()
    df["right_u"] = df["right"].astype(str).str.upper()
    rows = []
    for d, g in df.groupby("date"):
        puts_all, calls_all = g[g["right_u"] == "PUT"], g[g["right_u"] == "CALL"]
        vol_c, oi_c = calls_all["volume"].sum(), calls_all["open_interest"].sum()
        pcr_vol = puts_all["volume"].sum() / vol_c if vol_c > 0 else np.nan
        pcr_oi = puts_all["open_interest"].sum() / oi_c if oi_c > 0 else np.nan

        if "dte" in g.columns and g["dte"].notna().any():
            exps = g.groupby("expiration")["dte"].first()
            gt = g[g["expiration"] == (exps - cfg.iv_target_dte).abs().idxmin()]
        else:
            gt = g

        spot = _spot(gt)
        atm_iv = np.nan
        if len(gt) and not np.isnan(spot):
            atm_iv = gt.loc[(gt["strike"] - spot).abs().idxmin(), "implied_vol"]

        skew = np.nan
        puts, calls = gt[gt["right_u"] == "PUT"], gt[gt["right_u"] == "CALL"]
        if "delta" in gt.columns and len(puts) and len(calls):
            ip = (puts["delta"] - cfg.skew_put_delta).abs().idxmin()
            ic = (calls["delta"] - cfg.skew_call_delta).abs().idxmin()
            skew = puts.loc[ip, "implied_vol"] - calls.loc[ic, "implied_vol"]

        rows.append({"date": d, "atm_iv": atm_iv, "skew_slope": skew,
                     "pcr_volume": pcr_vol, "pcr_oi": pcr_oi})

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["iv_rank"] = _rolling_rank(out["atm_iv"], cfg.rank_lookback)
    out["iv_percentile"] = _rolling_percentile(out["atm_iv"], cfg.rank_lookback)
    return out


# ---------------------------------------------------------------------------
# Macro and momentum
# ---------------------------------------------------------------------------
def macro_momentum(underlying: pd.DataFrame, macro: pd.DataFrame,
                   cfg: FeatureConfig) -> pd.DataFrame:
    """SPX log-return momentum, VIX/SKEW levels and changes, rates and curve slope."""
    out = underlying[["date", "close"]].copy()
    for w in cfg.momentum_windows:
        out[f"ret_{w}d"] = np.log(out["close"] / out["close"].shift(w))
    out = out.drop(columns=["close"])

    if macro is None or macro.empty:
        return out
    m = macro.sort_values("date").rename(columns={
        "macro_VIX": "vix", "macro_UST_3M": "ust_3m",
        "macro_UST_10Y": "ust_10y", "macro_SKEW": "skew"})
    if "vix" in m.columns:
        m["vix_chg"] = m["vix"].diff()
    if "skew" in m.columns:
        m["skew_chg"] = m["skew"].diff()
    if {"ust_10y", "ust_3m"} <= set(m.columns):
        m["term_slope"] = m["ust_10y"] - m["ust_3m"]
    return out.merge(m, on="date", how="left")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_features(underlying: pd.DataFrame, chain: pd.DataFrame, macro: pd.DataFrame,
                   cfg: Optional[FeatureConfig] = None) -> pd.DataFrame:
    """Feature matrix with 22 predictors plus ``rv_yz_forward`` and ``target``.

    The row set is the SPX trading calendar; option features are NaN on days
    without a cleaned chain (notably all of 2021, see README > Limitations).
    """
    cfg = cfg or FeatureConfig()
    u = underlying.sort_values("date").reset_index(drop=True)

    feats = realised_vol_panel(u, cfg.rv_window)
    feats = feats.merge(macro_momentum(u, macro, cfg), on="date", how="left")
    opt = option_microstructure(chain, cfg)
    if not opt.empty:
        feats = feats.merge(opt, on="date", how="left")
    feats = feats.merge(mfiv_from_vix(macro), on="date", how="left")

    fwd = pd.DataFrame({"date": u["date"].values,
                        "rv_yz_forward": forward_yang_zhang(u, cfg.target_horizon).values})
    feats = feats.merge(fwd, on="date", how="left")
    feats["target"] = vrp_target(feats["mfiv"], feats["rv_yz_forward"])
    return feats.sort_values("date").reset_index(drop=True)


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Predictor names (everything except date, target and the forward RV)."""
    return [c for c in features.columns if c not in NON_FEATURE_COLUMNS]
