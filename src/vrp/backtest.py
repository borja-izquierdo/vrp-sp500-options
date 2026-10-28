"""Event-driven backtest of a delta-hedged short straddle (thesis §4.3 and §5.2-5.5).

Mechanics
---------
* Entry: when flat and the day's signal exceeds ``signal_threshold``, sell one
  ATM straddle on the expiration closest to 30 DTE, at mid.
* Daily mark-to-market with chain mids and greeks. When a leg has no quote
  (notably the last 15 days before expiry, outside the cleaned 15-45 DTE
  chain) it is priced with Black-Scholes at its last observed IV.
* Delta hedge with the index: rebalance to delta-neutral when |net delta|
  exceeds ``delta_band`` x contracts x multiplier. Each hedge trade pays
  ``hedge_slippage_bps``; each option leg pays ``option_commission``.
* Exit: cash settlement at expiration.

Benchmarks: the same engine driven by the naive signal ``mfiv - rv_yz``
("systematic"), the same engine always selling ("always sell"), and SPX buy & hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Black-Scholes fallback pricer (no scipy dependency)
# ---------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price_delta(spot: float, strike: float, t_years: float, r: float,
                   sigma: float, right: str) -> tuple[float, float]:
    """Black-Scholes price and delta; intrinsic value when t <= 0 or sigma <= 0."""
    call = right.upper() == "CALL"
    if t_years <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        if call:
            return max(spot - strike, 0.0), (1.0 if spot > strike else 0.0)
        return max(strike - spot, 0.0), (-1.0 if spot < strike else 0.0)
    sq = sigma * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * t_years) / sq
    d2 = d1 - sq
    if call:
        return spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2), _norm_cdf(d1)
    return (strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1),
            _norm_cdf(d1) - 1.0)


# ---------------------------------------------------------------------------
# Configuration and state
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    """Strategy and cost parameters. ``delta_band=0.50`` is the thesis setting (§5.5)."""

    signal_threshold: float = 0.0
    structure: str = "straddle"          # 'straddle' | 'strangle' (thesis: straddle)
    strangle_target_delta: float = 0.25
    target_dte: int = 30
    contracts: int = 1
    multiplier: float = 100.0
    delta_band: float = 0.50
    risk_free_rate: float = 0.02
    option_commission: float = 1.0       # per leg and contract
    hedge_slippage_bps: float = 1.0
    initial_capital: float = 1_000_000.0
    entry_price: str = "mid"             # 'mid' (thesis) | 'bid' (sell at the bid)


@dataclass
class Leg:
    expiration: pd.Timestamp
    strike: float
    right: str
    qty: float         # negative = short
    price: float       # last mark
    iv: float          # last observed IV (for the BS fallback)


@dataclass
class Position:
    legs: list[Leg] = field(default_factory=list)
    hedge: float = 0.0            # index units
    open_date: Optional[pd.Timestamp] = None
    pnl: float = 0.0              # accumulated P&L net of costs (for trade-level analysis)

    @property
    def contracts(self) -> float:
        return max(abs(leg.qty) for leg in self.legs) if self.legs else 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class VolatilityBacktest:
    """Daily event loop over the SPX calendar."""

    def __init__(self, cfg: Optional[BacktestConfig] = None):
        self.cfg = cfg or BacktestConfig()

    def _t(self, leg: Leg, date: pd.Timestamp) -> float:
        return max((leg.expiration - date).days, 0) / 365.0

    def _mark(self, chain_day: Optional[pd.DataFrame], leg: Leg, spot: float,
              date: pd.Timestamp) -> tuple[float, float, float]:
        """Price, delta and IV of a leg: chain mid if quoted, else Black-Scholes."""
        if chain_day is not None and not chain_day.empty:
            m = chain_day[(chain_day["expiration"] == leg.expiration)
                          & (chain_day["strike"] == leg.strike)
                          & (chain_day["right_u"] == leg.right)]
            if len(m):
                row = m.iloc[0]
                price = float((row["bid"] + row["ask"]) / 2.0)
                delta = (float(row["delta"]) if pd.notna(row.get("delta"))
                         else bs_price_delta(spot, leg.strike, self._t(leg, date),
                                             self.cfg.risk_free_rate, leg.iv, leg.right)[1])
                iv = (float(row["implied_vol"])
                      if pd.notna(row.get("implied_vol")) and row["implied_vol"] > 0 else leg.iv)
                return price, delta, iv
        price, delta = bs_price_delta(spot, leg.strike, self._t(leg, date),
                                      self.cfg.risk_free_rate, leg.iv, leg.right)
        return price, delta, leg.iv

    def _select_legs(self, chain_day: pd.DataFrame, spot: float,
                     date: pd.Timestamp) -> list[tuple]:
        """(expiration, strike, right, mid, delta, iv) for each leg to sell."""
        g = chain_day
        if "dte" not in g.columns:
            g = g.assign(dte=(g["expiration"] - date).dt.days)
        exps = g.groupby("expiration")["dte"].first()
        gt = g[g["expiration"] == (exps - self.cfg.target_dte).abs().idxmin()]
        calls, puts = gt[gt["right_u"] == "CALL"], gt[gt["right_u"] == "PUT"]
        if calls.empty or puts.empty:
            return []
        if self.cfg.structure == "strangle" and "delta" in gt.columns:
            chosen = [calls.iloc[(calls["delta"] - self.cfg.strangle_target_delta).abs().values.argmin()],
                      puts.iloc[(puts["delta"] + self.cfg.strangle_target_delta).abs().values.argmin()]]
        else:
            atm = gt.iloc[(gt["strike"] - spot).abs().values.argmin()]["strike"]
            c, p = calls[calls["strike"] == atm], puts[puts["strike"] == atm]
            if c.empty or p.empty:
                return []
            chosen = [c.iloc[0], p.iloc[0]]
        legs = []
        for row in chosen:
            iv = float(row["implied_vol"]) if row.get("implied_vol", 0) and row["implied_vol"] > 0 else 0.2
            delta = float(row["delta"]) if pd.notna(row.get("delta")) else 0.0
            price = (float(row["bid"]) if self.cfg.entry_price == "bid"
                     else float((row["bid"] + row["ask"]) / 2.0))
            legs.append((row["expiration"], float(row["strike"]), row["right_u"], price, delta, iv))
        return legs

    def run(self, signal: pd.DataFrame, chain: pd.DataFrame, underlying: pd.DataFrame,
            cash_rate: Optional[pd.Series] = None,
            band_schedule: Optional[pd.Series] = None) -> dict:
        """Run the strategy.

        Parameters
        ----------
        signal : DataFrame with ``date`` and ``prediction``; sell when prediction > threshold.
            An optional ``size`` column sets the number of contracts (may be fractional;
            no trade when size <= 0); otherwise ``cfg.contracts`` is used.
        chain : cleaned option chain with quotes and greeks.
        underlying : SPX daily closes (``date``, ``close``).
        cash_rate : optional annual rate (decimal) indexed by date; equity earns
            ``rate_{t-1} / 252`` per session (collateral interest). None = no interest (thesis).
        band_schedule : optional delta band indexed by date (step function, ``asof``
            lookup); overrides ``cfg.delta_band``.

        Returns
        -------
        dict with ``daily`` (equity, pnl, cost, net_delta, rebalanced, position_open,
        returns), ``trades`` (list of open/close events) and ``final_equity``.
        """
        cfg = self.cfg
        u = underlying[["date", "close"]].sort_values("date").reset_index(drop=True)
        preds = signal.set_index("date")["prediction"]
        sizes = signal.set_index("date")["size"] if "size" in signal.columns else None
        rate = cash_rate.sort_index() if cash_rate is not None else None
        bands = band_schedule.sort_index() if band_schedule is not None else None
        ch = chain.assign(right_u=chain["right"].astype(str).str.upper())
        by_date = {d: g for d, g in ch.groupby("date")} if not ch.empty else {}

        equity, pos, spot_prev = cfg.initial_capital, None, None
        rows, trades = [], []
        for date, spot in zip(u["date"], u["close"].astype(float)):
            chain_day = by_date.get(date)
            pnl, cost, rebalanced, net_delta, interest = 0.0, 0.0, False, 0.0, 0.0
            if rate is not None and spot_prev is not None:
                r_prev = rate.asof(prev_date)
                interest = equity * (0.0 if pd.isna(r_prev) else r_prev) / TRADING_DAYS

            if pos is not None:
                opt_delta = 0.0
                for leg in pos.legs:
                    price, delta, iv = self._mark(chain_day, leg, spot, date)
                    pnl += leg.qty * (price - leg.price) * cfg.multiplier
                    leg.price, leg.iv = price, iv
                    opt_delta += leg.qty * delta * cfg.multiplier
                if spot_prev is not None:
                    pnl += pos.hedge * (spot - spot_prev)
                net_delta = opt_delta + pos.hedge
                band_t = cfg.delta_band if bands is None else float(bands.asof(date))
                band = band_t * pos.contracts * cfg.multiplier
                if abs(net_delta) > max(band, 1e-9):
                    target = -opt_delta
                    cost += abs(target - pos.hedge) * spot * cfg.hedge_slippage_bps * 1e-4
                    pos.hedge, net_delta, rebalanced = target, 0.0, True
                if date >= pos.legs[0].expiration:
                    cost += cfg.option_commission * len(pos.legs) * pos.contracts
                    pos.pnl += pnl - cost
                    trades.append({"action": "close", "date": date, "open_date": pos.open_date,
                                   "pnl": pos.pnl})
                    pos = None
                else:
                    pos.pnl += pnl - cost

            if (pos is None and chain_day is not None and date in preds.index
                    and preds[date] > cfg.signal_threshold):
                n = cfg.contracts if sizes is None else float(sizes.get(date, 0.0))
                spec = self._select_legs(chain_day, spot, date) if n > 0 else []
                if spec:
                    legs = [Leg(e, k, r, -n, px, iv) for (e, k, r, px, _, iv) in spec]
                    opt_delta = sum(-n * d * cfg.multiplier for (*_, d, _) in spec)
                    pos = Position(legs=legs, hedge=-opt_delta, open_date=date)
                    open_cost = (cfg.option_commission * len(legs) * n
                                 + abs(pos.hedge) * spot * cfg.hedge_slippage_bps * 1e-4)
                    cost += open_cost
                    pos.pnl -= open_cost
                    trades.append({"action": "open", "date": date, "contracts": n})

            equity += pnl - cost + interest
            rows.append({"date": date, "spot": spot, "equity": equity, "pnl": pnl, "cost": cost,
                         "interest": interest,
                         "net_delta": net_delta, "rebalanced": rebalanced,
                         "position_open": pos is not None})
            spot_prev, prev_date = spot, date

        daily = pd.DataFrame(rows)
        daily["returns"] = daily["equity"].pct_change()
        return {"daily": daily, "trades": trades, "final_equity": equity}


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
METRICS = ["Total return (%)", "CAGR (%)", "Annual vol (%)", "Sharpe", "Sortino",
           "Max drawdown (%)", "Calmar", "VaR 95% daily (%)", "CVaR 95% daily (%)", "Trades"]


def performance_metrics(equity: pd.Series, rf: float = 0.02, var_level: float = 0.95,
                        n_trades: Optional[int] = None,
                        rf_series: Optional[pd.Series] = None) -> dict[str, float]:
    """Return, risk and tail metrics from a daily equity curve.

    Sharpe and Sortino use excess returns over ``rf`` (annual, 2 % in the thesis)
    in the numerator; VaR/CVaR are historical and reported as positive losses.
    ``rf_series`` (annual decimal, same length as ``equity``) replaces the flat
    ``rf`` with the previous day's rate.
    """
    eq = equity.astype(float).reset_index(drop=True)
    ret = eq.pct_change().dropna()
    if len(ret) < 2 or eq.iloc[0] <= 0:
        return dict.fromkeys(METRICS, np.nan)
    years = len(ret) / TRADING_DAYS
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    sd = ret.std(ddof=1)
    if rf_series is None:
        excess = ret - rf / TRADING_DAYS
    else:
        rfs = pd.Series(np.asarray(rf_series, float)).ffill().bfill()
        excess = ret - rfs.shift(1).iloc[1:].values / TRADING_DAYS
    down = ret[ret < 0]
    dsd = down.std(ddof=1)
    dd = eq / eq.cummax() - 1
    q = np.percentile(ret, (1 - var_level) * 100)
    tail = ret[ret <= q]
    return {
        "Total return (%)": 100 * (eq.iloc[-1] / eq.iloc[0] - 1),
        "CAGR (%)": 100 * cagr,
        "Annual vol (%)": 100 * sd * np.sqrt(TRADING_DAYS),
        "Sharpe": excess.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan,
        "Sortino": (excess.mean() / dsd * np.sqrt(TRADING_DAYS)
                    if len(down) > 1 and dsd > 0 else np.nan),
        "Max drawdown (%)": 100 * dd.min(),
        "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "VaR 95% daily (%)": -100 * q,
        "CVaR 95% daily (%)": -100 * tail.mean() if len(tail) else np.nan,
        "Trades": n_trades if n_trades is not None else np.nan,
    }


def count_trades(result: dict, start: Optional[pd.Timestamp] = None) -> int:
    """Number of positions opened (on or after ``start`` if given)."""
    return sum(1 for t in result["trades"]
               if t["action"] == "open" and (start is None or t["date"] >= start))


def _window(daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return daily[(daily["date"] >= start) & (daily["date"] <= end)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Benchmarks (thesis §5.4) and sensitivity (§5.5)
# ---------------------------------------------------------------------------
def systematic_signal(features: pd.DataFrame) -> pd.DataFrame:
    """Naive VRP signal: implied vol minus trailing 21d Yang-Zhang vol."""
    s = features[["date"]].assign(prediction=features["mfiv"] - features["rv_yz"])
    return s.dropna(subset=["prediction"]).reset_index(drop=True)


def run_benchmark(predictions: pd.DataFrame, features: pd.DataFrame, chain: pd.DataFrame,
                  underlying: pd.DataFrame, cfg: Optional[BacktestConfig] = None) -> dict:
    """ML vs always-sell vs systematic vs buy & hold over the ML evaluation window.

    As in the thesis, the systematic strategy is simulated from the start of the
    sample and then cut to the window, so it may hold a position on day one.
    Trades are counted inside the window only (the thesis table counted the
    systematic strategy's 14 trades before the window as well: 76 vs 62).
    """
    cfg = cfg or BacktestConfig()
    start, end = pd.Timestamp(predictions["date"].min()), pd.Timestamp(underlying["date"].max())
    runs = {
        "ML": VolatilityBacktest(cfg).run(predictions, chain, underlying),
        "Always sell": VolatilityBacktest(cfg).run(
            predictions[["date"]].assign(prediction=1.0), chain, underlying),
        "Systematic": VolatilityBacktest(cfg).run(systematic_signal(features), chain, underlying),
    }
    curves = {name: _window(r["daily"], start, end)[["date", "equity"]] for name, r in runs.items()}
    u = _window(underlying[["date", "close"]].sort_values("date"), start, end)
    curves["Buy & hold"] = pd.DataFrame(
        {"date": u["date"], "equity": cfg.initial_capital * u["close"] / u["close"].iloc[0]})
    table = pd.DataFrame({
        name: performance_metrics(c["equity"], cfg.risk_free_rate,
                                  n_trades=count_trades(runs[name], start) if name in runs else None)
        for name, c in curves.items()}).reindex(METRICS)
    return {"table": table, "curves": curves, "runs": runs, "window": (start, end)}


def hedge_band_sensitivity(predictions: pd.DataFrame, chain: pd.DataFrame,
                           underlying: pd.DataFrame,
                           bands: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.00, np.inf),
                           base: Optional[BacktestConfig] = None) -> pd.DataFrame:
    """Re-run the ML strategy for several hedge bands (``inf`` = unhedged).

    Metrics are computed on the full simulated curve, as in thesis §5.5.
    """
    base = base or BacktestConfig()
    rows = []
    for band in bands:
        cfg = BacktestConfig(**{**base.__dict__, "delta_band": 1e9 if np.isinf(band) else band})
        res = VolatilityBacktest(cfg).run(predictions, chain, underlying)
        d = res["daily"]
        m = performance_metrics(d["equity"], cfg.risk_free_rate, n_trades=count_trades(res))
        rows.append({"delta_band": "unhedged" if np.isinf(band) else f"{band:.2f}",
                     "total_return_pct": m["Total return (%)"], "annual_vol_pct": m["Annual vol (%)"],
                     "sharpe": m["Sharpe"], "sortino": m["Sortino"],
                     "max_drawdown_pct": m["Max drawdown (%)"],
                     "rebalances": int(d["rebalanced"].sum()), "total_cost": float(d["cost"].sum())})
    return pd.DataFrame(rows)
