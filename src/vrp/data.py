"""Data acquisition, option-chain cleaning and local storage.

Sources
-------
* SPX option chains (end of day): ThetaData, *Options Standard* plan, via the
  official ``thetadata`` Python library. Requires ``THETADATA_API_KEY``.
* SPX daily OHLC: Stooq (no key), with yfinance (``^GSPC``) as fallback.
* VIX and US Treasury yields (3M, 10Y): FRED. Requires ``FRED_API_KEY``.
* CBOE SKEW index: yfinance (``^SKEW``).

Vendor data is **not** redistributed with this repository. ``download_data.py``
rebuilds the raw files in ``data/raw/`` given your own API access.
"""

from __future__ import annotations

import io
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

logger = logging.getLogger("vrp.data")

RAW_FILES = ("underlying", "macro", "chain")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Parameters for data download and the option-chain filters (thesis §3.2)."""

    # Study horizon
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"

    # Options (ThetaData). The thesis uses the monthly SPX contracts only.
    option_root: str = "SPX"
    # Calendar days requested before each expiration. 120 is the setting used for
    # the thesis run; the API caps a single request at 365 days.
    chain_window_days: int = 120

    # Underlying OHLC
    underlying_source: str = "stooq"  # 'stooq' | 'yfinance'
    stooq_symbol: str = "^spx"
    yfinance_symbol: str = "^GSPC"

    # Macro (FRED) and SKEW (yfinance)
    fred_url: str = "https://api.stlouisfed.org/fred/series/observations"
    fred_series: dict[str, str] = field(default_factory=lambda: {
        "VIXCLS": "macro_VIX", "DGS3MO": "macro_UST_3M", "DGS10": "macro_UST_10Y"})
    skew_symbol: str = "^SKEW"

    # Cleaning filters (§3.2)
    min_open_interest: int = 100
    max_rel_spread: float = 0.20
    dte_min: int = 15
    dte_max: int = 45
    moneyness_min: float = 0.85
    moneyness_max: float = 1.15

    request_timeout: int = 30


def load_credentials(env_file: str | Path = ".env") -> dict[str, bool]:
    """Load API keys from a ``.env`` file into the environment.

    Returns a dict telling which keys are present, never the keys themselves.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:  # python-dotenv is optional if keys are already exported
        pass
    return {k: bool(os.environ.get(k)) for k in ("THETADATA_API_KEY", "FRED_API_KEY")}


def _as_date(d: str | date | datetime) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    return pd.Timestamp(d).date()


def third_fridays(first_year: int, last_year: int) -> list[date]:
    """Standard monthly SPX expirations (third Friday of each month)."""
    out = []
    for y in range(first_year, last_year + 1):
        for m in range(1, 13):
            first = pd.Timestamp(year=y, month=m, day=1)
            first_fri = first + pd.Timedelta(days=(4 - first.weekday()) % 7)
            out.append((first_fri + pd.Timedelta(days=14)).date())
    return out


# ---------------------------------------------------------------------------
# Underlying OHLC (free sources)
# ---------------------------------------------------------------------------
def _normalise_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    keep = [c for c in ("date", "open", "high", "low", "close") if c in df.columns]
    return df[keep].sort_values("date").reset_index(drop=True)


def fetch_stooq_ohlc(symbol: str, start: str, end: str, timeout: int = 30) -> pd.DataFrame:
    """Daily OHLC from Stooq. ``urllib`` keeps the ``^`` in the symbol unescaped."""
    d1, d2 = (pd.Timestamp(x).strftime("%Y%m%d") for x in (start, end))
    url = f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")))
    if df.empty or "Date" not in df.columns:
        logger.warning("Stooq returned no data for %s.", symbol)
        return pd.DataFrame()
    return _normalise_ohlc(df)


def fetch_yfinance_ohlc(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Daily OHLC from yfinance (``^GSPC``, ``^SKEW``...)."""
    import yfinance as yf

    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(symbol, start=str(_as_date(start)), end=end_excl,
                      progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        logger.warning("yfinance returned no data for %s.", symbol)
        return pd.DataFrame()
    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    return _normalise_ohlc(raw)


def fetch_underlying(cfg: DataConfig) -> pd.DataFrame:
    """SPX OHLC from the configured source, falling back to the other one."""
    sources = [cfg.underlying_source] + [s for s in ("stooq", "yfinance")
                                         if s != cfg.underlying_source]
    for src in sources:
        try:
            df = (fetch_stooq_ohlc(cfg.stooq_symbol, cfg.start_date, cfg.end_date,
                                   cfg.request_timeout)
                  if src == "stooq"
                  else fetch_yfinance_ohlc(cfg.yfinance_symbol, cfg.start_date, cfg.end_date))
            if not df.empty:
                return df
        except Exception as exc:  # noqa: BLE001 - try the next source
            logger.warning("Underlying source '%s' failed: %s", src, exc)
    raise RuntimeError("Could not download SPX OHLC from any source.")


# ---------------------------------------------------------------------------
# Macro: FRED (VIX, Treasury yields) + SKEW
# ---------------------------------------------------------------------------
def fetch_fred_series(series_id: str, cfg: DataConfig) -> pd.Series:
    """One FRED series as a float Series indexed by date (missing '.' dropped)."""
    params = {"series_id": series_id, "api_key": os.environ.get("FRED_API_KEY", ""),
              "file_type": "json", "observation_start": cfg.start_date,
              "observation_end": cfg.end_date}
    resp = requests.get(cfg.fred_url, params=params, timeout=cfg.request_timeout)
    resp.raise_for_status()
    obs = pd.DataFrame(resp.json().get("observations", []))
    if obs.empty:
        return pd.Series(dtype=float, name=series_id)
    s = pd.to_numeric(obs["value"], errors="coerce")
    s.index = pd.to_datetime(obs["date"])
    return s.dropna().rename(series_id)


def fetch_macro(cfg: DataConfig) -> pd.DataFrame:
    """Panel with columns date, macro_VIX, macro_UST_3M, macro_UST_10Y, macro_SKEW."""
    panel: Optional[pd.DataFrame] = None
    for sid, name in cfg.fred_series.items():
        s = fetch_fred_series(sid, cfg)
        if s.empty:
            continue
        df = s.rename(name).rename_axis("date").reset_index()
        panel = df if panel is None else panel.merge(df, on="date", how="outer")
    skew = fetch_yfinance_ohlc(cfg.skew_symbol, cfg.start_date, cfg.end_date)
    if not skew.empty:
        skew = skew[["date", "close"]].rename(columns={"close": "macro_SKEW"})
        panel = skew if panel is None else panel.merge(skew, on="date", how="outer")
    if panel is None:
        raise RuntimeError("No macro series could be downloaded.")
    return panel.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# ThetaData option chains
# ---------------------------------------------------------------------------
class ThetaDataClient:
    """Thin wrapper on the official ``thetadata`` library (API-key authentication)."""

    CHAIN_COLUMNS = ["date", "expiration", "strike", "right", "bid", "ask", "volume",
                     "open_interest", "delta", "gamma", "theta", "vega",
                     "implied_vol", "underlying_price"]

    def __init__(self, cfg: DataConfig):
        from thetadata import ThetaClient
        from thetadata.errors import NoDataFoundError

        if not os.environ.get("THETADATA_API_KEY"):
            raise EnvironmentError("THETADATA_API_KEY is not set (see .env.example).")
        self.cfg = cfg
        self._no_data = NoDataFoundError
        self.client = ThetaClient(dataframe_type="pandas")

    def _call(self, method: str, **kwargs) -> pd.DataFrame:
        try:
            out = getattr(self.client, method)(**kwargs)
            return out if out is not None else pd.DataFrame()
        except self._no_data:
            return pd.DataFrame()

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        """Lower-case columns, derive the New York trading date, drop duplicates."""
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={c: c.lower() for c in df.columns})
        if "timestamp" in df.columns:
            ny = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
            df["date"] = pd.to_datetime(ny.dt.date)
        if "expiration" in df.columns:
            df["expiration"] = pd.to_datetime(df["expiration"])
        keys = [k for k in ("date", "expiration", "strike", "right") if k in df.columns]
        return df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)

    def option_chain_eod(self, expiration: date) -> pd.DataFrame:
        """End-of-day quotes, greeks, IV and open interest for one expiration.

        Two calls: ``option_history_greeks_eod`` (quotes, greeks, IV, underlying)
        and ``option_history_open_interest``, merged on (date, expiration, strike, right).
        """
        exp = _as_date(expiration)
        start = max(_as_date(self.cfg.start_date), exp - timedelta(days=self.cfg.chain_window_days))
        end = min(_as_date(self.cfg.end_date), exp)
        query = dict(symbol=self.cfg.option_root, expiration=exp, strike="*", right="both",
                     start_date=start, end_date=end)
        greeks = self._prepare(self._call("option_history_greeks_eod", **query))
        if greeks.empty:
            return greeks
        oi = self._prepare(self._call("option_history_open_interest", **query))
        if not oi.empty and "open_interest" in oi.columns:
            keys = ["date", "expiration", "strike", "right"]
            greeks = greeks.merge(oi[keys + ["open_interest"]], on=keys, how="left")
        return greeks[[c for c in self.CHAIN_COLUMNS if c in greeks.columns]]


def clean_chain(chain: pd.DataFrame, underlying: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Apply the §3.2 filters to a raw chain.

    Liquidity (volume > 0, open interest >= 100), quote sanity (0 < bid < ask,
    relative spread <= 20 %), maturity (15-45 DTE) and moneyness (0.85-1.15).
    Adds ``mid``, ``spot``, ``dte`` and ``moneyness``.
    """
    if chain.empty:
        return chain
    df = chain.copy()
    n0 = len(df)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df = df.merge(underlying[["date", "close"]].rename(columns={"close": "spot"}),
                  on="date", how="left")
    df = df[(df["volume"] > 0) & (df["open_interest"] >= cfg.min_open_interest)]
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df = df[df["bid"] < df["ask"]]
    df = df[((df["ask"] - df["bid"]) / df["mid"]) <= cfg.max_rel_spread]
    df["dte"] = (df["expiration"] - df["date"]).dt.days
    df = df[(df["dte"] >= cfg.dte_min) & (df["dte"] <= cfg.dte_max)]
    df["moneyness"] = df["strike"] / df["spot"]
    df = df[(df["moneyness"] >= cfg.moneyness_min) & (df["moneyness"] <= cfg.moneyness_max)]
    logger.info("Cleaning: %d -> %d rows (%.1f%% kept).", n0, len(df), 100 * len(df) / max(n0, 1))
    return df.reset_index(drop=True)


def download_chains(cfg: DataConfig, underlying: pd.DataFrame,
                    expirations: Iterable[date], parts_dir: Path) -> pd.DataFrame:
    """Download and clean one expiration at a time, checkpointing each to disk.

    Re-running skips expirations already saved in ``parts_dir`` (resumable; the
    full 2018-2025 download took about 19 hours on the Options Standard plan).
    """
    client = ThetaDataClient(cfg)
    parts_dir.mkdir(parents=True, exist_ok=True)
    expirations = list(expirations)
    for i, exp in enumerate(expirations, 1):
        path = parts_dir / f"{exp.isoformat()}.parquet"
        if path.exists():
            continue
        try:
            cleaned = clean_chain(client.option_chain_eod(exp), underlying, cfg)
            cleaned.to_parquet(path)  # saved even if empty, so it is not retried forever
        except Exception as exc:  # noqa: BLE001 - log and continue with the next expiry
            logger.warning("Expiration %s skipped: %s: %s", exp, type(exc).__name__, exc)
        logger.info("%d/%d expirations processed", i, len(expirations))
    parts = [pd.read_parquet(p) for p in sorted(parts_dir.glob("*.parquet"))]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# Local storage
# ---------------------------------------------------------------------------
def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file and coerce ``date``/``expiration`` columns to datetime."""
    df = pd.read_parquet(path)
    for col in ("date", "expiration"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def load_raw(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load ``underlying``, ``macro`` and ``chain`` from ``raw_dir``."""
    raw_dir = Path(raw_dir)
    missing = [n for n in RAW_FILES if not (raw_dir / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {raw_dir}. Run scripts/download_data.py first "
            "(requires ThetaData and FRED API keys).")
    return {n: read_parquet(raw_dir / f"{n}.parquet") for n in RAW_FILES}
