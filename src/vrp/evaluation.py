"""Statistical inference for forecasts and strategies.

* Diebold-Mariano (1995) test of equal predictive accuracy, with a Newey-West
  (1987) long-run variance (lag h-1 = 20, because the 21-day target overlaps)
  and the Harvey, Leybourne & Newbold (1997) small-sample correction.
* Stationary block bootstrap (Politis & Romano, 1994) for the Sharpe ratio and
  for paired Sharpe differences between strategies (same resampled blocks for
  both series, in the spirit of Ledoit & Wolf, 2008).
"""

from __future__ import annotations

import math

import numpy as np

SEED = 42


def _normal_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def newey_west_variance(x: np.ndarray, lags: int) -> float:
    """Long-run variance of a mean with Bartlett weights."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    lrv = x @ x / n
    for k in range(1, lags + 1):
        lrv += 2.0 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n
    return float(lrv)


def diebold_mariano(e_model: np.ndarray, e_bench: np.ndarray, horizon: int = 21,
                    loss: str = "se") -> dict[str, float]:
    """DM test of H0: equal expected loss. Negative statistic = model more accurate.

    Returns the HLN-corrected statistic and the one-sided p-value for H1: model better,
    i.e. P(Z <= stat).
    """
    e1, e2 = np.asarray(e_model, float), np.asarray(e_bench, float)
    d = (e1**2 - e2**2) if loss == "se" else (np.abs(e1) - np.abs(e2))
    n = len(d)
    lrv = newey_west_variance(d, horizon - 1)
    dm = d.mean() / math.sqrt(lrv / n)
    hln = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = dm * hln
    # n > 1,000 in every use here, so the t(n-1) reference is replaced by the normal.
    return {"n": n, "mean_loss_diff": float(d.mean()), "dm_stat": float(stat),
            "p_one_sided": float(1.0 - _normal_sf(stat)), "loss": loss}


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator,
                                 n_boot: int = 1) -> np.ndarray:
    """``n_boot`` resamples of 0..n-1 with geometric block lengths (wrap-around), shape (n_boot, n)."""
    new = rng.random((n_boot, n)) < 1.0 / mean_block
    new[:, 0] = True
    starts = rng.integers(n, size=(n_boot, n))
    pos = np.where(new, np.arange(n), 0)
    block_start = np.maximum.accumulate(pos, axis=1)
    first = np.take_along_axis(starts, block_start, axis=1)
    return (first + (np.arange(n) - block_start)) % n


def sharpe(x: np.ndarray, periods: float = 252) -> np.ndarray | float:
    """Annualised Sharpe of the last axis."""
    x = np.asarray(x, float)
    sd = x.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = x.mean(axis=-1) / sd * np.sqrt(periods)
    return float(out) if np.ndim(out) == 0 else out


def bootstrap_sharpe(excess: np.ndarray, other: np.ndarray | None = None, n_boot: int = 10000,
                     mean_block: float = 21.0, periods: float = 252,
                     seed: int = SEED) -> dict[str, float]:
    """Bootstrap CI for the Sharpe of ``excess`` and, if given, of Sharpe(excess) - Sharpe(other).

    ``excess`` and ``other`` must be aligned excess-return series (same dates);
    both are resampled with the same blocks. Reports the point estimate, 90 % and
    95 % percentile intervals and the share of resamples <= 0 (one-sided p-value).
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(excess, float)
    ix = stationary_bootstrap_indices(len(a), mean_block, rng, n_boot)
    stats = sharpe(a[ix], periods)
    point = sharpe(a, periods)
    if other is not None:
        b = np.asarray(other, float)
        stats = stats - sharpe(b[ix], periods)
        point = point - sharpe(b, periods)
    q = np.nanpercentile(stats, [2.5, 5, 95, 97.5])
    return {"estimate": float(point), "ci95_lo": float(q[0]), "ci90_lo": float(q[1]),
            "ci90_hi": float(q[2]), "ci95_hi": float(q[3]),
            "p_le_zero": float(np.mean(stats <= 0))}
