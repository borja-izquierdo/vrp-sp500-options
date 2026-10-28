"""Figures for the README and research summary (static matplotlib, PNG).

Colour roles: implied / model / ML = blue, realised / systematic = orange,
third series = aqua, buy & hold = neutral grey. Text is never coloured by series.
No figure uses two y-axes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK_2, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
})


def save(fig: plt.Figure, path: str | Path, dpi: int = 200) -> Path:
    """Save as PNG and close the figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# ---------------------------------------------------------------------------
# 1. The premium itself
# ---------------------------------------------------------------------------
def vrp_timeseries(features: pd.DataFrame) -> plt.Figure:
    """Implied (VIX) vs subsequently realised 21-day volatility, 2018-2025."""
    df = features.dropna(subset=["target"])
    share = 100 * (df["target"] > 0).mean()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(df["date"], df["rv_yz_forward"], color=ORANGE, lw=1.2,
            label="Realised over the next 21 sessions (Yang-Zhang)")
    ax.plot(df["date"], df["mfiv"], color=BLUE, lw=1.2, label="Implied today (VIX / 100)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Annualised volatility")
    ax.set_title("Implied volatility usually overstates what the S&P 500 then realises")
    ax.text(0.01, 0.97, f"VRP > 0 on {share:.1f}% of days · mean {df['target'].mean():.4f}",
            transform=ax.transAxes, va="top", color=INK_2, fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    _date_axis(ax)
    return fig


# ---------------------------------------------------------------------------
# 2. Forecast skill vs naive baselines
# ---------------------------------------------------------------------------
def forecast_vs_baselines(fold_metrics: pd.DataFrame, metric: str = "rmse") -> plt.Figure:
    """Per-fold out-of-sample error of the model and two naive forecasts."""
    names = {"model": ("XGBoost", BLUE), "train_mean": ("Training-set mean", ORANGE),
             "naive_vrp": ("Naive: VIX − trailing 21d RV", AQUA)}
    piv = fold_metrics.pivot(index="fold", columns="forecast", values=metric)
    spans = fold_metrics.drop_duplicates("fold").set_index("fold")
    labels = [f"Fold {k}\n{pd.Timestamp(spans.loc[k, 'start']):%b %Y}–"
              f"{pd.Timestamp(spans.loc[k, 'end']):%b %Y}" for k in piv.index]
    x = np.arange(len(piv))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, key in enumerate(names):
        label, colour = names[key]
        ax.bar(x + (i - 1) * w, piv[key], width=w - 0.03, color=colour,
               label=f"{label} (mean {piv[key].mean():.4f})")
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylabel(f"Out-of-sample {metric.upper()}\n(annualised vol, 0.05 = 5 vol pts)")
    ax.set_title("Forecast error by purged walk-forward fold: a modest edge, not in every fold")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=9, loc="upper right")
    return fig


# ---------------------------------------------------------------------------
# 3. Strategy equity and drawdown
# ---------------------------------------------------------------------------
def equity_and_drawdown(curves: dict[str, pd.DataFrame]) -> plt.Figure:
    """Equity (base 100) of the option strategies and drawdowns incl. buy & hold."""
    style = {"ML": (BLUE, "ML signal (= always sell)"), "Systematic": (ORANGE, "Systematic rule"),
             "Buy & hold": (GREY, "SPX buy & hold")}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.3, 1]})
    for name in ("ML", "Systematic"):
        c = curves[name]
        colour, label = style[name]
        ax1.plot(c["date"], 100 * c["equity"] / c["equity"].iloc[0], color=colour, label=label)
    ax1.axhline(100, color=INK_2, lw=0.8, ls=":")
    ax1.set_ylabel("Equity (start = 100)")
    ax1.set_title("Delta-hedged short straddle, Apr 2019 – Dec 2025 (2021: no option data, no trades)")
    ax1.legend(fontsize=9, loc="upper left")
    for name in ("Buy & hold", "Systematic", "ML"):
        c = curves[name]
        dd = c["equity"] / c["equity"].cummax() - 1
        colour, label = style[name]
        if name == "Buy & hold":
            ax2.fill_between(c["date"], dd, 0, color=colour, alpha=0.35, lw=0, label=label)
        else:
            ax2.plot(c["date"], dd, color=colour, lw=1.3, label=label)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax2.set_ylabel("Drawdown")
    ax2.legend(fontsize=9, loc="lower left", ncol=3)
    _date_axis(ax2)
    return fig


# ---------------------------------------------------------------------------
# 4. Hedge-band sensitivity
# ---------------------------------------------------------------------------
def hedge_band(sensitivity: pd.DataFrame) -> plt.Figure:
    """Total return and max drawdown against the delta-hedge band (small multiples)."""
    s = sensitivity
    x = np.arange(len(s))
    labels = [f"{b}\n{r} rebal." for b, r in zip(s["delta_band"], s["rebalances"])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(x, s["total_return_pct"], color=BLUE, width=0.6)
    ax1.axhline(0, color=INK_2, lw=0.8)
    ax1.set_title("Total return (%)")
    ax2.bar(x, s["max_drawdown_pct"], color=ORANGE, width=0.6)
    ax2.set_title("Max drawdown (%)")
    for ax, col in ((ax1, "total_return_pct"), (ax2, "max_drawdown_pct")):
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_xlabel("Hedge band (|net delta| per contract before rebalancing)")
        ax.grid(axis="x", visible=False)
        for xi, v in zip(x, s[col]):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8, color=INK)
    fig.suptitle("Hedging too often eats the premium in costs; hedging too little lets gamma eat it",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Volmageddon (descriptive)
# ---------------------------------------------------------------------------
def volmageddon(features: pd.DataFrame, start: str = "2018-01-02",
                end: str = "2018-03-15", event: str = "2018-02-05") -> plt.Figure:
    """Implied, trailing and forward realised vol, and the realised VRP, around 5 Feb 2018.

    Descriptive only: no strategy is simulated here (the backtest starts in 2019).
    """
    df = features[(features["date"] >= start) & (features["date"] <= end)]
    ev = pd.Timestamp(event)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True,
                                   gridspec_kw={"height_ratios": [1.3, 1]})
    ax1.plot(df["date"], df["mfiv"], color=BLUE, label="Implied (VIX / 100)")
    ax1.plot(df["date"], df["rv_yz"], color=AQUA, label="Realised, trailing 21d")
    ax1.plot(df["date"], df["rv_yz_forward"], color=ORANGE, label="Realised, next 21d")
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax1.set_ylabel("Annualised volatility")
    ax1.set_title("Volmageddon: implied vol was far below the volatility that followed")
    ax1.legend(fontsize=9, loc="upper left")
    pos = df["target"] >= 0
    ax2.bar(df["date"][pos], df["target"][pos], color=BLUE, width=0.8,
            label="Implied above next-21d realised")
    ax2.bar(df["date"][~pos], df["target"][~pos], color=ORANGE, width=0.8,
            label="Implied below next-21d realised")
    ax2.axhline(0, color=INK_2, lw=0.8)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax2.set_ylabel("Realised VRP\n(implied − next-21d realised)")
    ax2.legend(fontsize=9, loc="lower right")
    for ax in (ax1, ax2):
        ax.axvline(ev, color=INK_2, lw=0.8, ls="--")
    ax1.annotate("5 Feb 2018", (ev, ax1.get_ylim()[1]), xytext=(4, -12),
                 textcoords="offset points", fontsize=8.5, color=INK_2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.set_xlabel(f"{ev.year} · descriptive only, no trades simulated (the backtest starts in 2019)",
                   color=INK_2, fontsize=9)
    return fig


def save_all(figures: dict[str, plt.Figure], out_dir: str | Path,
             names: Optional[list[str]] = None) -> list[Path]:
    """Save a dict of figures as ``<out_dir>/<name>.png``."""
    return [save(fig, Path(out_dir) / f"{name}.png")
            for name, fig in figures.items() if names is None or name in names]
