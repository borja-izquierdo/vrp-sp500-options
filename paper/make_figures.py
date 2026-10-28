"""Figures for the research paper, in Spanish or English.

    python paper/make_figures.py --lang es
    python paper/make_figures.py --lang en

Requires data/raw/ and the outputs of scripts/run_pipeline.py and
scripts/run_extensions.py. Writes vector PDFs to paper/figures/<lang>/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrp import backtest as bt  # noqa: E402
from vrp import data, features, varswap  # noqa: E402
from vrp.plots import AQUA, BLUE, GREY, GRID, INK, INK_2, ORANGE  # noqa: E402  (also sets rcParams)

YELLOW = "#eda100"
EXT = ROOT / "reports" / "extensions"
RES = ROOT / "reports" / "results"

L = {
    "es": dict(
        implied="Implícita hoy (VIX/100)", realised_fwd="Realizada en las 21 sesiones siguientes (Yang-Zhang)",
        ann_vol="Volatilidad anualizada", vrp_share="VRP > 0 en el {s:.1f} % de los días · media {m:.4f}",
        year="Año", swap_pnl="P&L medio del swap corto\n(puntos de vol. por nocional vega)",
        no_options="sin datos\nde opciones", fold="Pliegue", rmse="RMSE fuera de muestra",
        xgb="XGBoost (tesis)", train_mean="Media de entrenamiento", naive="Ingenua: VIX − RV 21d",
        mean="media", target="VRP realizado", prediction="VRP previsto (XGBoost, tesis)",
        zero="cero", pred_entry="VRP previsto al abrir", vix_entry="VIX al abrir",
        trade_pnl="P&L de la operación (USD)", rho="Spearman ρ = {r:.2f} (p = {p:.2f})",
        ml_eq="ML (= vender siempre)", sys2018="Sistemática (simulada desde 2018)",
        sys_al="Sistemática (desde el inicio de la ventana)", equity="Patrimonio (inicio = 100)",
        drawdown="Máxima caída", bh="Buy & hold SPX",
        steps=["Tesis\n(punto medio,\nrf 2 %)", "+ colateral\nal T-Bill 3M", "+ venta\nal bid",
               "+ banda\nanidada"],
        sharpe="Ratio de Sharpe", band="Banda de cobertura (|delta neta| por contrato)",
        total_ret="Rendimiento total (%)", mdd="Máxima caída (%)", rebal="rebal.", unhedged="sin cob.", rebal_note="entre paréntesis, nº de rebalanceos",
        always="Vender siempre", ml_med="ML > mediana", ml_sized="ML con tamaño", naive_med="Ingenua > mediana",
        panel_nested="(a) Banda elegida en cada pliegue (anidada)", panel_050="(b) Banda fija 0,50",
        sharpe_diff="Diferencia de Sharpe frente a la referencia (IC 90 % y 95 %)",
        boot_labels={"Sharpe diff: ML > 0 - Always sell": "ML > 0 − vender siempre",
                     "Sharpe diff: ML > train median - Always sell": "ML > mediana − vender siempre",
                     "Sharpe diff: ML sized - Always sell": "ML con tamaño − vender siempre",
                     "Sharpe diff: Naive > train median - Always sell": "Ingenua > mediana − vender siempre",
                     "Var-swap Sharpe diff: ML > median - unconditional": "Swap: ML > mediana − incondicional"},
        cum_pnl="P&L acumulado\n(puntos de vol. por nocional vega)", uncond="Incondicional",
        cum_title="(a) P&L acumulado, entrada cada 21 sesiones",
        offset_title="(b) Sharpe en los 21 calendarios de entrada",
        uncapped_s="sin límite", capped_s="con límite 2,5·K",
        entry_date="Fecha de entrada del swap (vencimiento a 21 sesiones)",
        swap_strike="Strike (VIX)", swap_rv="Volatilidad realizada del swap",
        vol_pts="Volatilidad (%)", swap_pnl_short="P&L swap corto\n(pts. vol./nocional vega)",
        event="5 feb 2018",
    ),
    "en": dict(
        implied="Implied today (VIX/100)", realised_fwd="Realised over the next 21 sessions (Yang-Zhang)",
        ann_vol="Annualised volatility", vrp_share="VRP > 0 on {s:.1f}% of days · mean {m:.4f}",
        year="Year", swap_pnl="Mean short-swap P&L\n(vol points per unit vega notional)",
        no_options="no option\ndata", fold="Fold", rmse="Out-of-sample RMSE",
        xgb="XGBoost (thesis)", train_mean="Training-set mean", naive="Naive: VIX − 21d RV",
        mean="mean", target="Realised VRP", prediction="Predicted VRP (XGBoost, thesis)",
        zero="zero", pred_entry="Predicted VRP at entry", vix_entry="VIX at entry",
        trade_pnl="Trade P&L (USD)", rho="Spearman ρ = {r:.2f} (p = {p:.2f})",
        ml_eq="ML (= always sell)", sys2018="Systematic (simulated from 2018)",
        sys_al="Systematic (from window start)", equity="Equity (start = 100)",
        drawdown="Drawdown", bh="SPX buy & hold",
        steps=["Thesis\n(mid,\nrf 2%)", "+ collateral\nat 3M T-bill", "+ sell\nat bid", "+ nested\nband"],
        sharpe="Sharpe ratio", band="Hedge band (|net delta| per contract)",
        total_ret="Total return (%)", mdd="Max drawdown (%)", rebal="rebal.", unhedged="unhedged", rebal_note="number of rebalances in brackets",
        always="Always sell", ml_med="ML > median", ml_sized="ML sized", naive_med="Naive > median",
        panel_nested="(a) Band chosen in each fold (nested)", panel_050="(b) Band fixed at 0.50",
        sharpe_diff="Sharpe difference vs reference (90% and 95% CI)",
        boot_labels={"Sharpe diff: ML > 0 - Always sell": "ML > 0 − always sell",
                     "Sharpe diff: ML > train median - Always sell": "ML > median − always sell",
                     "Sharpe diff: ML sized - Always sell": "ML sized − always sell",
                     "Sharpe diff: Naive > train median - Always sell": "Naive > median − always sell",
                     "Var-swap Sharpe diff: ML > median - unconditional": "Swap: ML > median − unconditional"},
        cum_pnl="Cumulative P&L\n(vol points per unit vega notional)", uncond="Unconditional",
        cum_title="(a) Cumulative P&L, entry every 21 sessions",
        offset_title="(b) Sharpe over the 21 entry calendars",
        uncapped_s="uncapped", capped_s="capped at 2.5·K",
        entry_date="Swap entry date (21-session maturity)",
        swap_strike="Strike (VIX)", swap_rv="Realised vol of the swap",
        vol_pts="Volatility (%)", swap_pnl_short="Short-swap P&L\n(vol pts / vega notional)",
        event="5 Feb 2018",
    ),
}

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "font.family": "serif",
                     "mathtext.fontset": "dejavuserif"})
W = 6.3  # inches, text width


def dec(lang: str):
    """Decimal-comma formatter for Spanish axes."""
    if lang == "en":
        return None
    return FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=170)
    plt.close(fig)


def years(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=["es", "en"], default="es")
    lang = p.parse_args().lang
    t = L[lang]
    out = ROOT / "paper" / "figures" / lang
    out.mkdir(parents=True, exist_ok=True)
    fmt = dec(lang)

    def num(ax, axis="y"):
        if fmt is not None:
            (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)

    raw = data.load_raw(ROOT / "data" / "raw")
    U, C, M = raw["underlying"], raw["chain"], raw["macro"]
    feats = features.build_features(U, C, M)
    thesis = pd.read_csv(ROOT / "reports/thesis/oos_predictions.csv", parse_dates=["date"])

    # 01 VRP series ----------------------------------------------------------
    df = feats.dropna(subset=["target"])
    fig, ax = plt.subplots(figsize=(W, 2.7))
    ax.plot(df["date"], df["rv_yz_forward"], color=ORANGE, lw=0.9, label=t["realised_fwd"])
    ax.plot(df["date"], df["mfiv"], color=BLUE, lw=0.9, label=t["implied"])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel(t["ann_vol"])
    txt = t["vrp_share"].format(s=100 * (df.target > 0).mean(), m=df.target.mean())
    ax.text(0.01, 0.96, txt.replace(".", ",") if lang == "es" else txt, transform=ax.transAxes,
            va="top", color=INK_2, fontsize=7.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    years(ax)
    save(fig, out, "fig01_vrp_series")

    # 02 var swap by year ----------------------------------------------------
    yr = pd.read_csv(EXT / "varswap_by_year.csv")
    fig, ax = plt.subplots(figsize=(W, 2.4))
    cols = [BLUE if v >= 0 else ORANGE for v in yr["mean_pnl"]]
    ax.bar(yr["year"].astype(str), yr["mean_pnl"], color=cols, width=0.6)
    ax.axhline(0, color=INK_2, lw=0.7)
    i21 = list(yr["year"]).index(2021)
    ax.annotate(t["no_options"], (i21, yr["mean_pnl"].iloc[i21]), xytext=(0, 13),
                textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK_2)
    for i, v in enumerate(yr["mean_pnl"]):
        s = f"{v:+.1f}".replace(".", ",") if lang == "es" else f"{v:+.1f}"
        ax.text(i, v, s, ha="center", va="bottom" if v >= 0 else "top", fontsize=7, color=INK)
    ax.set_ylabel(t["swap_pnl"])
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.25)
    num(ax)
    save(fig, out, "fig02_varswap_by_year")

    # 03 forecast vs baselines by fold --------------------------------------
    fm = pd.read_csv(EXT / "fold_metrics_thesis.csv")
    piv = fm.pivot(index="fold", columns="forecast", values="rmse")
    spans = fm.drop_duplicates("fold").set_index("fold")
    x = np.arange(len(piv))
    w = 0.26
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for i, (key, lab, col) in enumerate([("model", t["xgb"], BLUE), ("train_mean", t["train_mean"], ORANGE),
                                         ("naive_vrp", t["naive"], AQUA)]):
        mval = piv[key].mean()
        ms = f"{mval:.4f}".replace(".", ",") if lang == "es" else f"{mval:.4f}"
        ax.bar(x + (i - 1) * w, piv[key], width=w - 0.03, color=col, label=f"{lab} ({t['mean']} {ms})")
    ax.set_xticks(x, [f"{t['fold']} {k}\n{pd.Timestamp(spans.loc[k, 'start']):%m/%Y}–"
                      f"{pd.Timestamp(spans.loc[k, 'end']):%m/%Y}" for k in piv.index], fontsize=7)
    ax.set_ylabel(t["rmse"])
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    num(ax)
    save(fig, out, "fig03_forecast_folds")

    # 04 prediction vs realised over time -----------------------------------
    fig, ax = plt.subplots(figsize=(W, 2.5))
    ax.plot(thesis["date"], thesis["target"], color=GREY, lw=0.8, label=t["target"])
    ax.plot(thesis["date"], thesis["prediction"], color=BLUE, lw=1.1, label=t["prediction"])
    ax.axhline(0, color=INK, lw=0.7, ls="--")
    ax.set_ylabel("VRP")
    ax.legend(loc="lower left", ncol=2)
    years(ax)
    num(ax)
    save(fig, out, "fig04_prediction_vs_target")

    # 05 trade-level scatter ---------------------------------------------------
    tr = pd.read_csv(EXT / "trades_always_sell.csv")
    info = pd.read_csv(EXT / "trade_level_information.csv").set_index("signal")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True)
    for ax, col, lab in ((axes[0], "pred_thesis", t["pred_entry"]), (axes[1], "vix", t["vix_entry"])):
        ax.scatter(tr[col], tr["pnl"], s=12, color=BLUE, alpha=0.75, edgecolors="white", linewidths=0.4)
        ax.axhline(0, color=INK_2, lw=0.7)
        ax.set_xlabel(lab)
        r = info.loc[col]
        s = t["rho"].format(r=r["spearman"], p=r["p_one_sided"])
        ax.set_title(s.replace("0.", "0,") if lang == "es" else s, loc="left", fontweight="normal")
        num(ax, "x")
    axes[0].set_ylabel(t["trade_pnl"])
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "." if lang == "es" else ",")))
    save(fig, out, "fig05_trade_level")

    # 06 corrected comparison (thesis settings) -----------------------------
    start, end = thesis["date"].min(), U["date"].max()
    runs = {
        t["ml_eq"]: (thesis[["date", "prediction"]], BLUE, "-"),
        t["sys2018"]: (bt.systematic_signal(feats), ORANGE, "-"),
        t["sys_al"]: (bt.systematic_signal(feats).query("date >= @start"), ORANGE, "--"),
    }
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for lab, (sig, col, ls) in runs.items():
        d = bt.VolatilityBacktest().run(sig, C, U)["daily"]
        d = d[(d.date >= start) & (d.date <= end)]
        ax.plot(d["date"], 100 * d["equity"] / d["equity"].iloc[0], color=col, ls=ls, lw=1.2, label=lab)
    ax.axhline(100, color=INK_2, lw=0.6, ls=":")
    ax.set_ylabel(t["equity"])
    ax.legend(loc="upper left")
    years(ax)
    save(fig, out, "fig06_corrected_comparison")

    # 07 Sharpe by accounting step ------------------------------------------
    dec_ = pd.read_csv(EXT / "cost_decomposition_always_sell.csv", index_col=0)
    sh = dec_.loc["Sharpe"].values
    fig, ax = plt.subplots(figsize=(W, 2.4))
    ax.bar(range(4), sh, color=[GREY, BLUE, BLUE, ORANGE], width=0.55)
    ax.axhline(0, color=INK_2, lw=0.7)
    for i, v in enumerate(sh):
        s = f"{v:+.2f}".replace(".", ",") if lang == "es" else f"{v:+.2f}"
        ax.text(i, v, s, ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax.set_xticks(range(4), t["steps"], fontsize=7)
    ax.set_ylabel(t["sharpe"])
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.2)
    num(ax)
    save(fig, out, "fig07_sharpe_steps")

    # 08 hedge band sensitivity (thesis) ------------------------------------
    s = pd.read_csv(RES / "hedge_band_sensitivity.csv")
    labs = [(t["unhedged"] if b == "unhedged" else (b.replace(".", ",") if lang == "es" else b))
            + f"\n({r})" for b, r in zip(s["delta_band"], s["rebalances"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.5))
    a1.bar(range(len(s)), s["total_return_pct"], color=BLUE, width=0.6)
    a2.bar(range(len(s)), s["max_drawdown_pct"], color=ORANGE, width=0.6)
    for ax, col, ttl in ((a1, "total_return_pct", t["total_ret"]), (a2, "max_drawdown_pct", t["mdd"])):
        ax.axhline(0, color=INK_2, lw=0.7)
        ax.set_title(ttl)
        ax.set_xticks(range(len(s)), labs, fontsize=6.8)
        ax.grid(axis="x", visible=False)
        for i, v in enumerate(s[col]):
            vs = f"{v:.2f}".replace(".", ",") if lang == "es" else f"{v:.2f}"
            ax.text(i, v, vs, ha="center", va="bottom" if v >= 0 else "top", fontsize=6.5)
        ax.margins(y=0.15)
        num(ax)
    fig.supxlabel(t["band"] + " · " + t["rebal_note"], fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, out, "fig08_hedge_band")

    # 09 honest strategies equity -------------------------------------------
    eq = pd.read_csv(EXT / "equity_honest.csv", parse_dates=["date"])
    keys = [("Always sell", t["always"], GREY), ("ML > train median", t["ml_med"], BLUE),
            ("ML sized", t["ml_sized"], AQUA), ("Naive > train median", t["naive_med"], ORANGE)]
    rate = (M.set_index("date")["macro_UST_3M"] / 100).reindex(U["date"]).ffill()
    sig050 = {}
    nested = pd.read_csv(EXT / "nested_predictions.csv", parse_dates=["date"])
    folds = pd.read_csv(EXT / "nested_folds.csv", parse_dates=["test_start"])
    thr = pd.Series(folds["train_target_median"].values, index=folds["test_start"]).reindex(
        nested["date"], method="ffill").values
    naive = feats[["date"]].assign(prediction=feats["mfiv"] - feats["rv_yz"]).dropna()
    naive = naive[naive.date >= start]
    nthr = pd.Series(folds["train_target_median"].values, index=folds["test_start"]).reindex(
        naive["date"], method="ffill").values
    sig050["Always sell"] = nested[["date"]].assign(prediction=1.0)
    sig050["ML > train median"] = nested[["date"]].assign(prediction=(nested.prediction.values > thr) * 1.0)
    sig050["ML sized"] = nested[["date"]].assign(prediction=1.0, size=np.clip(nested.prediction.values / thr, 0, 3))
    sig050["Naive > train median"] = naive[["date"]].assign(prediction=(naive.prediction.values > nthr) * 1.0)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 4.4), sharex=True)
    for key, lab, col in keys:
        a1.plot(eq["date"], 100 * eq[key] / eq[key].iloc[0], color=col, lw=1.1, label=lab)
        d = bt.VolatilityBacktest(bt.BacktestConfig(entry_price="bid")).run(sig050[key], C, U, cash_rate=rate)["daily"]
        d = d[(d.date >= start) & (d.date <= end)]
        a2.plot(d["date"], 100 * d["equity"] / d["equity"].iloc[0], color=col, lw=1.1, label=lab)
    for ax, ttl in ((a1, t["panel_nested"]), (a2, t["panel_050"])):
        ax.set_title(ttl)
        ax.set_ylabel(t["equity"])
        ax.axhline(100, color=INK_2, lw=0.6, ls=":")
    a1.legend(loc="upper left", ncol=2)
    years(a2)
    save(fig, out, "fig09_honest_equity")

    # 10 bootstrap forest ----------------------------------------------------
    bs = pd.read_csv(EXT / "bootstrap.csv")
    bs = bs[bs["what"].isin(t["boot_labels"].keys())].iloc[::-1]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    y = np.arange(len(bs))
    ax.hlines(y, bs["ci95_lo"], bs["ci95_hi"], color=GREY, lw=1.2)
    ax.hlines(y, bs["ci90_lo"], bs["ci90_hi"], color=BLUE, lw=3.0)
    ax.plot(bs["estimate"], y, "o", color=INK, ms=4)
    ax.axvline(0, color=INK_2, lw=0.7, ls="--")
    ax.set_yticks(y, [t["boot_labels"][w] for w in bs["what"]], fontsize=7.5)
    ax.set_xlabel(t["sharpe_diff"])
    ax.grid(axis="y", visible=False)
    num(ax, "x")
    save(fig, out, "fig10_bootstrap")

    # 11 var swap: cumulative (offset 0) + Sharpe across the 21 entry calendars --------------
    fig, axes = plt.subplots(2, 1, figsize=(W, 5.0))
    pan = varswap.swap_panel(U, M)
    pan_c = varswap.swap_panel(U, M, cap=2.5)
    thr_p = pd.Series(folds["train_target_median"].values, index=folds["test_start"]).reindex(
        pan["date"], method="ffill")
    thr_c = pd.Series(folds["train_target_median"].values, index=folds["test_start"]).reindex(
        pan_c["date"], method="ffill")
    strats = ((t["uncond"], None, GREY), (t["ml_med"], nested, BLUE), (t["naive_med"], naive, ORANGE))
    ax = axes[0]
    for lab, sig, col in strats:
        kw = {} if sig is None else dict(signal=sig, threshold=thr_p)
        m = varswap.monthly_strategy(pan, start=start, **kw)
        ax.step(m["date"], m["pnl_vega"].cumsum(), where="post", color=col, lw=1.1, label=lab)
    ax.axhline(0, color=INK_2, lw=0.6)
    ax.set_title(t["cum_title"])
    ax.set_ylabel(t["cum_pnl"])
    years(ax)
    ax.legend(loc="lower right")
    num(ax)
    ax = axes[1]
    rng = np.random.default_rng(0)
    for i, (lab, sig, col) in enumerate(strats):
        for j, (pp, th, fill, leglab) in enumerate(((pan, thr_p, col, t["uncapped_s"]),
                                                    (pan_c, thr_c, "white", t["capped_s"]))):
            kw = {} if sig is None else dict(signal=sig, threshold=th)
            st = varswap.offset_averaged_stats(pp, start=start, **kw)["sharpe_ann"]
            xc = i + (j - 0.5) * 0.36
            ax.scatter(xc + rng.uniform(-0.07, 0.07, len(st)), st, s=11, facecolor=fill,
                       edgecolor=col, linewidths=0.8, label=leglab if i == 0 else None)
            ax.hlines(st.mean(), xc - 0.13, xc + 0.13, color=INK, lw=1.2)
    ax.axhline(0, color=INK_2, lw=0.6)
    ax.set_xticks(range(3), [x[0] for x in strats], fontsize=7.5)
    ax.set_xlim(-0.6, 2.6)
    ax.set_title(t["offset_title"])
    ax.set_ylabel(t["sharpe"])
    ax.grid(axis="x", visible=False)
    leg = ax.legend(loc="upper left", fontsize=6.8, ncol=2)
    for h in leg.legend_handles:
        h.set_edgecolor(INK_2)
    num(ax)
    fig.tight_layout()
    save(fig, out, "fig11_varswap")

    # 12 Volmageddon via variance swaps -----------------------------------------
    vm = pd.read_csv(EXT / "varswap_volmageddon.csv", parse_dates=["date"])
    ev = pd.Timestamp("2018-02-05")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 3.8), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    a1.plot(vm["date"], vm["k_vol"], color=BLUE, lw=1.2, label=t["swap_strike"])
    a1.plot(vm["date"], vm["rv_vol"], color=ORANGE, lw=1.2, label=t["swap_rv"])
    a1.set_ylabel(t["vol_pts"])
    a1.legend(loc="upper left")
    pos = vm["pnl_vega"] >= 0
    a2.bar(vm["date"][pos], vm["pnl_vega"][pos], color=BLUE, width=0.8)
    a2.bar(vm["date"][~pos], vm["pnl_vega"][~pos], color=ORANGE, width=0.8)
    a2.axhline(0, color=INK_2, lw=0.7)
    a2.set_ylabel(t["swap_pnl_short"])
    for ax in (a1, a2):
        ax.axvline(ev, color=INK_2, lw=0.7, ls="--")
        num(ax)
    a1.annotate(t["event"], (ev, a1.get_ylim()[1]), xytext=(3, -10), textcoords="offset points",
                fontsize=7, color=INK_2)
    a2.set_xlabel(t["entry_date"])
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m" if lang == "es" else "%d %b"))
    save(fig, out, "fig12_volmageddon")
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
