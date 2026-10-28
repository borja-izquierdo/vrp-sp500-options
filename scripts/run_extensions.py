"""Post-thesis robustness analyses (used in the public research paper).

    python scripts/run_extensions.py            # reuses reports/extensions/nested_predictions.csv
    python scripts/run_extensions.py --rerun-nested

Outputs go to reports/extensions/ (CSV) and paper/figures/ (PDF + PNG).

1. Forecast tests: Diebold-Mariano vs naive baselines, thesis and nested predictions.
2. Trade-level information: does the forecast at entry rank straddle P&L?
3. Option strategies under (a) thesis settings and (b) honest settings: nested
   predictions, nested hedge band, sell at the bid, collateral earning the 3M T-bill.
   Signals: always sell, forecast > 0, forecast > training median VRP, forecast-
   proportional size, and the naive signal with the same threshold.
4. Synthetic one-month variance swaps 2018-2025, incl. 2021 and Volmageddon.
5. Block-bootstrap confidence intervals for Sharpe ratios and differences.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrp import backtest as bt  # noqa: E402
from vrp import data, evaluation, features, models, varswap  # noqa: E402

log = logging.getLogger("run_extensions")
BANDS = (0.05, 0.10, 0.25, 0.50, 1.00, 1e9)


def band_label(b: float) -> str:
    return "unhedged" if b >= 1e9 else f"{b:.2f}"


def per_fold_series(dates: pd.Series, folds: pd.DataFrame, col: str) -> pd.Series:
    """Step function: value of ``col`` for the fold whose test block contains each date."""
    starts = pd.to_datetime(folds["test_start"])
    s = pd.Series(folds[col].values, index=starts).sort_index()
    return pd.Series(s.reindex(pd.to_datetime(dates), method="ffill").values, index=pd.to_datetime(dates))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--rerun-nested", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = ROOT / "reports" / "extensions"
    out.mkdir(parents=True, exist_ok=True)

    raw = data.load_raw(args.data_dir / "raw")
    U, C, M = raw["underlying"], raw["chain"], raw["macro"]
    feats = features.build_features(U, C, M)
    thesis = pd.read_csv(ROOT / "reports/thesis/oos_predictions.csv", parse_dates=["date"])

    # --- nested predictions --------------------------------------------------
    npath, fpath = out / "nested_predictions.csv", out / "nested_folds.csv"
    if args.rerun_nested or not npath.exists():
        res = models.nested_walk_forward(feats)
        res["predictions"].to_csv(npath, index=False)
        res["folds"].to_csv(fpath, index=False)
    nested = pd.read_csv(npath, parse_dates=["date"])
    folds = pd.read_csv(fpath, parse_dates=["train_end", "test_start", "test_end"])

    # =========================================================================
    # 1. Forecast tests
    # =========================================================================
    rows = []
    for name, pr in (("thesis", thesis), ("nested", nested)):
        base = models.baseline_fold_metrics(feats, pr)
        base.round(5).to_csv(out / f"fold_metrics_{name}.csv", index=False)
        for bench in ("train_mean", "naive_vrp"):
            # aligned error series per fold
            errs_m, errs_b = [], []
            df = (feats[feats["target"].notna()].sort_values("date").reset_index(drop=True)
                  .merge(pr[["date", "prediction"]], on="date", how="left"))
            for k, (tr, te) in enumerate(models.PurgedTimeSeriesSplit().split(len(df)), 1):
                t = df.iloc[te]
                yb = (np.full(len(t), df["target"].iloc[tr].mean()) if bench == "train_mean"
                      else (t["mfiv"] - t["rv_yz"]).values)
                ok = ~np.isnan(yb)
                errs_m.append((t["target"] - t["prediction"]).values[ok])
                errs_b.append((t["target"].values - yb)[ok])
            em, eb = np.concatenate(errs_m), np.concatenate(errs_b)
            for loss in ("se", "ae"):
                rows.append({"predictions": name, "benchmark": bench,
                             **evaluation.diebold_mariano(em, eb, 21, loss)})
    pd.DataFrame(rows).to_csv(out / "dm_tests.csv", index=False)
    log.info("DM tests:\n%s", pd.DataFrame(rows).round(4))

    # =========================================================================
    # 2. Trade-level information (thesis settings, always sell)
    # =========================================================================
    always = thesis[["date"]].assign(prediction=1.0)
    r_always = bt.VolatilityBacktest(bt.BacktestConfig()).run(always, C, U)
    tr = pd.DataFrame([t for t in r_always["trades"] if t["action"] == "close"])
    tr["open_date"] = pd.to_datetime(tr["open_date"])
    for name, pr in (("thesis", thesis), ("nested", nested)):
        tr[f"pred_{name}"] = pr.set_index("date")["prediction"].reindex(tr["open_date"]).values
    tr["naive"] = (feats.set_index("date")["mfiv"] - feats.set_index("date")["rv_yz"]) \
        .reindex(tr["open_date"]).values
    tr["vix"] = feats.set_index("date")["vix"].reindex(tr["open_date"]).values
    tr.to_csv(out / "trades_always_sell.csv", index=False)
    rng = np.random.default_rng(42)
    info = []
    for col in ("pred_thesis", "pred_nested", "naive", "vix"):
        d = tr[["pnl", col]].dropna()
        rho = d["pnl"].rank().corr(d[col].rank())
        perm = np.array([d["pnl"].rank().corr(pd.Series(rng.permutation(d[col].rank().values),
                                                          index=d.index)) for _ in range(5000)])
        info.append({"signal": col, "n_trades": len(d), "spearman": rho,
                     "p_one_sided": float(np.mean(perm >= rho))})
    pd.DataFrame(info).to_csv(out / "trade_level_information.csv", index=False)
    log.info("Trade-level rank correlation with straddle P&L:\n%s", pd.DataFrame(info).round(3))

    # =========================================================================
    # 3. Option strategies
    # =========================================================================
    start, end = nested["date"].min(), U["date"].max()
    rate = (M.set_index("date")["macro_UST_3M"] / 100).reindex(U["date"]).ffill()
    honest = bt.BacktestConfig(entry_price="bid")

    # nested hedge band: best Sharpe of always-sell on data before each test block
    band_runs = {b: bt.VolatilityBacktest(bt.BacktestConfig(**{**honest.__dict__, "delta_band": b}))
                 .run(U[["date"]].assign(prediction=1.0), C, U, cash_rate=rate) for b in BANDS}
    choice = []
    for _, f in folds.iterrows():
        sc = {}
        for b, r in band_runs.items():
            d = r["daily"][r["daily"]["date"] <= f["train_end"]]
            sc[b] = bt.performance_metrics(d["equity"], rf_series=rate.reindex(d["date"]).values)["Sharpe"]
        best = max(sc, key=lambda b: -np.inf if np.isnan(sc[b]) else sc[b])
        choice.append({"fold": f["fold"], "train_end": f["train_end"].date(),
                       **{band_label(b): v for b, v in sc.items()}, "chosen": band_label(best),
                       "chosen_value": best})
    choice = pd.DataFrame(choice)
    choice.to_csv(out / "nested_band_choice.csv", index=False)
    band_sched = pd.Series(choice["chosen_value"].values, index=pd.to_datetime(folds["test_start"]))
    log.info("Nested hedge band per fold:\n%s", choice[["fold", "train_end", "chosen"]])

    thr = per_fold_series(nested["date"], folds, "train_target_median")
    naive_sig = feats[["date"]].assign(prediction=feats["mfiv"] - feats["rv_yz"])
    naive_oos = naive_sig[(naive_sig.date >= start)].dropna()
    signals = {
        "Always sell": nested[["date"]].assign(prediction=1.0),
        "ML > 0": nested[["date", "prediction"]],
        "ML > train median": nested[["date"]].assign(
            prediction=(nested["prediction"].values > thr.values).astype(float)),
        "ML sized": nested[["date"]].assign(
            prediction=1.0, size=np.clip(nested["prediction"].values / thr.values, 0, 3)),
        "Naive > train median": naive_oos[["date"]].assign(
            prediction=(naive_oos["prediction"].values
                        > per_fold_series(naive_oos["date"], folds, "train_target_median").values)
            .astype(float)),
    }

    def run_set(cfg: bt.BacktestConfig, cash, bands, sigs: dict) -> tuple[pd.DataFrame, dict]:
        cols, curves = {}, {}
        for name, sig in sigs.items():
            r = bt.VolatilityBacktest(cfg).run(sig, C, U, cash_rate=cash, band_schedule=bands)
            d = r["daily"]
            d = d[(d["date"] >= start) & (d["date"] <= end)].reset_index(drop=True)
            rfs = rate.reindex(d["date"]).values if cash is not None else None
            cols[name] = bt.performance_metrics(d["equity"], n_trades=bt.count_trades(r, start),
                                                rf_series=rfs)
            avg = np.mean([t["contracts"] for t in r["trades"]
                           if t["action"] == "open" and t["date"] >= start]) if bt.count_trades(r, start) else np.nan
            cols[name]["Avg contracts"] = avg
            curves[name] = d
        return pd.DataFrame(cols), curves

    # (a) thesis settings with thesis predictions: the corrected comparison
    sig_a = {"ML (thesis)": thesis[["date", "prediction"]],
             "Always sell": thesis[["date"]].assign(prediction=1.0),
             "Systematic (thesis, from 2018)": bt.systematic_signal(feats),
             "Systematic (from window start)": bt.systematic_signal(feats).query("date >= @start")}
    tab_a, curves_a = run_set(bt.BacktestConfig(), None, None, sig_a)
    tab_a.round(4).to_csv(out / "strategies_thesis_settings.csv")
    log.info("Thesis settings:\n%s", tab_a.round(3))

    # (b) honest settings
    tab_b, curves_b = run_set(honest, rate, band_sched, signals)
    bh = U[(U.date >= start) & (U.date <= end)].reset_index(drop=True)
    tab_b["Buy & hold"] = pd.Series(bt.performance_metrics(bh["close"], rf_series=rate.reindex(bh["date"]).values))
    tab_b.round(4).to_csv(out / "strategies_honest.csv")
    log.info("Honest settings:\n%s", tab_b.round(3))

    # (b') honest settings but fixed band 0.50, to isolate the band choice
    tab_c, curves_c = run_set(bt.BacktestConfig(entry_price="bid"), rate, None, signals)
    tab_c.round(4).to_csv(out / "strategies_honest_band050.csv")
    log.info("Honest settings, band fixed at 0.50:\n%s", tab_c.round(3))

    # cost decomposition for always sell: mid->bid and collateral
    steps = {
        "Thesis (mid, no interest, rf 2%)": (bt.BacktestConfig(), None, None, False),
        "+ collateral at 3M T-bill": (bt.BacktestConfig(), rate, None, True),
        "+ sell at bid": (bt.BacktestConfig(entry_price="bid"), rate, None, True),
        "+ nested hedge band": (honest, rate, band_sched, True),
    }
    dec = {}
    for name, (cfg, cash, bands, tb) in steps.items():
        r = bt.VolatilityBacktest(cfg).run(signals["Always sell"], C, U, cash_rate=cash, band_schedule=bands)
        d = r["daily"][(r["daily"]["date"] >= start)].reset_index(drop=True)
        dec[name] = bt.performance_metrics(d["equity"], rf_series=rate.reindex(d["date"]).values if tb else None,
                                           n_trades=bt.count_trades(r, start))
    pd.DataFrame(dec).round(4).to_csv(out / "cost_decomposition_always_sell.csv")
    log.info("Cost decomposition (always sell):\n%s", pd.DataFrame(dec).round(3))

    eq = pd.DataFrame({"date": curves_b["Always sell"]["date"]})
    for name, d in curves_b.items():
        eq[name] = d["equity"].values
    eq["Buy & hold"] = (1e6 * bh["close"] / bh["close"].iloc[0]).values
    eq["rf"] = rate.reindex(eq["date"]).values
    eq.to_csv(out / "equity_honest.csv", index=False)

    # =========================================================================
    # 4. Synthetic variance swaps
    # =========================================================================
    panel = varswap.swap_panel(U, M)
    panel.to_csv(out / "varswap_panel.csv", index=False)
    vs_rows = []
    specs = {
        "Unconditional, 2018-2025": dict(),
        "Unconditional, OOS window": dict(start=start),
        "ML > train median, OOS window": dict(start=start, signal=nested,
                                              threshold=per_fold_series(panel["date"], folds, "train_target_median")),
        "ML > 0, OOS window": dict(start=start, signal=nested, threshold=0.0),
        "Naive > train median, OOS window": dict(start=start, signal=naive_sig.dropna(),
                                                 threshold=per_fold_series(panel["date"], folds, "train_target_median")),
    }
    for name, kw in specs.items():
        st = varswap.offset_averaged_stats(panel, **kw)
        vs_rows.append({"strategy": name, **st.drop(columns="offset").mean().to_dict(),
                        "worst_month_min": st["worst_month"].min()})
    capped = varswap.swap_panel(U, M, cap=2.5)
    for name, kw in specs.items():
        st = varswap.offset_averaged_stats(capped, **kw)
        vs_rows.append({"strategy": f"{name} [cap 2.5K]", **st.drop(columns="offset").mean().to_dict(),
                        "worst_month_min": st["worst_month"].min()})
    vs = pd.DataFrame(vs_rows)
    vs.to_csv(out / "varswap_strategies.csv", index=False)
    log.info("Variance swaps (averaged over 21 entry offsets):\n%s", vs.round(3))
    vm = panel[(panel.date >= "2018-01-02") & (panel.date <= "2018-02-28")]
    vm.to_csv(out / "varswap_volmageddon.csv", index=False)
    log.info("Volmageddon window:\n%s", vm.round({"k_vol": 2, "rv_vol": 2, "pnl_var": 2, "pnl_vega": 2}).to_string(index=False))

    # yearly table of the daily-entered swap P&L (mean per entry day)
    yr = panel.assign(year=panel.date.dt.year).groupby("year").agg(
        mean_k=("k_vol", "mean"), mean_rv=("rv_vol", "mean"), mean_pnl=("pnl_vega", "mean"),
        share_positive=("pnl_vega", lambda s: (s > 0).mean()), worst=("pnl_vega", "min"))
    yr.to_csv(out / "varswap_by_year.csv")

    # =========================================================================
    # 5. Bootstrap
    # =========================================================================
    boot = []
    ex = {}
    for name, d in curves_b.items():
        r = d["equity"].pct_change().dropna().values
        ex[name] = r - rate.reindex(d["date"]).shift(1).values[1:] / 252
    ex["Buy & hold"] = bh["close"].pct_change().dropna().values - rate.reindex(bh["date"]).shift(1).values[1:] / 252
    for name, x in ex.items():
        boot.append({"what": f"Sharpe: {name}", **evaluation.bootstrap_sharpe(x)})
    for name in ("ML > 0", "ML > train median", "ML sized", "Naive > train median"):
        boot.append({"what": f"Sharpe diff: {name} - Always sell",
                     **evaluation.bootstrap_sharpe(ex[name], ex["Always sell"])})
    # variance swaps, monthly, offset 0
    for name in ("Unconditional, OOS window", "ML > train median, OOS window"):
        m = varswap.monthly_strategy(panel, **specs[name])["pnl_vega"].values
        boot.append({"what": f"Var-swap Sharpe (monthly): {name}",
                     **evaluation.bootstrap_sharpe(m, mean_block=3, periods=12)})
    m1 = varswap.monthly_strategy(panel, **specs["ML > train median, OOS window"])["pnl_vega"].values
    m0 = varswap.monthly_strategy(panel, **specs["Unconditional, OOS window"])["pnl_vega"].values
    boot.append({"what": "Var-swap Sharpe diff: ML > median - unconditional",
                 **evaluation.bootstrap_sharpe(m1, m0, mean_block=3, periods=12)})
    pd.DataFrame(boot).to_csv(out / "bootstrap.csv", index=False)
    log.info("Bootstrap:\n%s", pd.DataFrame(boot).round(3).to_string())


if __name__ == "__main__":
    main()
