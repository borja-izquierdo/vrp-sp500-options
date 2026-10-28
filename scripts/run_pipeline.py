"""Reproduce every result and figure from the raw data in ``data/raw/``.

    python scripts/run_pipeline.py              # thesis predictions, ~1 min
    python scripts/run_pipeline.py --retrain    # also re-run model selection, ~3 min

Steps
-----
1. Build the feature matrix and VRP target           -> data/processed/features.parquet
2. Out-of-sample predictions: the frozen thesis file (reports/thesis/) by default,
   or a fresh LightGBM vs XGBoost search with --retrain -> reports/results/
3. Forecast vs naive baselines on the purged folds    -> reports/results/forecast_vs_baselines.csv
4. Backtest: ML vs always-sell vs systematic vs B&H  -> reports/results/benchmark_table.csv
5. Hedge-band sensitivity                             -> reports/results/hedge_band_sensitivity.csv
6. Figures                                            -> reports/figures/*.png

Why a frozen prediction file? XGBoost does not give bit-identical predictions
across platforms and library versions (we observed fold RMSE differences of up
to 0.003 between the thesis run on Windows and a Linux re-run). With
--retrain the backtest is unchanged because every prediction is still positive.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vrp import backtest, data, features, models, plots  # noqa: E402

log = logging.getLogger("run_pipeline")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    p.add_argument("--retrain", action="store_true",
                   help="re-run hyper-parameter search and walk-forward instead of "
                        "using the frozen thesis predictions")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()

    results = args.reports_dir / "results"
    thesis = args.reports_dir / "thesis"
    processed = args.data_dir / "processed"
    for d in (results, processed, args.reports_dir / "figures"):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Features --------------------------------------------------------------
    raw = data.load_raw(args.data_dir / "raw")
    feats = features.build_features(raw["underlying"], raw["chain"], raw["macro"])
    feats.to_parquet(processed / "features.parquet")
    tgt = feats["target"].dropna()
    log.info("Features: %d sessions x %d predictors; VRP mean %.4f, positive %.1f%%",
             len(feats), len(features.feature_columns(feats)), tgt.mean(), 100 * (tgt > 0).mean())

    # 2. Predictions -----------------------------------------------------------
    if args.retrain:
        cmp_ = models.compare_models(feats)
        preds = cmp_["predictions"]
        cmp_["summary"].round(4).to_csv(results / "model_comparison.csv")
        pd.concat({m: r["fold_metrics"] for m, r in cmp_["runs"].items()}, names=["model"]) \
            .round(4).to_csv(results / "fold_metrics_retrained.csv")
        cmp_["importances"].round(4).to_csv(results / "feature_importance_retrained.csv")
        preds.to_csv(results / "oos_predictions_retrained.csv", index=False)
        log.info("Model comparison:\n%s\nSelected: %s", cmp_["summary"].round(4), cmp_["selected"])
    else:
        preds = pd.read_csv(thesis / "oos_predictions.csv", parse_dates=["date"])
        log.info("Using frozen thesis predictions (%d rows).", len(preds))
    log.info("Share of positive predictions: %.1f%%", 100 * (preds["prediction"] > 0).mean())

    # 3. Baselines -------------------------------------------------------------
    base = models.baseline_fold_metrics(feats, preds)
    base.round(4).to_csv(results / "forecast_vs_baselines.csv", index=False)
    log.info("Mean fold metrics:\n%s",
             base.groupby("forecast")[["rmse", "mae", "dir_acc", "corr"]].mean().round(4))

    # 4. Backtest --------------------------------------------------------------
    bench = backtest.run_benchmark(preds, feats, raw["chain"], raw["underlying"])
    bench["table"].round(4).to_csv(results / "benchmark_table.csv")
    for name, c in bench["curves"].items():
        c.to_parquet(processed / f"equity_{name.lower().replace(' & ', '_').replace(' ', '_')}.parquet")
    log.info("Benchmark %s -> %s:\n%s", *(d.date() for d in bench["window"]),
             bench["table"].round(3))

    # 5. Sensitivity -----------------------------------------------------------
    sens = backtest.hedge_band_sensitivity(preds, raw["chain"], raw["underlying"])
    sens.round(4).to_csv(results / "hedge_band_sensitivity.csv", index=False)
    log.info("Hedge-band sensitivity:\n%s", sens.round(3))

    # 6. Figures ---------------------------------------------------------------
    figs = {
        "fig1_vrp_implied_vs_realised": plots.vrp_timeseries(feats),
        "fig2_forecast_vs_baselines": plots.forecast_vs_baselines(base),
        "fig3_equity_drawdown": plots.equity_and_drawdown(bench["curves"]),
        "fig4_hedge_band": plots.hedge_band(sens),
        "fig5_volmageddon": plots.volmageddon(feats),
    }
    for path in plots.save_all(figs, args.reports_dir / "figures"):
        log.info("Saved %s", path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    log.info("Done in %.0f s.", time.time() - t0)


if __name__ == "__main__":
    main()
