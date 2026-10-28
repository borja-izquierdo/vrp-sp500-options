"""Forecasting the VRP with gradient boosting (thesis §3.4 and §5.1).

* Huber loss (delta = 0.02) as a custom objective, identical for XGBoost and LightGBM.
* Expanding-window cross-validation with purging and embargo (López de Prado,
  2018): between the end of each training block and the start of its test block
  ``horizon + embargo = 21 + 5`` sessions are dropped, so no training label
  overlaps the test period.
* Random search of 40 hyper-parameter combinations, scored by out-of-sample
  Huber loss inside the purged CV.
* Walk-forward out-of-sample predictions with the best parameters.
* Naive baselines on the same folds, to judge whether the model adds anything.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd

from vrp.features import feature_columns

logger = logging.getLogger("vrp.models")

SEED = 42


@dataclass
class MLConfig:
    """Model, loss, validation and search settings (thesis values as defaults)."""

    model_type: str = "xgboost"            # 'xgboost' | 'lightgbm'
    huber_delta: float = 0.02
    n_estimators: int = 400
    n_splits: int = 5
    horizon: int = 21                      # label length -> purge
    embargo: int = 5
    random_state: int = SEED
    n_iter: int = 40                       # random-search draws from the grid
    param_grid: dict = field(default_factory=lambda: {
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 63, 127],       # LightGBM only
        "max_depth": [3, 5, 7],            # XGBoost only
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_alpha": [0.0, 0.1, 1.0, 10.0],
        "reg_lambda": [0.0, 0.1, 1.0, 10.0],
    })


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def make_huber_objective(delta: float) -> Callable[[np.ndarray, np.ndarray],
                                                  tuple[np.ndarray, np.ndarray]]:
    """Huber objective for the scikit-learn API of XGBoost and LightGBM.

    Gradient of the residual ``r = y_pred - y_true`` is ``r`` inside ``|r| <= delta``
    and ``delta * sign(r)`` outside; the Hessian is set to 1 for numerical stability.
    """
    def objective(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = y_pred - y_true
        grad = np.where(np.abs(r) <= delta, r, delta * np.sign(r))
        return grad, np.ones_like(r)
    return objective


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float) -> float:
    """Mean Huber loss."""
    r = np.abs(y_true - y_pred)
    return float(np.mean(np.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta))))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, delta: float) -> dict[str, float]:
    """RMSE, MAE, Huber loss, directional accuracy and Pearson correlation."""
    err = y_true - y_pred
    corr = (float(np.corrcoef(y_true, y_pred)[0, 1])
            if len(y_true) > 1 and np.ptp(y_pred) > 0 else np.nan)
    return {"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err))),
            "huber": huber_loss(y_true, y_pred, delta),
            "dir_acc": float(np.mean(np.sign(y_true) == np.sign(y_pred))), "corr": corr}


# ---------------------------------------------------------------------------
# Purged cross-validation
# ---------------------------------------------------------------------------
class PurgedTimeSeriesSplit:
    """Expanding-window splitter with a ``horizon + embargo`` gap before each test block.

    The sample is cut into ``n_splits + 1`` equal blocks; fold k tests on block k
    (the last fold absorbs the remainder) and trains on everything up to
    ``test_start - horizon - embargo``.
    """

    def __init__(self, n_splits: int = 5, horizon: int = 21, embargo: int = 5):
        self.n_splits, self.horizon, self.embargo = n_splits, horizon, embargo

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        test_size = n_samples // (self.n_splits + 1)
        if test_size == 0:
            raise ValueError("Sample too small for the number of folds.")
        idx = np.arange(n_samples)
        for k in range(1, self.n_splits + 1):
            test_start = k * test_size
            test_end = n_samples if k == self.n_splits else (k + 1) * test_size
            train_end = test_start - self.horizon - self.embargo
            if train_end > 0:
                yield idx[:train_end], idx[test_start:test_end]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class MLEngine:
    """Hyper-parameter search and walk-forward prediction for one model family."""

    def __init__(self, cfg: Optional[MLConfig] = None):
        self.cfg = cfg or MLConfig()

    def make_model(self, params: dict):
        """Instantiate the regressor with the Huber objective and fixed seed."""
        ints = {"num_leaves", "max_depth", "min_child_samples"}
        params = {k: (int(v) if k in ints else float(v)) for k, v in params.items()}
        objective = make_huber_objective(self.cfg.huber_delta)
        if self.cfg.model_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMRegressor(objective=objective, n_estimators=self.cfg.n_estimators,
                                     random_state=self.cfg.random_state, verbose=-1, **params)
        import xgboost as xgb
        return xgb.XGBRegressor(objective=objective, n_estimators=self.cfg.n_estimators,
                                random_state=self.cfg.random_state, verbosity=0, **params)

    def splitter(self) -> PurgedTimeSeriesSplit:
        return PurgedTimeSeriesSplit(self.cfg.n_splits, self.cfg.horizon, self.cfg.embargo)

    def candidate_params(self) -> list[dict]:
        """Random draw (seeded) of ``n_iter`` combinations from the grid."""
        grid = dict(self.cfg.param_grid)
        grid.pop("max_depth" if self.cfg.model_type == "lightgbm" else "num_leaves", None)
        combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
        if len(combos) > self.cfg.n_iter:
            rng = np.random.default_rng(self.cfg.random_state)
            combos = [combos[i] for i in rng.choice(len(combos), self.cfg.n_iter, replace=False)]
        return combos

    def search(self, X: pd.DataFrame, y: pd.Series) -> tuple[dict, pd.DataFrame]:
        """Score every candidate by mean out-of-sample Huber loss across purged folds."""
        splits = list(self.splitter().split(len(X)))
        results = []
        for params in self.candidate_params():
            scores = []
            for tr, te in splits:
                model = self.make_model(params).fit(X.iloc[tr], y.iloc[tr])
                scores.append(huber_loss(y.iloc[te].values, model.predict(X.iloc[te]),
                                         self.cfg.huber_delta))
            results.append({**params, "cv_huber": float(np.mean(scores))})
        res = pd.DataFrame(results).sort_values("cv_huber").reset_index(drop=True)
        best = {k: v for k, v in res.iloc[0].items() if k != "cv_huber"}
        return best, res

    def walk_forward(self, X: pd.DataFrame, y: pd.Series,
                     params: dict) -> tuple[pd.Series, pd.DataFrame]:
        """Out-of-sample predictions and per-fold metrics with fixed parameters."""
        oos = pd.Series(np.nan, index=X.index, dtype=float)
        rows = []
        for k, (tr, te) in enumerate(self.splitter().split(len(X)), 1):
            pred = self.make_model(params).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
            oos.iloc[te] = pred
            rows.append({"fold": k, "n_train": len(tr), "n_test": len(te),
                         **regression_metrics(y.iloc[te].values, pred, self.cfg.huber_delta)})
        return oos, pd.DataFrame(rows)

    def run(self, features: pd.DataFrame) -> dict:
        """Search, then walk-forward predict. Returns predictions, params and metrics."""
        df = features[features["target"].notna()].sort_values("date").reset_index(drop=True)
        cols = feature_columns(df)
        X, y = df[cols], df["target"]
        best, search_results = self.search(X, y)
        logger.info("%s best parameters: %s", self.cfg.model_type, best)
        oos, folds = self.walk_forward(X, y, best)
        preds = df[["date", "target"]].assign(prediction=oos.values).dropna(subset=["prediction"])
        return {"predictions": preds.reset_index(drop=True), "best_params": best,
                "search_results": search_results, "fold_metrics": folds, "feature_cols": cols}


def compare_models(features: pd.DataFrame,
                   model_types: tuple[str, ...] = ("lightgbm", "xgboost")) -> dict:
    """Thesis §5.1: fit each family, pick the lowest mean fold RMSE, refit on all data.

    Returns a dict with ``summary`` (mean fold metrics per model), ``runs``,
    ``selected`` (model name), ``predictions`` and ``importances`` of the
    selected model (gain-based, refit on the full sample).
    """
    runs = {m: MLEngine(MLConfig(model_type=m)).run(features) for m in model_types}
    summary = pd.DataFrame({m: r["fold_metrics"][["rmse", "mae", "huber", "dir_acc", "corr"]].mean()
                            for m, r in runs.items()}).T.rename_axis("model")
    selected = summary["rmse"].idxmin()

    run = runs[selected]
    df = features[features["target"].notna()].sort_values("date").reset_index(drop=True)
    engine = MLEngine(MLConfig(model_type=selected))
    final = engine.make_model(run["best_params"]).fit(df[run["feature_cols"]], df["target"])
    importances = (pd.Series(final.feature_importances_, index=run["feature_cols"])
                   .sort_values(ascending=False).rename("importance"))
    return {"summary": summary, "runs": runs, "selected": selected,
            "predictions": run["predictions"], "importances": importances}


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def baseline_fold_metrics(features: pd.DataFrame, predictions: pd.DataFrame,
                          cfg: Optional[MLConfig] = None) -> pd.DataFrame:
    """Score the model against two naive forecasts on the same purged folds.

    * ``train_mean``: the mean target of the fold's (purged) training block.
    * ``naive_vrp``: today's implied minus trailing 21d Yang-Zhang vol, i.e.
      assume realised vol stays where it is (rows with missing rv_yz dropped).

    Not part of the thesis; added as a sanity check for this repository.
    """
    cfg = cfg or MLConfig()
    df = (features[features["target"].notna()].sort_values("date").reset_index(drop=True)
          .merge(predictions[["date", "prediction"]], on="date", how="left"))
    rows = []
    splitter = PurgedTimeSeriesSplit(cfg.n_splits, cfg.horizon, cfg.embargo)
    for k, (tr, te) in enumerate(splitter.split(len(df)), 1):
        test = df.iloc[te]
        y = test["target"].values
        naive = (test["mfiv"] - test["rv_yz"]).values
        ok = ~np.isnan(naive)
        for name, yhat, mask in [
            ("model", test["prediction"].values, np.ones(len(y), bool)),
            ("train_mean", np.full(len(y), df["target"].iloc[tr].mean()), np.ones(len(y), bool)),
            ("naive_vrp", naive, ok),
        ]:
            rows.append({"fold": k, "forecast": name, "start": test["date"].iloc[0].date(),
                         "end": test["date"].iloc[-1].date(), "n": int(mask.sum()),
                         **regression_metrics(y[mask], yhat[mask], cfg.huber_delta)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Nested walk-forward (added after the thesis)
# ---------------------------------------------------------------------------
def nested_walk_forward(features: pd.DataFrame,
                        model_types: tuple[str, ...] = ("lightgbm", "xgboost"),
                        inner_splits: int = 3, cfg: Optional[MLConfig] = None) -> dict:
    """Out-of-sample predictions with model choice and hyper-parameters chosen inside each fold.

    For every outer purged fold, each model family runs its random search on an
    inner purged CV restricted to the outer training block; the family and
    parameters with the lowest inner Huber loss are refit on the whole training
    block and predict the test block. Nothing from a test block is used to
    choose anything that is evaluated on it.

    Returns ``predictions`` (date, target, prediction, fold) and ``folds``
    (one row per outer fold: dates, chosen model and parameters, training-block
    target median and mean, and out-of-sample metrics).
    """
    cfg = cfg or MLConfig()
    df = features[features["target"].notna()].sort_values("date").reset_index(drop=True)
    cols = feature_columns(df)
    X, y = df[cols], df["target"]
    outer = PurgedTimeSeriesSplit(cfg.n_splits, cfg.horizon, cfg.embargo)
    preds, rows = [], []
    for k, (tr, te) in enumerate(outer.split(len(df)), 1):
        best = None
        for m in model_types:
            eng = MLEngine(MLConfig(**{**cfg.__dict__, "model_type": m, "n_splits": inner_splits}))
            params, res = eng.search(X.iloc[tr], y.iloc[tr])
            score = float(res["cv_huber"].iloc[0])
            if best is None or score < best[2]:
                best = (m, params, score, eng)
        m, params, score, eng = best
        pred = eng.make_model(params).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        preds.append(df.iloc[te][["date", "target"]].assign(prediction=pred, fold=k))
        rows.append({"fold": k, "train_end": df["date"].iloc[tr[-1]].date(),
                     "test_start": df["date"].iloc[te[0]].date(),
                     "test_end": df["date"].iloc[te[-1]].date(),
                     "model": m, "inner_cv_huber": score, "params": params,
                     "train_target_median": float(y.iloc[tr].median()),
                     "train_target_mean": float(y.iloc[tr].mean()),
                     **regression_metrics(y.iloc[te].values, pred, cfg.huber_delta)})
        logger.info("Outer fold %d: %s %s", k, m, params)
    return {"predictions": pd.concat(preds, ignore_index=True), "folds": pd.DataFrame(rows)}
