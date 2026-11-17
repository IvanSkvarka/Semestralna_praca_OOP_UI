from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate

from base import BasePipelineComponent
from utils.config_manager import ConfigManager


class Metrics:
    def __init__(self, log_target: bool = True) -> None:
        if log_target:
            self.funcs = {
                "r2_log10": (r2_score, True),
                "rmse_log10": (root_mean_squared_error, False),
                "mae_log10": (mean_absolute_error, False),
                "rmse_orig": (lambda y, yp: root_mean_squared_error(10**y, 10**yp), False),
                "mae_orig": (lambda y, yp: mean_absolute_error(10**y, 10**yp), False),
            }
        else:
            self.funcs = {
                "r2": (r2_score, True),
                "rmse": (root_mean_squared_error, False),
                "mae": (mean_absolute_error, False),
            }

    def get_scorers(self) -> Dict[int, Any]:
        return {k: make_scorer(fn, greater_is_better=hib) for k, (fn, hib) in self.funcs.items()}

    def compute(self, y_true, y_pred) -> dict:
        y_t, y_p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
        return {k: float(fn(y_t, y_p)) for k, (fn, _) in self.funcs.items()}


class Evaluator(BasePipelineComponent):
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        super().__init__(config_manager, name)
        self.n_splits = int(self.config_manager.get_param("pipeline.evaluation.n_splits", 5))
        self.log_target = bool(self.config_manager.get_param("pipeline.preprocessing.log_target", True))
        self.save_reports = bool(self.config_manager.get_param("pipeline.evaluation.save_reports", True))
        self.metrics = Metrics(self.log_target)
        self.comparison_table: Optional[pd.DataFrame] = None
        self.oof_predictions: Optional[pd.DataFrame] = None

    def run(self, estimators, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> pd.DataFrame:
        if not isinstance(estimators, (list, tuple)):
            estimators = [estimators]

        splits = list(GroupKFold(n_splits=self.n_splits).split(X, y, groups))
        scorers = self.metrics.get_scorers()
        oof_dict, rows = {}, []

        for est in estimators:
            name = est.estimator.name if isinstance(est, GridSearchCV) else est.name
            res = cross_validate(
                est, X, y, cv=splits, groups=groups,
                scoring=scorers, return_estimator=True
            )

            oof = np.empty(len(y))
            for fitted_est, (_, test_idx) in zip(res["estimator"], splits):
                oof[test_idx] = fitted_est.predict(X.iloc[test_idx])
            oof_dict[name] = oof

            summary = {}
            for m, (_, higher_is_better) in self.metrics.funcs.items():
                vals = res[f"test_{m}"] if higher_is_better else -res[f"test_{m}"]
                summary[f"{m}_mean"] = float(np.mean(vals))
                summary[f"{m}_std"] = float(np.std(vals, ddof=1))

            rows.append({"model": name, **self.metrics.compute(y, oof), **summary})

        sort_col = "r2_log10" if self.log_target else "r2"
        self.comparison_table = pd.DataFrame(rows).set_index("model").sort_values(sort_col, ascending=False)
        self.oof_predictions = pd.DataFrame(oof_dict, index=y.index)

        if self.save_reports:
            out_dir = Path(self.config_manager.get_param("settings.paths.output_dir", "output")) / "evaluation"
            out_dir.mkdir(parents=True, exist_ok=True)
            self.comparison_table.to_json(out_dir / "evaluation_summary.json", orient="index", indent=4)

        return self.comparison_table