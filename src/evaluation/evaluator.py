from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold

from base import BasePipelineComponent # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error
from models.base_model import BaseModel # pylint: disable=import-error
from visualisation.evaluation_visualiser import EvaluationVisualiser # pylint: disable=import-error

ERR_MSG = "Evaluator has not been run yet. " \
                 "First, call the `run()` method or call the instance."


class Metrics:
    """A set of regression metrics computed on the preprocessed target.

    If the target is on the log10 scale, the metrics are reported on the log10 scale
    and on the original scale (`*_orig`). Otherwise only the plain metrics are used.
    """
    def __init__(self, log_target: bool = True) -> None:
        """Configures regression metrics for the target scale used by the pipeline.

        Uses log-scale scores and inverse-transformed error metrics when logging is enabled.
        Otherwise configures R-squared, root mean squared error and mean absolute error.
        Preserves metric insertion order so the first metric can serve as the main score.

        Args:
            log_target: Whether the target is on the log10 scale.

        Returns:
            None: The metric functions are stored on this instance.
        """
        self.funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]]
        if log_target:
            self.funcs = {
                "r2_log10": r2_score,
                "rmse_log10": root_mean_squared_error,
                "mae_log10": mean_absolute_error,
                "rmse_orig": lambda y, yp: root_mean_squared_error(10.0 ** y, 10.0 ** yp),
                "mae_orig": lambda y, yp: mean_absolute_error(10.0 ** y, 10.0 ** yp),
            }
        else:
            self.funcs = {
                "r2": r2_score,
                "rmse": root_mean_squared_error,
                "mae": mean_absolute_error,
            }



    @property
    def main_metric(self) -> str:
        """Returns the first configured metric, used to sort model results."""
        return next(iter(self.funcs))



    def compute(
            self,
            y_true: Union[pd.Series, np.ndarray], 
            y_pred: Union[pd.Series, np.ndarray]
        ) -> Dict[str, float]:
        """Computes every configured metric for true and predicted targets.

        Converts both inputs to floating-point NumPy arrays before evaluation.
        Applies each configured metric function to the same pair of arrays.
        Returns the scores under their configured names.

        Args:
            y_true: True target values.
            y_pred: Predicted target values.

        Returns:
            Dict[str, float]: Metric values keyed by metric name.
        """
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        return {name: float(func(y_t, y_p)) for name, func in self.funcs.items()}



class Evaluator(BasePipelineComponent):
    """A pipeline component that compares models by group-wise cross-validation.

    The outer loop is a `GroupKFold` over the groups (geometric families), so rows of one
    group never appear in both the training and the test part of a fold. Each model is
    fitted on the training rows of every fold and evaluated on the held-out rows. A model
    can be a `BaseModel` or a `GridSearchCV` wrapping one (nested cross-validation); the
    groups of the training rows are passed to its `fit()` so that the inner group-wise
    search does not leak data.
    """
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        """Initializes evaluation settings and optional visualisation support.

        Reads the number of outer folds and the target-scale option from configuration.
        Initializes stored evaluation results as unavailable until `run()` completes.
        Creates an evaluation visualiser only when plot generation is enabled.

        Args:
            config_manager: Configuration manager instance.
            name: Optional component name.
        """
        super().__init__(config_manager, name)
        self._n_splits: int = int(
            self.config_manager.get_param("pipeline.evaluation.n_splits", 5)
        )
        self._metrics = Metrics(bool(
            self.config_manager.get_param("pipeline.preprocessing.log_target", True)
        ))
        self._comparison_table: Optional[pd.DataFrame] = None
        self._predictions: Optional[pd.DataFrame] = None
        self._y_true: Optional[pd.Series] = None
        self._best_params: Dict[str, List[Dict[str, Any]]] = {}

        self._visualiser: Optional[EvaluationVisualiser] = (
            EvaluationVisualiser(config_manager, name=f"{self.name}Visualizer")
            if self.config_manager.get_param("pipeline.evaluation.generate_plots", True)
            else None
        )



    @property
    def comparison_table(self) -> pd.DataFrame:
        """Returns the metrics sorted from the best model, or raises RuntimeError if evaluation has not run.

        Each row holds the metrics computed on all out-of-fold predictions together and
        the mean and standard deviation (`*_mean`, `*_std`) of each metric over the folds.
        """
        if self._comparison_table is None:
            raise RuntimeError(ERR_MSG)
        return self._comparison_table


    @property
    def predictions(self) -> pd.DataFrame:
        """Returns the out-of-fold predictions of each model, or raises RuntimeError if evaluation has not run."""
        if self._predictions is None:
            raise RuntimeError(ERR_MSG)
        return self._predictions


    @property
    def y_true(self) -> pd.Series:
        """Returns the true target of all rows, or raises RuntimeError if evaluation has not run."""
        if self._y_true is None:
            raise RuntimeError(ERR_MSG)
        return self._y_true



    @staticmethod
    def _model_name(model: Union[BaseModel, GridSearchCV]) -> str:
        """Resolves the display name for a model or a model wrapped by grid search.

        Reads the estimator name from `GridSearchCV` instances.
        Uses the model's own name for unwrapped base models.

        Args:
            model: Model instance or grid search wrapper.

        Returns:
            str: Name used to identify the model in evaluation output.
        """
        return model.estimator.name if isinstance(model, GridSearchCV) else model.name # type: ignore



    def _plot(self) -> None:
        """Renders metric comparisons and out-of-fold predictions when plotting is enabled.

        Does nothing when no evaluation visualiser was created.
        Otherwise passes the stored comparison table, target and predictions to it.
        """
        if self._visualiser is None:
            return

        self._visualiser.plot_metrics(self.comparison_table)
        self._visualiser.plot_predictions(self.y_true, self.predictions)


    def _save_report(self) -> None:
        """Writes the metrics table and the best grid-search parameters to the output directory.

        Returns immediately when `pipeline.evaluation.save_reports` is not true.
        Otherwise creates the output directory and serializes the table as JSON. The best
        parameters found by the inner search in each outer fold are saved per model, if
        any model was tuned.
        """
        if not self.config_manager.get_param("pipeline.evaluation.save_reports", True):
            return

        output_dir = Path(
            self.config_manager.get_param("settings.paths.output_dir", "output")
        ) / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.comparison_table.to_json(
            output_dir / "evaluation_summary.json", orient="index", indent=4
        )
        if self._best_params:
            with open(
                output_dir / "GridSearchCV_best_params.json", mode="w", encoding="utf-8"
            ) as file:
                json.dump(self._best_params, file, indent=4, ensure_ascii=False)



    def run(
        self,
        models: Union[BaseModel, GridSearchCV, Sequence[Union[BaseModel, GridSearchCV]]],
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        *args: Any,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Evaluates models by group-wise cross-validation and stores their predictions.

        Splits the rows with `GroupKFold`, so rows from one group never cross the
        train-test boundary of a fold. Clones and fits each model on every training part,
        passing the groups to grid-search estimators, and predicts the held-out part.
        Computes the metrics on every fold and on all out-of-fold predictions together.

        Args:
            models: One model or a sequence of models and grid-search wrappers.
            X: The feature matrix.
            y: The preprocessed target.
            groups: Group label for each row.

        Returns:
            pd.DataFrame: Metrics with one row per model, sorted by the main metric.

        Raises:
            ValueError: If the models have duplicate names or `n_splits` is greater
                than the number of groups.
        """
        model_list: List[Union[BaseModel, GridSearchCV]] = (
            list(models) if isinstance(models, (list, tuple)) else [models] # type: ignore
        )
        names = [self._model_name(model) for model in model_list]
        if len(set(names)) != len(names):
            raise ValueError(f"Model names must be unique, got {names}.")
        if self._n_splits > groups.nunique():
            raise ValueError(
                f"n_splits={self._n_splits} is greater than the number of groups "
                f"({groups.nunique()})."
            )

        splits = list(GroupKFold(n_splits=self._n_splits).split(X, y, groups))
        predictions: Dict[str, np.ndarray] = {}
        rows: List[Dict[str, Any]] = []
        self._best_params = {}

        for name, model in zip(names, model_list):
            oof = np.empty(len(y))
            fold_metrics: List[Dict[str, float]] = []

            for train_idx, test_idx in splits:
                fitted = clone(model)
                if isinstance(fitted, GridSearchCV):
                    fitted.fit(X.iloc[train_idx], y.iloc[train_idx], groups=groups.iloc[train_idx])
                    self._best_params.setdefault(name, []).append(fitted.best_params_)
                else:
                    fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
                oof[test_idx] = fitted.predict(X.iloc[test_idx])
                fold_metrics.append(self._metrics.compute(y.iloc[test_idx], oof[test_idx]))

            summary: Dict[str, float] = {}
            for metric in self._metrics.funcs:
                values = [fold[metric] for fold in fold_metrics]
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = float(np.std(values, ddof=1))

            predictions[name] = oof
            rows.append({"model": name, **self._metrics.compute(y, oof), **summary})

        self._comparison_table = (
            pd.DataFrame(rows).set_index("model").sort_values(
                self._metrics.main_metric, ascending=False
            )
        )
        self._predictions = pd.DataFrame(predictions, index=y.index)
        self._y_true = y

        self._plot()
        self._save_report()
        return self._comparison_table
