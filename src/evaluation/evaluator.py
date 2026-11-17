from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

from base import BasePipelineComponent # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error
from models.base_model import BaseModel # pylint: disable=import-error
from visualization.evaluation_visualizer import EvaluationVisualizer # pylint: disable=import-error

ERR_MSG = "Evaluator has not been run yet. First, call the `run()` method or call the instance."


class Metrics:
    """A set of regression metrics computed on the preprocessed target.

    If the target is on the log10 scale, the metrics are reported on the log10 scale
    and on the original scale (`*_orig`). Otherwise only the plain metrics are used.
    """
    def __init__(self, log_target: bool = True) -> None:
        """Defines the metric functions.

        Args:
            log_target (bool): Whether the target is on the log10 scale.
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
        """Provides the name of the metric used to sort the models."""
        return next(iter(self.funcs))

    def compute(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """Computes all metrics.

        Args:
            y_true (pd.Series | np.ndarray): True target values.
            y_pred (pd.Series | np.ndarray): Predicted target values.

        Returns:
            Dict[str, float]: The metric values by name.
        """
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        return {name: float(func(y_t, y_p)) for name, func in self.funcs.items()}


class Evaluator(BasePipelineComponent):
    """A pipeline component that compares models on a held-out test set.

    The groups (geometric families) are split into a training and a test part with
    `train_test_split`, so that rows of one group never appear in both parts. Each model
    is fitted on the training rows and evaluated on the test rows. A model can be a
    `BaseModel` or a `GridSearchCV` wrapping one; the groups of the training rows are
    passed to its `fit()` so that an inner group-wise search does not leak data.
    """
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        """Reads the evaluation settings and creates the visualizer if plotting is enabled.

        Args:
            config_manager (ConfigManager): Configuration manager instance.
            name (str | None): Component name.
        """
        super().__init__(config_manager, name)
        self._test_size: float = float(self.config_manager.get_param("pipeline.evaluation.test_size", 0.2))
        self._random_state: Optional[int] = self.config_manager.get_param("pipeline.random_state")
        self._metrics = Metrics(bool(self.config_manager.get_param("pipeline.preprocessing.log_target", True)))
        self._comparison_table: Optional[pd.DataFrame] = None
        self._predictions: Optional[pd.DataFrame] = None
        self._y_test: Optional[pd.Series] = None

        self._visualizer: Optional[EvaluationVisualizer] = (
            EvaluationVisualizer(config_manager, name=f"{self.name}Visualizer")
            if self.config_manager.get_param("pipeline.evaluation.generate_plots", True)
            else None
        )


    @property
    def comparison_table(self) -> pd.DataFrame:
        """Provides the table of test metrics, one row per model, sorted from the best.

        Raises:
            RuntimeError: If `run()` has not been called yet.
        """
        if self._comparison_table is None:
            raise RuntimeError(ERR_MSG)
        return self._comparison_table


    @property
    def predictions(self) -> pd.DataFrame:
        """Provides the predictions for the test rows, one column per model.

        Raises:
            RuntimeError: If `run()` has not been called yet.
        """
        if self._predictions is None:
            raise RuntimeError(ERR_MSG)
        return self._predictions


    @property
    def y_test(self) -> pd.Series:
        """Provides the true target of the test rows.

        Raises:
            RuntimeError: If `run()` has not been called yet.
        """
        if self._y_test is None:
            raise RuntimeError(ERR_MSG)
        return self._y_test


    @staticmethod
    def _model_name(model: Union[BaseModel, GridSearchCV]) -> str:
        """Returns the name of a model or of the model wrapped by a `GridSearchCV`."""
        return model.estimator.name if isinstance(model, GridSearchCV) else model.name


    def _plot(self) -> None:
        """Renders the comparison of the metrics, the predictions and the residuals, if plotting is enabled."""
        if self._visualizer is None:
            return

        self._visualizer.plot_metrics(self.comparison_table)
        self._visualizer.plot_predictions(self.y_test, self.predictions)


    def _save_report(self) -> None:
        """Saves the table of test metrics as JSON, if `pipeline.evaluation.save_reports` is true."""
        if self.config_manager.get_param("pipeline.evaluation.save_reports", True):
            return

        out_dir = Path(self.config_manager.get_param("settings.paths.output_dir", "output")) / "evaluation"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.comparison_table.to_json(out_dir / "evaluation_summary.json", orient="index", indent=4)


    def run(
        self,
        models: Union[BaseModel, GridSearchCV, Sequence[Union[BaseModel, GridSearchCV]]],
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
    ) -> pd.DataFrame:
        """Splits the groups, fits the models, evaluates them on the test rows and saves the results.

        Args:
            models (BaseModel | GridSearchCV | Sequence): One model or a list of models.
            X (pd.DataFrame): The features.
            y (pd.Series): The preprocessed target.
            groups (pd.Series): The group number of each row.

        Returns:
            pd.DataFrame: One row per model with the metrics computed on the test rows,
            sorted from the best model.

        Raises:
            ValueError: If the models have duplicate names.
        """
        model_list: List[Union[BaseModel, GridSearchCV]] = (
            list(models) if isinstance(models, (list, tuple)) else [models] # type: ignore
        )
        names = [self._model_name(model) for model in model_list]
        if len(set(names)) != len(names):
            raise ValueError(f"Model names must be unique, got {names}.")

        train_groups, _ = train_test_split(
            groups.unique(), test_size=self._test_size, random_state=self._random_state
        )
        train_mask = groups.isin(train_groups).to_numpy()
        X_train, y_train, groups_train = X[train_mask], y[train_mask], groups[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask] # ~ je inverzia masky

        predictions: Dict[str, np.ndarray] = {}
        rows: List[Dict[str, Any]] = []

        for name, model in zip(names, model_list):
            fitted = clone(model)
            if isinstance(fitted, GridSearchCV):
                fitted.fit(X_train, y_train, groups=groups_train)
            else:
                fitted.fit(X_train, y_train)
            predictions[name] = fitted.predict(X_test)
            rows.append({"model": name, **self._metrics.compute(y_test, predictions[name])})

        self._comparison_table = (
            pd.DataFrame(rows).set_index("model").sort_values(self._metrics.main_metric, ascending=False)
        )
        self._predictions = pd.DataFrame(predictions, index=y_test.index)
        self._y_test = y_test

        self._plot()
        self._save_report()
        return self._comparison_table