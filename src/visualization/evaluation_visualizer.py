from typing import Any, List, Optional
import math
import numpy as np
import pandas as pd

from visualization.base_visualizer import BaseVisualizer, GRID_COLUMNS # pylint: disable=import-error
from utils.config_manager import ConfigManager  # pylint: disable=import-error



class EvaluationVisualizer(BaseVisualizer):
    """A plotting helper for the evaluation of the models.

    Renders the comparison of the metrics, the predicted versus the true target and
    the residuals of the test rows from the results of Evaluator. If `log_target` is enabled,
    the target and the predictions are converted back to the original scale before plotting,
    and the axes of the target are logarithmic so that the values spanning several orders of
    magnitude stay readable. Figures are saved in the `evaluation` figures subdirectory.
    """
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        """Initializes the visualizer with the `evaluation` output subdirectory.

        Args:
            config_manager (ConfigManager): Configuration manager instance.
            name (str | None): Component name.
        """
        super().__init__(config_manager, name, subfolder="evaluation")
        self._log_target: bool = bool(
            config_manager.get_param("pipeline.preprocessing.log_target", False)
        )


    def run(self) -> None:
        """Does nothing: the plots are rendered on demand by the `plot_*` methods."""


    def _original(self, values: Any) -> Any:
        """Converts the preprocessed target values back to the original scale.

        Args:
            values (Any): Series, DataFrame or array of the preprocessed target values.

        Returns:
            Any: The values in the original scale, or unchanged if `log_target` is disabled.
        """
        return 10.0 ** values if self._log_target else values


    def _target_label(self, target_name: str, prefix: str = "") -> str:
        """Builds an axis label of the target in the original scale with its unit.

        Args:
            target_name (str): Target column name.
            prefix (str): Text placed before the label, e.g. ``"Predicted"``.

        Returns:
            str: The label, e.g. ``Predicted $\\varepsilon_{\\mathrm{p50}}$ [$\\mathrm{m^2\\,s^{-3}}$]``.
        """
        return f"{prefix} {self._build_label(target_name, include_unit=True)}".strip()


    def _model_grid(self, n_models: int) -> Any:
        """Creates a figure with one square panel per model and removes the unused panels.

        Args:
            n_models (int): Number of models.

        Returns:
            Tuple[Figure, Any]: The figure and the flat array of the used axes.
        """
        n_rows = math.ceil(n_models / GRID_COLUMNS)
        fig, axes = self._create_figure(n_rows, GRID_COLUMNS, square=True)
        for axis in axes[n_models:]:
            fig.delaxes(axis)
        return fig, axes[:n_models]


    def plot_metrics(self, comparison_table: pd.DataFrame) -> None:
        """Saves bar charts comparing the models, one panel per metric.

        Args:
            comparison_table (pd.DataFrame): The table from `Evaluator.comparison_table`
                with one row per model and one column per metric.
        """
        metrics: List[str] = list(comparison_table.columns)
        if not metrics:
            return

        n_rows = math.ceil(len(metrics) / GRID_COLUMNS)
        fig, axes = self._create_figure(n_rows, GRID_COLUMNS)

        if not self._exportable:
            fig.suptitle("Comparison of the models")  # type: ignore

        for index, (axis, metric) in enumerate(zip(axes, metrics)):
            axis.barh(comparison_table.index, comparison_table[metric].to_numpy())
            axis.invert_yaxis()
            axis.set_xlabel(metric.replace("_", " "))
            if index % GRID_COLUMNS:
                axis.set_yticklabels([])

        for axis in axes[len(metrics):]:
            fig.delaxes(axis)

        self._finalize_plot(fig, "metrics_comparison")


    def plot_predictions(self, y_true: pd.Series, predictions: pd.DataFrame) -> None:
        """Saves scatter plots of the predicted versus the true target, one panel per model.

        The values are in the original scale. The dashed diagonal marks a perfect prediction.

        Args:
            y_true (pd.Series): The preprocessed target.
            predictions (pd.DataFrame): The preprocessed predictions, one column per model.
        """
        if predictions.empty:
            return

        fig, axes = self._model_grid(predictions.shape[1])
        if not self._exportable:
            fig.suptitle("Predicted versus true target")  # type: ignore

        true = self._original(y_true.to_numpy(dtype=float))
        predicted_all = self._original(predictions.to_numpy(dtype=float))
        low = min(true.min(), predicted_all.min())
        high = max(true.max(), predicted_all.max())
        scale = "log" if self._log_target else "linear"

        for axis, model in zip(axes, predictions.columns):
            axis.scatter(true, self._original(predictions[model].to_numpy(dtype=float)), s=6, alpha=0.5)
            axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
            axis.set_xscale(scale)
            axis.set_yscale(scale)
            axis.set_xlim(low, high)
            axis.set_ylim(low, high)
            axis.set_title(model)
            axis.set_xlabel(self._target_label(str(y_true.name), "True"))
            axis.set_ylabel(self._target_label(str(y_true.name), "Predicted"))

        self._finalize_plot(fig, "predicted_vs_true")