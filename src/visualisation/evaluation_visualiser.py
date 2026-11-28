from typing import Any, List, Optional
import math
import pandas as pd

from visualisation.base_visualiser import BaseVisualiser, GRID_COLUMNS # pylint: disable=import-error
from utils.config_manager import ConfigManager  # pylint: disable=import-error



class EvaluationVisualiser(BaseVisualiser):
    """A plotting helper for the evaluation of the models.

    Renders the comparison of the metrics and the predicted versus the true target from
    the results of Evaluator. If `log_target` is enabled, the target and the predictions
    are plotted on the log10 scale and the target axes are marked with a log prefix and
    a dimensionless unit. Figures are saved in the `evaluation` figures subdirectory.
    """
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        """Initializes plotting support for model evaluation results.

        Configures the evaluation-specific output subdirectory.
        Reads whether target values are represented on a log10 scale.

        Args:
            config_manager: Configuration manager instance.
            name: Optional component name.

        Returns:
            None: The visualiser is ready to render plots.
        """
        super().__init__(config_manager, name, subfolder="evaluation")
        self._log_target: bool = bool(
            config_manager.get_param("pipeline.preprocessing.log_target", False)
        )


    def _target_label(self, target_name: str, prefix: str = "") -> str:
        """Builds a labeled target name with an optional prefix and unit.

        Uses the shared label builder to resolve the target symbol and unit.
        When log-target mode is enabled, the label is marked with a log10 prefix
        and a dimensionless unit.

        Args:
            target_name: Target column name.
            prefix: Text placed before the label, e.g. ``"Predicted"``.

        Returns:
            str: Formatted target axis label.
        """
        if self._log_target:
            label = f"$\\log_{{10}}$ {self._build_label(target_name)} [$-$]"
        else:
            label = self._build_label(target_name, include_unit=True)
        return f"{prefix} {label}".strip()


    def _model_grid(self, n_models: int) -> Any:
        """Creates a square-panel figure with one used axis for each model.

        Computes the required number of rows for the shared grid width.
        Removes unused axes from the figure and returned array.
        Returns only the axes assigned to model plots.

        Args:
            n_models: Number of models.

        Returns:
            Tuple[Figure, Any]: Figure and flat array of axes used for models.
        """
        n_rows = math.ceil(n_models / GRID_COLUMNS)
        fig, axes = self._create_figure(n_rows, GRID_COLUMNS, square=True)
        for axis in axes[n_models:]:
            fig.delaxes(axis)
        return fig, axes[:n_models]



    def plot_metrics(self, comparison_table: pd.DataFrame) -> None:
        """Renders and saves a horizontal bar chart for each evaluation metric.

        Each bar is the mean of the metric over the cross-validation folds and the error
        bar is its standard deviation. Arranges metric plots in a grid and removes unused
        panels. Orders model bars according to the already sorted comparison table.
        Returns without creating a figure when the table has no metrics.

        Args:
            comparison_table: Table of `Evaluator.comparison_table` with the `*_mean`
                and `*_std` columns and one row per model.

        Returns:
            None: The charts are saved as a side effect.
        """
        metrics: List[str] = [
            col.removesuffix("_mean") for col in comparison_table.columns if col.endswith("_mean")
        ]
        if not metrics:
            return

        n_rows = math.ceil(len(metrics) / GRID_COLUMNS)
        fig, axes = self._create_figure(n_rows, GRID_COLUMNS)

        if not self._exportable:
            fig.suptitle("Comparison of the models")  # type: ignore

        for index, (axis, metric) in enumerate(zip(axes, metrics)):
            axis.barh(
                comparison_table.index,
                comparison_table[f"{metric}_mean"].to_numpy(),
                xerr=comparison_table[f"{metric}_std"].to_numpy(),
                capsize=3,
            )
            axis.invert_yaxis()
            axis.set_xlabel(metric.replace("_", " "))
            if index % GRID_COLUMNS:
                axis.set_yticklabels([])

        for axis in axes[len(metrics):]:
            fig.delaxes(axis)

        self._finalize_plot(fig, "metrics_comparison")


    def plot_predictions(self, y_true: pd.Series, predictions: pd.DataFrame) -> None:
        """Renders and saves predicted-versus-true scatter plots for each model.

        Plots the preprocessed values, i.e. the log10 scale when log-target mode is active.
        Uses a shared axis range and a dashed identity line to indicate perfect predictions.
        Returns without creating a figure when no predictions are supplied.

        Args:
            y_true: The preprocessed target.
            predictions: Preprocessed predictions, with one column per model.

        Returns:
            None: The scatter plots are saved as a side effect.
        """
        if predictions.empty:
            return

        fig, axes = self._model_grid(predictions.shape[1])
        if not self._exportable:
            fig.suptitle("Predicted versus true target")  # type: ignore

        true = y_true.to_numpy(dtype=float)
        predicted_all = predictions.to_numpy(dtype=float)
        low = min(true.min(), predicted_all.min())
        high = max(true.max(), predicted_all.max())

        for axis, model in zip(axes, predictions.columns):
            axis.scatter(true, predictions[model].to_numpy(dtype=float), s=6, alpha=0.5)
            axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
            axis.set_xlim(low, high)
            axis.set_ylim(low, high)
            axis.set_title(model)
            axis.set_xlabel(self._target_label(str(y_true.name), "True"))
            axis.set_ylabel(self._target_label(str(y_true.name), "Predicted"))

        self._finalize_plot(fig, "predicted_vs_true")