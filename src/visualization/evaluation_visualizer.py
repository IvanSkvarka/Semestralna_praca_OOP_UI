from typing import Any, List, Optional
import math
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
        """Initializes plotting support for model evaluation results.

        Configures the evaluation-specific output subdirectory.
        Reads whether target values are represented on a log10 scale.

        Args:
            config_manager: Configuration manager instance.
            name: Optional component name.

        Returns:
            None: The visualizer is ready to render plots.
        """
        super().__init__(config_manager, name, subfolder="evaluation")
        self._log_target: bool = bool(
            config_manager.get_param("pipeline.preprocessing.log_target", False)
        )


    def _original(self, values: Any) -> Any:
        """Converts preprocessed target values back to their original scale when needed.

        Applies the inverse power-of-ten transformation when log-target mode is enabled.
        Leaves the input unchanged when targets are already represented on the original scale.
        Supports pandas objects and NumPy arrays through their exponentiation operation.

        Args:
            values: Series, DataFrame or array of preprocessed target values.

        Returns:
            Any: Values on the original scale, or unchanged values when logging is disabled.
        """
        return 10.0 ** values if self._log_target else values


    def _target_label(self, target_name: str, prefix: str = "") -> str:
        """Builds a labeled target name with an optional prefix and configured unit.

        Uses the shared label builder to resolve the target symbol and unit.
        Strips leading or trailing whitespace when the prefix is empty or supplied.
        The label represents the original target scale used by evaluation plots.

        Args:
            target_name: Target column name.
            prefix: Text placed before the label, e.g. ``"Predicted"``.

        Returns:
            str: Formatted target axis label.
        """
        return f"{prefix} {self._build_label(target_name, include_unit=True)}".strip()


    def run(self, *args: Any, **kwargs: Any) -> None:
        """Provides the pipeline interface without rendering plots eagerly.

        Plot generation is performed only when an explicit `plot_*` method is called.
        This no-op allows the visualizer to follow the common pipeline component API.

        Returns:
            None: No work is performed.

        """


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

        Arranges metric plots in a grid and removes unused panels.
        Orders model bars according to the already sorted comparison table.
        Returns without creating a figure when the table has no metrics.

        Args:
            comparison_table: Table with one row per model and one column per metric.

        Returns:
            None: The charts are saved as a side effect.
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
        """Renders and saves predicted-versus-true scatter plots for each model.

        Converts targets and predictions to the original scale when log-target mode is active.
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

        true = self._original(y_true.to_numpy(dtype=float))
        predicted_all = self._original(predictions.to_numpy(dtype=float))
        low = min(true.min(), predicted_all.min())
        high = max(true.max(), predicted_all.max())
        scale = "log" if self._log_target else "linear"

        for axis, model in zip(axes, predictions.columns):
            axis.scatter(
                true, self._original(predictions[model].to_numpy(dtype=float)), s=6, alpha=0.5
            )
            axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
            axis.set_xscale(scale)
            axis.set_yscale(scale)
            axis.set_xlim(low, high)
            axis.set_ylim(low, high)
            axis.set_title(model)
            axis.set_xlabel(self._target_label(str(y_true.name), "True"))
            axis.set_ylabel(self._target_label(str(y_true.name), "Predicted"))

        self._finalize_plot(fig, "predicted_vs_true")
