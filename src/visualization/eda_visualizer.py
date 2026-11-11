from typing import Any, Optional
import pandas as pd
import seaborn as sns

from utils.config_manager import ConfigManager # pylint: disable=import-error
from visualization.base_visualizer import BaseVisualizer # pylint: disable=import-error



class EDAVisualizer(BaseVisualizer):
    """A plotting helper for the exploratory data analysis.

    Renders distributions, boxplots and the correlation matrix from data
    prepared by EDAnalyzer. Figures are saved in the `eda` figures subdirectory.
    """
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        """Initializes the visualizer with the `eda` output subdirectory.

        Args:
            config_manager (ConfigManager): Configuration manager instance.
            name (str | None): Component name.
        """
        super().__init__(config_manager, name, subfolder="eda")


    def run(self) -> None:
        """Does nothing: the plots are rendered on demand by the `plot_*` methods."""


    def plot_distributions(self, df_subset: pd.DataFrame, group_name: str) -> None:
        """Saves histograms with a KDE curve for each column of `df_subset`.

        Args:
            df_subset (pd.DataFrame): The columns to plot.
            group_name (str): Name of the column group, e.g. ``"target_columns"``.
        """
        def _draw_hist(series: pd.Series, axis: Any) -> None:
            sns.histplot(series, kde=True, ax=axis) # type: ignore

        self._plot_grid(df_subset, group_name, _draw_hist, "Histogram plot", "histplot")


    def plot_boxplots(self, df_subset: pd.DataFrame, group_name: str) -> None:
        """Saves horizontal boxplots for each column of `df_subset`.

        Args:
            df_subset (pd.DataFrame): The columns to plot.
            group_name (str): Name of the column group, e.g. ``"target_columns"``.
        """
        def _draw_box(series: pd.Series, axis: Any) -> None:
            sns.boxplot(x=series, orient="h", ax=axis)

        self._plot_grid(
            df_subset, group_name, _draw_box, "Box-and-whisker diagram", "boxplot"
        )


    def plot_correlation_matrix(self, corr_matrix: pd.DataFrame) -> None:
        """Saves a heatmap of a precomputed correlation matrix.

        Args:
            corr_matrix (pd.DataFrame): Correlation matrix computed by EDAnalyzer.
        """
        fig, axes = self._create_figure()

        if not self._exportable:
            fig.suptitle("Correlation matrix")  # type: ignore

        labels = [self._build_label(col) for col in corr_matrix.columns]

        sns.heatmap(  # type: ignore
            corr_matrix,
            cmap="coolwarm",
            square=True,
            ax=axes[0],
            xticklabels=labels,
            yticklabels=labels,
        )

        self._finalize_plot(fig, "correlation_matrix")