"""
Drawing heplper for exploratory data analysis.
Author: Ivan Skvarka
"""

from typing import Any, Optional
import pandas as pd
import seaborn as sns

from utils.config_manager import ConfigManager
from visualisation.base_visualiser import BaseVisualiser



class EDAVisualiser(BaseVisualiser):
    """A plotting helper for the exploratory data analysis.

    Renders boxplots and the correlation matrix (distributions are inherited) from data
    prepared by EDAnalyser. Figures are saved in the `eda` figures subdirectory.
    """
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        """Initializes plotting support for exploratory data analysis.

        Delegates shared plotting configuration to the base visualiser.
        Routes generated figures to the EDA-specific output subdirectory.

        Args:
            config_manager: Configuration manager instance.
            name: Optional component name.

        Returns:
            None: The visualiser is ready to render plots.
        """
        super().__init__(config_manager, name, subfolder="eda")



    def plot_boxplots(self, df_subset: pd.DataFrame, group_name: str) -> None:
        """Plots and saves a horizontal boxplot for each supplied column.

        Defines a per-column drawing callback and uses the shared grid renderer.
        Empty subsets produce no figure.
        The group name is included in the output file name and optional title.

        Args:
            df_subset: The columns to plot.
            group_name: Column group name, e.g. ``"target_columns"``.

        Returns:
            None: The figure is saved as a side effect.
        """
        def _draw_box(series: pd.Series, axis: Any) -> None:
            sns.boxplot(x=series, orient="h", ax=axis)

        self._plot_grid(
            df_subset, group_name, _draw_box, "Box-and-whisker diagram", "boxplot"
        )


    def plot_correlation_matrix(self, corr_matrix: pd.DataFrame) -> None:
        """Renders and saves a heatmap for a precomputed correlation matrix.

        Builds display labels for both matrix dimensions from configured metadata.
        Uses a diverging color map and square cells to make correlation strength readable.
        Adds an interactive title only when exportable output is disabled.

        Args:
            corr_matrix: Correlation matrix computed by `EDAnalyzer`.

        Returns:
            None: The heatmap is saved as a side effect.
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
