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
        """Initializes plotting support for exploratory data analysis.

        Delegates shared plotting configuration to the base visualizer.
        Routes generated figures to the EDA-specific output subdirectory.

        Args:
            config_manager: Configuration manager instance.
            name: Optional component name.

        Returns:
            None: The visualizer is ready to render plots.
        """
        super().__init__(config_manager, name, subfolder="eda")



    def run(self, *args: Any, **kwargs: Any) -> None:
        """Provides the pipeline interface without rendering plots eagerly.

        Plot generation is performed only when an explicit `plot_*` method is called.
        This no-op allows the visualizer to follow the common pipeline component API.

        Returns:
            None: No work is performed.
        """


    def plot_distributions(self, df_subset: pd.DataFrame, group_name: str) -> None:
        """Plots and saves a histogram with a KDE curve for each supplied column.

        Defines a per-column drawing callback and delegates grid layout and file output.
        Empty column subsets are ignored by the shared grid helper.
        The requested group name is used to identify the resulting figure.

        Args:
            df_subset: The columns to plot.
            group_name: Column group name, e.g. ``"target_columns"``.

        Returns:
            None: The figure is saved as a side effect.
        """
        def _draw_hist(series: pd.Series, axis: Any) -> None:
            """Draws one column's histogram and KDE curve on the supplied axes.

            Uses seaborn to render the distribution and a smooth density estimate.
            The caller supplies the axes so the plot can be embedded in a figure grid.

            Args:
                series: Column values to visualize.
                axis: Matplotlib axes that receives the plot.

            Returns:
                None: The histogram is drawn as a side effect.
            """
            sns.histplot(series, kde=True, ax=axis) # type: ignore

        self._plot_grid(df_subset, group_name, _draw_hist, "Histogram plot", "histplot")


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
            """Draws a horizontal boxplot for a single data column.

            Uses the series as the horizontal value axis for the boxplot.
            The caller supplies the axes so the plot can be embedded in a figure grid.

            Args:
                series: Column values to visualize.
                axis: Matplotlib axes that receives the plot.

            Returns:
                None: The boxplot is drawn as a side effect.
            """
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
