from pathlib import Path
from typing import Union, Optional, Any
import pandas as pd
import seaborn as sns

from utils.config_manager import ConfigManager # pylint: disable=import-error
from visualization.base_visualizer import BaseVisualizer # pylint: disable=import-error

class EDAVisualizer(BaseVisualizer):
    """A stateless plotting helper used directly by EDAnalyzer subroutines.

    Provides dedicated rendering methods for distributions, boxplots, and
    correlation matrices without duplicating data filtering or statistical
    computations performed by EDAnalyzer.
    """
    def __init__(
            self,
            config_manager: ConfigManager,
            name: Optional[str] = None,
            subfolder: Optional[Union[str, Path]] = "eda"
        ) -> None:
        """Initializes the EDA visualizer and configures output paths.

        Args:
            config_manager (ConfigManager): Configuration manager instance.
            name (str | None): Component name for logging and identification.
            subfolder (str | Path | None): Target subdirectory for saving EDA plots.
        """
        super().__init__(config_manager, name, subfolder)


    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Satisfies the abstract BasePipelineComponent interface.

        Note: Plotting is triggered granularly inside EDAnalyzer methods rather
        than via a monolithic `run()` call.
        """
        self.logger.debug(
            "EDAVisualizer is triggered per-method by EDAnalyzer; standalone run()"
            " skipped."
        )


    def plot_distributions(
        self,df_subset: pd.DataFrame, group_name: str, **kwargs: Any
    ) -> Any:
        """Renders histograms with KDE for the pre-selected DataFrame subset.

        Args:
            df_subset (pd.DataFrame): Pre-filtered DataFrame slice to visualize.
            group_name (str): Group identifier (e.g., 'target_columns').
            **kwargs (Any): Additional keyword arguments forwarded to
            `sns.histplot`.

        Returns:
            Any: Output of `_finalize_plot`.
        """
        kde_default: bool = kwargs.pop("kde", True)

        def _draw_hist(series: pd.Series, ax: Any) -> None:
            sns.histplot(series, kde=kde_default, ax=ax, **kwargs) # type: ignore

        return self._plot_grid(
            df_subset=df_subset,
            group_name=group_name,
            plot_fn=_draw_hist,
            title_prefix="Histogram plot",
            file_suffix="histplot",
        )

    def plot_boxplots(
        self, df_subset: pd.DataFrame, group_name: str, **kwargs: Any
    ) -> Any:
        """Renders horizontal box-and-whisker diagrams for outlier inspection.

        Args:
            df_subset (pd.DataFrame): Pre-filtered DataFrame slice to visualize.
            group_name (str): Group identifier (e.g., 'target_columns').
            **kwargs (Any): Additional keyword arguments forwarded to `sns.boxplot`.

        Returns:
            Any: Output of `_finalize_plot`.
        """

        def _draw_box(series: pd.Series, ax: Any) -> None:
            sns.boxplot(x=series, orient="h", ax=ax, **kwargs)

        return self._plot_grid(
            df_subset=df_subset,
            group_name=group_name,
            plot_fn=_draw_box,
            title_prefix="Box-and-whisker diagram",
            file_suffix="boxplot",
        )

    def plot_correlation_matrix(
        self, corr_matrix: pd.DataFrame, **kwargs: Any
    ) -> Any:
        """Renders a square heatmap from an already computed correlation matrix.

        Args:
            corr_matrix (pd.DataFrame): Precomputed correlation matrix from
            EDAnalyzer.
            **kwargs (Any): Optional overrides such as `annot` (bool).

        Returns:
            Any: Output of `_finalize_plot`.
        """
        annot: bool = kwargs.get("annot", False)

        fig, axis = self._create_figure(square=True)

        if not self._exportable:
            fig.suptitle(self._build_label(prefix="Correlation matrix"))  # type: ignore

        labels = [
            self._build_label(col, include_unit=False)
            for col in corr_matrix.columns
        ]

        sns.heatmap(  # type: ignore
            corr_matrix,
            cmap="coolwarm",
            square=True,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            ax=axis[0],
            xticklabels=labels,
            yticklabels=labels,
            annot=annot,
            fmt=".2f",
            cbar_kws={"shrink": 0.8, "anchor": (0.0, 0.0), "panchor": (1.0, 0.0)},
        )

        return self._finalize_plot(fig, "correlation_matrix")