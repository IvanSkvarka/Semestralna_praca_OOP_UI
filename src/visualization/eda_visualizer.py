from pathlib import Path
from typing import Union, Optional, Any, List
import pandas as pd
import seaborn as sns

from utils.config_manager import ConfigManager # pylint: disable=import-error
from visualization.base_visualizer import BaseVisualizer # pylint: disable=import-error

class EDAVisualizer(BaseVisualizer):
    """
    
    """
    def __init__(
            self,
            config_manager: ConfigManager,
            name: Optional[str] = None,
            subfolder: Optional[Union[str, Path]] = "eda"
        ) -> None:
        """
        
        """
        super().__init__(config_manager, name, subfolder)
        self._target_cols: List[str] = self.config_manager["data.target_columns"]
        self._feature_cols: List[str] = self.config_manager["data.feature_columns"]
        self._corr_method: str = self.config_manager["pipeline.eda.correlation_method"]

    def run(self, *args: Any, **kwargs: Any) -> None:
        """Executes the full suite of exploratory data analysis visualizations.

        Extracts the input DataFrame and optional precomputed correlation matrix.
        Filters out missing records and constant columns before rendering
        distributions, boxplots, and the correlation heatmap.

        Args:
            *args (Any): Variable positional arguments; expects DataFrame at index 0
            and optionally a precomputed correlation DataFrame at index 1.
            **kwargs (Any): Optional keyword arguments such as `constant_cols`
            (List[str]) to exclude invariant columns from plotting.

        Returns:
            None: Renders and exports plots directly to the configured output path.
        """
        if not args or not isinstance(args[0], pd.DataFrame):
            self.logger.error(
                "Expected a pandas DataFrame as the first positional argument."
            )
            raise ValueError(
                "Expected a pandas DataFrame as the first positional argument."
            )

        df: pd.DataFrame = args[0].dropna()
        constant_cols: List[str] = kwargs.get("constant_cols") or [
            col for col in df.columns if df[col].nunique(dropna=False) <= 1
        ]

        corr_matrix: pd.DataFrame = (
            args[1]
            if len(args) > 1 and isinstance(args[1], pd.DataFrame)
            else df.drop(columns=constant_cols, errors="ignore").corr(
                method=self._corr_method, numeric_only=True # type: ignore
            )
        )

        active_features = [col for col in self._feature_cols if col not in constant_cols]
        active_targets = [col for col in self._target_cols if col not in constant_cols]

        self.plot_distributions(df, active_targets, group_name="target_columns")
        self.plot_distributions(df, active_features, group_name="feature_columns")
        self.plot_boxplots(df, active_targets, group_name="target_columns")
        self.plot_boxplots(df, active_features, group_name="feature_columns")
        self.plot_correlation_matrix(corr_matrix, annot=False)


    def plot_distributions(
        self,
        df: pd.DataFrame,
        columns: Union[str, List[str]],
        group_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Renders histograms with Kernel Density Estimation (KDE) for specified columns.

        Args:
            df (pd.DataFrame): DataFrame containing the variables to plot.
            columns (str | List[str]): Configuration key name or list of column
            names.
            group_name (str | None): Optional group label for titles and filenames.
            **kwargs (Any): Additional keyword arguments forwarded to
            `sns.histplot`.

        Returns:
            Any: Output of `_finalize_plot`.
        """
        kde_default = kwargs.pop("kde", True)

        def _draw_hist(series: pd.Series, ax: Any) -> None:
            has_variance = series.nunique(dropna= True) > 1
            sns.histplot(series, kde=(kde_default and has_variance), ax=ax, **kwargs) # type: ignore

        return self._plot_grid(
            df=df,
            columns=columns,
            plot_fn=_draw_hist,
            title_prefix="Histogram plot",
            file_suffix="histplot",
            group_name=group_name,
        )


    def plot_boxplots(
        self,
        df: pd.DataFrame,
        columns: Union[str, List[str]],
        group_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Renders horizontal box-and-whisker diagrams to inspect spread and outliers.

        Args:
            df (pd.DataFrame): DataFrame containing the variables to plot.
            columns (str | List[str]): Configuration key name or list of column
            names.
            group_name (str | None): Optional group label for titles and filenames.
            **kwargs (Any): Additional keyword arguments forwarded to `sns.boxplot`.

        Returns:
            Any: Output of `_finalize_plot`.
        """

        def _draw_box(series: pd.Series, ax: Any) -> None:
            sns.boxplot(x=series, orient="h", ax=ax, **kwargs)

        return self._plot_grid(
            df=df,
            columns=columns,
            plot_fn=_draw_box,
            title_prefix="Box-and-whisker diagram",
            file_suffix="boxplot",
            group_name=group_name,
        )


    def plot_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        """Renders a square heatmap of the feature and target correlation matrix.

        Maps correlation coefficients across the full [-1, 1] interval centered
        at zero using a diverging colormap.

        Args:
            corr_matrix (pd.DataFrame): Symmetric correlation matrix DataFrame.
            **kwargs (Any): Optional overrides such as `annot` (bool) for cell
            values.

        Returns:
            Any: Output of `_finalize_plot`.
        """
        annot: bool = kwargs.get("annot", True)

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