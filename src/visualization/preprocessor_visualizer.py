from typing import Optional, Any, Set, List
import pandas as pd
import seaborn as sns

from visualization.base_visualizer import BaseVisualizer # pylint: disable=import-error
from utils.config_manager import ConfigManager  # pylint: disable=import-error



class PreprocessorVisualizer(BaseVisualizer):
    """A plotting helper for the preprocessed data.

    Renders distributions of the preprocessed columns. The axis labels of the columns
    transformed by log10 (`pipeline.preprocessing.log_features` and, if `log_target`
    is enabled, the target) are marked with a log prefix and a dimensionless unit.
    Figures are saved in the `preprocessing` figures subdirectory.
    """
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        """Initializes the visualizer with the `preprocessing` output subdirectory.

        Args:
            config_manager (ConfigManager): Configuration manager instance.
            name (str | None): Component name.
        """
        super().__init__(config_manager, name, subfolder="preprocessing")

        target = config_manager.get_param("pipeline.preprocessing.target")
        log_target = config_manager.get_param("pipeline.preprocessing.log_target", False)
        log_features: List[str] = config_manager.get_param("pipeline.preprocessing.log_features", []) or []
        self._logged_columns: Set[str] = set(log_features)
        if log_target and target:
            self._logged_columns.add(target)


    def run(self) -> None:
        """Does nothing: the plots are rendered on demand by the `plot_*` methods."""


    def _build_label(self, col: str, include_unit: bool = False) -> str:
        """Builds an axis label for a data column, marking the log10 transformed columns.

        Args:
            col (str): Column name.
            include_unit (bool): If True, appends the unit. A log10 transformed column
                is dimensionless.

        Returns:
            str: The label, e.g. ``$N$ [$\\mathrm{min^{-1}}$]`` or ``$\\log_{10}$ $N$ [$-$]``.
        """
        if col not in self._logged_columns:
            return super()._build_label(col, include_unit)

        symbol = super()._build_label(col)
        label = f"$\\log_{{10}}$ {symbol}"
        return f"{label} [$-$]" if include_unit else label


    def plot_distributions(self, df_subset: pd.DataFrame, group_name: str) -> None:
        """Saves histograms with a KDE curve for each column of `df_subset`.

        Args:
            df_subset (pd.DataFrame): The columns to plot.
            group_name (str): Name of the column group, e.g. ``"target"``.
        """
        def _draw_hist(series: pd.Series, axis: Any) -> None:
            sns.histplot(series, kde=True, ax=axis) # type: ignore

        self._plot_grid(df_subset, group_name, _draw_hist, "Histogram plot", "histplot")


    def plot_group_sizes(self, groups: pd.Series) -> None:
        """Saves a bar chart with the number of rows in each group.

        Args:
            groups (pd.Series): Group number of each row.
        """
        sizes = groups.value_counts().sort_index()
        fig, axes = self._create_figure()

        if not self._exportable:
            fig.suptitle("Number of rows per group")  # type: ignore

        axes[0].bar(sizes.index, sizes.to_numpy())
        axes[0].set_xlabel("Group")
        axes[0].set_ylabel("Number of rows")

        self._finalize_plot(fig, "group_sizes")