"""
Drawing heplper for Preprocessor.
Author: Ivan Skvarka
"""

from typing import Optional, Set, List
import pandas as pd

from visualisation.base_visualiser import BaseVisualiser # pylint: disable=import-error
from utils.config_manager import ConfigManager  # pylint: disable=import-error



class PreprocessorVisualiser(BaseVisualiser):
    """A plotting helper for the preprocessed data.

    Renders distributions (inherited) of the preprocessed columns and the group sizes.
    The axis labels of the columns transformed by log10 (`pipeline.preprocessing.log_features`
    and, if `log_target` is enabled, the target) are marked with a log prefix and
    a dimensionless unit. Figures are saved in the `preprocessing` figures subdirectory.
    """
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        """Initializes plotting support for preprocessing results.

        Configures the preprocessing-specific output subdirectory.
        Collects target and feature names that should be labeled as log-transformed.

        Args:
            config_manager: Configuration manager instance.
            name: Optional component name.

        Returns:
            None: The visualiser is ready to render plots.
        """
        super().__init__(config_manager, name, subfolder="preprocessing")

        target = config_manager.get_param("pipeline.preprocessing.target")
        log_target = config_manager.get_param("pipeline.preprocessing.log_target", False)
        log_features: List[str] = config_manager.get_param(
            "pipeline.preprocessing.log_features", []
        ) or []
        self._logged_columns: Set[str] = set(log_features)
        if log_target and target:
            self._logged_columns.add(target)



    def _build_label(self, col: str, include_unit: bool = False) -> str:
        """Builds an axis label that marks columns transformed with log10.

        Delegates untransformed columns to the base visualiser's label builder.
        Marks transformed columns with a log10 prefix and dimensionless unit when requested.
        Uses the configured label dictionary to render the original column symbol.

        Args:
            col: Column name.
            include_unit: If true, appends the unit; transformed values are dimensionless.

        Returns:
            str: The label, e.g. ``$N$ [$\\mathrm{min^{-1}}$]`` or ``$\\log_{10}$ $N$ [$-$]``.
        """
        if col not in self._logged_columns:
            return super()._build_label(col, include_unit)

        symbol = super()._build_label(col)
        label = f"$\\log_{{10}}$ {symbol}"
        return f"{label} [$-$]" if include_unit else label


    def plot_group_sizes(self, groups: pd.Series) -> None:
        """Renders and saves a bar chart of the row count for every group.

        Counts rows per group and orders the bars by group identifier.
        Labels the chart axes and adds a title for non-exportable figures.
        Writes the result through the shared figure finalization helper.

        Args:
            groups: Group number of each row.

        Returns:
            None: The chart is saved as a side effect.
        """
        sizes = groups.value_counts().sort_index()
        fig, axes = self._create_figure()

        if not self._exportable:
            fig.suptitle("Number of rows per group")  # type: ignore

        axes[0].bar(sizes.index, sizes.to_numpy())
        axes[0].set_xlabel("Group")
        axes[0].set_ylabel("Number of rows")

        self._finalize_plot(fig, "group_sizes")
