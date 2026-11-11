from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import math
import shutil
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import pandas as pd

from base import BasePipelineComponent  # pylint: disable=import-error
from utils.config_manager import ConfigManager  # pylint: disable=import-error

GOLDEN_RATIO = (5**0.5 - 1) / 2
GRID_COLUMNS = 2



class BaseVisualizer(BasePipelineComponent):
    """An abstract base class for components that render figures.

    Reads the figure settings from the `visualization` configuration, prepares the
    matplotlib style and provides helpers for figure creation, axis labels,
    multi-panel grids and saving. Subclasses add the concrete plots.
    """
    def __init__(
        self,
        config_manager: ConfigManager,
        name: Optional[str] = None,
        subfolder: str = "",
    ) -> None:
        """Reads the visualization settings and prepares the matplotlib style.

        Figures are saved as PDF with the style from the configuration if
        `visualization.exportable` is true, otherwise as PNG and optionally shown.
        Exportable figures are never shown interactively.

        Args:
            config_manager (ConfigManager): Configuration manager instance.
            name (str | None): Component name.
            subfolder (str): Subdirectory of `settings.paths.figures_dir` for the figures.
        """
        super().__init__(config_manager, name)
        self._exportable: bool = bool(config_manager["visualization.exportable"])
        self._show_plots: bool = bool(config_manager["visualization.show_plots"]) \
                                 and not self._exportable
        self._dpi: int = int(config_manager["visualization.dpi"])
        self._width_pt: float = float(config_manager["visualization.width_pt"])
        self._units: Dict[str, str] = config_manager.get_param("dictionary.units", {})
        self._labels: Dict[str, str] = config_manager.get_param("dictionary.labels", {})

        self._save_format: str = "pdf" if self._exportable else "png"
        self._output_dir: Path = Path(config_manager["settings.paths.figures_dir"]) / subfolder

        sns.set_style("white")
        if self._exportable:
            self._setup_style()


    def _setup_style(self) -> None:
        """Applies the font and size settings from the configuration to matplotlib.

        LaTeX text rendering is enabled only if `visualization.use_tex` is true
        and a LaTeX installation is found, otherwise a warning is raised.
        """
        use_tex = bool(self.config_manager["visualization.use_tex"])
        if use_tex and shutil.which("latex") is None:
            warnings.warn("LaTeX not found in PATH. Text rendering with LaTeX is disabled.")
            use_tex = False

        font_size = self.config_manager["visualization.font_size"]
        tick_size = font_size * float(self.config_manager["visualization.tick_scale"])
        legend_size = font_size * float(self.config_manager["visualization.legend_scale"])

        plt.rcParams.update({
            "font.family": self.config_manager["visualization.font_family"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "text.usetex": use_tex,
        })


    def _create_figure(
        self, nrows: int = 1, ncols: int = 1, square: bool = False
    ) -> Tuple[Figure, Any]:
        """Creates a figure with a grid of axes sized from `visualization.width_pt`.

        Args:
            nrows (int): Number of grid rows.
            ncols (int): Number of grid columns.
            square (bool): If True, each panel is square, otherwise it follows
                the golden ratio.

        Returns:
            Tuple[Figure, Any]: The figure and a flat array of its axes.
        """
        width_in = self._width_pt / 72.27
        ratio = 1.0 if square else GOLDEN_RATIO
        height_in = width_in * ratio * nrows / ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(width_in, height_in), squeeze=False)
        return fig, axes.ravel()


    def _build_label(self, col: str, include_unit: bool = False) -> str:
        """Builds an axis label for a data column.

        Uses the symbol from the dictionary, or the column name with spaces instead
        of underscores if there is no symbol.

        Args:
            col (str): Column name.
            include_unit (bool): If True, appends the unit from the dictionary.

        Returns:
            str: The label, e.g. ``$N$ [$\\mathrm{min^{-1}}$]``.
        """
        label = self._labels.get(col, col.replace("_", " "))
        if include_unit:
            label = f"{label} [${self._units.get(col, '-')}$]"
        return label


    def _finalize_plot(self, fig: Figure, filename: str) -> None:
        """Saves the figure to the output directory, optionally shows it, and closes it.

        Args:
            fig (Figure): The figure to save.
            filename (str): File name without extension.
        """
        fig.tight_layout()

        self._output_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._output_dir / f"{filename}.{self._save_format}"
        fig.savefig(file_path, dpi=self._dpi, bbox_inches="tight") # type: ignore

        if self._show_plots:
            fig.show()

        plt.close(fig)


    def _plot_grid(
        self,
        df_subset: pd.DataFrame,
        group_name: str,
        plot_fn: Callable[[pd.Series, Any], None],
        title_prefix: str,
        file_suffix: str,
    ) -> None:
        """Draws every column of `df_subset` into its own panel of one saved figure.

        Does nothing if `df_subset` has no columns.

        Args:
            df_subset (pd.DataFrame): The columns to plot.
            group_name (str): Name of the column group, used in the title and the file name.
            plot_fn (Callable): Function that draws one column (Series) on the given axes.
            title_prefix (str): Start of the figure title, shown only for non-exportable figures.
            file_suffix (str): Suffix of the file name.
        """
        columns = df_subset.columns
        if columns.empty:
            return

        n_rows = math.ceil(len(columns) / GRID_COLUMNS)
        fig, axes = self._create_figure(n_rows, GRID_COLUMNS)

        if not self._exportable:
            fig.suptitle(f"{title_prefix} of {group_name.replace('_', ' ')}")  # type: ignore

        for axis, col in zip(axes, columns):
            plot_fn(df_subset[col], axis)
            axis.set_xlabel(self._build_label(col, include_unit=True))

        for axis in axes[len(columns):]:
            fig.delaxes(axis)

        self._finalize_plot(fig, f"{group_name}_{file_suffix}")
