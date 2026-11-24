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
    """A base class for components that render figures.

    Reads the figure settings from the `visualization` configuration, prepares the
    matplotlib style and provides helpers for figure creation, axis labels,
    multi-panel grids and saving, and the shared histogram plot. Subclasses add the
    concrete plots. Plots are rendered by explicit `plot_*` calls, never by `run()`.
    """
    def __init__(
        self,
        config_manager: ConfigManager,
        name: Optional[str] = None,
        subfolder: str = "",
    ) -> None:
        """Initializes plotting settings, labels and the output location.

        Figures are saved as PDF with the style from the configuration if
        `visualization.exportable` is true, otherwise as PNG and optionally shown.
        Exportable figures are never shown interactively. Configures shared seaborn styling
        and applies the publication style when exportable output is requested.

        Args:
            config_manager: Configuration manager instance.
            name: Optional component name.
            subfolder: Subdirectory of `settings.paths.figures_dir` for figures.
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
        """Applies configured font, scale and text-rendering options to matplotlib.

        LaTeX text rendering is enabled only if `visualization.use_tex` is true
        and a LaTeX installation is found, otherwise a warning is raised. Derives
        tick and legend sizes from the configured font scale before updating rcParams.
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
        """Creates a matplotlib figure with axes sized from the configured print width.

        Converts the configured width from points to inches and derives figure height.
        Uses a square aspect ratio when requested, otherwise applies the golden ratio.
        Returns the axes flattened into a one-dimensional array for convenient iteration.

        Args:
            nrows: Number of grid rows.
            ncols: Number of grid columns.
            square: If true, each panel is square; otherwise it follows the golden ratio.

        Returns:
            Tuple[Figure, Any]: Figure and a flat array of its axes.
        """
        width_in = self._width_pt / 72.27
        ratio = 1.0 if square else GOLDEN_RATIO
        height_in = width_in * ratio * nrows / ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(width_in, height_in), squeeze=False)
        return fig, axes.ravel()


    def _build_label(self, col: str, include_unit: bool = False) -> str:
        """Builds a display label from a column name and optional unit metadata.

        Uses the symbol from the dictionary, or the column name with spaces instead
        of underscores if there is no symbol. When requested, appends the configured
        unit or a dash when the column has no unit entry.

        Args:
            col: Column name.
            include_unit: If true, appends the unit from the dictionary.

        Returns:
            str: Formatted axis label.
        """
        label = self._labels.get(col, col.replace("_", " "))
        if include_unit:
            label = f"{label} [${self._units.get(col, '-')}$]"
        return label


    def _finalize_plot(self, fig: Figure, filename: str) -> None:
        """Lays out, saves, optionally displays and closes a matplotlib figure.

        Creates the configured output directory before writing the selected file format.
        Displays the figure only when interactive display is enabled.
        Closes the figure after output handling to release plotting resources.

        Args:
            fig: The figure to save.
            filename: File name without extension.

        Returns:
            None: The figure is written and closed as a side effect.
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
        """Creates a multi-panel figure by applying a drawing callback to each column.

        Divides columns across a two-column grid and labels each panel with its unit.
        Removes unused axes, applies a title for non-exportable figures and saves the result.
        Does nothing when the supplied frame contains no columns.

        Args:
            df_subset: The columns to plot.
            group_name: Column group name used in the title and file name.
            plot_fn: Callback that draws one column on its axes.
            title_prefix: Title prefix for non-exportable figures.
            file_suffix: Suffix appended to the output file name.

        Returns:
            None: The figure is saved as a side effect.
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


    def run(self, *args: Any, **kwargs: Any) -> None:
        """Provides the pipeline interface without rendering plots eagerly.

        Plot generation is performed only when an explicit `plot_*` method is called.
        This no-op allows every visualizer to follow the common pipeline component API.

        Returns:
            None: No work is performed.
        """


    def plot_distributions(self, df_subset: pd.DataFrame, group_name: str) -> None:
        """Plots and saves a histogram with a KDE curve for each supplied column.

        Uses the shared grid renderer to label, arrange and save the panels, and seaborn
        to draw the distribution and a smooth density estimate of each column.
        Empty column subsets are ignored by the grid helper.

        Args:
            df_subset: The columns to plot.
            group_name: Column group name, e.g. ``"target_columns"``.

        Returns:
            None: The figure is saved as a side effect.
        """
        def _draw_hist(series: pd.Series, axis: Any) -> None:
            sns.histplot(series, kde=True, ax=axis) # type: ignore

        self._plot_grid(df_subset, group_name, _draw_hist, "Histogram plot", "histplot")
