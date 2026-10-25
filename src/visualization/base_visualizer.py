from typing import Union, Any, Dict, Optional, Tuple, Callable, List
from abc import abstractmethod
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import pandas as pd
import math

from base import BasePipelineComponent  # pylint: disable=import-error
from utils.config_manager import ConfigManager  # pylint: disable=import-error


class BaseVisualizer(BasePipelineComponent):
    """
    
    """
    def __init__(
        self,
        config_manager: ConfigManager,
        name: Optional[str] = None,
        subfolder: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        
        """
        super().__init__(config_manager, name)
        self._show_plots: bool = bool(config_manager["visualization.show_plots"])
        self._dpi: int = int(config_manager["visualization.dpi"])
        self._width_pt: float = float(config_manager["visualization.width_pt"])
        self._names: Dict[str, str] = config_manager.get_param("dictionary.names", {})
        self._units: Dict[str, str] = config_manager.get_param("dictionary.units", {})
        self._labels: Dict[str, str] = config_manager.get_param("dictionary.labels", {})

        self._exportable: bool = bool(config_manager["visualization.exportable"])
        if self._exportable and self._show_plots:
            self.logger.warning(
                "Exportable=True: vypínam interaktívne zobrazovanie (show_plots=False)."
            )
            self._show_plots = False


        self._save_format: str = "pdf" if self._exportable else "png"
        base_dir = Path(config_manager["settings.paths.figures_dir"])
        self._output_dir: Path = base_dir / subfolder if subfolder else base_dir

        sns.set_style('white')
        if self._exportable:
            self._setup_style()


    def _setup_style(self) -> None:
        """ """
        use_tex = bool(
            self.config_manager["visualization.use_tex"] and 
            (shutil.which("latex") or shutil.which("pdflatex")
            or shutil.which("tex")) is not None
        )
        if not use_tex and self.config_manager["visualization.use_tex"]:
            self.logger.warning("LaTeX not found in PATH. Shutting down text.usetex.")

        tick_scale = float(self.config_manager["visualization.tick_scale"])
        legend_scale = float(self.config_manager["visualization.legend_scale"])

        plt.rcParams.update({
            "font.family": self.config_manager["visualization.font_family"],
            "font.size": self.config_manager["visualization.font_size"],
            "axes.labelsize": self.config_manager["visualization.font_size"],
            "xtick.labelsize": self.config_manager["visualization.font_size"] * tick_scale,
            "ytick.labelsize": self.config_manager["visualization.font_size"] * tick_scale,
            "legend.fontsize": self.config_manager["visualization.font_size"] * legend_scale,
            "text.usetex": use_tex,
            "pgf.rcfonts": False,
            "pgf.texsystem": "pdflatex",
            "pgf.preamble": r"\usepackage{mathptmx}",
        })


    def _create_figure(
        self,
        nrows: int = 1,
        ncols: int = 1,
        fraction: float = 1.0,
        square: bool = False,
    ) -> Tuple[Figure, Any]:
        """ """
        width_in = (self._width_pt * fraction) / 72.27
        ratio = 1.0 if square else (5**0.5 - 1) / 2
        height_in = width_in * ratio * (nrows / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(width_in, height_in), squeeze=False
        )
        return (fig, axes.flatten())


    def _build_label(
        self,
        col: str = "",
        prefix: str = "",
        suffix: str = "",
        include_name: bool = False,
        include_symbol: bool = True,
        include_unit: bool = False
    ) -> str:
        """ """

        symbol = self._labels.get(col, " ") if include_symbol else " "
        name = self._names.get(col, col.replace("_", " ")) if include_name else " "
        unit = f" [${self._units.get(col, "-")}$]" if include_unit else " "

        prefix = prefix.replace("_", " ")
        suffix = suffix.replace("_", " ")

        return f"{prefix}{name}{symbol}{unit}{suffix}"


    def _finalize_plot(self, fig: Figure, filename: str) -> None:
        """ """
        fig.tight_layout()

        self._output_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._output_dir / f"{filename}.{self._save_format}"

        fig.savefig(file_path, dpi=self._dpi, bbox_inches="tight") # type: ignore
        self.logger.info("Graph saved in '%s'", file_path)

        if self._show_plots:
            fig.show()

        plt.close(fig)


    def _plot_grid(
        self,
        df: pd.DataFrame,
        columns: Union[str, List[str]],
        plot_fn: Callable[[pd.Series, Any], None],
        title_prefix: str,
        file_suffix: str,
        group_name: Optional[str] = None,
        n_cols: int = 2,
    ) -> Any:
        """Helper method that renders a multi-panel grid of univariate plots.

        Resolves the requested column list, constructs a subplot grid, applies
        the provided plotting callback to each column, removes unused axes,
        and finalizes the figure.

        Args:
            df (pd.DataFrame): Source DataFrame containing the data to plot.
            columns (str | List[str]): Config key suffix (e.g., 'target_columns') or
            explicit list of column names to visualize.
            plot_fn (Callable): Callback function accepting (Series, Axis) to draw.
            title_prefix (str): Prefix string for the figure suptitle.
            file_suffix (str): Suffix appended to the exported filename.
            group_name (str | None): Identifier used for title and filename when
            `columns` is passed as a list.
            n_cols (int): Number of columns in the subplot grid. Defaults to 2.

        Returns:
            Any: Output returned by `self._finalize_plot`.
        """
        if isinstance(columns, str):
            cols_to_plot: List[str] = self.config_manager[f"data.{columns}"]
            label_name = columns
        else:
            cols_to_plot = columns
            label_name = group_name or "selected_columns"

        cols_to_plot = [col for col in cols_to_plot if col in df.columns]
        n_rows = math.ceil(len(cols_to_plot) / n_cols)

        fig, axes = self._create_figure(n_rows, n_cols)

        if not self._exportable:
            fig.suptitle(self._build_label(prefix=f"{title_prefix} of {label_name}"))  # type: ignore

        for idx, col in enumerate(cols_to_plot):
            plot_fn(df[col], axes[idx])
            axes[idx].set_xlabel(self._build_label(col, include_unit=True))

        for i in range(len(cols_to_plot), len(axes)):
            fig.delaxes(axes[i])

        return self._finalize_plot(fig, f"{label_name}_{file_suffix}")



    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """ """
