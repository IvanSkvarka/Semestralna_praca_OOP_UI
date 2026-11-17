from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import json
import pandas as pd
import numpy as np

from base import BasePipelineComponent  # pylint: disable=import-error
from utils.config_manager import ConfigManager  # pylint: disable=import-error
from visualization.preprocessor_visualizer import PreprocessorVisualizer  # pylint: disable=import-error

ERR_MSG = "Preprocessor has not been run yet. " \
                 "First, call the `run()` method or call the instance."



class Preprocessor(BasePipelineComponent):
    """A pipeline component that prepares the raw data for modelling.

    Removes duplicate rows, other target columns and constant columns, assigns each row
    to a group (geometric family), applies the log10 transformation to the configured
    columns and optionally renders plots of the result. The results are available
    through read-only properties after `run()`.
    """
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        """Initializes preprocessing options and optional visualization support.

        Reads target, grouping, transformation and cleanup options from configuration.
        Initializes result storage before any input data has been processed.
        Creates a visualizer only when preprocessing plots are enabled.

        Args:
            config_manager: Configuration manager instance.
            name: Optional component name.
        """
        super().__init__(config_manager, name)
        self._target_name: Optional[str] = self.config_manager.get_param(
            "pipeline.preprocessing.target"
        )
        self._group_cols: List[str] = self.config_manager.get_param(
            "pipeline.preprocessing.group_columns", []
        ) or []
        self._log_target: bool = bool(
            self.config_manager.get_param("pipeline.preprocessing.log_target", False)
        )
        self._log_features: List[str] = self.config_manager.get_param(
            "pipeline.preprocessing.log_features", []
        ) or []
        self._drop_duplicates: bool = bool(
            self.config_manager.get_param("pipeline.preprocessing.drop_duplicates", False)
        )
        self._drop_constant_columns: bool = bool(
            self.config_manager.get_param("pipeline.preprocessing.drop_constant_columns", False)
        )
        self._target_cols: List[str] = self.config_manager.get_param("data.target_columns", [])
        self._feature_cols: List[str] = self.config_manager.get_param("data.feature_columns", [])

        self._features: Optional[pd.DataFrame] = None
        self._target: Optional[pd.Series] = None
        self._groups: Optional[pd.Series] = None
        self._logged_features: List[str] = []

        self._visualizer: Optional[PreprocessorVisualizer] = (
            PreprocessorVisualizer(config_manager, name=f"{self.name}Visualizer")
            if self.config_manager.get_param("pipeline.preprocessing.generate_plots", True)
            else None
        )



    @property
    def target(self) -> pd.Series:
        """Returns the preprocessed target or raises RuntimeError if preprocessing has not run."""
        if self._target is None:
            raise RuntimeError(ERR_MSG)
        return self._target


    @property
    def features(self) -> pd.DataFrame:
        """Returns the preprocessed features or raises RuntimeError if preprocessing has not run."""
        if self._features is None:
            raise RuntimeError(ERR_MSG)
        return self._features


    @property
    def groups(self) -> pd.Series:
        """Returns each row's group label or raises RuntimeError if preprocessing has not run."""
        if self._groups is None:
            raise RuntimeError(ERR_MSG)
        return self._groups



    @staticmethod
    def _log10(data: pd.DataFrame) -> pd.DataFrame:
        """Applies the base-10 logarithm to every column after validating its values.

        Checks each column for zero or negative values before transforming the frame.
        Rejects the entire operation if any value is not strictly positive.
        Returns a new frame containing the transformed values.

        Args:
            data: The columns to transform.

        Returns:
            pd.DataFrame: The transformed columns.

        Raises:
            ValueError: If any column contains zero or negative values.
        """
        non_positive = data.columns[(data <= 0).any(axis=0)].to_list()
        if non_positive:
            raise ValueError(
                f"Cannot apply log10 to non-positive values in columns {non_positive}."
            )
        return np.log10(data) # type: ignore



    def _plot(self) -> None:
        """Renders target, feature and group-size plots when a visualizer is configured.

        Does nothing when plot generation was disabled during initialization.
        Otherwise sends the current preprocessing results to their corresponding plots.
        The results are accessed through properties that validate preprocessing state.
        """
        if self._visualizer is None:
            return

        self._visualizer.plot_distributions(self.target.to_frame(), "target")
        self._visualizer.plot_distributions(self.features, "features")
        self._visualizer.plot_group_sizes(self.groups)


    def _save_report(self) -> None:
        """Writes a JSON summary of the prepared training data when reports are enabled.

        Creates the preprocessing output directory when needed.
        Records dataset dimensions, grouping information and applied log transformations.
        Does nothing when report saving is disabled in the configuration.
        """
        if not self.config_manager.get_param("pipeline.preprocessing.save_reports", True):
            return

        output_dir = Path(self.config_manager.get_param("settings.paths.output_dir", "output"))
        output_dir = output_dir / "preprocessing"
        output_dir.mkdir(parents=True, exist_ok=True)

        report: Dict[str, Any] = {
            "n_rows": int(len(self.features)),
            "n_features": int(self.features.shape[1]),
            "n_unique_groups": int(self.groups.nunique()),
            "features": [
                {"name": col, "log10_transformed": col in self._logged_features}
                for col in self.features.columns
            ],
            "target": {"name": self.target.name, "log10_transformed": self._log_target},
        }
        with open(output_dir / "preprocessing_report.json", mode="w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)



    def run(
        self, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepares model features, target values and leakage-safe group labels.

        Validates configured columns, optionally removes duplicates and constant features,
        and applies requested log10 transformations.
        Rejects non-finite results before storing them as the latest preprocessing output.
        Finally generates enabled plots and a JSON report.

        Args:
            df: The raw data.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.Series]: Features, target and row group labels,
                all sharing the same index.

        Raises:
            ValueError: If the target or group columns are not configured, if a configured
                column is missing in `df`, if log10 is applied to non-positive values or if
                the result contains NaN or infinite values.
        """
        target = self._target_name
        if not target:
            raise ValueError("Target column 'pipeline.preprocessing.target' is not configured.")
        if not self._group_cols:
            raise ValueError(
                "Group columns 'pipeline.preprocessing.group_columns' are not configured."
            )

        required = [target, *self._group_cols, *self._log_features]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Configured columns {missing} are missing in the data.")

        data = df.drop_duplicates().reset_index(drop=True) if self._drop_duplicates else df

        groups = data.groupby(self._group_cols, dropna=False).ngroup().rename("geom_family")

        feature_cols = self._feature_cols or [c for c in data.columns if c not in self._target_cols]
        features = data[[c for c in feature_cols if c != target]]
        if self._drop_constant_columns:
            features = features.loc[:, features.nunique(dropna=False) > 1]

        log_cols = [c for c in self._log_features if c in features.columns]
        if log_cols:
            features = features.assign(**self._log10(features[log_cols])) # type: ignore

        target_values = data[target]
        if self._log_target:
            target_values = self._log10(target_values.to_frame())[target]

        if not (np.isfinite(features.to_numpy(dtype=float)).all()
                and np.isfinite(target_values.to_numpy(dtype=float)).all()):
            raise ValueError("The preprocessed data contain NaN or infinite values.")

        self._features, self._target, self._groups = features, target_values, groups
        self._logged_features = log_cols
        self._plot()
        self._save_report()
        return features, target_values, groups
