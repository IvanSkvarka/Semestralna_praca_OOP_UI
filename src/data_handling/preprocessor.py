from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import json
import pandas as pd
import numpy as np

from base import BasePipelineComponent  # pylint: disable=import-error
from utils.config_manager import ConfigManager  # pylint: disable=import-error
from visualization.preprocessor_visualizer import PreprocessorVisualizer  # pylint: disable=import-error

ERR_MSG = "Preprocessor has not been run yet. First, call the `run()` method or call the instance."



class Preprocessor(BasePipelineComponent):
    """A pipeline component that prepares the raw data for modelling.

    Removes duplicate rows, other target columns and constant columns, assigns each row
    to a group (geometric family), applies the log10 transformation to the configured
    columns and optionally renders plots of the result. The results are available
    through read-only properties after `run()`.
    """
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        """Reads the preprocessing settings and creates the visualizer if plotting is enabled.

        Args:
            config_manager (ConfigManager): Configuration manager instance.
            name (str | None): Component name.
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
        """Provides the preprocessed target (log10 scale if `log_target` is enabled).

        Raises:
            RuntimeError: If `run()` has not been called yet.
        """
        if self._target is None:
            raise RuntimeError(ERR_MSG)
        return self._target


    @property
    def features(self) -> pd.DataFrame:
        """Provides the preprocessed features.

        Raises:
            RuntimeError: If `run()` has not been called yet.
        """
        if self._features is None:
            raise RuntimeError(ERR_MSG)
        return self._features


    @property
    def groups(self) -> pd.Series:
        """Provides the group number of each row, used to split the data without leakage.

        Raises:
            RuntimeError: If `run()` has not been called yet.
        """
        if self._groups is None:
            raise RuntimeError(ERR_MSG)
        return self._groups



    @staticmethod
    def _log10(data: pd.DataFrame) -> pd.DataFrame:
        """Applies log10 to all columns after checking that all their values are positive.

        Args:
            data (pd.DataFrame): The columns to transform.

        Returns:
            pd.DataFrame: The transformed columns.

        Raises:
            ValueError: If any column contains zero or negative values.
        """
        non_positive = data.columns[(data <= 0).any(axis=0)].to_list()
        if non_positive:
            raise ValueError(f"Cannot apply log10 to non-positive values in columns {non_positive}.")
        return np.log10(data) # type: ignore


    def _plot(self) -> None:
        """Renders the distributions and the group sizes, if plotting is enabled."""
        if self._visualizer is None:
            return

        #logged_target = [self.target.name] if self._log_target else []
        self._visualizer.plot_distributions(self.target.to_frame(), "target")
        self._visualizer.plot_distributions(self.features, "features")
        self._visualizer.plot_group_sizes(self.groups)


    def _save_report(self) -> None:
        """Saves a JSON report of the data prepared for training."""
        if not self.config_manager.get_param("pipeline.preprocessing.save_reports", True):
            return

        output_dir = Path(self.config_manager.get_param("settings.paths.output_dir", "output"))
        output_dir = output_dir / "preprocessing"
        output_dir.mkdir(parents=True, exist_ok=True)

        report: Dict[str, Any] = {
            "n_unique_groups": int(self.groups.nunique()),
            "features": [
                {"name": col, "log10_transformed": col in self._logged_features}
                for col in self.features.columns
            ],
            "target": {"name": self.target.name, "log10_transformed": self._log_target},
        }
        with open(output_dir / "preprocessing_report.json", mode="w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)



    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Runs the preprocessing, renders the plots and saves the report.

        Args:
            df (pd.DataFrame): The raw data.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.Series]: The features, the target and the group
            number of each row. All three share the same index.

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


    def inverse_target(self, y_log: Union[pd.Series, np.ndarray, float]) -> Union[pd.Series, np.ndarray, float]:
        """Converts the target back to its original scale.

        Args:
            y_log (pd.Series | np.ndarray | float): Target values in the preprocessed scale.

        Returns:
            pd.Series | np.ndarray | float: Values in the original scale, or unchanged
            if `log_target` is disabled.
        """
        return 10.0 ** y_log if self._log_target else y_log
