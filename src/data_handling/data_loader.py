"""
A module that uses the base class for the pipeline and implements data 
loading from the directory configured in .settings.yaml.
Author: Ivan Skvarka
"""

from typing import Union, List, Optional, Any
from numbers import Integral
from pathlib import Path
import pandas as pd

from base import BasePipelineComponent
from utils.config_manager import ConfigManager

ERR_MSG = "The data has not been loaded yet. " \
                 "First, call the `run()` method or call the instance."



class DataLoader(BasePipelineComponent):
    """A pipeline component that loads and validates the raw CSV dataset.

    The file path and the column names are read from the configuration.
    The loaded data frame is available through read-only properties after `run()`.
    """
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        """Initializes the loader from the configured dataset path and columns.

        If `data.feature_columns` is empty, all columns that are not targets
        are used as features once the data is loaded. Target and configured
        feature names are copied to protect the configuration values from changes.
        The data frame remains unset until `run()` successfully reads the CSV.

        Args:
            config_manager: Configuration manager instance.
            name: Optional component name.
        """
        super().__init__(config_manager, name)
        self._file_path = Path(self.config_manager["settings.paths.raw_data"])
        self._target_columns: List[str] = list(self.config_manager["data.target_columns"])
        feature_columns = self.config_manager["data.feature_columns"]
        self._feature_columns: Optional[List[str]] = (
            list(feature_columns) if feature_columns else None
        )
        self._data: Optional[pd.DataFrame] = None



    @property
    def data(self) -> pd.DataFrame:
        """Returns the loaded data frame, or raises RuntimeError if loading has not completed."""
        if self._data is None:
            raise RuntimeError(ERR_MSG)
        return self._data


    @property
    def target_columns(self) -> List[str]:
        """Returns a copy of the configured target column names."""
        return list(self._target_columns)


    @property
    def feature_columns(self) -> List[str]:
        """Returns the feature column names or raises RuntimeError if they are not yet available."""
        if self._feature_columns is None:
            raise RuntimeError(ERR_MSG)
        return list(self._feature_columns)


    @property
    def feature_data(self) -> pd.DataFrame:
        """Returns the loaded data restricted to its configured feature columns."""
        return self.data[self.feature_columns]


    @property
    def target_data(self) -> pd.DataFrame:
        """Returns the loaded data restricted to its configured target columns."""
        return self.data[self.target_columns]



    def run(
        self, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        """Reads and validates the configured CSV dataset.

        Loads the file into a data frame and rejects empty datasets.
        Infers feature columns from non-target columns when none were configured.
        Stores the validated frame for subsequent property and index access.

        Returns:
            pd.DataFrame: The loaded and validated data frame.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the file is empty or configured columns are missing in it.
        """
        df = pd.read_csv(self._file_path)
        if df.empty:
            raise ValueError(f"Loaded dataset '{self._file_path}' is empty.")

        if self._feature_columns is None:
            self._feature_columns = [c for c in df.columns if c not in self._target_columns]

        missing = [c for c in self._target_columns + self._feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} from the configuration are missing in the data.")

        self._data = df
        return df



    def __len__(self) -> int:
        """Returns the number of rows in the loaded dataset.

        Delegates to the data frame accessor, so accessing the length before loading
        also raises the same runtime error as accessing the data directly.

        Returns:
            int: Number of rows in the loaded data frame.
        """
        return len(self.data)


    def __getitem__(self, key: Union[int, slice, str, List[str]]) -> Union[pd.Series, pd.DataFrame]:
        """Selects dataset rows by position or columns by name.

        Integer and slice keys are applied through positional indexing.
        String keys and lists of strings select data frame columns.
        Unsupported key types are rejected.

        Args:
            key: Row position or slice, or column name(s).

        Returns:
            pd.Series | pd.DataFrame: The selected row or rows, or selected column data.

        Raises:
            TypeError: If the key type is not supported.
        """
        if isinstance(key, (Integral, slice)):
            return self.data.iloc[key]
        if isinstance(key, (str, list)):
            return self.data[key]
        raise TypeError(f"Unsupported index type: {type(key).__name__}")
