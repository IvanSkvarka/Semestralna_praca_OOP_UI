from typing import Union, List, Optional
from numbers import Integral
from pathlib import Path
import pandas as pd

from base import BasePipelineComponent # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error

NOT_LOADED_MSG = "The data has not been loaded yet." \
                 " First, call the `run()` method or call the instance."


class DataLoader(BasePipelineComponent):
    """A pipeline component that loads and validates the raw CSV dataset.

    The file path and the column names are read from the configuration.
    The loaded data frame is available through read-only properties after `run()`.
    """
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        """Reads the data path and the column lists from the configuration.

        If `data.feature_columns` is empty, all columns that are not targets
        are used as features once the data is loaded.

        Args:
            config_manager (ConfigManager): Configuration manager instance.
            name (str | None): Component name.
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
        """Provides read-only access to the loaded data frame.

        Raises:
            RuntimeError: If the data has not been loaded yet.
        """
        if self._data is None:
            raise RuntimeError(NOT_LOADED_MSG)
        return self._data


    @property
    def target_columns(self) -> List[str]:
        """Provides a copy of the target column names."""
        return list(self._target_columns)


    @property
    def feature_columns(self) -> List[str]:
        """Provides a copy of the feature column names.

        Raises:
            RuntimeError: If no feature columns are configured and the data has not been loaded.
        """
        if self._feature_columns is None:
            raise RuntimeError(NOT_LOADED_MSG)
        return list(self._feature_columns)


    @property
    def feature_data(self) -> pd.DataFrame:
        """Provides the feature columns of the loaded data."""
        return self.data[self.feature_columns]


    @property
    def target_data(self) -> pd.DataFrame:
        """Provides the target columns of the loaded data."""
        return self.data[self.target_columns]


    def run(self) -> pd.DataFrame:
        """Loads the CSV file and checks it against the configured columns.

        Returns:
            pd.DataFrame: The loaded data frame.

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
        """Calculates the total number of row entries within the dataset."""
        return len(self.data)


    def __getitem__(self, key: Union[int, slice, str, List[str]]) -> Union[pd.Series, pd.DataFrame]:
        """Selects rows by position or columns by name.

        Args:
            key (int | slice | str | List[str]): Row position or slice, or column name(s).

        Returns:
            pd.Series | pd.DataFrame: The selected row(s) or column(s).

        Raises:
            TypeError: If the key type is not supported.
        """
        if isinstance(key, (Integral, slice)):
            return self.data.iloc[key]
        if isinstance(key, (str, list)):
            return self.data[key]
        raise TypeError(f"Unsupported index type: {type(key).__name__}")
