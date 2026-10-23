from typing import Union, List
from pathlib import Path
import pandas as pd
from base import BasePipelineComponent # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error

class DataLoader(BasePipelineComponent):
    """
    
    """

    def __init__(self, config_manager: ConfigManager, name: Union[str, None] = None) -> None:
        """
        
        """
        super().__init__(config_manager, name)
        self._file_path = Path(self.config_manager["settings.paths.raw_data"])
        self._target_columns = list(self.config_manager["config.data.target_columns"])
        _feature_columns = self.config_manager["config.data.feature_columns"]
        self._feature_columns = list(_feature_columns) if _feature_columns else None

        self._data = pd.DataFrame(None)


    @property
    def data(self) -> pd.DataFrame:
        """ """
        if self._data.empty:
            err_msg = "The data has not been loaded yet." \
            " First, call the `run()` method or an instance of the object."
            self.logger.error(err_msg)
            raise RuntimeError(err_msg)
        return self._data

    @property
    def target_columns(self) -> List[str]:
        """ """
        return self._target_columns

    @property
    def feature_columns(self) -> List[str]:
        """ """
        if self._feature_columns is None:
            self._feature_columns = [col for col in self.data.columns
                                     if col not in self._target_columns]

        return self._feature_columns

    @property
    def X(self) -> pd.DataFrame:
        """ """
        return self.data[self.feature_columns].copy()

    @property
    def y(self) -> pd.DataFrame:
        """ """
        return self.data[self.target_columns].copy()


    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        
        """
        if df.empty:
            self.logger.error("Loaded dataset is empty.")

        missing_targets = [col for col in self._target_columns if col not in df.columns]
        if missing_targets:
            self.logger.error(f"Missing target columns in dataset: {missing_targets}")

        if self._feature_columns is not None:
            # NOTE: Prepisanie premennej `missing_targets`
            missing_targets = [col for col in self._feature_columns if col not in df.columns]
            if missing_targets:
                self.logger.error(f"Missing feature columns in dataset: {missing_targets}")


    def run(self) -> pd.DataFrame:
        """
        
        """

        df = pd.read_csv(self._file_path)
        self._validate_dataframe(df)

        self._data = df
        self.logger.info(f"Data {df.shape} successfully loaded.")
        return self._data

    def __len__(self) -> int:
        """ """
        return len(self.data)

    def __getitem__(self, key: int | slice | str | List[str]) -> Union[pd.Series, pd.DataFrame]:
        """ """
        if isinstance(key, (int, slice)):
            return self.data.iloc[key]
        elif isinstance(key, str):
            return self.data[key]
        else:
            raise TypeError(f"Unsupported index type: {type(key).__name__}")