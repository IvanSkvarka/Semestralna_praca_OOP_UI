from typing import Union, List, Optional, Any
from pathlib import Path
import pandas as pd

from base import BasePipelineComponent # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error



class DataLoader(BasePipelineComponent):
    """A pipeline component responsible for loading and validating tabular datasets.

    Reads raw CSV files from paths specified in the shared configuration manager.
    The component dynamically extracts target and feature column sets from settings.
    Encapsulates the structural dataset matrix and provides safe slice access.
    """
    def __init__(self, config_manager: ConfigManager, name: Union[str, None] = None) -> None:
        """Initializes the data loader component and resolves data storage paths.

        Invokes the base component constructor and extracts specific dataset configurations.
        The method parses target columns and resolves optional explicit feature requirements.
        Internal data storage remains uninitialized until the component is explicitly executed.

        Args:
            config_manager (ConfigManager): Configuration manager instance.
            name (str | None): Module name.
        """
        super().__init__(config_manager, name)
        self._file_path = Path(self.config_manager["settings.paths.raw_data"])
        self._target_columns = list(self.config_manager["data.target_columns"])
        _feature_columns = self.config_manager["data.feature_columns"]
        self._feature_columns = list(_feature_columns) if _feature_columns else None

        self._data: Optional[pd.DataFrame] = None



    @property
    def data(self) -> pd.DataFrame:
        """Provides read-only access to the completely loaded data frame matrix.

        Checks whether the internal dataset placeholder is currently empty or unassigned.
        If data has not been processed yet, it logs a severe state error and raises a runtime exception.

        Returns:
            pd.DataFrame: The validated underlying Pandas DataFrame structure.

        Raises:
            RuntimeError: If data has not been initialized via run() method yet.
        """
        if self._data is None or self._data.empty:
            err_msg = "The data has not been loaded yet." \
                      " First, call the `run()` method or an instance of the object."
            self.logger.error(err_msg)
            raise RuntimeError(err_msg)
        return self._data


    @property
    def target_columns(self) -> List[str]:
        """Provides read-only access the current list of designated target column identifiers.

        Generates a fresh list representation of the registered target label parameters.
        This provides external callers with visibility into columns used for modeling predictions.

        Returns:
            List[str]: A list containing the target column names.
        """
        return list(self._target_columns)


    @property
    def feature_columns(self) -> List[str]:
        """Provides the list of active predictive feature column names.

        Lazily evaluates features if no explicit column listing was defined in configuration.
        The method filters out all current target identifiers from the underlying dataset schema.

        Returns:
            List[str]: A list containing the designated training feature names.
        """
        if self._feature_columns is None:
            self._feature_columns = [col for col in self.data.columns
                                     if col not in self._target_columns]

        return list(self._feature_columns)


    @property
    def feature_data(self) -> pd.DataFrame:
        """Slices and returns the feature matrix from the loaded dataset.

        Queries the internal data frame utilizing the resolved feature column list.
        This isolates the independent variable attributes required for preprocessing inputs.

        Returns:
            pd.DataFrame: A data frame subset containing only training features.
        """
        return self.data[self.feature_columns]


    @property
    def target_data(self) -> pd.DataFrame:
        """Slices and returns the target vector matrix from the loaded dataset.

        Queries the internal data frame utilizing the verified target column identifier list.
        This isolates the required dependent label attributes.

        Returns:
            pd.DataFrame: A data frame subset containing only target labels.
        """
        return self.data[self.target_columns]



    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validates the structural integrity and columns of the loaded data frame.

        Raises errors for empty files and checks for missing targets or features.
        The method prints warnings and dynamically drops columns not present in the layout.

        Args:
            df (pd.DataFrame): The raw dataframe instance loaded from storage.

        Raises:
            ValueError: If the passed dataframe contains zero data entries.
        """
        if df.empty:
            self.logger.error("Loaded dataset is empty.")
            raise ValueError("Loaded dataset is empty.")

        missing_targets = [col for col in self._target_columns if col not in df.columns]
        if missing_targets:
            self.logger.warning("Missing target columns in dataset: %s. Dropped.", missing_targets)
            self._target_columns = [col for col in self._target_columns if col in df.columns]

        if self._feature_columns is not None:
            # NOTE: Overwrite of `missing_targets`
            missing_targets = [col for col in self._feature_columns if col not in df.columns]
            if missing_targets:
                self.logger.warning(
                    "Missing feature columns in dataset: %s. Dropped.", missing_targets
                )
                self._feature_columns = [col for col in self._feature_columns if col in df.columns]


    def run(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Executes the complete CSV file retrieval and validation routine.

        Verifies physical file existence on disk before triggering the parsing layer.
        The method reads comma-separated inputs, runs validations, and populates states.

        Args:
            *args (Any): Variable keyword parameters, currently unsupported.
            **kwargs (Any): Variable keyword parameters, currently unsupported.

        Returns:
            pd.DataFrame: The completely parsed and structurally verified data structure.

        Raises:
            FileNotFoundError: If the data file path does not point to a valid resource.
        """
        if not self._file_path.exists():
            self.logger.error("Data file not found: %s.", self._file_path)
            raise FileNotFoundError(f"Data file not found: {self._file_path}")

        if args or kwargs:
            self.logger.warning("Args and kwargs are not implemented yet.")

        df = pd.read_csv(self._file_path)
        self._validate_dataframe(df)

        self._data = df
        self.logger.info("Data %s successfully loaded.", df.shape)
        return self._data



    def __len__(self) -> int:
        """Calculates the total number of row entries within the dataset."""
        return len(self.data)


    def __getitem__(self, key: Union[int, slice, str, List[str]]) -> Union[pd.Series, pd.DataFrame]:
        """Enables direct index and key slice routing over the dataset.

        Args:
            key (int | slice | str | List[str]): The target index row or column label selection.

        Returns:
            Union[pd.Series, pd.DataFrame]: The extracted row series or column slice projection.

        Raises:
            TypeError: If the supplied index key does not match any valid slice types.
        """
        if isinstance(key, (int, slice)):
            return self.data.iloc[key]
        elif isinstance(key, (str, list)): #type: ignore
            return self.data[key]
        else:
            raise TypeError(f"Unsupported index type: {type(key).__name__}")
