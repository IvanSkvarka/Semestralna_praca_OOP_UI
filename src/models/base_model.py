from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from base import BasePipelineComponent # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error

ERR_MSG = (
    "The model has not been fitted yet. "
    "First, call the `fit()` or `run()` method."
)


class BaseModel(BaseEstimator, RegressorMixin, BasePipelineComponent, ABC):
    """

    """
    def __init__(self, config_manager: ConfigManager) -> None:
        """
        """
        super().__init__(config_manager)



    @property
    def name(self) -> str:
        return self.__class__.__name__



    @abstractmethod
    def _build_estimator(self) -> "BaseEstimator":
        pass


    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, groups: Optional[pd.Series] = None) -> "BaseModel":
        pass


    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        pass



    def run(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        self.fit(X_train, y_train)

        if X_test is not None:
            return self.predict(X_test)

        return None