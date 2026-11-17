from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

from base import BasePipelineComponent # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error

ERR_MSG = (
    "The model has not been fitted yet. "
    "First, call the `fit()` or `run()` method."
)


class BaseModel(BasePipelineComponent, ABC):
    """

    """
    def __init__(
            self,
            config_manager: ConfigManager,
            model_key: str,
            params_override: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None
        ) -> None:
        """
        """
        super().__init__(config_manager, name)

        self._model_key = model_key
        base_params = config_manager.get_param(f"config.pipeline.models.{model_key}.params", {})
        self._params: Dict[str, Any] = {**base_params, **(params_override or {})} # Dictionary merge

        self._is_fitted: bool = False
        self._estimator: Optional[BaseEstimator] = None
        self._feature_names: List[str] = []



    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


    @property
    def params(self) -> Dict[str, Any]:
        return dict(self._params)


    @property
    def feature_names(self) -> List[str]:
        self._check_is_fitted()
        return list(self._feature_names)


    @abstractmethod
    def _build_estimator(self) -> "BaseEstimator":
        pass



    def _check_is_fitted(self) -> None:
        if not self._is_fitted or self._estimator is None:
            raise RuntimeError(ERR_MSG)



    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseModel":
        self._feature_names = X_train.columns.to_list()
        self._estimator = self._build_estimator()
        self._estimator.fit(X_train[self._feature_names], y_train)
        self._is_fitted = True
        return self


    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        missing = [col for col in self._feature_names if col not in X_test.columns]
        if missing:
            raise ValueError(f"Input data is missing required features: {missing}")

        return np.asarray(self._estimator.predict(X_test[self._feature_names]))


    def clone(self, params_override: Optional[Dict[str, Any]] = None) -> "BaseModel":
        new_params = {**self._params, **(params_override or {})} 
        return self.__class__(self.config_manager, self._model_key, params_override=new_params, name=self.name)


    def run(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        self.fit(X_train, y_train)

        if X_test is not None:
            return self.predict(X_test)

        return None