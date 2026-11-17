from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from models.base_model import BaseModel # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error


class GradientBoostingModel(BaseModel):
    """
    """
    def __init__(
        self,
        config_manager: ConfigManager,
        n_estimators: Optional[int] = None,
        learning_rate: Optional[float] = None,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        """
        super().__init__(config_manager)
        self._params: Dict[str, Any] = self.config_manager.get_param("pipeline.models.gradient_boosting") or {}

        self.n_estimators = n_estimators if n_estimators is not None else self._params.get("n_estimators", 100)
        self.learning_rate = learning_rate if learning_rate is not None else self._params.get("learning_rate", 0.1)
        self.max_depth = max_depth if max_depth is not None else self._params.get("max_depth", 3)
        self.random_state = random_state if random_state is not None else self._params.get("random_state", 42)



    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "GradientBoostingModel":
        self.model_ = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.model_.fit(X_train, y_train)
        return self


    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X_test)