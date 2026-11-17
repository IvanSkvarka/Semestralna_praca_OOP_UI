from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from models.base_model import BaseModel # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error


class LogLogLinearModel(BaseModel):
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
        self._params: Dict[str, Any] = self.config_manager.get_param("pipeline.models.loglog_linear", {})



    def fit(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> "LogLogLinearModel":
        self.model_ = LinearRegression()
        self.model_.fit(X, y)
        return self


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)