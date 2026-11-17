from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from models.base_model import BaseModel # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error


class MeanBaselineModel(BaseModel):
    """
    """
    def __init__(self, config_manager: ConfigManager) -> None:
        """
        """
        super().__init__(config_manager)
        self._params: Dict[str, Any] = self.config_manager.get_param("pipeline.models.mean_baseline", {})



    def fit(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> "DummyRegressor":
        self.model_ = DummyRegressor()
        self.model_.fit(X, y)
        return self


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)