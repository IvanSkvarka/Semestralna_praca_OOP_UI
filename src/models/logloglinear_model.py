import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from models.base_model import BaseModel # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error


class LogLogLinearModel(BaseModel):
    """
    """
    def __init__(self, config_manager: ConfigManager) -> None:
        """
        """
        super().__init__(config_manager)



    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LogLogLinearModel":
        self.model_ = LinearRegression()
        self.model_.fit(X_train, y_train)
        return self


    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X_test)