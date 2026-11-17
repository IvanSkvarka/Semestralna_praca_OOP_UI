from typing import Optional, Dict, Any
from sklearn.linear_model import LinearRegression
import numpy as np

from models.base_model import BaseModel # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error

class LogLogLinearModel(BaseModel):
    """
    """
    def __init__(
            self,
            config_manager: ConfigManager,
            params_override: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None
        ) -> None:
        """
        """
        super().__init__(config_manager, "loglog_linear", params_override, name)

    def _build_estimator(self) -> "LinearRegression":
        return LinearRegression(**self._params)

    @property
    def coefficients(self) -> Dict[str, float]:
        self._check_is_fitted()
        return {
            feature: float(coef)
            for feature, coef in zip(self._feature_names, np.ravel(self._estimator.coef_))
        }
    
    @property
    def intercept(self) -> float:
        self._check_is_fitted()
        return float(np.asarray(self._estimator.intercept_).item())