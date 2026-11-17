from typing import Optional, Dict, Any
from sklearn.dummy import DummyRegressor

from models.base_model import BaseModel # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error

class MeanBaselineModel(BaseModel):
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
        super().__init__(config_manager, "mean_baseline", params_override, name)

    def _build_estimator(self) -> "DummyRegressor":
        return DummyRegressor(strategy="mean")

    @property
    def constant(self) -> float:
        self._check_is_fitted()
        return float(np.asarray(self._estimator.constant_).item())