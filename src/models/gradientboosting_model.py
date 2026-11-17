from typing import Optional, Dict, Any
from sklearn.ensemble import GradientBoostingRegressor

from models.base_model import BaseModel # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error

class GradientBoostingModel(BaseModel):
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
        super().__init__(config_manager, "gradient_boosting", params_override, name)

    def _build_estimator(self):
        random_state = self.config_manager.get_param("config.pipeline.random_state")
        params = {"random_state": random_state, **self._params}
        return GradientBoostingRegressor(**params)
    
    @property
    def feature_importances(self) -> Dict[str, float]:
        self._check_is_fitted()
        importances = {
            feature: float(importance)
            for feature, importance in zip(
                self._feature_names, self._estimator.feature_importances_
            )
        }
        return dict(
            sorted(importances.items(), key=lambda item: item[1], reverse=True)
        )