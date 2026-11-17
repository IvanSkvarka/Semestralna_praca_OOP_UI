from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from models.base_model import BaseModel # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error


class GradientBoostingModel(BaseModel):
    """Regression model backed by scikit-learn's gradient boosting estimator.

    Initializes estimator parameters from explicit arguments when supplied.
    Missing arguments are filled from pipeline configuration and then defaults.
    """
    def __init__(
        self,
        config_manager: ConfigManager,
        n_estimators: Optional[int] = None,
        learning_rate: Optional[float] = None,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """Initializes gradient boosting parameters from arguments or configuration.

        Reads the model settings and applies built-in defaults for unspecified values.
        The estimator itself is created later when `fit()` is called.

        Args:
            config_manager: Configuration manager used by the pipeline.
            n_estimators: Number of boosting stages, or `None` to use configuration.
            learning_rate: Shrinkage applied to each boosting stage, or `None` for config.
            max_depth: Maximum depth of individual regression trees, or `None` for config.
            random_state: Seed for reproducible fitting, or `None` for config.

        Returns:
            None: The model is initialized but not fitted.
        """
        super().__init__(config_manager)
        self._params: Dict[str, Any] = self.config_manager.get_param(
            "pipeline.models.gradient_boosting"
        ) or {}
        self.n_estimators = n_estimators if n_estimators is not None else self._params.get(
            "n_estimators", 100
        )
        self.learning_rate = learning_rate if learning_rate is not None else self._params.get(
            "learning_rate", 0.1
        )
        self.max_depth = max_depth if max_depth is not None else self._params.get(
            "max_depth", 3
        )
        self.random_state = random_state if random_state is not None else self._params.get(
            "random_state", 42
        )



    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "GradientBoostingModel":
        """Trains a gradient boosting regressor on the supplied training data.

        Creates the estimator using the parameters resolved at initialization.
        Fits it to the training features and targets and stores the fitted estimator.

        Args:
            X_train: Training feature matrix.
            y_train: Training target values.

        Returns:
            GradientBoostingModel: This fitted model instance.
        """
        self.model_ = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.model_.fit(X_train, y_train) # type: ignore
        return self


    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predicts target values using the fitted gradient boosting estimator.

        Checks the fitted state before passing the feature matrix to the estimator.
        Returns one prediction for each test row.

        Args:
            X_test: Feature matrix to predict.

        Returns:
            np.ndarray: Predicted target values.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        return self.model_.predict(X_test)