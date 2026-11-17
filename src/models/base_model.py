from typing import Optional, Any, Union
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from base import BasePipelineComponent # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error

ERR_MSG = "The model has not been fitted yet. " \
                 "First, call the `fit()` or `run()` method."



class BaseModel(BaseEstimator, RegressorMixin, BasePipelineComponent, ABC):
    """Defines the shared interface and fitted-state checks for regression models.

    Concrete subclasses implement fitting and prediction with their chosen estimator.
    The class also exposes a convenience `run()` method for fit-and-predict workflows.
    """
    def __init__(self, config_manager: ConfigManager) -> None:
        """Initializes the model component with its shared configuration.

        Delegates component setup to the pipeline base class.
        Concrete models can read model-specific settings from the same manager.

        Args:
            config_manager: Configuration manager used by the pipeline.

        Returns:
            None: The model is initialized but not fitted.
        """
        super().__init__(config_manager)
        self.model_ = None



    @property
    def name(self) -> str:
        """Returns the concrete model class name."""
        return self.__class__.__name__



    def _check_is_fitted(self) -> None:
        """Verifies that fitting created the estimator state required for prediction.

        Checks for the fitted `model_` attribute used by concrete implementations.
        Raises a runtime error when the model has not yet been fitted.

        Raises:
            RuntimeError: If no fitted estimator is available.
        """
        if not hasattr(self, "model_"):
            raise RuntimeError(ERR_MSG)



    @abstractmethod
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "BaseModel":
        """Fits the concrete regression estimator to the training data.

        Subclasses define the estimator and its fitting procedure.

        Args:
            X_train: Training feature matrix.
            y_train: Training target values.

        Returns:
            BaseModel: The fitted model instance.
        """


    @abstractmethod
    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """Predicts target values for the supplied feature matrix.

        Subclasses implement prediction using their fitted estimator.

        Args:
            X_test: Feature matrix to predict.

        Returns:
            np.ndarray: Predicted target values.
        """



    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        *args: Any,
        **kwargs: Any
    ) -> Optional[np.ndarray]:
        """Fits the model and optionally predicts values for a separate feature set.

        Fits on the supplied training features and targets first.
        Calls `predict()` only when a test feature matrix is provided.
        Returns `None` when the caller requests fitting without prediction.

        Args:
            X_train: Training feature matrix.
            y_train: Training target values.
            X_test: Optional feature matrix for prediction after fitting.

        Returns:
            Optional[np.ndarray]: Predictions, or `None` when `X_test` is omitted.
        """
        self.fit(X_train, y_train)

        if X_test is not None:
            return self.predict(X_test)

        return None
