import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from models.base_model import BaseModel # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error


class MeanBaselineModel(BaseModel):
    """Regression baseline that predicts the mean target learned from training data.

    Wraps scikit-learn's `DummyRegressor` with its default mean strategy.
    The inherited fitted-state check prevents prediction before training.
    """
    def __init__(self, config_manager: ConfigManager) -> None:
        """Initializes the mean baseline with the shared pipeline configuration.

        The configuration is retained by the base model for interface consistency.
        The underlying estimator is created when `fit()` is called.

        Args:
            config_manager: Configuration manager used by the pipeline.

        Returns:
            None: The model is initialized but not fitted.
        """
        super().__init__(config_manager)



    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "MeanBaselineModel":
        """Fits a dummy regressor that predicts the training-target mean.

        Creates a fresh scikit-learn estimator for each fit operation.
        Stores the fitted estimator on the model and returns this model.

        Args:
            x_train: Training feature matrix.
            y_train: Training target values.

        Returns:
            MeanBaselineModel: This fitted model instance.
        """
        self._model = DummyRegressor()
        self._model.fit(x_train, y_train)
        return self


    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """Predicts the training-target mean for each test row.

        Checks that fitting has completed before delegating prediction.
        The output contains one prediction for every row in the test matrix.

        Args:
            x_test: Feature matrix to predict.

        Returns:
            np.ndarray: Mean target predictions.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        return self._model.predict(x_test) # type: ignore


    @property
    def constant(self) -> float:
        """Returns the training-target mean predicted for every row.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        return float(np.ravel(self._model.constant_)[0]) # type: ignore
