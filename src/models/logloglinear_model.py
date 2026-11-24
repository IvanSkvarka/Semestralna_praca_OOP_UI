from typing import Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from models.base_model import BaseModel # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error


class LogLogLinearModel(BaseModel):
    """Fits an ordinary least-squares linear regression to the supplied values.

    The class expects preprocessing to apply any desired logarithmic transforms.
    It delegates regression fitting and prediction to scikit-learn's `LinearRegression`.
    """
    def __init__(self, config_manager: ConfigManager) -> None:
        """Initializes the linear regression model with shared pipeline configuration.

        Stores the manager through the base class for a consistent pipeline interface.
        The scikit-learn estimator is instantiated when the model is fitted.

        Args:
            config_manager: Configuration manager used by the pipeline.

        Returns:
            None: The model is initialized but not fitted.
        """
        super().__init__(config_manager)



    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "LogLogLinearModel":
        """Fits a linear regression estimator to the training data.

        Creates a new least-squares estimator and trains it on the supplied arrays.
        Stores the fitted estimator for later prediction.

        Args:
            x_train: Training feature matrix.
            y_train: Training target values.

        Returns:
            LogLogLinearModel: This fitted model instance.
        """
        self._model = LinearRegression()
        self._model.fit(x_train, y_train)
        return self


    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """Predicts target values with the fitted linear regression estimator.

        Verifies that the model has been fitted before delegating prediction.
        Produces one numeric prediction for each row in the test matrix.

        Args:
            x_test: Feature matrix to predict.

        Returns:
            np.ndarray: Predicted target values.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        return self._model.predict(x_test)


    @property
    def coefficients(self) -> Dict[str, float]:
        """Returns the fitted regression coefficient of each feature.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        return {
            str(feature): float(coef)
            for feature, coef in zip(self._model.feature_names_in_, np.ravel(self._model.coef_)) # type: ignore
        }


    @property
    def intercept(self) -> float:
        """Returns the fitted intercept of the regression.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        return float(np.ravel(self._model.intercept_)[0]) # type: ignore
