"""Provide shared fixtures and helpers for the test suite.

This module creates synthetic tabular data and temporary configuration
files. It also supplies a preprocessing fixture, an exception assertion
helper, and a recording model used to verify cross-validation behavior.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# pylint: disable=wrong-import-position
from data_handling.preprocessor import Preprocessor
from models.base_model import BaseModel
from utils.config_manager import ConfigManager

TMP = tempfile.TemporaryDirectory()
ROOT = Path(TMP.name)
FEATURES = ["volume", "reactor_diameter", "liquid_height", "n_impeller", "rpm", "constant_col"]
TARGETS = ["epsilon_p50", "k_p50"]


def make_raw_df() -> pd.DataFrame:
    """Create a synthetic dataframe for tests.

    The function generates 48 rows across six combinations of reactor
    diameter and impeller count, with eight reactor-speed observations per
    combination. It then appends the first row as a duplicate and adds a
    constant feature, one target, and one randomized target.

    Returns:
        pd.DataFrame: A synthetic dataframe containing features and targets.
    """
    rng = np.random.default_rng(0)
    rows = []
    for diameter in (1.0, 2.0, 3.0):
        for n_impeller in (1, 2):
            for rpm in np.linspace(50.0, 400.0, 8):
                rows.append({
                    "volume": diameter ** 3,
                    "reactor_diameter": diameter,
                    "liquid_height": diameter * 1.5,
                    "n_impeller": n_impeller,
                    "rpm": rpm,
                    "constant_col": 7.0,
                    "epsilon_p50": 1e-6 * rpm ** 2 * n_impeller / diameter * rng.uniform(0.9, 1.1),
                    "k_p50": rng.uniform(0.1, 1.0),
                })
    df = pd.DataFrame(rows)
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


def make_config(**evaluation: object) -> ConfigManager:
    """Create and load a temporary test configuration.

    The function constructs a fresh configuration directory under the shared
    temporary root. It writes all required YAML sections, applies optional
    evaluation overrides, and initializes the ConfigManager with the data.

    Args:
        **evaluation (object): Optional evaluation settings to merge into the
            default evaluation configuration.

    Returns:
        ConfigManager: A manager populated with the generated test configuration.
    """
    config_dir = Path(tempfile.mkdtemp(dir=ROOT))
    configs = {
        "settings": {"paths": {
            "raw_data": str(config_dir / "data.csv"),
            "output_dir": str(config_dir / "report"),
            "figures_dir": str(config_dir / "report" / "figures"),
        }},
        "data": {"feature_columns": FEATURES, "target_columns": TARGETS},
        "pipeline": {
            "random_state": 42,
            "eda": {"save_reports": False, "generate_plots": False},
            "preprocessing": {
                "target": "epsilon_p50",
                "group_columns": ["reactor_diameter", "liquid_height", "n_impeller"],
                "log_target": True,
                "log_features": ["volume", "rpm"],
                "drop_duplicates": True,
                "drop_constant_columns": True,
                "save_reports": False,
                "generate_plots": False,
            },
            "evaluation": {"n_splits": 3, "save_reports": False, "generate_plots": False, **evaluation},
            "tuning": {
                "n_splits": 2,
                "n_jobs": 1,
                "param_grids": {"GradientBoostingModel": {"n_estimators": [5, 10], "max_depth": [1, 2]}},
            },
        },
        "visualization": {},
        "dictionary": {},
    }
    for name, content in configs.items():
        with open(config_dir / f"{name}.yaml", "w", encoding="utf-8") as file:
            yaml.safe_dump(content, file)
    manager = ConfigManager(config_dir)
    manager.load_config(list(configs))
    return manager


def prepare(cfg: ConfigManager):
    """Prepare synthetic data through the configured preprocessor.

    The function passes the synthetic raw dataframe to Preprocessor. It then
    returns the processed feature matrix, target series, and group identifiers
    needed by model and evaluation tests.

    Args:
        cfg (ConfigManager): Configuration containing preprocessing settings.

    Returns:
        tuple: A tuple containing features, target, and group identifiers.
    """
    return Preprocessor(cfg)(make_raw_df())


def raises(exception: type, func, *args, match: str = "", **kwargs) -> None:
    """Assert that a callable raises a matching exception.

    The function invokes the supplied callable with its positional and keyword
    arguments. It succeeds only when the requested exception type is raised
    and, when supplied, the exception message contains the expected text.

    Args:
        exception (type): Exception type that the callable must raise.
        func (Callable): Callable to invoke.
        *args: Positional arguments forwarded to the callable.
        match (str): Text that must occur in the exception message.
        **kwargs: Keyword arguments forwarded to the callable.

    Raises:
        AssertionError: If the callable raises no exception or an unmatched
            exception message.
    """
    try:
        func(*args, **kwargs)
    except exception as error:
        assert match in str(error), f"message '{error}' does not contain '{match}'"
        return
    raise AssertionError(f"{exception.__name__} was not raised")


class RecordingModel(BaseModel):
    """Record dataframe indices used for fitting and prediction.

    This test model stores row-index sets in class-level lists. It uses the
    mean training target as its constant prediction, which makes it useful
    for verifying that evaluator folds do not mix training and test rows.
    """
    fitted_on: list = []
    predicted_on: list = []

    def fit(self, x_train, y_train):
        """Fit the model and record training indices.

        Args:
            x_train (pd.DataFrame): Training feature dataframe.
            y_train (pd.Series): Training target values.

        Returns:
            RecordingModel: The fitted model instance.
        """
        type(self).fitted_on.append(set(x_train.index))
        self._model = y_train.mean()
        return self

    def predict(self, x_test):
        """Generate constant predictions and record prediction indices.

        Args:
            x_test (pd.DataFrame): Feature dataframe to predict.

        Returns:
            np.ndarray: One constant prediction for every input row.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        type(self).predicted_on.append(set(x_test.index))
        return np.full(len(x_test), self._model)
