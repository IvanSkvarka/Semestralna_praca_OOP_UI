"""Test the implemented model classes.

The tests verify fitted-state checks, cloning, baseline and linear fitting,
gradient-boosting parameters and feature importances, and reproducibility.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# pylint: disable=wrong-import-position
from models.gradientboosting_model import GradientBoostingModel
from models.logloglinear_model import LogLogLinearModel
from models.meanbaseline_model import MeanBaselineModel
from configuration import make_config, prepare, raises

MODEL_CLASSES = (MeanBaselineModel, LogLogLinearModel, GradientBoostingModel)


def test_models_predict_before_fit_raises_runtime_error():
    """Test prediction before fitting for every model type.

    The test invokes prediction on prepared features for baseline, linear, and
    gradient-boosting models and verifies that all raise RuntimeError.
    """
    cfg = make_config()
    features, _, _ = prepare(cfg)
    for model_class in MODEL_CLASSES:
        raises(RuntimeError, model_class(cfg).predict, features)


def test_models_fit_returns_self_and_predict_returns_one_value_per_row():
    """Test model fitting and prediction result conventions.

    The test verifies that fit returns the model instance, prediction returns
    one value per feature row, and each model reports the expected class name.
    """
    cfg = make_config()
    features, target, _ = prepare(cfg)
    for model_class in MODEL_CLASSES:
        model = model_class(cfg)
        assert model.fit(features, target) is model
        assert model.predict(features).shape == (len(features),)
        assert model.name == model_class.__name__


def test_models_clone_is_unfitted():
    """Test that cloning a fitted model produces an unfitted model.

    The test fits each model, clones it with scikit-learn, and verifies that
    prediction on the clone raises RuntimeError.
    """
    cfg = make_config()
    features, target, _ = prepare(cfg)
    for model_class in MODEL_CLASSES:
        fitted = model_class(cfg).fit(features, target)
        raises(RuntimeError, clone(fitted).predict, features)


def test_mean_baseline_predicts_training_mean():
    """Test mean-baseline fitting and prediction behavior.

    The test verifies the constant property is unavailable before fitting, then
    checks that fitting stores the training target mean and predicts that value.
    """
    cfg = make_config()
    features, target, _ = prepare(cfg)
    model = MeanBaselineModel(cfg)
    raises(RuntimeError, lambda: model.constant)
    model.fit(features, target)
    assert np.isclose(model.constant, target.mean())
    assert np.allclose(model.predict(features), target.mean())


def test_linear_model_recovers_a_known_relation():
    """Test recovery of a known linear relationship.

    The test fits the log-log linear model to an exact synthetic relation and
    verifies coefficient sets, a near-zero unrelated coefficient, the intercept,
    and close agreement between predictions and the target.
    """
    cfg = make_config()
    features, _, _ = prepare(cfg)
    target = 2.0 * features["rpm"] - 1.0 * features["volume"] + 3.0
    model = LogLogLinearModel(cfg)
    raises(RuntimeError, lambda: model.coefficients)
    raises(RuntimeError, lambda: model.intercept)
    model.fit(features, target)
    assert set(model.coefficients) == set(features.columns)
    assert abs(model.coefficients["n_impeller"]) < 1e-8
    assert abs(model.intercept - 3.0) < 1e-6
    assert np.allclose(model.predict(features), target, atol=1e-8)


def test_gradient_boosting_hyperparameters_and_importances():
    """Test gradient-boosting parameters and feature importances.

    The test verifies constructor parameters, unavailable importances before
    fitting, sorted positive feature importance values, normalization, and
    coverage of all feature columns after fitting.
    """
    cfg = make_config()
    features, target, _ = prepare(cfg)
    model = GradientBoostingModel(cfg, n_estimators=7, min_samples_leaf=2, subsample=0.8)
    params = model.get_params(deep=False)
    assert (params["n_estimators"], params["min_samples_leaf"], params["subsample"]) == (7, 2, 0.8)
    raises(RuntimeError, lambda: model.feature_importances)

    model.fit(features, target)
    importances = model.feature_importances
    assert list(importances.values()) == sorted(importances.values(), reverse=True)
    assert np.isclose(sum(importances.values()), 1.0)
    assert set(importances) == set(features.columns)


def test_gradient_boosting_is_reproducible():
    """Test reproducibility of gradient-boosting predictions.

    The test fits two independently configured models with identical parameters
    and verifies that their prediction arrays are elementwise identical.
    """
    cfg = make_config()
    features, target, _ = prepare(cfg)
    first = GradientBoostingModel(cfg, n_estimators=10, subsample=0.7).fit(features, target)
    second = GradientBoostingModel(cfg, n_estimators=10, subsample=0.7).fit(features, target)
    assert np.array_equal(first.predict(features), second.predict(features))