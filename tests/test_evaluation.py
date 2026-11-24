"""Test model tuning and evaluation behavior.

The tests cover tuner wrapping rules, invalid parameter grids, group-aware
cross-validation, predictions, comparison reports, and nested validation
outputs.
"""

import json
import sys
from pathlib import Path

from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# pylint: disable=wrong-import-position
from evaluation.evaluator import Evaluator, Metrics
from evaluation.tuner import HyperparameterTuner
from models.gradientboosting_model import GradientBoostingModel
from models.logloglinear_model import LogLogLinearModel
from models.meanbaseline_model import MeanBaselineModel
from configuration import RecordingModel, make_config, raises, prepare

# ----- Tuner -----
def test_tuner_wraps_only_models_with_a_grid():
    """Test model wrapping by the hyperparameter tuner.

    The test supplies one baseline model and one tunable model, then verifies
    that only the configured gradient-boosting model becomes a GridSearchCV
    wrapper while the other model remains unchanged.
    """
    cfg = make_config()
    models = HyperparameterTuner(cfg)([MeanBaselineModel(cfg), GradientBoostingModel(cfg)])
    assert isinstance(models[0], MeanBaselineModel)
    assert isinstance(models[1], GridSearchCV)


def test_tuner_rejects_invalid_grids():
    """Test rejection of invalid hyperparameter grid entries.

    The test first supplies an unknown estimator parameter and then an unknown
    model name, verifying that both configurations raise ValueError.
    """
    cfg = make_config()
    cfg._config["pipeline"]["tuning"]["param_grids"] = {  # pylint: disable=protected-access
        "GradientBoostingModel": {"not_a_parameter": [1]}
    }
    raises(
        ValueError, HyperparameterTuner(cfg), [GradientBoostingModel(cfg)],
        match="GradientBoostingModel are not parameters",
    )

    cfg._config["pipeline"]["tuning"]["param_grids"] = {  # pylint: disable=protected-access
        "UnknownModel": {"a": [1]}
    }
    raises(ValueError, HyperparameterTuner(cfg), [GradientBoostingModel(cfg)], match="UnknownModel")

# ----- Metrics -----
def test_metrics_instances_are_independent():
    """Test that each Metrics object keeps its own metric set.

    The test creates metrics for the log10 target and for the plain target and
    verifies that the second one does not change the first, which would happen
    if the instances were shared.
    """
    logged, plain = Metrics(log_target=True), Metrics(log_target=False)
    assert logged is not plain
    assert logged.main_metric == "r2_log10"
    assert plain.main_metric == "r2"
    assert "rmse_orig" in logged.funcs and "rmse_orig" not in plain.funcs


# ----- Evaluator -----
def test_evaluator_results_raise_before_run():
    """Test evaluator result properties before execution.

    The test accesses comparison, prediction, and true-target properties before
    evaluation and verifies that each access raises RuntimeError.
    """
    evaluator = Evaluator(make_config())
    for attribute in ("comparison_table", "predictions", "y_true"):
        raises(RuntimeError, getattr, evaluator, attribute)

# ----- Evaluator -----
def test_evaluator_never_splits_a_group_between_training_and_test_rows():
    """Test that evaluator folds preserve row and group separation.

    The recording model captures training and test indices during three folds.
    The test verifies index disjointness and group disjointness, and confirms
    that every input row is predicted exactly across the fold collection.
    """
    cfg = make_config()
    features, target, groups = prepare(cfg)
    RecordingModel.fitted_on, RecordingModel.predicted_on = [], []
    Evaluator(cfg)(RecordingModel(cfg), features, target, groups)

    assert len(RecordingModel.predicted_on) == 3
    for train_rows, test_rows in zip(RecordingModel.fitted_on, RecordingModel.predicted_on):
        assert not train_rows & test_rows
        assert not set(groups[list(train_rows)]) & set(groups[list(test_rows)])
    assert set.union(*RecordingModel.predicted_on) == set(features.index)


def test_evaluator_comparison_table_and_out_of_fold_predictions():
    """Test comparison results and out-of-fold prediction storage.

    The test compares baseline and linear models, checks ranking and metric
    columns, and verifies prediction and true-target result shapes and values.
    """
    cfg = make_config()
    features, target, groups = prepare(cfg)
    evaluator = Evaluator(cfg)
    table = evaluator([MeanBaselineModel(cfg), LogLogLinearModel(cfg)], features, target, groups)

    assert list(table.index) == ["LogLogLinearModel", "MeanBaselineModel"]
    for column in ("r2_log10", "rmse_orig", "r2_log10_mean", "r2_log10_std"):
        assert column in table.columns
    assert table.loc["LogLogLinearModel", "r2_log10"] > table.loc["MeanBaselineModel", "r2_log10"]
    assert evaluator.predictions.shape == (len(features), 2)
    assert not evaluator.predictions.isna().any().any()
    assert evaluator.y_true.equals(target)


def test_evaluator_rejects_duplicate_names_and_too_many_splits():
    """Test invalid evaluator model lists and split counts.

    The test rejects duplicate model names and a requested split count larger
    than the available grouped fold count, checking both ValueError messages.
    """
    cfg = make_config()
    features, target, groups = prepare(cfg)
    raises(ValueError, Evaluator(cfg), [MeanBaselineModel(cfg), MeanBaselineModel(cfg)],
           features, target, groups, match="unique")

    cfg = make_config(n_splits=10)
    raises(ValueError, Evaluator(cfg), MeanBaselineModel(cfg), features, target, groups, match="n_splits")


def test_nested_cross_validation_saves_reports():
    """Test nested cross-validation report generation.

    The test runs tuning and evaluation with report saving enabled, then verifies
    the summary JSON and best-parameter JSON files and their contents.
    """
    cfg = make_config(save_reports=True)
    features, target, groups = prepare(cfg)
    models = HyperparameterTuner(cfg)([MeanBaselineModel(cfg), GradientBoostingModel(cfg)])
    table = Evaluator(cfg)(models, features, target, groups)

    assert set(table.index) == {"MeanBaselineModel", "GradientBoostingModel"}
    report_dir = Path(cfg["settings.paths.output_dir"]) / "evaluation"
    assert (report_dir / "evaluation_summary.json").is_file()
    with open(report_dir / "GridSearchCV_best_params.json", encoding="utf-8") as file:
        best_params = json.load(file)
    assert list(best_params) == ["GradientBoostingModel"]
    assert len(best_params["GradientBoostingModel"]) == 3
    for params in best_params["GradientBoostingModel"]:
        assert set(params) == {"n_estimators", "max_depth"}