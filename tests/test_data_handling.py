"""Test data loading, analysis, and preprocessing behavior.

The tests use small synthetic CSV and dataframe inputs. They verify input
validation, duplicate and constant-column removal, transformations, grouping,
and protection against unintended dataframe mutation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# pylint: disable=wrong-import-position
from data_handling.data_loader import DataLoader
from data_handling.eda_analyzer import EDAnalyzer
from data_handling.preprocessor import Preprocessor
from configuration import make_config, make_raw_df, raises, prepare, FEATURES, TARGETS


# ----- DataLoader -----
def test_data_loader_loads_csv():
    """Test successful CSV loading and data exposure.

    The test writes the synthetic dataframe, verifies that data is unavailable
    before loading, and checks dimensions, configured columns, target data,
    and indexed feature access after loading.
    """
    cfg = make_config()
    raw = make_raw_df()
    raw.to_csv(cfg["settings.paths.raw_data"], index=False)
    loader = DataLoader(cfg)
    raises(RuntimeError, lambda: loader.data)
    df = loader()
    assert len(loader) == len(df) == len(raw)
    assert loader.feature_columns == FEATURES
    assert list(loader.target_data.columns) == TARGETS
    assert loader["rpm"].equals(raw["rpm"])


def test_data_loader_rejects_empty_and_incomplete_files():
    """Test CSV validation for empty and incomplete files.

    The test replaces the input CSV with empty or missing-feature data and
    verifies that DataLoader raises ValueError with a relevant message.
    """
    cfg = make_config()
    make_raw_df().iloc[:0].to_csv(cfg["settings.paths.raw_data"], index=False)
    raises(ValueError, DataLoader(cfg), match="empty")

    make_raw_df().drop(columns=["rpm"]).to_csv(cfg["settings.paths.raw_data"], index=False)
    raises(ValueError, DataLoader(cfg), match="missing")


def test_data_loader_missing_file_raises():
    """Test loading a CSV that is not present on disk.

    The test asserts that DataLoader raises FileNotFoundError when invoked
    for a configuration whose raw-data path has not been created.
    """
    raises(FileNotFoundError, DataLoader(make_config()))

# ----- EDA -----
def test_eda_analyzer_reports():
    """Test exploratory-data-analysis reports and fitted-state checks.

    The test verifies that report properties raise before analysis, then checks
    row, duplicate, missing, constant, statistical, and correlation results.
    """
    cfg = make_config()
    analyzer = EDAnalyzer(cfg)
    raises(RuntimeError, lambda: analyzer.quality_report)

    report, stats, correlations = analyzer(make_raw_df())
    assert report["num_rows"] == 49
    assert report["duplicate_rows"] == 1
    assert report["total_missing"] == 0
    assert report["constant_columns"] == ["constant_col"]
    assert "rpm" in stats.index
    assert correlations.shape[0] == correlations.shape[1]
    assert analyzer.quality_report is report

# ----- Preprocessor -----
def test_preprocessor_properties_raise_before_run():
    """Test preprocessor output properties before execution.

    The test accesses features, target, and groups before invoking the
    preprocessor and verifies that each access raises RuntimeError.
    """
    preprocessor = Preprocessor(make_config())
    for attribute in ("features", "target", "groups"):
        raises(RuntimeError, getattr, preprocessor, attribute)


def test_preprocessor_removes_duplicates_constants_and_other_targets():
    """Test duplicate, constant, and target-column removal.

    The test preprocesses the synthetic dataframe and verifies duplicate-row
    removal, unchanged input size assumptions, output dimensions, and removal
    of constant features and nonselected targets.
    """
    raw = make_raw_df()
    features, target, groups = Preprocessor(make_config())(raw)
    assert len(raw) == 49
    assert len(features) == len(target) == len(groups) == 48
    for column in ("constant_col", "epsilon_p50", "k_p50"):
        assert column not in features.columns


def test_preprocessor_applies_log10_only_to_configured_columns():
    """Test configured base-10 logarithmic transformations.

    The test verifies that configured feature and target columns are transformed
    while an unconfigured numeric feature remains unchanged.
    """
    features, target, _ = prepare(make_config())
    raw = make_raw_df().drop_duplicates().reset_index(drop=True)
    assert np.allclose(features["rpm"], np.log10(raw["rpm"]))
    assert np.allclose(features["volume"], np.log10(raw["volume"]))
    assert np.allclose(features["n_impeller"], raw["n_impeller"])
    assert np.allclose(target, np.log10(raw["epsilon_p50"]))


def test_preprocessor_groups_follow_the_group_columns():
    """Test generation of preprocessor group identifiers.

    The test verifies that the configured grouping fields create six groups and
    that every group contains eight processed observations.
    """
    _, _, groups = prepare(make_config())
    assert groups.nunique() == 6
    assert (groups.value_counts() == 8).all()


def test_preprocessor_rejects_non_positive_values_and_missing_columns():
    """Test invalid values and missing preprocessor columns.

    The test supplies a zero for a configured logarithmic feature and removes
    another required feature, then verifies ValueError results and messages.
    """
    raw = make_raw_df()
    broken = raw.copy()
    broken.loc[3, "rpm"] = 0.0
    raises(ValueError, Preprocessor(make_config()), broken, match="non-positive")
    raises(ValueError, Preprocessor(make_config()), raw.drop(columns=["rpm"]), match="missing")


def test_preprocessor_does_not_modify_its_input():
    """Test that preprocessing leaves the source dataframe unchanged.

    The test saves a complete copy of the input dataframe, runs preprocessing,
    and compares the original and copy with pandas dataframe assertions.
    """
    raw = make_raw_df()
    before = raw.copy()
    Preprocessor(make_config())(raw)
    pd.testing.assert_frame_equal(raw, before)