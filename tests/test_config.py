"""Test configuration loading and access behavior.

The tests exercise nested parameter lookup, membership checks, read-only
configuration-directory access, and missing configuration resources.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# pylint: disable=wrong-import-position
from utils.config_manager import ConfigManager
from configuration import make_config, raises, ROOT

def test_config_get_param_reads_nested_values_and_defaults():
    """Test nested configuration parameter lookup and defaults.

    The test loads a temporary configuration and verifies direct nested access,
    bracket-style access, missing keys, and missing intermediate path segments.
    """
    cfg = make_config()
    assert cfg.get_param("pipeline.random_state") == 42
    assert cfg["pipeline.evaluation.n_splits"] == 3
    assert cfg.get_param("pipeline.missing.key", "fallback") == "fallback"
    assert cfg.get_param("pipeline.random_state.deeper", "fallback") == "fallback"


def test_config_contains_distinguishes_present_and_missing_keys():
    """Test configuration membership for present and missing paths.

    The test checks that section and leaf paths are recognized, while unknown
    sections and paths below existing leaves are reported as absent.
    """
    cfg = make_config()
    assert "pipeline.random_state" in cfg
    assert "pipeline.evaluation" in cfg
    assert "pipeline.missing_key" not in cfg
    assert "missing_section.key" not in cfg
    assert "pipeline.random_state.deeper" not in cfg


def test_config_contains_is_true_for_a_stored_none():
    """Test membership when a stored configuration value is None.

    The test adds an explicit null value and verifies that membership checks
    distinguish a stored key from a missing configuration path.
    """
    cfg = make_config()
    cfg._config["pipeline"]["nothing"] = None  # pylint: disable=protected-access
    assert "pipeline.nothing" in cfg


def test_config_dir_is_read_only():
    """Test that the configuration directory path is read-only.

    The test verifies that config_dir is a Path and that assignment raises
    AttributeError because the property is implemented without a setter.
    """
    cfg = make_config()
    assert isinstance(cfg.config_dir, Path)
    raises(AttributeError, setattr, cfg, "config_dir", Path("."))


def test_config_missing_directory_raises():
    """Test construction with a missing configuration directory.

    The test asserts that ConfigManager rejects a directory that does not exist
    and raises the standard FileNotFoundError exception.
    """
    raises(FileNotFoundError, ConfigManager, ROOT / "does_not_exist")


def test_config_missing_file_gives_a_warning():
    """Test warnings raised for a missing configuration file.

    The test loads a nonexistent named configuration under warning capture and
    verifies that at least one warning is emitted to the caller.
    """
    cfg = make_config()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg.load_config("does_not_exist")
    assert caught, "no warning was issued"