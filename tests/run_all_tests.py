"""Simple tests of the pipeline components using plain `assert` statements.

Run from any directory with `python tests/run_all_tests.py`. Every function whose name
starts with `test_` is executed on synthetic data, so the real dataset is not needed.
The script prints the result of each test and exits with a non-zero code if any failed.
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# pylint: disable=wrong-import-position
from configuration import TMP
import importlib

moduls = [ # Pozadovane moduly
    "test_config",
    "test_data_handling",
    "test_models",
    "test_evaluation",
]

for modul_name in moduls: # Pre kazdy modul
    modul = importlib.import_module(modul_name)
    globals().update({ # update globalnych
        name: getattr(modul, name)
        for name in dir(modul)
        if name.startswith("test_")
    })

def main() -> int:
    """Run every test function and report the outcome.

    The function discovers callable globals whose names start with test_. It
    invokes each test independently, prints pass or failure information, and
    captures unexpected exceptions as test failures.

    Returns:
        int: The number of tests that failed.
    """
    tests = [(name, func) for name, func in globals().items() if name.startswith("test_") and callable(func)]
    failed = 0
    for name, func in tests:
        try:
            func()
        except Exception:  # pylint: disable=broad-exception-caught
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            print(f"OK    {name}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    TMP.cleanup()
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)