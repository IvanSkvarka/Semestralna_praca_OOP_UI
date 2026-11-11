from typing import Any, Dict, List, Union
from pathlib import Path
from abc import ABC, abstractmethod



class BaseConfigManager(ABC):
    """An abstract base class for configuration managers.

    Stores the configuration directory and the loaded configuration values.
    Subclasses implement loading and parameter lookup, while this class adds
    dictionary-style access: ``manager["a.b"]`` and ``"a.b" in manager``.
    """
    def __init__(self, config_dir: Union[str, Path]) -> None:
        """Initializes the manager and checks that the configuration directory exists.

        Args:
            config_dir (str | Path): Path to the configuration directory.

        Raises:
            FileNotFoundError: If the path does not exist or is not a directory.
        """
        self.config_dir = Path(config_dir)
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"Configuration directory '{self.config_dir}' not found.")

        self._config: Dict[str, Any] = {}



    @abstractmethod
    def load_config(self, config_name: Union[str, List[str]]) -> None:
        """Loads one or more configuration files into the manager."""

    @abstractmethod
    def get_param(self, key: str, default: Any = None) -> Any:
        """Returns the value stored under `key`, or `default` if the key does not exist."""



    def __getitem__(self, key: str) -> Any:
        """Returns the value stored under `key`, or None if the key does not exist.

        Args:
            key (str): Parameter key, in dot notation for nested values.

        Returns:
            Any: The stored value or None.
        """
        return self.get_param(key)

    def __contains__(self, key: str) -> bool:
        """Checks whether a parameter exists, even if its stored value is None.

        Args:
            key (str): Parameter key, in dot notation for nested values.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return self.get_param(key, object()) is not object()

    def __repr__(self) -> str:
        """Returns a text representation with the config directory and loaded files."""
        return (
            f"<{self.__class__.__name__}(config_dir='{self.config_dir}', "
            f"loaded={list(self._config)})>"
        )