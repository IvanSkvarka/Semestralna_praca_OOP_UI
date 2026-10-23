from typing import Any, Dict, Union
from pathlib import Path
from abc import ABC, abstractmethod

class BaseConfigManager(ABC):
    """Define the interface for configuration managers.

    Concrete subclasses provide configuration loading and parameter lookup.
    """
    def __init__(self, config_dir: Union[str, Path]) -> None:
        """Initializes the configuration manager.

        Args:
            config_dir (str | Path): Directory containing YAML configuration files.

        Raises:
            FileNotFoundError: If the configuration directory does not exist.
        """
        self.config_dir = Path(config_dir)

        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"Configuration directory '{self.config_dir}' not found.")

        self._config: Dict[str, Any] = {}
        
    @abstractmethod
    def load_config(self, config_name: str) -> None:
        """Abstract method"""

    @abstractmethod
    def get_param(self, key: str, default: Any = None) -> Any:
        """Abstract method"""

    def __getitem__(self, key: str) -> Any:
        """Return a configuration parameter using dictionary-style access.

        Args:
            key (str): Name of the configuration parameter to retrieve.

        Returns:
            The value associated with ``key``.

        Raises:
            KeyError: If the parameter cannot be found.
        """
        return self.get_param(key, None)

    def __contains__(self, key: str) -> bool:
        """Check whether a configuration parameter can be retrieved.

        Args:
            key (str): Name of the configuration parameter to check.

        Returns:
            ``True`` if the parameter exists, otherwise ``False``.
        """
        try:
            self.get_param(key)
            return True
        except KeyError:
            return False

    def __repr__(self):
        """Return a string representation of the manager and loaded configs.

        Returns:
            A string describing the configuration directory and loaded
            configuration names.
        """
        loaded = list(self._config.keys())
        return f"<ConfigManager(config_dir='{self.config_dir}', loaded={loaded})>"