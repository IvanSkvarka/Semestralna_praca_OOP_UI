from typing import Any, Dict, Union
from pathlib import Path
from abc import ABC, abstractmethod



class BaseConfigManager(ABC):
    """An abstract base class for managing system setup profiles and parameter retrieval.

    Establishes a centralized configuration directory context used for resolving settings.
    The class defines standardized lookup interfaces and dictionary access routing protocols.
    Provides a framework that underlying implementations inherit to safely pull key pairs.
    """
    def __init__(self, config_dir: Union[str, Path]) -> None:
        """Initializes the base configuration manager and validates the reference path.

        Converts the input directory path to a standard object and checks its presence on disk.
        An empty internal dictionary placeholder is initialized to record future settings maps.

        Args:
            config_dir (str | Path): The system path pointing to the configuration directory.

        Raises:
            FileNotFoundError: If the provided directory path does not exist or is not a directory.
        """
        self.config_dir = Path(config_dir)
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"Configuration directory '{self.config_dir}' not found.")
        
        self._config: Dict[str, Any] = {}



    @abstractmethod
    def load_config(self, config_name: str) -> None:
        """Loads a specific configuration target source file into the manager storage memory."""

    @abstractmethod
    def get_param(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value using a specified key or falls back to a default."""



    def __getitem__(self, key: str) -> Any:
        """Retrieves a configuration parameter value using index key bracket notation.

        Args:
            key (str): The structural parameter notation key string queried from outside.

        Returns:
            Any: The found configuration entry data or None if the parameter does not exist.
        """
        return self.get_param(key, None)

    def __contains__(self, key: str) -> bool:
        """Checks if a specific configuration parameter key exists inside the current state.

        Args:
            key (str): The parameter target indicator path evaluated for actual existence.

        Returns:
            bool: True if the key is registered in configurations, False otherwise.
        """
        try:
            self.get_param(key)
            return True
        except KeyError:
            return False

    def __repr__(self):
        """Generates an informative text string layout representing the instance state.

        Returns:
            str: A formatted debug string containing instance attributes and memory keys.
        """
        loaded = list(self._config.keys())
        return f"<ConfigManager(config_dir='{self.config_dir}', loaded={loaded})>"
