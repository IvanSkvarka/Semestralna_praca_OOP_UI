from typing import Any, Dict, List, Union
from pathlib import Path
from abc import ABC, abstractmethod

_MISSING = object()



class BaseConfigManager(ABC):
    """An abstract base class for configuration managers.

    Stores the configuration directory and the loaded configuration values.
    Subclasses implement loading and parameter lookup, while this class adds
    dictionary-style access: ``manager["a.b"]`` and ``"a.b" in manager``.
    """
    def __init__(self, config_dir: Union[str, Path]) -> None:
        """Initializes the manager and validates its configuration directory.

        Converts the supplied path to a `Path` instance for later file operations.
        Rejects missing paths and non-directory paths before creating the empty config store.
        Stores loaded configuration values in the manager's internal mapping.

        Args:
            config_dir: Path to the configuration directory.

        Raises:
            FileNotFoundError: If the path does not exist or is not a directory.
        """
        self._config_dir = Path(config_dir)
        if not self._config_dir.is_dir():
            raise FileNotFoundError(f"Configuration directory '{self._config_dir}' not found.")

        self._config: Dict[str, Any] = {}



    @property
    def config_dir(self) -> Path:
        """Returns the configuration directory."""
        return self._config_dir



    @abstractmethod
    def load_config(self, config_name: Union[str, List[str]]) -> None:
        """Loads the named configuration file or files into the manager.

        Concrete managers determine the file format and loading behavior.
        Loaded values are stored for later parameter lookup.

        Args:
            config_name: A configuration name or a list of configuration names.

        Returns:
            None: This method updates the manager in place.
        """


    @abstractmethod
    def get_param(self, key: str, default: Any = None) -> Any:
        """Looks up a configuration value, using a fallback when the key is absent.

        Concrete managers define how the key is interpreted.
        The default value is returned when the requested parameter cannot be found.

        Args:
            key: Parameter key to look up.
            default: Value to return when the key does not exist.

        Returns:
            Any: The stored value or the supplied default.
        """



    def __getitem__(self, key: str) -> Any:
        """Provides dictionary-style access to a configuration parameter.

        Delegates lookup to `get_param()` using its default behavior.
        Nested parameter paths may be supported by concrete manager implementations.
        Missing parameters therefore resolve to the manager's default value.

        Args:
            key: Parameter key, in dot notation for nested values.

        Returns:
            Any: The stored value or None.
        """
        return self.get_param(key)


    def __contains__(self, key: str) -> bool:
        """Checks whether a configuration parameter is present.

        Uses a sentinel default to distinguish a missing path from an explicit value.
        The result is intended to be true even when the stored value is `None`.

        Args:
            key: Parameter key, in dot notation for nested values.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return self.get_param(key, _MISSING) is not _MISSING


    def __repr__(self) -> str:
        """Builds a diagnostic representation of the manager state.

        Includes the configuration directory and names of loaded top-level entries.
        The representation is intended to help identify the manager during debugging.

        Returns:
            str: A concise representation of this configuration manager.
        """
        return (
            f"<{self.__class__.__name__}(config_dir='{self.config_dir}', "
            f"loaded={list(self._config)})>"
        )