import yaml
from pathlib import Path 
from typing import Any, Union, Dict
from config.base import BaseConfigManager

class ConfigManager(BaseConfigManager):
    """Loads YAML configuration files and provides access to their values.

    Configuration files are expected to be named ``<config_name>.yaml`` and
    stored in the configured directory.
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
        # self.project_root = self.config_dir.parent

    def load_config(self, config_name: str = "config") -> None:
        """Loads a YAML configuration file into memory.

        Args:
            config_name (str): Name of the configuration file, without the ``.yaml`` extension.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        filepath = Path(self.config_dir / f"{config_name}.yaml")

        if not filepath.is_file():
            raise FileNotFoundError(f"No configuration file found at '{filepath}'.")
        
        with open(filepath, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        for name in data.keys():
            self._config[name] = data.get(name) if data is not None else {}

        return None

    def get_param(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value using a dot-separated key.

        Args:
            key (str): Dot-separated path to the value, such as ``database.host``.
            default (Any): Value to return if the key is not found. If omitted, a missing 
                key raises ``KeyError``.

        Returns:
            The value at the specified key.

        Raises:
            KeyError: If the key is not found and no default is provided.
        """
        parts = key.split('.')
        current_config = self._config

        try:
            for part in parts:
                if not isinstance(current_config, dict):
                    raise KeyError(f"Path '{key}' doesn't exist, because part before '{part}' is not dictionary.")
                current_config = current_config.get(part)

            if current_config is None:
                if default is not None:
                    return default
                raise KeyError(f"Key '{key}' not found in configuration.")
            return current_config
        
        except (KeyError, TypeError, AttributeError) as e:
            raise KeyError(f"Configuration parameter '{key}' not found and no default provided.") from e