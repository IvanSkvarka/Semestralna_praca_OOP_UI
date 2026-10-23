from typing import Any, Dict, Union, List
from pathlib import Path
import yaml
from .base import BaseConfigManager

class ConfigManager(BaseConfigManager):
    """Loads YAML configuration files and provides access to their values.

    Configuration files are expected to be named ``<config_name>.yaml`` and
    stored in the configured directory.
    """

    def load_config(self, config_name: Union[str, List[str]]) -> None:
        """Loads a YAML configuration file into memory.

        Args:
            config_name (str | list[str]): Name of the configuration file, without the ``.yaml`` extension.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        config_names = config_name if isinstance(config_name, list) else list(config_name)
        for config_name in config_names:
            filepath = Path(self.config_dir / f"{config_name}.yaml")

            if not filepath.is_file():
                raise FileNotFoundError(f"No configuration file found at '{filepath}'.")

            with open(filepath, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)

            self._config[config_name] = data if data is not None else {}

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
        current_config: Dict[str, Any] = self._config

        try:
            for part in parts:
                current_config = current_config[part]
            return current_config

        except KeyError:
            return default
