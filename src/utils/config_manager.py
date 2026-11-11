from typing import Any, List, Union
import warnings
import yaml

from .base_config_manager import BaseConfigManager



class ConfigManager(BaseConfigManager):
    """A configuration manager that reads YAML files from the configuration directory.

    Each file is stored under its file name without extension, so the value
    ``a: {b: 1}`` loaded from ``settings.yaml`` is available as ``settings.a.b``.
    """
    def load_config(self, config_name: Union[str, List[str]]) -> None:
        """Loads one or more YAML files from the configuration directory.

        A missing file only raises a warning and is skipped. An empty file is
        stored as an empty dictionary.

        Args:
            config_name (str | List[str]): File name without extension, or a list of them.
        """
        names = [config_name] if isinstance(config_name, str) else config_name
        for name in names:
            filepath = self.config_dir / f"{name}.yaml"

            if not filepath.is_file():
                warnings.warn(f"No configuration file found at '{filepath}'. Skipped.")
                continue

            with open(filepath, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)

            self._config[name] = data if data is not None else {}


    def get_param(self, key: str, default: Any = None) -> Any:
        """Returns a nested value addressed by a dot-separated key.

        Args:
            key (str): Dot-separated path, e.g. ``"settings.paths.raw_data"``.
            default (Any): Value returned if any part of the path does not exist.

        Returns:
            Any: The stored value or `default`.
        """
        value: Any = self._config
        for part in key.split("."):
            try:
                value = value[part]
            except (KeyError, TypeError):
                return default
        return value