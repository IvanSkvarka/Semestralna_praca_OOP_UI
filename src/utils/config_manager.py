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
        """Loads named YAML configuration files from the configured directory.

        Resolves each name to a `.yaml` file and parses existing files with `yaml.safe_load`.
        Missing files produce a warning and are skipped, while empty files become dictionaries.
        Parsed mappings are stored under the corresponding file name.

        Args:
            config_name: File name without extension, or a list of file names.

        Returns:
            None: Loaded values are stored on this manager.
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
        """Retrieves a nested configuration value using a dotted key.

        Starts from the loaded configuration mapping and resolves each key segment in order.
        If a segment is missing or the current value cannot be indexed, lookup stops.
        In that case the supplied default is returned.

        Args:
            key: Dot-separated path, e.g. ``"settings.paths.raw_data"``.
            default: Value returned if any part of the path does not exist.

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