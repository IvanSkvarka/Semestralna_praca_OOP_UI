from typing import Any, Dict, Union, List
from pathlib import Path
import warnings
import yaml
from yaml import CSafeLoader as SafeLoader

from .base_config_manager import BaseConfigManager



class ConfigManager(BaseConfigManager):
    """A concrete configuration manager that processes and serves structured YAML profile files.

    Implements the base management interface to load configuration assets from local disk paths.
    The class parses standard files using an optimized C-based safe loader for memory efficiency.
    Resolves multi-level nested parameters by navigating internal storage maps via dot notation.
    """
    def load_config(self, config_name: Union[str, List[str]]) -> None:
        """Loads and parses one or more target YAML configuration files into internal storage.

        Normalizes input parameters into a uniform iterable list sequence of string keys.
        The method loops through targets and triggers a warning instead of halting if missing.
        Valid paths are decoded via safe YAML parser and saved into the internal data registry.

        Args:
            config_name (str | List[str]): Single identifier or a list of config file names.
        """
        config_names = config_name if isinstance(config_name, list) else [config_name]
        for config_name in config_names:
            filepath = Path(self.config_dir / f"{config_name}.yaml")

            if not filepath.is_file():
                warnings.warn(f"No configuration file found at '{filepath}'. Passed.")
                continue

            with open(filepath, "r", encoding="utf-8") as file:
                data = yaml.load(file, Loader=SafeLoader)

            self._config[config_name] = data if data is not None else {}


    def get_param(self, key: str, default: Any = None) -> Any:
        """Retrieves a nested parameter value from the loaded state utilizing dot notation paths.

        Splits the incoming query string path by dot characters to locate deep structural keys.
        The mechanism recursively traverses internal storage blocks layer by layer until matching.
        If any step triggers a key error missing event, it falls back to return the default item.

        Args:
            key (str): The dot-separated sequence path pointing to the desired setup parameter.
            default (Any): The fallback item returned if the exact key path is not registered.

        Returns:
            Any: The located configuration variable value or the designated default fallback.
        """
        parts = key.split('.')
        current_config: Dict[str, Any] = self._config

        try:
            for part in parts: # Recursion for layer by layer matching
                current_config = current_config[part]
            return current_config

        except KeyError:
            return default
