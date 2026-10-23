from abc import ABC, abstractmethod
from typing import Any, Union
import logging
from utils.config_manager import ConfigManager # pylint: disable=import-error
from utils.logger import get_logger # pylint: disable=import-error

class BasePipelineComponent(ABC):
    """An abstract base class for all steps in the ML pipeline.
    It provides a unified interface, access to configuration,
    and the ability to call the object as a function.
    """
    def __init__(self, config_manager: ConfigManager, name: Union[str, None] = None) -> None:
        """Initializes the base pipeline component.

        Args:
            config_manager (ConfigManager): Configuration manager instance. 
            name (str | None): Module name.

        Raises:
            TypeError: Config_manager is not an instance of ConfigManager class.
        """
        # if not isinstance(config_manager, ConfigManager):
        #     raise TypeError("Parameter 'config_manager' must be ConfigManager instance.")

        self._config_manager = config_manager
        self._name = name if name is not None else self.__class__.__name__
        self._logger = get_logger(self._name, self._config_manager)
        self._logger.info("Initialized component '%s'", self._name)

    @property
    def config_manager(self) -> ConfigManager:
        """Returns a reference to ConfigManager (read-only)."""
        return self._config_manager

    @property
    def name(self) -> str:
        """Returns the component name (read-only)."""
        return self._name

    @property
    def logger(self) -> logging.Logger:
        """Returns a reference to ConfigManager (read-only)."""
        return self._logger

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Abstract method"""

    def __call__(self, *args: Any, **kwargs: Any):
        """Delegates the call to the self.run() method. 
        Automatically logs the progress or any errors that may occur."""
        return self.run(*args, **kwargs)

    def __repr__(self):
        """Returns a human-readable text representation of the component for debugging."""
        return f"<{self.__class__.__name__}(name={self._name}, {self._config_manager.config_dir})>"