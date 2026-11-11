from abc import ABC, abstractmethod
from typing import Any, Optional

from utils.config_manager import ConfigManager # pylint: disable=import-error



class BasePipelineComponent(ABC):
    """An abstract base class for all steps in the ML pipeline.

    Gives every step access to the shared configuration and a common interface:
    a step is executed by `run()` or by calling the instance like a function.
    """
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        """Stores the shared configuration manager and the component name.

        Args:
            config_manager (ConfigManager): Configuration manager instance.
            name (str | None): Component name. Defaults to the class name.
        """
        self._config_manager = config_manager
        self._name = name if name is not None else self.__class__.__name__



    @property
    def config_manager(self) -> ConfigManager:
        """Provides read-only access to the configuration manager."""
        return self._config_manager


    @property
    def name(self) -> str:
        """Provides read-only access to the component name."""
        return self._name



    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the step. Each subclass defines its own explicit arguments."""



    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Calls `run()` with the given arguments.

        Args:
            *args (Any): Positional arguments forwarded to `run()`.
            **kwargs (Any): Keyword arguments forwarded to `run()`.

        Returns:
            Any: The result of `run()`.
        """
        return self.run(*args, **kwargs)


    def __repr__(self) -> str:
        """Returns a text representation with the class, name and config directory."""
        return (
            f"<{self.__class__.__name__}(name={self._name}, "
            f"config_dir={self._config_manager.config_dir})>"
        )
