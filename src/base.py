from abc import ABC, abstractmethod
from typing import Any, Optional

from utils.config_manager import ConfigManager # pylint: disable=import-error



class BasePipelineComponent(ABC):
    """An abstract base class for all steps in the ML pipeline.

    Gives every step access to the shared configuration and a common interface:
    a step is executed by `run()` or by calling the instance like a function.
    """
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        """Initializes a pipeline component with shared configuration and a display name.

        Stores the configuration manager for use by the component and its subclasses.
        Uses the class name when no explicit component name is supplied.
        The resulting name is also used to identify related pipeline helpers.

        Args:
            config_manager: Configuration manager shared with this component.
            name: Optional component name; defaults to the class name.
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
        """Executes the pipeline step implemented by a concrete component.

        Subclasses provide the operation and define its required arguments.
        The result depends on the work performed by the component.

        Args:
            *args: Positional arguments accepted by the concrete implementation.
            **kwargs: Keyword arguments accepted by the concrete implementation.

        Returns:
            Any: The result produced by the concrete pipeline step.
        """



    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the component by forwarding arguments to its `run()` method.

        This makes pipeline components callable as a convenient shorthand.
        Positional and keyword arguments are passed through unchanged.
        The return value is the one produced by `run()`.

        Args:
            *args: Positional arguments forwarded to `run()`.
            **kwargs: Keyword arguments forwarded to `run()`.

        Returns:
            Any: The result of `run()`.
        """
        return self.run(*args, **kwargs)


    def __repr__(self) -> str:
        """Builds a diagnostic string containing the component identity and config path.

        Uses the concrete class name and configured component name.
        Reads the configuration directory from the shared configuration manager.

        Returns:
            str: A concise representation of this pipeline component.
        """
        return (
            f"<{self.__class__.__name__}(name={self._name}, "
            f"config_dir={self._config_manager.config_dir})>"
        )
