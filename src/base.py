from abc import ABC, abstractmethod
from typing import Any, Union
import logging

from utils.config_manager import ConfigManager # pylint: disable=import-error
from utils.logger import get_logger # pylint: disable=import-error



class BasePipelineComponent(ABC):
    """An abstract base class for all steps in the ML pipeline.
    Provides a unified interface, access to configuration,
    and the ability to call the object as a function.
    """
    def __init__(self, config_manager: ConfigManager, name: Union[str, None] = None) -> None:
        """Initializes the base pipeline component with configuration and logging.

        Assigns the provided config manager and determines the component's name.
        The method instantiates a dedicated logger and logs an initialization message.

        Args:
            config_manager (ConfigManager): Configuration manager instance.
            name (str | None): Module name.
        """
        self._config_manager = config_manager
        self._name = name if name is not None else self.__class__.__name__
        self._logger = get_logger(self._name, self._config_manager)
        self._logger.info("Initialized component '%s'", self._name)



    @property
    def config_manager(self) -> ConfigManager:
        """Provides read-only access to the configuration manager."""
        return self._config_manager


    @property
    def name(self) -> str:
        """Provides read-only access to the component name."""
        return self._name


    @property
    def logger(self) -> logging.Logger:
        """Provides read-only access to the component logger."""
        return self._logger



    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the core data processing logic of the pipeline component."""



    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invokes the component using function call syntax.

        Intercepts the call signature and directly forwards all arguments to the run method.
        This enables treating pipeline steps as callable functions within orchestration workflows.

        Args:
            *args (Any): Variable length argument list forwarded to run.
            **kwargs (Any): Arbitrary keyword arguments forwarded to run.

        Returns:
            Any: The resulting execution output returned by the internal run method.
        """
        return self.run(*args, **kwargs)


    def __repr__(self) -> str:
        """Generates a detailed text representation of the component.

        Returns:
            str: A formatted string representing the component instance state.
        """
        return f"<{self.__class__.__name__}(name={self._name}, {self._config_manager.config_dir})>"
