import logging
from abc import ABC
from typing import Any
from pydantic import BaseModel, PrivateAttr

from rag_pipeline_setup.rag_language_model.language_model_plugins import LanguageModelPlugins

class AbstractBaseLLM(LanguageModelPlugins, BaseModel, ABC):
    """The model name reference."""
    model_name: str
    """Internal name used for referencing."""
    alias: str

    _logger: logging.Logger = PrivateAttr()
    _llm_model: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_logger()
        self._logger.info(f"Initializing language model: {self.alias}")

    def _setup_logger(self) -> None:
        """
        Create a class-specific logger with a unique name.
        """
        # Use the class name and a unique identifier (alias) to create a distinct logger
        logger_name = f"{self.__class__.__name__}.{self.alias}"
        self._logger = logging.getLogger(logger_name)

        # Only add handler if no handlers exist to prevent duplicate logging
        if not self._logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self._logger.addHandler(console_handler)

            # Set default logging level to INFO
            self._logger.setLevel(logging.INFO)


    @property
    def llm_model(self) -> Any:
        if self._llm_model is None:
            self._logger.error("Language model has not been initialized correctly")
            raise RuntimeError("Language model has not been initialized correctly")
        return self._llm_model

    def set_log_level(self, level: int) -> None:
        """
        Allow dynamic setting of log level.

        Args:
            level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.ERROR)
        """
        self._logger.setLevel(level)