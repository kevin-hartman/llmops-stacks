import logging
from abc import ABC
from typing import Any
from pydantic import BaseModel, PrivateAttr

from rag_pipeline_setup.rag_embedding_model.embedding_model_plugins import EmbeddingModelPlugins

class AbstractBaseEmbeddingModel(EmbeddingModelPlugins, BaseModel, ABC):
    """The model name reference."""
    model_name: str
    """Internal name used for referencing."""
    alias: str

    _logger: logging.Logger = PrivateAttr()
    _embedding_model: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_logger()
        self._logger.info(f"Initializing embedding model: {self.alias}")

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
    def embedding_model(self) -> Any:
        if self._embedding_model is None:
            self._logger.error("Embedding model has not been initialized correctly")
            raise RuntimeError("Embedding model has not been initialized correctly")
        return self._embedding_model


    def set_log_level(self, level: int) -> None:
        """
        Allow dynamic setting of log level.

        Args:
            level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.ERROR)
        """
        self._logger.setLevel(level)
