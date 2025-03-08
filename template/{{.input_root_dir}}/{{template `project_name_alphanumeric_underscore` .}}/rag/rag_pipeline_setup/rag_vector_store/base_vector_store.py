import logging
from abc import ABC
from typing import Any
from pydantic import BaseModel, PrivateAttr

from rag_pipeline_setup.rag_vector_store.vector_store_plugins import VectorStorePlugins


class AbstractBaseVectorStore(VectorStorePlugins, BaseModel, ABC):
    """Internal name used for referencing."""
    alias: str

    _logger: logging.Logger = PrivateAttr()
    _vector_store: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_logger()
        self._logger.info(f"Initializing Vector Store: {self.alias}")

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
    def vector_store(self) -> Any:
        if self._vector_store is None:
            self._logger.error(f"Vector Store: {self.alias} has not been initialized correctly")
            raise RuntimeError(f"Vector Store: {self.alias} has not been initialized correctly")
        return self._vector_store

    def set_log_level(self, level: int) -> None:
        """
        Allow dynamic setting of log level.

        Args:
            level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.ERROR)
        """
        self._logger.setLevel(level)
