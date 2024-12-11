import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, PrivateAttr, ConfigDict
import importlib
from typing import Literal

from embedding_models import LangchainEmbeddingModel

# TODO - update pydantic variables to fields with descriptions
class BaseVectorStore(BaseModel, ABC):
    module: str
    alias: str

    _logger: logging.Logger = PrivateAttr()
    _vector_store: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_logger()
        self._logger.info(f"Initializing embedding model: {self.alias}")

        try:
            self._pre_setup_steps()
            self._setup_vector_store()
            self._post_setup_steps()
            self._logger.info(f"Successfully initialized embedding model: {self.alias}")
        except Exception as e:
            self._logger.error(f"Failed to initialize Vector Store {self.alias}: {e}")
            raise

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

    @abstractmethod
    def _pre_setup_steps(self) -> None:
        pass

    @abstractmethod
    def _post_setup_steps(self) -> None:
        pass

    @abstractmethod
    def _setup_vector_store(self) -> None:
        pass

    @property
    def vector_store(self) -> Any:
        if self._vector_store is None:
            self._logger.error("Vector Store has not been initialized correctly")
            raise RuntimeError("Vector Store has not been initialized correctly")
        return self._vector_store

    def set_log_level(self, level: int) -> None:
        """
        Allow dynamic setting of log level.

        Args:
            level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.ERROR)
        """
        self._logger.setLevel(level)


class LangChainVectorStore(BaseVectorStore):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    flavor: Literal["langchain"]
    index_name: str
    endpoint_name: Optional[str] = None,
    embedding_model_ref: Optional[LangchainEmbeddingModel] = None,
    text_column: Optional[str] = None,
    columns: Optional[List[str]] = None

    _vector_store: VectorStore = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # TODO - Standardize package loading and move to common module
    def _setup_vector_store(self) -> None:
        try:
            package_name, module_name = self.module.split(".")
            package = importlib.import_module(package_name)
            module = getattr(package, module_name)

            # TODO - Check initialization from local based on api keys
            self._vector_store = module(
                index_name=self.index_name,
                endpoint=self.endpoint_name,
                embedding=self.embedding_model_ref.embedding_model,
                text_column=self.text_column,
                columns=self.columns
            )
        except (ImportError, AttributeError) as e:
            self._logger.error(f"Failed to initialize Vector Store: '{e}'")
            raise RuntimeError(f"Failed to initialize Vector Store: '{e}'")

    # TODO - Setup configs for non DB workspace deployment
    def _pre_setup_steps(self) -> None:
        self._logger.warning(f"No pre-setup steps defined for vector store '{self.alias}'")

    def _post_setup_steps(self) -> None:
        self._logger.warning(f"No post-setup steps defined for vector store '{self.alias}'")
