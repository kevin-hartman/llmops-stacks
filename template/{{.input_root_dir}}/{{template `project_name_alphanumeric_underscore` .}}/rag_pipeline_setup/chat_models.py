import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, PrivateAttr
from enum import Enum
from databricks_langchain import ChatDatabricks

from flavor_enums import LanguageModelFlavor
from plugin_registries import LLMPlugins


# TODO - update pydantic variables to fields with descriptions
class BaseLLM(BaseModel, ABC):
    model_name: str
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


class LangChainLLM(BaseLLM, ABC):
    _llm_model: BaseChatModel = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self._pre_setup_steps()
            self._setup_llm_model()
            self._post_setup_steps()
            self._logger.info(f"Successfully initialized embedding model: {self.alias}")
        except Exception as e:
            self._logger.error(f"Failed to initialize embedding model {self.alias}: {e}")
            raise

    @abstractmethod
    def _pre_setup_steps(self) -> None:
        pass

    @abstractmethod
    def _post_setup_steps(self) -> None:
        pass

    @abstractmethod
    def _setup_llm_model(self) -> None:
        pass


class DatabricksLLM(LangChainLLM, LLMPlugins):
    model_name: str
    temperature: float = 0.0
    n: int = 1
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    extra_params: Optional[Dict[str, Any]] = None
    stream_usage: bool = False

    _llm_model: ChatDatabricks = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _setup_llm_model(self) -> None:
        try:
            self._llm_model = ChatDatabricks(
                endpoint=self.model_name,
                temperature=self.temperature,
                n=self.n,
                stop=self.stop,
                max_tokens=self.max_tokens,
                extra_params=self.extra_params,
                stream_usage=self.stream_usage
            )
        except (ImportError, AttributeError) as e:
            self._logger.error(f"Failed to initialize language model: '{e}'")
            raise RuntimeError(f"Failed to initialize language model: '{e}'")

    def _pre_setup_steps(self) -> None:
        self._logger.warning(f"No pre-setup steps defined for language model '{self.alias}'")
        pass

    def _post_setup_steps(self) -> None:
        self._logger.warning(f"No post-setup steps defined for language model '{self.alias}'")
        pass

    @staticmethod
    def llm_flavor() -> str:
        return LanguageModelFlavor.LANGCHAIN_CHAT_DATABRICKS.value