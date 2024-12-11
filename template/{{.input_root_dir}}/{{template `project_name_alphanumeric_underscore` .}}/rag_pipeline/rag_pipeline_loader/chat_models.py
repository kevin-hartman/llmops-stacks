import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, PrivateAttr
import importlib
from typing import Literal

# TODO - update pydantic variables to fields with descriptions
class BaseLLMModel(BaseModel, ABC):
    module: str
    model_name: str
    alias: str

    _logger: logging.Logger = PrivateAttr()
    _llm_model: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_logger()
        self._logger.info(f"Initializing embedding model: {self.alias}")

        try:
            self._pre_setup_steps()
            self._setup_llm_model()
            self._post_setup_steps()
            self._logger.info(f"Successfully initialized embedding model: {self.alias}")
        except Exception as e:
            self._logger.error(f"Failed to initialize embedding model {self.alias}: {e}")
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
    def _setup_llm_model(self) -> None:
        pass

    @property
    def llm_model(self) -> Any:
        if self._llm_model is None:
            self._logger.error("LLM model has not been initialized correctly")
            raise RuntimeError("LLM model has not been initialized correctly")
        return self._llm_model

    def set_log_level(self, level: int) -> None:
        """
        Allow dynamic setting of log level.

        Args:
            level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.ERROR)
        """
        self._logger.setLevel(level)


class LangChainLLMModel(BaseLLMModel):
    temperature: float = 0.0
    n: int = 1
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    extra_params: Optional[Dict[str, Any]] = None
    stream_usage: bool = False
    flavor: Literal["langchain"]

    _llm_model: BaseChatModel = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # TODO - Standardize package loading and move to common module
    def _setup_llm_model(self) -> None:
        try:
            package_name, module_name = self.module.split(".")
            package = importlib.import_module(package_name)
            module = getattr(package, module_name)

            self._llm_model = module(
                endpoint=self.model_name,
                temperature=self.temperature,
                n=self.n,
                stop=self.stop,
                max_tokens=self.max_tokens,
                extra_params=self.extra_params,
                stream_usage=self.stream_usage
            )
        except (ImportError, AttributeError) as e:
            self._logger.error(f"Failed to initialize llm model: '{e}'")
            raise RuntimeError(f"Failed to initialize llm model: '{e}'")

    def _pre_setup_steps(self) -> None:
        self._logger.warning(f"No pre-setup steps defined for llm model '{self.alias}'")
        pass

    def _post_setup_steps(self) -> None:
        self._logger.warning(f"No post-setup steps defined for llm model '{self.alias}'")
        pass
