from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pydantic import PrivateAttr

from langchain_core.language_models import BaseChatModel
from databricks_langchain import ChatDatabricks

from rag_pipeline.rag_pipeline_setup.flavor_enums import LanguageModelFlavor
from rag_pipeline.rag_pipeline_setup.rag_language_model.base_language_model import AbstractBaseLLM


class AbstractLangChainLLM(AbstractBaseLLM, ABC):
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


class DatabricksLLM(AbstractLangChainLLM):
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