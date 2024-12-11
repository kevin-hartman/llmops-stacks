import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel, PrivateAttr, ConfigDict, Field
from yaml import safe_load
from typing import Literal

from embedding_models import BaseEmbeddingModel, LangchainEmbeddingModel
from vector_store import BaseVectorStore, LangChainVectorStore
from chat_models import BaseLLMModel, LangChainLLMModel

# TODO - check need of having component level flavor vs one for the entire pipeline config
class BaseRagPipeline(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # TODO - move static references to constants
    base_config_yaml_location: str = "configs/rag_config.yaml"
    alias: str

    _logger: logging.Logger = PrivateAttr()
    # TODO - Check pros/cons of setting attributes directly on class based on naming instead of explicit dict references
    _embedding_models: Dict[str, BaseEmbeddingModel] = PrivateAttr(default={})
    _llms: Dict[str, BaseLLMModel] = PrivateAttr(default={})
    _vector_stores: Dict[str, BaseVectorStore] = PrivateAttr(default={})
    _base_config_dictionary: dict = PrivateAttr(default={})
    _base_inst_dictionary: Dict[str, str] = PrivateAttr(default={})

    # TODO - Add exception handling
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_logger()
        self._logger.info(f"Initializing Rag pipeline : {self.alias}")

        self._setup_base_config_yaml()
        self._setup_embedding_models()
        self._setup_chat_models()
        self._setup_vector_stores()
        self._setup_instruction_dictionary()

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

    def set_log_level(self, level: int) -> None:
        """
        Allow dynamic setting of log level.

        Args:
            level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.ERROR)
        """
        self._logger.setLevel(level)

    @abstractmethod
    def _pre_setup_steps(self) -> None:
        pass

    @abstractmethod
    def _post_setup_steps(self) -> None:
        pass

    @abstractmethod
    def _setup_embedding_models(self) -> None:
        pass

    @abstractmethod
    def _setup_chat_models(self) -> None:
        pass

    @abstractmethod
    def _setup_vector_stores(self) -> None:
        pass

    def _setup_base_config_yaml(self) -> None:
        with open(self.base_config_yaml_location) as stream:
            self._base_config_dictionary = safe_load(stream)

    @property
    def embedding_models(self) -> Dict[str, BaseEmbeddingModel]:
        return self._embedding_models

    @property
    def llms(self) -> Dict[str, BaseLLMModel]:
        return self._llms

    @property
    def vector_stores(self) -> Dict[str, BaseVectorStore]:
        return self._vector_stores

    @property
    def base_inst_dictionary(self) -> Dict:
        return self._base_inst_dictionary

    # TODO - Set appropriate validations/checks
    def get_embedding_model(self, alias: str) -> BaseEmbeddingModel:
        return self.embedding_models[alias]

    # TODO - Set appropriate validations/checks
    def get_chat_model(self, alias: str) -> BaseLLMModel:
        return self.llms[alias]

    # TODO - Set appropriate validations/checks
    def get_vector_store(self, alias: str) -> BaseVectorStore:
        return self.vector_stores[alias]

    # TODO - Set appropriate validations/checks
    def get_instruction(self, alias: str) -> str:
        return self.base_inst_dictionary[alias]

class LangChainRagPipeline(BaseRagPipeline):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    flavor: Literal["langchain"]

    _embedding_models: Dict[str, LangchainEmbeddingModel] = PrivateAttr(default={})
    _llms: Dict[str, LangChainLLMModel] = PrivateAttr(default={})
    _vector_stores: Dict[str, LangChainVectorStore] = PrivateAttr(default={})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # TODO - Move all static string references to constants
    # TODO - Add validations and error checks
    def _setup_embedding_models(self) -> None:
        for model in self._base_config_dictionary['rag_chain_configs'].get('embedding_models', {}):
            self._embedding_models[model['alias']] = LangchainEmbeddingModel(**model)

    # TODO - Move all static string references to constants
    # TODO - Add validations and error checks
    def _setup_chat_models(self) -> None:
        for llm_config in self._base_config_dictionary['rag_chain_configs'].get('llms', {}):
            self._llms[llm_config['alias']] = LangChainLLMModel(**llm_config)

    # TODO - Move all static string references to constants
    # TODO - Add validations and error checks
    def _setup_vector_stores(self) -> None:
        for vs_config in self._base_config_dictionary['rag_chain_configs'].get('vector_stores', {}):
            embedding_model_ref = self.get_embedding_model(vs_config['dependent_embedding_module_alias'])
            del vs_config['dependent_embedding_module_alias']

            vs_config['embedding_model_ref'] = embedding_model_ref
            self._vector_stores[vs_config['alias']] = LangChainVectorStore(**vs_config)

    def _pre_setup_steps(self) -> None:
        self._logger.warning(f"No pre-setup steps defined for rag pipeline '{self.alias}'")

    def _post_setup_steps(self) -> None:
        self._logger.warning(f"No post-setup steps defined for rag pipeline '{self.alias}'")
