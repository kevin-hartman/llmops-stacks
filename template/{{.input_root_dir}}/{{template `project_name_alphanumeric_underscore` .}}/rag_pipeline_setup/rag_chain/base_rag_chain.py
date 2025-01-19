import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel, PrivateAttr

from rag_pipeline_setup.rag_embedding_model.embedding_model_plugins import EmbeddingModelPlugins
from rag_pipeline_setup.rag_embedding_model.embedding_model_plugins import EmbeddingModelPluginsDiscovery
from rag_pipeline_setup.rag_language_model.language_model_plugins import LanguageModelPlugins
from rag_pipeline_setup.rag_language_model.language_model_plugins import LanguageModelPluginsDiscovery
from rag_pipeline_setup.rag_vector_store.vector_store_plugins import VectorStorePlugins
from rag_pipeline_setup.rag_vector_store.vector_store_plugins import VectorStorePluginsDiscovery
from rag_pipeline_setup.rag_chain.rag_chain_plugins import RagChainPlugins
from rag_pipeline_setup.rag_embedding_model.base_embedding_model import  AbstractBaseEmbeddingModel
from rag_pipeline_setup.rag_language_model.base_language_model import  AbstractBaseLLM
from rag_pipeline_setup.rag_vector_store.base_vector_store import  AbstractBaseVectorStore
from rag_pipeline_setup.consts import *

# TODO - update pydantic variables to fields with descriptions
class AbstractRagChain(RagChainPlugins, BaseModel, ABC):
    alias: str

    _logger: logging.Logger = PrivateAttr()
    _embedding_models: Dict[str, AbstractBaseEmbeddingModel] = PrivateAttr(default={})
    _llms: Dict[str, AbstractBaseLLM] = PrivateAttr(default={})
    _vector_stores: Dict[str, AbstractBaseVectorStore] = PrivateAttr(default={})
    _base_config_dictionary: dict = PrivateAttr(default={})
    _base_template_dictionary: Dict[str, str] = PrivateAttr(default={})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_logger()
        self._logger.info(f"Initializing Rag pipeline: {self.alias}")

        assert kwargs.get(PIPELINE_CONFIGS, False), f"Pipeline configuration missing. Expected key {PIPELINE_CONFIGS} in configuration dictionary"
        self._base_config_dictionary = kwargs[PIPELINE_CONFIGS]
        self._logger.info(f"Setting up rag pipeline configurations using config values: {self._base_config_dictionary}")

        self._setup_embedding_models()
        self._setup_language_models()
        self._setup_vector_stores()
        # self._setup_template_dictionary()

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


    def _setup_embedding_models(self) -> None:
        EmbeddingModelPluginsDiscovery().discover_embedding_model_plugins()
        if not self._base_config_dictionary.get(EMBEDDING_MODELS, False):
            self._logger.warning(f"Missing configuration key for embedding models: '{EMBEDDING_MODELS}'")
            return

        for em_model in self._base_config_dictionary[EMBEDDING_MODELS]:
            embedding_model_class = (EmbeddingModelPlugins
                               .embedding_model_plugins
                               .get(em_model[FLAVOR]))

            if embedding_model_class is None:
                raise RuntimeError(f"No Embedding model plugin with flavor: '{em_model[FLAVOR]}'")

            embedding_model = embedding_model_class(**em_model)
            self._embedding_models[em_model[ALIAS]] = embedding_model

    def _setup_language_models(self) -> None:
        LanguageModelPluginsDiscovery().discover_language_model_plugins()
        if not self._base_config_dictionary.get(LLMS, False):
            self._logger.warning(f"Missing configuration key for language models: '{LLMS}'")
            return

        for ll_model in self._base_config_dictionary[LLMS]:
            ll_model_class = (LanguageModelPlugins
                              .language_model_plugins
                              .get(ll_model[FLAVOR]))

            if ll_model_class is None:
                raise RuntimeError(f"No Language model plugin with flavor: '{ll_model[FLAVOR]}'")

            language_model = ll_model_class(**ll_model)
            self._llms[ll_model[ALIAS]] = language_model

    def _setup_vector_stores(self) -> None:
        VectorStorePluginsDiscovery().discover_vector_store_plugins()
        if not self._base_config_dictionary.get(VECTOR_STORES, False):
            self._logger.warning(f"Missing configuration key for Vector Stores: '{VECTOR_STORES}'")
            return

        for vcs in self._base_config_dictionary[VECTOR_STORES]:
            if vcs.get(DEPENDENT_MODEL_ALIAS, False):
                assert self.get_embedding_model(vcs[DEPENDENT_MODEL_ALIAS]), \
                    f"Dependent embedding model '{vcs[DEPENDENT_MODEL_ALIAS]}' not found for Vector Store '{vcs['alias']}'"
                dependent_embedding_model = self.get_embedding_model(vcs[DEPENDENT_MODEL_ALIAS])
                vcs[EMBEDDING_MODEL_REF] = dependent_embedding_model

            vc_store_class = (VectorStorePlugins
                         .vector_store_plugins
                         .get(vcs[FLAVOR]))

            if vc_store_class is None:
                raise RuntimeError(f"No Language model plugin with flavor: '{vcs[FLAVOR]}'")

            vc_store = vc_store_class(**vcs)
            self._vector_stores[vcs[ALIAS]] = vc_store

    def _setup_template_dictionary(self) -> None:
        self._base_template_dictionary = self._base_config_dictionary.get(PROMPT_TEMPLATES, {})

    @property
    def embedding_models(self) -> Dict[str, AbstractBaseEmbeddingModel]:
        return self._embedding_models

    @property
    def llms(self) -> Dict[str, AbstractBaseLLM]:
        return self._llms

    @property
    def vector_stores(self) -> Dict[str, AbstractBaseVectorStore]:
        return self._vector_stores

    @property
    def base_template_dictionary(self) -> Dict:
        return self._base_template_dictionary

    def get_embedding_model(self, alias: str) -> AbstractBaseEmbeddingModel:
        if not self.embedding_models.get(alias, False):
            raise RuntimeError(f"Embedding model with alias '{alias}' not found")
        return self.embedding_models[alias]

    def get_chat_model(self, alias: str) -> AbstractBaseLLM:
        if not self.llms.get(alias, False):
            raise RuntimeError(f"Language model with alias '{alias}' not found")
        return self.llms[alias]

    def get_vector_store(self, alias: str) -> AbstractBaseVectorStore:
        if not self.vector_stores.get(alias, False):
            raise RuntimeError(f"Vector Store with alias '{alias}' not found")
        return self.vector_stores[alias]
