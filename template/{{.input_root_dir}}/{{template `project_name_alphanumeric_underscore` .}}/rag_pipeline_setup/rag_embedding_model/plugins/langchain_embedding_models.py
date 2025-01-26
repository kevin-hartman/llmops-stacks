from abc import abstractmethod, ABC
from pydantic import PrivateAttr
from typing import Any, Dict
import os

from langchain_core.embeddings import Embeddings
from databricks_langchain import DatabricksEmbeddings

from rag_pipeline_setup import constants
from rag_pipeline_setup.flavor_enums import EmbeddingModelFlavor
from rag_pipeline_setup.rag_embedding_model.base_embedding_model import AbstractBaseEmbeddingModel

class AbstractLangchainEmbeddingModel(AbstractBaseEmbeddingModel, ABC):
    _embedding_model: Embeddings = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self._pre_setup_steps()
            self._setup_embedding_model()
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
    def _setup_embedding_model(self) -> None:
        pass

class DatabricksEmbeddingModel(AbstractLangchainEmbeddingModel):
    query_params: Dict[str, Any] = {}
    documents_params: Dict[str, Any] = {}

    _embedding_model: DatabricksEmbeddings = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _setup_embedding_model(self) -> None:
        try:
            self._embedding_model = DatabricksEmbeddings(
                endpoint=self.model_name,
                query_params=self.query_params,
                documents_params=self.documents_params
            )
        except Exception as e:
            self._logger.error(f"Failed to initialize embedding model: '{e}'")
            raise RuntimeError(f"Failed to initialize embedding model: '{e}'")

    def _pre_setup_steps(self) -> None:
        self._logger.info(f"Checking Databricks authentication using Service Principal for Embedding Model: '{self.alias}'")
        if not os.environ.get(constants.DATABRICKS_CLIENT_ID, False):
            raise RuntimeError(f"Missing '{constants.DATABRICKS_CLIENT_ID}' in environment variable")
        if not os.environ.get(constants.DATABRICKS_CLIENT_SECRET, False):
            raise RuntimeError(f"Missing '{constants.DATABRICKS_CLIENT_SECRET}' in environment variable")
        if not os.environ.get(constants.DATABRICKS_HOST, False):
            raise RuntimeError(f"Missing '{constants.DATABRICKS_HOST}' in environment variable")

    def _post_setup_steps(self) -> None:
        self._logger.warning(f"No post-setup steps defined for embedding model: '{self.alias}'")

    @staticmethod
    def embedding_model_flavor() -> str:
        return EmbeddingModelFlavor.LANGCHAIN_DATABRICKS.value
