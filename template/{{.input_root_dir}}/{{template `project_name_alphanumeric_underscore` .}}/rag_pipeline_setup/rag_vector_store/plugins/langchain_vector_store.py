from abc import abstractmethod, ABC
from pydantic import PrivateAttr, ConfigDict
from typing import Optional, List, Any

#from rag_pipeline.rag_pipeline_setup.rag_embedding_model.plugins.langchain_embedding_models import DatabricksEmbeddingModel
from langchain_core.vectorstores import VectorStore
from databricks_langchain import DatabricksVectorSearch

from rag_pipeline.rag_pipeline_setup.flavor_enums import VectorStoreFlavor
from rag_pipeline.rag_pipeline_setup.rag_vector_store.base_vector_store import AbstractBaseVectorStore


class AbstractLangChainVectorStore(AbstractBaseVectorStore, ABC):
    _vector_store: VectorStore = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self._pre_setup_steps()
            self._setup_vector_store()
            self._post_setup_steps()
            self._logger.info(f"Successfully initialized embedding model: {self.alias}")
        except Exception as e:
            self._logger.error(f"Failed to initialize Vector Store {self.alias}: {e}")
            raise

    # TODO - Standardize package loading and move to common module
    @abstractmethod
    def _pre_setup_steps(self) -> None:
        pass

    @abstractmethod
    def _post_setup_steps(self) -> None:
        pass

    @abstractmethod
    def _setup_vector_store(self) -> None:
        pass


class DatabricksVectorStore(AbstractLangChainVectorStore):
    # model_config = ConfigDict(arbitrary_types_allowed=True)

    index_name: str
    endpoint_name: Optional[str] = None,
    embedding_model_ref: Optional[Any] = None,
    text_column: Optional[str] = None,
    columns: Optional[List[str]] = None

    _vector_store: DatabricksVectorSearch = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _setup_vector_store(self) -> None:
        try:
            # TODO - Check initialization from local based on api keys
            self._vector_store = DatabricksVectorSearch(
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

    @staticmethod
    def vector_store_flavor() -> str:
        return VectorStoreFlavor.LANGCHAIN_DATABRICKS_VECTOR_SEARCH.value
