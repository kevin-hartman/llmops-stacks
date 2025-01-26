import os
from abc import abstractmethod, ABC
from pydantic import PrivateAttr, ConfigDict
from typing import Optional, List, Any
from typing_extensions import TypedDict

from langchain_core.vectorstores import VectorStore
from databricks_langchain import DatabricksVectorSearch

from rag_pipeline_setup import constants
from rag_pipeline_setup.flavor_enums import VectorStoreFlavor
from rag_pipeline_setup.rag_vector_store.base_vector_store import AbstractBaseVectorStore

SERVICE_PRINCIPAL_CLIENT_ID_VALUE = "service_principal_client_id"
SERVICE_PRINCIPAL_CLIENT_ID_SECRET = "service_principal_client_secret"

class AbstractLangChainVectorStore(AbstractBaseVectorStore, ABC):
    _vector_store: VectorStore = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self._pre_setup_steps()
            self._setup_vector_store()
            self._post_setup_steps()
            self._logger.info(f"Successfully initialized Vector Store: '{self.alias}'")
        except Exception as e:
            self._logger.error(f"Failed to initialize Vector Store '{self.alias}': {e}")
            raise

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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    class ClientArgs(TypedDict):
        workspace_url: str

    index_name: str
    endpoint_name: Optional[str] = None
    embedding_model_ref: Optional[Any] = None
    text_column: Optional[str] = None
    columns: Optional[List[str]] = None
    client_args: ClientArgs

    _vector_store: DatabricksVectorSearch = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _setup_vector_store(self) -> None:
        try:
            # TODO - Check initialization from local based on api keys
            self._vector_store = DatabricksVectorSearch(
                index_name=self.index_name,
                endpoint=self.endpoint_name,
                embedding=None if self.embedding_model_ref is None else self.embedding_model_ref.embedding_model,
                text_column=self.text_column,
                columns=self.columns,
                client_args=self.client_args
            )
        except Exception as e:
            self._logger.error(f"Failed to initialize Vector Store: '{e}'")
            raise RuntimeError(f"Failed to initialize Vector Store: '{e}'")


    def _pre_setup_steps(self) -> None:
        self._logger.info(f"Setting up Databricks authentication using Service Principal for Vector Store: '{self.alias}'")
        if not os.environ.get(constants.DATABRICKS_CLIENT_ID, False):
            raise RuntimeError(f"Missing '{constants.DATABRICKS_CLIENT_ID}' in environment variable")
        if not os.environ.get(constants.DATABRICKS_CLIENT_SECRET, False):
            raise RuntimeError(f"Missing '{constants.DATABRICKS_CLIENT_SECRET}' in environment variable")

        self.client_args[SERVICE_PRINCIPAL_CLIENT_ID_VALUE] = os.environ[constants.DATABRICKS_CLIENT_ID]
        self.client_args[SERVICE_PRINCIPAL_CLIENT_ID_SECRET] = os.environ[constants.DATABRICKS_CLIENT_SECRET]

    def _post_setup_steps(self) -> None:
        self._logger.warning(f"No post-setup steps defined for vector store '{self.alias}'")

    @staticmethod
    def vector_store_flavor() -> str:
        return VectorStoreFlavor.LANGCHAIN_DATABRICKS_VECTOR_SEARCH.value
