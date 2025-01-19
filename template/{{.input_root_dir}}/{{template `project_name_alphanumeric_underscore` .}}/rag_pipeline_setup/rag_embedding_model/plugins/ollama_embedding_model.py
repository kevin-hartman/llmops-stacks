from pydantic import PrivateAttr
from typing import Optional

from langchain_ollama import OllamaEmbeddings

from rag_pipeline_setup.flavor_enums import EmbeddingModelFlavor
from rag_pipeline_setup.rag_embedding_model.plugins.langchain_embedding_models import AbstractLangchainEmbeddingModel


class OllamaEmbeddingModel(AbstractLangchainEmbeddingModel):
    base_url: Optional[str] = None
    http_client_kwargs: Optional[dict] = {}

    _embedding_model: OllamaEmbeddings = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _setup_embedding_model(self) -> None:
        try:
            self._embedding_model = OllamaEmbeddings(
                model=self.model_name,
                base_url=self.base_url,
                client_kwargs=self.http_client_kwargs
            )
        except Exception as e:
            self._logger.error(f"Failed to initialize embedding model: '{e}'")
            raise RuntimeError(f"Failed to initialize embedding model: '{e}'")

    def _pre_setup_steps(self) -> None:
        self._logger.warning(f"No pre-setup steps defined for embedding model: '{self.alias}'")

    def _post_setup_steps(self) -> None:
        self._logger.warning(f"No post-setup steps defined for embedding model: '{self.alias}'")

    @staticmethod
    def embedding_model_flavor() -> str:
        return EmbeddingModelFlavor.LANGCHAIN_OLLAMA.value