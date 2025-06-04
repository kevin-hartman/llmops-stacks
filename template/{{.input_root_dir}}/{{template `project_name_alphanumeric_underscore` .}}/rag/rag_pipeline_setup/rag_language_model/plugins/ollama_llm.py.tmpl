from typing import Optional, List, Literal, Union
from pydantic import PrivateAttr
from pydantic.json_schema import JsonSchemaValue

from langchain_ollama import ChatOllama

from rag_pipeline_setup.flavor_enums import LanguageModelFlavor
from rag_pipeline_setup.rag_language_model.plugins.langchain_llm import AbstractLangChainLLM

class OllamaLLM(AbstractLangChainLLM):
    model_name: str
    mirostat: Optional[int] = None
    mirostat_eta: Optional[float] = None
    mirostat_tau: Optional[float] = None
    num_ctx: Optional[int] = None
    num_gpu: Optional[int] = None
    num_thread: Optional[int] = None
    num_predict: Optional[int] = None
    repeat_last_n: Optional[int] = None
    repeat_penalty: Optional[float] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    tfs_z: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    format: Optional[Union[Literal["", "json"], JsonSchemaValue]] = None
    keep_alive: Optional[Union[int, str]] = None
    base_url: Optional[str] = None
    http_client_kwargs: Optional[dict] = {}

    _llm: ChatOllama = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _setup_llm_model(self) -> None:
        try:
            self._llm = ChatOllama(
                model=self.model_name,
                mirostat=self.mirostat,
                mirostat_eta=self.mirostat_eta,
                mirostat_tau=self.mirostat_tau,
                num_ctx=self.num_ctx,
                num_gpu=self.num_gpu,
                num_thread=self.num_thread,
                num_predict=self.num_predict,
                repeat_last_n=self.repeat_last_n,
                repeat_penalty=self.repeat_penalty,
                temperature=self.temperature,
                seed=self.seed,
                stop=self.stop,
                tfs_z=self.tfs_z,
                top_k=self.top_k,
                top_p=self.top_p,
                format=self.format,
                keep_alive=self.keep_alive,
                base_url=self.base_url,
                client_kwargs=self.http_client_kwargs
            )
        except Exception as e:
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
        return LanguageModelFlavor.LANGCHAIN_CHAT_OLLAMA.value