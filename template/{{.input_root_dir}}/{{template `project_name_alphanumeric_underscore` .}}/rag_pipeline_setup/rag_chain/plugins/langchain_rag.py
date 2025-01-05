from rag_pipeline_setup.flavor_enums import PipelineFlavor
from rag_pipeline_setup.rag_chain.base_rag_chain import AbstractRagChain

class LangChainRag(AbstractRagChain):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _pre_setup_steps(self) -> None:
        self._logger.warning(f"No pre-setup steps defined for Rag Pipeline: '{self.alias}'")

    #TODO - Add post setup steps for internal model types
    def _post_setup_steps(self) -> None:
        self._logger.warning(f"No post-setup steps defined for Rag Pipeline: '{self.alias}'")

    @staticmethod
    def rag_chain_flavor() -> str:
        return PipelineFlavor.LANGCHAIN.value