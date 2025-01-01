from abc import ABC, abstractmethod

class VectorStoresPlugins(ABC):
    vector_store_plugins = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.vector_store_plugins[cls.vector_store_flavor()] = cls

    @staticmethod
    @abstractmethod
    def vector_store_flavor() -> str:
        pass


# class LLMPlugins(ABC):
#     llm_plugins = {}
#
#     @classmethod
#     def __init_subclass__(cls, **kwargs):
#         super().__init_subclass__(**kwargs)
#         cls.llm_plugins[cls.llm_flavor()] = cls
#
#     @staticmethod
#     @abstractmethod
#     def llm_flavor() -> str:
#         pass
