from enum import Enum

class VectorStoreFlavor(Enum):
    LANGCHAIN_DATABRICKS_VECTOR_SEARCH = "langchain_databricks_vector_search"

class EmbeddingModelFlavor(Enum):
    LANGCHAIN_DATABRICKS = "langchain_databricks"

class LanguageModelFlavor(Enum):
    LANGCHAIN_CHAT_DATABRICKS = "langchain_chat_databricks"