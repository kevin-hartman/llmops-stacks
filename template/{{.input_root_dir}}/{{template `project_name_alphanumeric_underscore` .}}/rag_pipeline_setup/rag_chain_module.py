from yaml import safe_load

import os
os.getcwd()

# from rag_pipeline_setup.rag_embedding_model.embedding_model_plugins import EmbeddingModelPlugins
# from rag_pipeline_setup.rag_embedding_model.embedding_model_plugins import EmbeddingModelPluginsDiscovery
#
# from rag_pipeline_setup.rag_language_model.language_model_plugins import LanguageModelPlugins
# from rag_pipeline_setup.rag_language_model.language_model_plugins import LanguageModelPluginsDiscovery
#
# from rag_pipeline_setup.rag_vector_store.vector_store_plugins import VectorStorePlugins
# from rag_pipeline_setup.rag_vector_store.vector_store_plugins import VectorStorePluginsDiscovery

from rag_pipeline_setup.rag_chain.rag_chain_plugins import RagChainPlugins
from rag_pipeline_setup.rag_chain.rag_chain_plugins import RAGChainPluginsDiscovery

# def parse_embedding_params(mappings):
#     EmbeddingModelPluginsDiscovery().discover_embedding_model_plugins()
#     for model in mappings['embedding_models']:
#         # print(EmbeddingModelPlugins.embedding_model_plugins)
#         embedding_model = (EmbeddingModelPlugins
#                            .embedding_model_plugins
#                            .get(model['flavor']))
#         # print(embedding_model)
#         module = embedding_model(**model)
#         # print(module)
#     return module
#
#
# def parse_llm_params(mappings):
#     LanguageModelPluginsDiscovery().discover_language_model_plugins()
#     for model in mappings['llms']:
#         # print(LanguageModelPlugins.language_model_plugins)
#         language_model = (LanguageModelPlugins
#                           .language_model_plugins
#                           .get(model['flavor']))
#
#         module = language_model(**model)
#         # print(module)
#
# def parse_vc_params(mappings, embedding_model):
#     VectorStorePluginsDiscovery().discover_vector_store_plugins()
#     for model in mappings['vector_stores']:
#
#         del model['dependent_embedding_module_alias']
#         model['embedding_model_ref'] = embedding_model
#         print(model['flavor'])
#         print(embedding_model)
#         print(type(embedding_model))
#         model['embedding_model_ref'] = embedding_model
#         vc_module = (VectorStorePlugins
#                      .vector_store_plugins
#                      .get(model['flavor']))
#         print(model)
#         module = vc_module(**model)

RAGChainPluginsDiscovery().discover_rag_chain_plugins()
with open("configs/rag_config.yaml") as stream:
    mapping = safe_load(stream)

print(mapping)
print()

rag_chain = RagChainPlugins.rag_chain_plugins.get(mapping['rag_chain_configs']['flavor'])
rag_chain(**mapping['rag_chain_configs'])

# embedding_module = parse_embedding_params(mapping['rag_chain_configs'])
# parse_llm_params(mapping['rag_chain_configs'])
# parse_vc_params(mapping['rag_chain_configs'], embedding_module)

# rag_chain_config = {
#     "databricks_resources": {
#         "llm_endpoint_name": "databricks-dbrx-instruct",
#         "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT_NAME,
#     },
#     "input_example": {
#         "messages": [
#             {"role": "user", "content": "What is Apache Spark"},
#             {"role": "assistant", "content": "Apache spark is a distributed, OSS in-memory computation engine."},
#             {"role": "user", "content": "Does it support streaming?"}
#         ]
#     },
#     "llm_config": {
#         "llm_parameters": {"max_tokens": 1500, "temperature": 0.01},
#         "llm_prompt_template": "You are a trusted assistant that helps answer questions based only on the provided information. If you do not know the answer to a question, you truthfully say you do not know.  Here is some context which might or might not help you answer: {context}.  Answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer, do not mention the context or the question. Based on this context, answer this question: {question}",
#         "llm_prompt_template_variables": ["context", "question"],
#     },
#     "retriever_config": {
#         "embedding_model": "databricks-gte-large-en",
#         "chunk_template": "Passage: {chunk_text}\n",
#         "data_pipeline_tag": "poc",
#         "parameters": {"k": 3, "query_type": "ann"},
#         "schema": {"chunk_text": "content", "document_uri": "url", "primary_key": "id"},
#         "vector_search_index": f"{catalog}.{db}.databricks_pdf_documentation_self_managed_vs_index",
#     },