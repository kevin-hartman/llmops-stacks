from yaml import safe_load

from rag_pipeline.rag_pipeline_setup.rag_embedding_model.embedding_model_plugins import EmbeddingModelPlugins
from rag_pipeline.rag_pipeline_setup.rag_embedding_model.embedding_model_plugins import EmbeddingModelPluginsDiscovery

from rag_pipeline.rag_pipeline_setup.rag_language_model.language_model_plugins import LanguageModelPlugins
from rag_pipeline.rag_pipeline_setup.rag_language_model.language_model_plugins import LanguageModelPluginsDiscovery

from rag_pipeline.rag_pipeline_setup.rag_vector_store.vector_store_plugins import VectorStorePlugins
from rag_pipeline.rag_pipeline_setup.rag_vector_store.vector_store_plugins import VectorStorePluginsDiscovery



def parse_embedding_params(mappings):
    EmbeddingModelPluginsDiscovery().discover_embedding_model_plugins()
    for model in mappings['embedding_models']:
        # print(EmbeddingModelPlugins.embedding_model_plugins)
        embedding_model = (EmbeddingModelPlugins
                           .embedding_model_plugins
                           .get(model['flavor']))
        # print(embedding_model)
        module = embedding_model(**model)
        # print(module)
    return module


def parse_llm_params(mappings):
    LanguageModelPluginsDiscovery().discover_language_model_plugins()
    for model in mappings['llms']:
        # print(LanguageModelPlugins.language_model_plugins)
        language_model = (LanguageModelPlugins
                          .language_model_plugins
                          .get(model['flavor']))

        module = language_model(**model)
        # print(module)

def parse_vc_params(mappings, embedding_model):
    VectorStorePluginsDiscovery().discover_vector_store_plugins()
    for model in mappings['vector_stores']:

        del model['dependent_embedding_module_alias']
        model['embedding_model_ref'] = embedding_model
        print(model['flavor'])
        print(embedding_model)
        print(type(embedding_model))
        model['embedding_model_ref'] = embedding_model
        vc_module = (VectorStorePlugins
                     .vector_store_plugins
                     .get(model['flavor']))
        print(model)
        module = vc_module(**model)

with open("configs/rag_config.yaml") as stream:
    mapping = safe_load(stream)

print(mapping)
print()
embedding_module = parse_embedding_params(mapping['rag_chain_configs'])
parse_llm_params(mapping['rag_chain_configs'])
parse_vc_params(mapping['rag_chain_configs'], embedding_module)
