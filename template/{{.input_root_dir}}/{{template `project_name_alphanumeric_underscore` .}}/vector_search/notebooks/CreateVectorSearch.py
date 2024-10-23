# Databricks notebook source
# MAGIC %pip install mlflow==2.16.0 databricks-vectorsearch==0.40
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from utils import endpoint_exists, wait_for_vs_endpoint_to_be_ready

vsc = VectorSearchClient(disable_notice=True)

if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")


# this may throw an error on the first pass, once the endpoint is created we'd see correct messages
wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

from utils import index_exists, wait_for_index_to_be_ready
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

#The table we'd like to index
source_table_fullname = f"{CATALOG}.{SCHEMA}.{SOURCE_DATA_FOR_INDEX_TABLE_NAME}"
# Where we want to store our index
vs_index_fullname = f"{CATALOG}.{SCHEMA}.{SOURCE_DATA_FOR_INDEX_TABLE_NAME}_vs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_source_column=config.get("retriever_config")["schema"]["chunk_text"], #The column containing our text
    embedding_model_endpoint_name="databricks-gte-large-en" #The embedding endpoint used to create the embeddings
  )
  #Let's wait for the index to be ready and all our embeddings to be created and indexed
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).wait_until_ready()
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")
