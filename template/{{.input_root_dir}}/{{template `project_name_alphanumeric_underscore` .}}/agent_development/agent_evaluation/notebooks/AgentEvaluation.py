# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

##################################################################################
# Agent Evaluation
# 
# Notebook that downloads an evaluation dataset and evaluates the model using
# llm-as-a-judge with the Databricks agent framework.
#
# Parameters:
# * uc_catalog (required)           - Name of the Unity Catalog 
# * schema (required)               - Name of the schema inside Unity Catalog 
# * eval_table (required)           - Name of the table containing the evaluation dataset
# * registered_model (required)     - Name of the model registered in mlflow
# * model_version (required)        - Model verison to deploy
#
# Widgets:
# * Unity Catalog: Text widget to input the name of the Unity Catalog
# * Schema: Text widget to input the name of the database inside the Unity Catalog
# * Evaluation Table: Text widget to input the name of the table containing the evaluation dataset
# * Registered model name: Text widget to input the name of the model to register in mlflow
# * Model Vesion: Text widget to input the model version to deploy
#
# Usage:
# 1. Set the appropriate values for the widgets.
# 2. Run to evaluate your agent.
#
##################################################################################

# COMMAND ----------

# MAGIC %pip install -qqq -r ../../agent_requirements.txt
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

from typing_extensions import deprecated

# COMMAND ----------

# MAGIC %pip freeze | grep databricks

# COMMAND ----------

import databricks.agents

# COMMAND ----------

# List of input args needed to run the notebook as a job.
# Provide them via DB widgets or notebook arguments.

# A Unity Catalog containing the model
dbutils.widgets.text(
    "uc_catalog",
    "ai_agent_stacks",
    label="Unity Catalog",
)
# Name of schema
dbutils.widgets.text(
    "schema",
    "ai_agent_ops",
    label="Schema",
)
# Name of evaluation table
dbutils.widgets.text(
    "eval_table",
    "databricks_documentation_eval",
    label="Evaluation dataset",
)
# Name of model registered in mlflow
dbutils.widgets.text(
    "registered_model",
    "agent_function_chatbot",
    label="Registered model name",
)
# Model version
dbutils.widgets.text(
    "model_version",
    "1",
    label="Model Version",
)

# COMMAND ----------

uc_catalog = dbutils.widgets.get("uc_catalog")
schema = dbutils.widgets.get("schema")
eval_table = dbutils.widgets.get("eval_table")
registered_model = dbutils.widgets.get("registered_model")
model_version = dbutils.widgets.get("model_version")

assert uc_catalog != "", "uc_catalog notebook parameter must be specified"
assert schema != "", "schema notebook parameter must be specified"
assert eval_table != "", "eval_table notebook parameter must be specified"
assert registered_model != "", "registered_model notebook parameter must be specified"
assert model_version != "", "model_version notebook parameter must be specified"

# COMMAND ----------

import os

notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ../evaluation

# COMMAND ----------

# DBTITLE 1,Get Evaluation Dataset
from evaluation import get_reference_documentation

eval_dataset = get_reference_documentation(uc_catalog, schema, eval_table, spark)

# COMMAND ----------

display(eval_dataset)

# COMMAND ----------

print(f"models:/{uc_catalog}.{schema}.{registered_model}/{model_version}")

# COMMAND ----------

# DBTITLE 1,Run evaluation
import mlflow

# Workaround for serverless compatibility
mlflow.tracking._model_registry.utils._get_registry_uri_from_spark_session = lambda: "databricks-uc"

# No ability to currently evaluate on serverless
model_uri = dbutils.jobs.taskValues.get(taskKey = "AgentDevelopment", key = "model_uri")
with mlflow.start_run():
    # Evaluate the logged model
    eval_results = mlflow.evaluate(
        data=eval_dataset,
        #model="runs:/5968bbf4e46d4d90add88e72c6cdec03/model",
        model=model_uri,
        model_type="databricks-agent",
    )

# COMMAND ----------

