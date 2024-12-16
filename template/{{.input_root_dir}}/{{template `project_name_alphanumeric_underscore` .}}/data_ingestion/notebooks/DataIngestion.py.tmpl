# Databricks notebook source
################################################################################### Pipeline Documentation

# This pipeline is designed to process and evaluate raw documentation data from a specified data source URL. The data is stored in a Unity Catalog and processed within a specified database. The pipeline includes the following parameters:

# Optional Parameters:
# - uc_catalog (str): Name of the Unity Catalog containing the input data. Default is "llmops_stacks".
# - database_name (str): Name of the database inside the Unity Catalog. Default is "rag_chatbot".
# - raw_data_table_name (str): Name of the raw data table inside the database of the Unity Catalog. Default is "raw_documentation".
# - data_source_url (str): URL of the data source. Default is "https://docs.databricks.com/en/doc-sitemap.xml".

# Required Parameters:
# - None

# Widgets:
# - Name of Unity Catalog: Text widget to input the name of the Unity Catalog.
# - Name of database inside Unity Catalog: Text widget to input the name of the database inside the Unity Catalog.
# - Raw data Table Name: Text widget to input the name of the raw data table inside the database of the Unity Catalog.
# - Eval set data Table Name: Text widget to input the name of the evaluation table inside the database of the Unity Catalog.
# - Data Source URL: Text widget to input the URL of the data source.

# Usage:
# 1. Set the appropriate values for the widgets.
# 2. Run the pipeline to process and evaluate the raw documentation data.

# Example:
# dbutils.widgets.text("uc_catalog", "llmops_stacks", label="Name of Unity Catalog")
# dbutils.widgets.text("database_name", "rag_chatbot", label="Name of database inside Unity Catalog")
# dbutils.widgets.text("raw_data_table_name", "raw_documentation", label="Raw data Table Name")
# dbutils.widgets.text("eval_set_table_name", "eval_set_databricks_documentation", label="Eval set data Table Name")
# dbutils.widgets.text("data_source_url", "https://docs.databricks.com/en/doc-sitemap.xml", label="Data Source URL")
##################################################################################

# COMMAND ----------

# MAGIC %md
# MAGIC List of input args needed to run this notebook as a job.
# MAGIC Provide them via DB widgets or notebook arguments in your DAB resources.

# COMMAND ----------

# A Unity Catalog location containing the input data.
dbutils.widgets.text(
    "uc_catalog",
    "llmops_stacks",
    label="Name of Unity Catalog",
)
# Name of database inside Unity Catalog.
dbutils.widgets.text(
    "database_name",
    "rag_chatbot",
    label="Name of database inside Unity Catalog",
)
# Name of raw data table inside database of Unity Catalog.
dbutils.widgets.text(
    "raw_data_table_name",
    "raw_documentation",
    label="Raw data Table Name",
)

# data source url.
dbutils.widgets.text(
    "data_source_url",
    "https://docs.databricks.com/en/doc-sitemap.xml",
    label="Data Source URL",
)

# COMMAND ----------

import os

notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ../ingestion

# COMMAND ----------

# DBTITLE 1,Define input and output variables
uc_catalog = dbutils.widgets.get("uc_catalog")
database_name = dbutils.widgets.get("database_name")
raw_data_table_name = dbutils.widgets.get("raw_data_table_name")
data_source_url = dbutils.widgets.get("data_source_url")

assert uc_catalog != "", "uc_catalog notebook parameter must be specified"
assert database_name != "", "database_name notebook parameter must be specified"
assert raw_data_table_name != "", "raw_data_table_name notebook parameter must be specified"
assert data_source_url != "", "data_source_url notebook parameter must be specified"

# COMMAND ----------

# DBTITLE 1,Use the catalog and database specified in the notebook parameters
spark.sql(f"""USE `{uc_catalog}`.`{database_name}`""")

# COMMAND ----------

# DBTITLE 1,Download and store data to UC
from fetch_data import fetch_data_from_url

if not spark.catalog.tableExists(f"{raw_data_table_name}") or spark.table(f"{raw_data_table_name}").isEmpty():
    # Download the data to a DataFrame 
    doc_articles = fetch_data_from_url(spark, data_source_url)
    #Save them as to unity catalog
    doc_articles.write.mode('overwrite').saveAsTable(f"{raw_data_table_name}")