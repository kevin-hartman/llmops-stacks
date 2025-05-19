import pandas as pd

def get_reference_documentation(catalog, schema, table, spark):
    (spark.createDataFrame(pd.read_parquet('https://notebooks.databricks.com/demos/dbdemos-dataset/llm/databricks-documentation/databricks_doc_eval_set.parquet'))
    .write.mode('overwrite').saveAsTable(f"{catalog}.{schema}.{table}"))

    eval_df = spark.read.table(f"{catalog}.{schema}.{table}")

    return eval_df
