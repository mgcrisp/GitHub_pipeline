# Databricks notebook source
# MAGIC %md
# MAGIC # Landing Zone to Bronze Layer

# COMMAND ----------

import datetime
import numpy as np
from pyspark.sql import types as T 
from pyspark.sql import functions as F

# COMMAND ----------

### Getting oAuth for Azure Container
service_credential = dbutils.secrets.get(scope="databricks",key="databricks-test")
spark.conf.set("fs.azure.account.auth.type.ade20220919.dfs.core.windows.net", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type.ade20220919.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id.ade20220919.dfs.core.windows.net", "98a69314-1a2f-4eef-9b09-6ff1e73ead68")
spark.conf.set("fs.azure.account.oauth2.client.secret.ade20220919.dfs.core.windows.net", service_credential)
spark.conf.set("fs.azure.account.oauth2.client.endpoint.ade20220919.dfs.core.windows.net", "https://login.microsoftonline.com/33da9f3f-4c1a-4640-8ce1-3f63024aea1d/oauth2/token")

# COMMAND ----------

### Get all file paths in landing zone
y = dbutils.fs.ls(f"abfs://landing-zone@ade20220919.dfs.core.windows.net/gharchive-part")
rdd = spark.sparkContext.parallelize(y)
md = rdd.toDF()

# COMMAND ----------

### Organize table of file paths as a DataFrame
md = (md
      .withColumn('year', F.substring('name', 1, 4).cast(T.IntegerType()))
      .withColumn('month', F.substring('name', 6, 2).cast(T.IntegerType()))
      .withColumn('day', F.substring('name', 9, 2).cast(T.IntegerType()))
      .withColumn('hour', F.substring('name', 12, 2).cast(T.IntegerType()))
      .drop('name', 'modificationTime')
      .orderBy('year','month','day','hour')
     )

x = md.display()

# COMMAND ----------

### From miniro_exploration, we know that parquet files are about double 
# the size of our compressed json, we will use this information
# as well as the size of the json files to select the number of subpartitions(target 128MB)



json_hour_size = []
# for every day
for i in range(len(x)):
    json_hour_size.append(x[i][1])

# Get size of each day
json_day_size = np.sum(np.array(json_hour_size).reshape(-1, 24), axis=1)

parquet_day_size = json_day_size * 2

partition_sizes_max = np.round(parquet_day_size / 128000000, 1)
partition_sizes_min = np.round(parquet_day_size / 400000000, 1)

# COMMAND ----------

min(partition_sizes_max)

# COMMAND ----------

max(partition_sizes_min)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2 and 3 partitions per day would both seem reasonable choices. We pick 3. 

# COMMAND ----------

### Write data logically partitioned by day and in 3 
# Read in all of the landing zone data
df = spark.read.json(f"abfs://landing-zone@ade20220919.dfs.core.windows.net/gharchive-part")

# Adding day column and write files partitioning by day
df = df.withColumn("day", df.created_at.substr(1,10))
df.repartition(3).write.partitionBy('day').parquet(f"abfs://bronze-layer@ade20220919.dfs.core.windows.net/miniro/2017-01")

# COMMAND ----------

### Printing out all distinct event types
types = df.select("type").distinct().collect()
type_list = [types[i][0] for i in range(len(types))]
type_list