# Databricks notebook source
# MAGIC %md
# MAGIC # Silver to Gold

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing the modules, files, and setting oAuth

# COMMAND ----------

## Importing modules
import os
import datetime
import re
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark import StorageLevel
from datetime import datetime
from datetime import timezone
import pandas as pd
from sparknlp.base import *
from sparknlp.annotator import *


# to_date() function for translating dates
spark = SparkSession.builder.appName('PySpark to_date()').getOrCreate()

# Legacy mode to be able to use date_format()
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# COMMAND ----------

## Getting oAuth for Azure Container
service_credential = dbutils.secrets.get(scope="databricks",key="databricks-test")
spark.conf.set("fs.azure.account.auth.type.ade20220919.dfs.core.windows.net", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type.ade20220919.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id.ade20220919.dfs.core.windows.net", "98a69314-1a2f-4eef-9b09-6ff1e73ead68")
spark.conf.set("fs.azure.account.oauth2.client.secret.ade20220919.dfs.core.windows.net", service_credential)
spark.conf.set("fs.azure.account.oauth2.client.endpoint.ade20220919.dfs.core.windows.net", "https://login.microsoftonline.com/33da9f3f-4c1a-4640-8ce1-3f63024aea1d/oauth2/token")

# COMMAND ----------

## Reading necessary parquet files from silver layer 
main_df  = spark.read.parquet(f"abfs://silver-layer@ade20220919.dfs.core.windows.net/miniro/main_df")
push_event_df  = spark.read.parquet(f"abfs://silver-layer@ade20220919.dfs.core.windows.net/miniro/push_event_df")
push_event_df_commits_df  = spark.read.parquet(f"abfs://silver-layer@ade20220919.dfs.core.windows.net/miniro/push_event_df_commits_df")
create_event_df  = spark.read.parquet(f"abfs://silver-layer@ade20220919.dfs.core.windows.net/miniro/create_event_df")
commit_comment_event_df  = spark.read.parquet(f"abfs://silver-layer@ade20220919.dfs.core.windows.net/miniro/commit_comment_event_df")
issue_comment_event_df = spark.read.parquet(f"abfs://silver-layer@ade20220919.dfs.core.windows.net/miniro/issue_comment_event_df")
pull_request_review_comment_event_df = spark.read.parquet(f"abfs://silver-layer@ade20220919.dfs.core.windows.net/miniro/pull_request_review_comment_event_df")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Type of GitHub event per hour

# COMMAND ----------

# converting the UTC time stamp to a more readable format 
# and saving it to its own table
time_event_df = main_df\
    .select("id", "type", "created_at")\
    .withColumn("hour", 
            F.date_format(F.to_timestamp(F.col("created_at")),
            "MM/dd/yyyy HH:00:00"))

# grouping the hour and type columns and ordering by hour
# and then getting the count of each event type for each hour 
event_per_hour_df = time_event_df \
    .select("id", "type", "created_at", "hour") \
    .groupBy("hour") \
    .pivot("type") \
    .count() \
    .drop('type') \
    .orderBy("hour")

event_per_hour_df.display()

# COMMAND ----------

# writing to a file in gold 
event_per_hour_df.write.mode("overwrite") \
    .parquet('abfs://gold-layer@ade20220919.dfs.core.windows.net/miniro/event_per_hour_df')

# COMMAND ----------

# MAGIC %md
# MAGIC ## PushEvents on main/gh-pages/other branches

# COMMAND ----------

# PushEvent data aggregated by ref type – whether the commit is on the main branch 
push_event_ref_df = push_event_df \
    .withColumn("day", F.date_format(F.to_timestamp(F.col("created_at")),
                "MM/dd/yyyy")) \
    .withColumn("ma", (F.split(push_event_df.ref, "/")[2] == "master")) \
    .withColumn("gp", (F.split(push_event_df.ref, "/")[2] == "gh-pages")) \
    .withColumn("ot", ((F.split(push_event_df.ref, "/")[2] != "master") &
                        (F.split(push_event_df.ref, "/")[2] != "gh-pages"))) \
    .groupBy(F.col("day")) \
    .agg( \
        (F.sum(F.col("ma").cast("long"))).alias("main"), \
        (F.sum(F.col("gp").cast("long"))).alias("gh-pages"), \
        (F.sum(F.col("ot").cast("long"))).alias("other")) \
    .orderBy(F.col("day").asc())

push_event_ref_df.display()

# COMMAND ----------

# writing to a file in gold
push_event_ref_df.write.mode("overwrite") \
    .parquet('abfs://gold-layer@ade20220919.dfs.core.windows.net/miniro/push_event_ref_df')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Number of commits per PushEvent

# COMMAND ----------

# creating a number of commits data frame with
# formatted time 
# use distinct_size instead since commits array counts for the 20 most recent commits 

num_commits_df = push_event_df.join(main_df.drop("created_at"), "id")

num_commits_df = num_commits_df \
    .withColumn("branch",F.split(push_event_df.ref, "/" [2])\
    .select(\
        F.col("id").alias("fact_id"), "push_id",
        F.date_format(F.to_timestamp(F.col("created_at")), "MM/dd/yyyy HH:mm:ss").alias("time"),
        F.col("actor_display_login").alias("user"), 
        F.col("org_login").alias("organization"), 
        F.col("repo_name").alias("repository"),
        F.col("branch"),
        F.col("distinct_size").alias('number_commits'))
        

num_commits_df.display()

# COMMAND ----------

# save num_commits_df to parquet 
num_commits_df.write.mode('overwrite') \
    .parquet('abfs://gold-layer@ade20220919.dfs.core.windows.net/miniro/num_commits_df')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## User user activity by week

# COMMAND ----------

# displaying our main dataframe 
main_df.select("actor_id", "actor_display_login", "type", "created_at") \
    .display()

# COMMAND ----------

 # formatting the UTC timestamp 
 # Grouping by the week of the year
 # filling nulls with zero 
user_type_week = main_df \
    .select("actor_id", "actor_display_login", 
            "type", "created_at") \
    .withColumn("created_at", F.to_timestamp("created_at")) \
    .withColumn("date", F.to_date("created_at")) \
    .withColumn("week_of_year", F.date_format(F.to_date("date", "dd/MMM/yyyy"), "w")) \
    .groupBy('week_of_year', 'actor_display_login') \
    .pivot('type') \
    .count() \
    .fillna(0) \
    .orderBy('week_of_year',  'actor_display_login') \
    .withColumnRenamed('actor_display_login', 'user')

# COMMAND ----------

# example of filtering by user 
user_type_week.where(F.col('user') == 'torodev').display()

# COMMAND ----------

# save user_type_date to parquet
user_type_week.write.mode("overwrite") \
    .parquet('abfs://gold-layer@ade20220919.dfs.core.windows.net/miniro/user_type_week')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Breakdown of activity by project – find a unique use case

# COMMAND ----------

## Joining Push Event Table with Fact Table
repo_push_df = push_event_df \
    .select(F.date_format(
                F.to_timestamp("created_at"), "MM/dd/yyyy HH:00:00") \
                .alias('created_at'), 'id', 'push_id', 'distinct_size') \
    .join(
        main_df.select(
            'id', 'repo_name', "repo_id", 'actor_login') \
            .withColumn("repo_owner", F.split("repo_name", '\/')[0]) \
            .withColumn("project_name", F.split("repo_name", '\/')[1]) \
            .drop("repo_name"), 'id', 'left')

## Total number of push events by user, by project
# and when their first/last push was for that project
agg_push_df = repo_push_df \
    .groupBy('actor_login', 'repo_id', 'repo_owner', 'project_name') \
    .agg(F.sum("distinct_size").alias('number_of_push_events'),\
            F.max('created_at').alias('date_of_last_push'),
            F.min('created_at').alias('date_of_first_push'))

agg_push_df.display()

# COMMAND ----------

# save user activity push data
agg_push_df.write.mode('overwrite') \
    .parquet('abfs://gold-layer@ade20220919.dfs.core.windows.net/miniro/user_push_activity')

# COMMAND ----------

## You can use the aggregation to filter by project name to see what are the most active users by Push Event
# and what did their most recent push event did to the project based on the commit messages 
# for projects that are about any topic, such as "pokemon"
agg_push_df.filter(F.col("project_name").contains("pokemon")).display()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Add language to each event based on commit messages

# COMMAND ----------

### code and language dataframe for the nlp model
codes = ['ar', 'be', 'bg', 'cs', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'fa', 'fi', 'fr', 
    'he', 'hi', 'hu', 'ia', 'id', 'is', 'it', 'ja', 'ko', 'la', 'lt', 'lv', 'mk', 'mr', 'nl', 
    'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sv', 'tl', 'tr', 'tt', 'uk', 'vi', 'zh', 'Unknown']
langs = ['English', 'English', 'English', 'English', 'English', 'English', 'English', 'English', 
    'English', 'Spanish', 'English', 'English', 'English', 'English', 'English', 'English', 'English', 
    'English', 'English', 'English', 'English', 'Japanese', 'Korean', 'English', 'English', 'English', 
    'English', 'English', 'English', 'Polish', 'Portuguese', 'English', 'Russian', 'English', 'English', 
    'English', 'English', 'English', 'English', 'English', 'English', 'Vietnamese', 'Chinese', 'Unknown']

annuaire = spark.createDataFrame(zip(codes, langs), ['code','language'])
annuaire.display()

# COMMAND ----------

### 
main = main_df.select("created_at", "id", "type", "actor_login", "org_login")

# Select the event id and the comment for all Dataframes event types that have comments 
commits = commit_comment_event_df.select("id", "comment_body") \
    .withColumnRenamed("comment_body", "comment")
issues = issue_comment_event_df.select("id", "comment_body") \
    .withColumnRenamed("comment_body", "comment")
push = push_event_df_commits_df.groupby("fact_id") \
    .agg(F.concat_ws(',',F.collect_list("message")) \
    .alias("comment")).withColumnRenamed("fact_id","id").select("id","comment")
pull = pull_request_review_comment_event_df \
    .select("id","comment_body") \
    .withColumnRenamed("comment_body", "comment")

# Union all the id/comment dataframes for all the events with comments
comment_data = commits.union(issues).union(push).union(pull)


filt_main = main \
    .where((main.type == "PushEvent") |
        (main.type == "IssueCommentEvent") |
        (main.type == "CommitCommentEvent") |
        (main.type == "PullRequestReviewCommentEvent"))
all_comments = filt_main.join(comment_data,"id")
all_comments.persist(StorageLevel.DISK_ONLY)

# COMMAND ----------

# initializing document assembler 
documentAssembler = DocumentAssembler()\
.setInputCol("comment")\
.setOutputCol("document")

#initializing language detector 
language_detector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_43", "xx")\
.setInputCols(["document"])\
.setOutputCol("lang")\
.setThreshold(0.1)\
.setCoalesceSentences(True)

languagePipeline = Pipeline(stages=[
 documentAssembler, 
 language_detector
])

results = languagePipeline.fit(all_comments).transform(all_comments)


ready_results = results.select("id",results.lang.result[0]).withColumnRenamed("lang.result[0]","code")

event_languages = all_comments.join(ready_results,"id").join(annuaire,"code", "left")


event_languages.display()

# COMMAND ----------

event_languages.persist(StorageLevel.DISK_ONLY)

# COMMAND ----------

language_stats = event_languages.groupBy('type') \
    .pivot('language') \
    .count() \
    .fillna(0) \
    .orderBy('type') \
    .withColumnRenamed('actor_display_login', 'user')

language_stats.persist(StorageLevel.DISK_ONLY)

display(language_stats.drop("null"))

# COMMAND ----------

display(language_stats.drop("null"))

# COMMAND ----------

# writing to a file in gold 
event_languages.write.mode("overwrite") \
    .parquet('abfs://gold-layer@ade20220919.dfs.core.windows.net/miniro/event_languages_df')

# COMMAND ----------

# writing to a file in gold 
language_stats.write.mode("overwrite") \
    .parquet('abfs://gold-layer@ade20220919.dfs.core.windows.net/miniro/language_stats_df')

# COMMAND ----------

