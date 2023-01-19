# GitHub_pipeline

This was a collaborative ETL project with two great coworkers using GitHub's [GH Archive events](https://www.gharchive.org/) that holds data of the 20+ event types. 
All files were done in Databricks notebooks using PySpark.
- ```Bronze layer``` holds a script that was done to ingest the data and partition it no more than ```178 MB``` in size as parquet files.
- ```silver layer``` holds a script that completely flattened the data.
- ```gold layer``` holds the script of all aggregations done. 
