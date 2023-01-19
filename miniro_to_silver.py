# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer to Silver Layer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing modules and setting up oAuth for Azure

# COMMAND ----------

import datetime
import re
from pyspark.sql import types as T 
from pyspark.sql import functions as F 
from pyspark import StorageLevel

# COMMAND ----------

### Getting oAuth for Azure Container
service_credential = dbutils.secrets.get(scope="databricks",key="databricks-test")
spark.conf.set("fs.azure.account.auth.type.ade20220919.dfs.core.windows.net", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type.ade20220919.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id.ade20220919.dfs.core.windows.net", "98a69314-1a2f-4eef-9b09-6ff1e73ead68")
spark.conf.set("fs.azure.account.oauth2.client.secret.ade20220919.dfs.core.windows.net", service_credential)
spark.conf.set("fs.azure.account.oauth2.client.endpoint.ade20220919.dfs.core.windows.net", "https://login.microsoftonline.com/33da9f3f-4c1a-4640-8ce1-3f63024aea1d/oauth2/token")

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Defining Event Table types

# COMMAND ----------

### List of Event Table types, we remove PublicEvent because it contains no columns
type_list_camel = ['PushEvent', 
             'GollumEvent', 
             'ReleaseEvent', 
             'CommitCommentEvent', 
             'CreateEvent', 
             'PullRequestReviewCommentEvent', 
             'IssueCommentEvent', 
             'DeleteEvent', 
             'IssuesEvent', 
             'ForkEvent',
             'MemberEvent', 
             'WatchEvent', 
             'PullRequestEvent']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions for flattening & cleaning bronze layer

# COMMAND ----------

### Helper functions for flattening out data into a Fact Table and 14 Seperate Event Tables 
### with any array columns into their own separate tables

## Flattening JSON Struct Functions --------------------------------------------------------
def event_df_flatten(big_df, event_type, limit_rows=True, L=500):    
    '''
    Extracts 'id', 'created_at', and 'payload' for one Event Table
    from the Fact Table with 'payload' being nested

    Flattens the event dataframe once, which flattens the payload column by one layer
    then any null columns are dropped
    after it contains the only payload columns that have data in it
    then the dataframe is completely flattened 

    Limits rows by L if creating a template to save storage,
    if template behavior not wanted, it does not limit the rows read

    Params:
        big_df : Dataframe
            dataframe containing unnested columns with json structs in the 'payload' column
        event_type : str
            type of event that needs to be extracted and flattened into its own table
        limit_rows : bool
            primarlily used for making templates, 
            if True limits the amount of rows that are read in from the big_df
            if False it reads in all the rows from big_df 
        L : int
            number of rows that are read in from big_df if limit_rows is True

    Returns:
        Dataframe
            Completely null column removed and flattened Event Table
    '''
    
    # Filter out specific event type payload and flatten out one nested layer
    if limit_rows:
        event_df = big_df.filter(big_df.type == event_type).select('id','created_at','payload').limit(L)
    else:
        event_df = big_df.filter(big_df.type == event_type).select('id','created_at','payload')
    
    # Flatten one layer of payload, don't keep 'payload' in the column name
    event_df = flatten_df(event_df, False)
    
    # Find and drop any columns that are completely full of null values
    null_counts = event_df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in event_df.columns]).collect()[0].asDict()
    na_cols = [k for k in null_counts.keys() if null_counts[k] == L]   
    event_df = event_df.drop(*na_cols)
    
    # Completely flatten all layers of the dataframe (other than the arrays)
    return complete_flatten(event_df)


def flatten_df(nested_df, keep_parent_name = True):  
    '''
    Flattens one nested layer for every nested column in a DataFrame.

    Ex: actor contains a schema for id, login, and url nested inside -> Flattens to actor_id, actor_login, actor_url

    Params:
        nested_df : DataFrame
            DataFrame with nested information within columns
    Returns: 
        flat_df : DataFrame
            DataFrame with flattened columns
    '''
    
    # store the column names that contain structs as their data
    # in dtypes, 0 index is the col name, 1 index is the actual data
    flat_cols = [F.col(x[0]) for x in nested_df.dtypes if not re.match('^struct',x[1])] 
    nested_cols = [x[0] for x in nested_df.dtypes if re.match('^struct',x[1])]
    
    if keep_parent_name:
        # flatten nested columns using select(.*)
        # names flattened columns as "parentCol_childCol"
        flat_df = nested_df.select(flat_cols + [F.col(n+'.'+c).alias(n+'_'+c) 
                                                for n in nested_cols 
                                                for c in nested_df.select(n+'.*').columns])
    else:
        # names flattened columns as "childCol"
        flat_df = nested_df.select(flat_cols + [F.col(n+'.'+c).alias(c) 
                                                for n in nested_cols 
                                                for c in nested_df.select(n+'.*').columns])
        
    return flat_df


def complete_flatten(nested_df):
    '''
    Flattens all nested layers in a DataFrame

    Params:
        nested_df : DataFrame
            DataFrame with nested information within columns
    Returns: 
        flat_df : DataFrame
            DataFrame with completely flattened columns
    '''
    
    flat_df = nested_df
    # flatten until no columns are struct datatype
    while any([re.match('^struct',x[1]) for x in flat_df.dtypes]):
        flat_df = flatten_df(flat_df)
    
    return flat_df


## Template Function  ------------------------------------------------------------------------------
def templatize(new_df, template_df_name, path):
    '''
    Reformats an Event or Array Table to contain only the columns from a Template DataFrame 

    Params:
        new_df : DataFrame
            Event or Array table to be reformatted by dropping columns from a Template DataFrame
        model_df_name : str
            name of Template DataFrame file
        path : str
            folder path to silver_layer
    Returns:
        new_df : DataFrame
            Event or Array Dataframe matching the table structure of the Template DataFrame
    '''
    
    # Read in Template dataframe
    template_df = spark.read.parquet(path + model_df_name)
    
    # add null columns for columns that are in the template but not in the new_df
    # this doesn't get used for project 2 but is for future-proofing the code
    select_cols = new_df.columns + [F.lit(None).alias(c) for c in template_df.columns if c not in new_df.columns]
    new_df = new_df.select(*select_cols)
    
    # Select all the corresponding columns in the input Dataframe that are in the Template dataFrame
    new_df = new_df.select(template_df.columns)
    
    return new_df


## Writing Seperate Event and Array Dataframe Files Functions ----------------------------------------------
def save_array_table(event_df, event_df_name, template, L, partition_by, path):
    '''
    Extracts Dimension Array Tables from an Event Table
    flattens all structs in the Array Tables 
    writes Array Tables to a parquet file

    Params:
        event_df : DataFrame
            Event Table dataframe
        event_df_name : str
            name of Event Table with underscores between each word
        template : bool
            if True saves L rows after dropping any unhelpful columns using drop_useless()
            if False saves Event Table to a parquet file 
        L : int
            number of rows to save to a Template Event Table if template is True
        partition_by : str
            if template is false then the Event Table parquet file is partitioned by this column
        path : str
            save path for the file
    Returns:
        DataFrame
            Event Table with the array columns dropped
    '''
    
    # Find any columns that are arrays
    array_cols = [x[0] for x in event_df.dtypes if re.match('^array',x[1])]
    
    # If there are array columns, flatten them
    if len(array_cols) > 0:
        for a_col in array_cols:
            array_df_name = event_df_name + '_' + a_col + '_df'
            # Keep id as fact_id to tie back to Fact Table
            # Keep created_at as event_created_at to be able to partition arrays by day
            # explode array and flatten it
            array_df = flatten_df(
                event_df.select(F.col('id').alias('fact_id'),
                F.col('created_at').alias('event_created_at'),
                F.explode(a_col)), False)
            # Make sure there are no structs left
            array_df = complete_flatten(array_df)

            if template:
                # Save only L rows, drop any columns defined useless from drop_useless()
                array_df = array_df.limit(L)
                event_df = drop_useless(event_df)
                array_df.write.mode('overwrite') \
                .parquet(path + array_df_name)
            else:
                # Choose only the columns deemed important using the Template Dataframe
                array_df = templatize(array_df, array_df_name, path)
                array_df = add_day(array_df, 'event_created_at')
                array_df.write.partitionBy(partition_by).mode('overwrite') \
                .parquet(path + array_df_name)

    # Drop array columns and return so Event Table doesn't contain repeat information from Array Tables     
    return event_df.drop(*array_cols)


def save_event_table(event_df, event_df_name, template, L, partition_by, path):
    '''
    Writes Event Table to a parquet file

    Params:
        event_df : DataFrame
            Event Table dataframe
        event_df_name : str
            name of Event Table with underscores between each word
        template : bool
            if True saves L rows after dropping any unhelpful columns using drop_useless()
            if False saves Event Table to a parquet file 
        L : int
            number of rows to save to a Template Event Table if template is True
        partition_by : str
            if template is false then the Event Table parquet file is partitioned by this column
        path : str
            save path for the file

    ''' 
    
    if template:
        # Save only L rows, drop any columns defined useless from drop_useless()
        event_df = event_df.limit(L)
        event_df = drop_useless(event_df)
        event_df.write.mode('overwrite').parquet(path + event_df_name)
    else:
        event_df.write.partitionBy(partition_by).mode('overwrite') \
        .parquet(path + event_df_name)
     

## Cleaning/Partitioning Helper Functions ------------------------------------------------------
def drop_useless(df, keep_list=[]):
    '''
    Returns a dataframe without columns contain "gravatar" or that end in "url"
    url or gravatar columns can be kept if specified

    Params:
        df : DataFrame
            dataframe to transform
        keep_list : list of str
            list of columns that contain "gravatar" or end in "url" that should be kept
    Returns:
        DataFrame with "gravatar" and "url" columns removed
    '''
    df = df.drop(*[col for col in df.columns if 
                    (col[-3:] == "url" or "gravatar" in col) and 
                    (col not in keep_list)])
    return df

def add_day(df, date_col):
    '''
    Adds a day column to a DataFrame given a date column

    Params:
        df : DataFrame
            dataframe to transform
        date_col : str
            string containing the column name with the date in it
    Returns:
        DataFrame
            dataframe with day column
    '''
    return df.withColumn("day", F.col(date_col).substr(1,10))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating Templates

# COMMAND ----------

### Creating Template DataFrame Files
template_path = 'abfs://silver-layer@ade20220919.dfs.core.windows.net/miniro/templates/'
templates_exist = len(dbutils.fs.ls(template_path)) != 0

if not templates_exist:
    # Read in just one day to make a template
    template_file = f"abfs://bronze-layer@ade20220919.dfs.core.windows.net/miniro/2017-01/day=2017-01-01"
    row_limit = 500
    big_df = spark.read.parquet(template_file)

    # Make templates for each event type table
    for event_type in type_list_camel:
        event_df_name = re.sub(r'(?<!^)(?=[A-Z])', '_', event_type).lower() + "_df"

        # flatten, clean and save Event Table
        event_df = event_df_flatten(big_df, event_type, True, row_limit)
        # !! event tables are saved first on purpose here !!
        # templates need to include array cols in the Event Table so that they can be written later
        save_event_table(event_df, event_df_name, template=True, L=2, partition_by=None path=template_path)
        
        # flatten, clean and save Array Tables
        event_df = save_array_table(event_df, event_df_name, template=True, L=2, partition_by=None, path=template_path) 
    
    # Creating Fact Table as 'main_df'
    main_df = flatten_df(big_df.drop("payload"))
    main_df = main_df.limit(2)
    main_df = drop_useless(main_df)
    main_df.write.mode('overwrite').parquet('abfs://silver-layer@ade20220919.dfs.core.windows.net/miniro/templates/main_df')   

# COMMAND ----------

# MAGIC %md
# MAGIC ## Splitting Data into flat and cleaned Fact Table, 14 Event Tables and Dimension Tables

# COMMAND ----------

### Creating Fact, Event, and Dimension Tables
silver_path = 'abfs://silver-layer@ade20220919.dfs.core.windows.net/miniro/'

# Read in one month (can be changed to read in all the data)
big_df = spark.read.parquet(f"abfs://bronze-layer@ade20220919.dfs.core.windows.net/miniro/2017-01/")
big_df.cache()

# Writing Event Table and Dimension array Tables files
for event_type in type_list_camel:
    event_df_name = re.sub(r'(?<!^)(?=[A-Z])', '_', event_type).lower() + "_df"
    event_df = event_df_flatten(big_df, event_type, limit_rows = False)
    
    # Select only the columns that are in the template df
    event_df = templatize(event_df, event_df_name, template_path)
    event_df = add_day(event_df, 'created_at')
    event_df.cache()

    # Drop arrays and save them as seperate tables
    event_df = save_array_table(event_df, event_df_name, template=False, partition_by='day', 0, path=silver_path)

    # Save event table to parquet after removing unnecessary columns
    save_event_table(event_df, event_df_name, template=False, partition_by='day', 0, path=silver_path)
    event_df.unpersist()

# Flatten and clean Fact Table
main_df = flatten_df(big_df.drop("payload"))
main_df = templatize(main_df, "main_df", template_path)
main_df = add_day(main_df, 'created_at')

# Save Fact Table
main_df.write.partitionBy('day').mode('overwrite') \
.parquet(f'abfs://silver-layer@ade20220919.dfs.core.windows.net/miniro/main_df')
big_df.unpersist()

# COMMAND ----------

