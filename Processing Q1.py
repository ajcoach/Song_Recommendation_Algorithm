# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 12:41:51 2021

@author: ajc364
"""

# Increasing executers and cores
start_pyspark_shell -e 4 -c 2 -w 4 -m 4

### Question 1 a ###

# Python and pyspark modules required
import sys

from pyspark import SparkContext
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import *


# Required to allow the file to be submitted and run using spark-submit instead of using pyspark interactively
spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()


## /data/msd/audio/attributes/msd-jmir-area-of-moments-all-v1.0.attributes.csv
audio_attributes = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "true")
    .load("hdfs:////data/msd/audio/attributes/msd-jmir-area-of-moments-all-v1.0.attributes.csv")
    .limit(1000)
)


audio_attributes.printSchema()

""" Output
root
 |-- _c0: string (nullable = true)
 |-- _c1: string (nullable = true)
"""


audio_attributes.show(10, False)

""" Output
+----------------------------------------------------+----+
|_c0                                                 |_c1 |
+----------------------------------------------------+----+
|Area_Method_of_Moments_Overall_Standard_Deviation_1 |real|
|Area_Method_of_Moments_Overall_Standard_Deviation_2 |real|
|Area_Method_of_Moments_Overall_Standard_Deviation_3 |real|
|Area_Method_of_Moments_Overall_Standard_Deviation_4 |real|
|Area_Method_of_Moments_Overall_Standard_Deviation_5 |real|
|Area_Method_of_Moments_Overall_Standard_Deviation_6 |real|
|Area_Method_of_Moments_Overall_Standard_Deviation_7 |real|
|Area_Method_of_Moments_Overall_Standard_Deviation_8 |real|
|Area_Method_of_Moments_Overall_Standard_Deviation_9 |real|
|Area_Method_of_Moments_Overall_Standard_Deviation_10|real|
+----------------------------------------------------+----+
only showing top 10 rows
"""


audio_features = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "true")
    .load("hdfs:////data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/*.csv.gz")
    .limit(1000)
)


audio_features.printSchema()

""" Output
root
 |-- _c0: double (nullable = true)
 |-- _c1: double (nullable = true)
 |-- _c2: string (nullable = true)
 |-- _c3: double (nullable = true)
 |-- _c4: double (nullable = true)
 |-- _c5: double (nullable = true)
 |-- _c6: double (nullable = true)
 |-- _c7: double (nullable = true)
 |-- _c8: double (nullable = true)
 |-- _c9: double (nullable = true)
 |-- _c10: double (nullable = true)
 |-- _c11: double (nullable = true)
 |-- _c12: double (nullable = true)
 |-- _c13: double (nullable = true)
 |-- _c14: double (nullable = true)
 |-- _c15: double (nullable = true)
 |-- _c16: double (nullable = true)
 |-- _c17: double (nullable = true)
 |-- _c18: double (nullable = true)
 |-- _c19: double (nullable = true)
 |-- _c20: string (nullable = true)
"""


audio_statistics = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("hdfs:////data/msd/audio/statistics/*.csv.gz")
    .limit(1000)
)


audio_statistics.printSchema()

""" Output
root
 |-- track_id: string (nullable = true)
 |-- title: string (nullable = true)
 |-- artist_name: string (nullable = true)
 |-- duration: double (nullable = true)
 |-- 7digita_Id: integer (nullable = true)
 |-- sample_bitrate: integer (nullable = true)
 |-- sample_length: double (nullable = true)
 |-- sample_rate: integer (nullable = true)
 |-- sample_mode: integer (nullable = true)
 |-- sample_version: integer (nullable = true)
 |-- filesize: integer (nullable = true)
"""


genre = (
    spark.read.format("com.databricks.spark.csv")
    .option("delimiter", "\t")
    .option("header", "false")
    .option("inferSchema", "true")
    .load("hdfs:////data/msd/genre/*.tsv")
    .limit(1000)
)


genre.printSchema()

""" Output
root
 |-- _c0: string (nullable = true)
 |-- _c1: string (nullable = true)
"""


main_summary = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("hdfs:////data/msd/main/summary/*.csv.gz")
    .limit(1000)
)


main_summary.printSchema()

""" Output
root
 |-- analyzer_version: string (nullable = true)
 |-- artist_7digitalid: string (nullable = true)
 |-- artist_familiarity: string (nullable = true)
 |-- artist_hotttnesss: string (nullable = true)
 |-- artist_id: string (nullable = true)
 |-- artist_latitude: string (nullable = true)
 |-- artist_location: string (nullable = true)
 |-- artist_longitude: string (nullable = true)
 |-- artist_mbid: string (nullable = true)
 |-- artist_name: string (nullable = true)
 |-- artist_playmeid: string (nullable = true)
 |-- genre: string (nullable = true)
 |-- idx_artist_terms: string (nullable = true)
 |-- idx_similar_artists: string (nullable = true)
 |-- release: string (nullable = true)
 |-- release_7digitalid: string (nullable = true)
 |-- song_hotttnesss: string (nullable = true)
 |-- song_id: string (nullable = true)
 |-- title: string (nullable = true)
 |-- track_7digitalid: string (nullable = true)
"""


tasteprofile_mismatches = (
    spark.read.format("text")
    .option("header", "false")
    .option("inferSchema", "true")
    .load("hdfs:////data/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt")
    .limit(1000)
)


tasteprofile_mismatches.printSchema()

""" Output
root
 |-- value: string (nullable = true)
"""


tasteprofile_triplets = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("delimiter", "\t")
    .option("inferSchema", "true")
    .load("hdfs:////data/msd/tasteprofile/triplets.tsv/*.tsv.gz")
    .limit(1000)
)


tasteprofile_triplets.printSchema()

"""Output
root
 |-- _c0: string (nullable = true)
 |-- _c1: string (nullable = true)
 |-- _c2: integer (nullable = true)
"""


### Question 1 b ###

default_config = sc.getConf()
N = int(default_config.get("spark.executor.instances"))
M = int(default_config.get("spark.executor.cores"))

partitions = 4 * N * M


### Question 1 c ###

audio_attributes = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "true")
    .load("hdfs:////data/msd/audio/attributes/*.csv")
)

audio_attributes.count()

""" Output
Out[4]: 3929
"""

audio_features = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "true")
    .load("hdfs:////data/msd/audio/features/*/*.csv.gz")
)

audio_features.count()

""" Output
Out[6]: 12927867
"""

audio_statistics = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("hdfs:////data/msd/audio/statistics/*.csv.gz")
)

audio_statistics.count()

""" Output
Out[8]: 992865
"""

genre = (
    spark.read.format("com.databricks.spark.csv")
    .option("delimiter", "\t")
    .option("header", "false")
    .option("inferSchema", "true")
    .load("hdfs:////data/msd/genre/*.tsv")
)

genre.count()

""" Output
Out[10]: 1103077
"""


main_summary = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("hdfs:////data/msd/main/summary/*.csv.gz")
)

main_summary.count()

""" Output
Out[12]: 2000000
"""


tasteprofile_mismatches = (
    spark.read.format("text")
    .option("header", "false")
    .option("inferSchema", "true")
    .load("hdfs:////data/msd/tasteprofile/mismatches/*.txt")
)

tasteprofile_mismatches.count()

""" Output
Out[14]: 20032
"""


tasteprofile_triplets = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("delimiter", "\t")
    .option("inferSchema", "true")
    .load("hdfs:////data/msd/tasteprofile/triplets.tsv/*.tsv.gz")
)

tasteprofile_triplets.count()

""" Output
Out[16]: 48373586
"""


tasteprofile_triplets.select('_c1').distinct().count()

""" Output
384546
"""






 



 
