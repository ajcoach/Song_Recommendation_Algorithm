# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 20:08:32 2021

@author: ajc364
"""

from pyspark.ml.feature import StringIndexer


### Question 1 a ###

# Counts the number of unique songs in dataset "taste profiles"

triplets.select(countDistinct("song_id").alias("unique_songs")).show()

""" Output
+------------+
|unique_songs|
+------------+
|      384546|
+------------+
"""


# Counts number of unique songs unique songs after removing mismatched songs

triplets_not_mismatched.select(countDistinct("song_id").alias("unique_songs")).show()


""" Output
+------------+
|unique_songs|
+------------+
|      378310|
+------------+
"""


# Counts numner of unique users in dataset "taste profiles"

triplets.select(countDistinct("user_id").alias("unique_users")).show()

""" Output
+------------+
|unique_users|
+------------+
|     1019318|
+------------+
"""


### Question 1 b ###


user_play_count = (
    triplets_not_mismatched
    .groupBy("user_id")
    .agg(
        F.sum(col("plays")).alias("plays_count"),
        F.count(col("song_id")).alias("song_count"),
    )
    .orderBy(col("plays_count").desc())
    .cache()
)


user_play_count.show(truncate = False)

""" Output
+----------------------------------------+-----------+----------+
|user_id                                 |plays_count|song_count|
+----------------------------------------+-----------+----------+
|093cb74eb3c517c5179ae24caf0ebec51b24d2a2|13074      |195       |
|119b7c88d58d0c6eb051365c103da5caf817bea6|9104       |1362      |
|3fa44653315697f42410a30cb766a4eb102080bb|8025       |146       |
|a2679496cd0af9779a92a13ff7c6af5c81ea8c7b|6506       |518       |
|d7d2d888ae04d16e994d6964214a1de81392ee04|6190       |1257      |
|4ae01afa8f2430ea0704d502bc7b57fb52164882|6153       |453       |
|b7c24f770be6b802805ac0e2106624a517643c17|5827       |1364      |
|113255a012b2affeab62607563d03fbdf31b08e7|5471       |1096      |
|99ac3d883681e21ea68071019dba828ce76fe94d|5385       |939       |
|6d625c6557df84b60d90426c0116138b617b9449|5362       |1307      |
|6b36f65d2eb5579a8b9ed5b4731a7e13b8760722|5318       |145       |
|ec6dfcf19485cb011e0b22637075037aae34cf26|5146       |4316      |
|3325fe1d8da7b13dd42004ede8011ce3d7cd205d|5100       |149       |
|281deab3afccc906251ef67a8eda2b9f9baec459|5057       |336       |
|c1255748c06ee3f6440c51c439446886c7807095|4977       |1498      |
|18c1dd917693fd929e3f99dd7906c2aafe9ff17f|4883       |1062      |
|6a58f480d522814c087fd3f8c77b3f32bb161f9d|4764       |31        |
|3b93435988354b1889de1e71810d1dd65c4ba17c|4625       |1083      |
|c11dea7d1f4d227b98c5f2a79561bf76884fcf10|4356       |176       |
|31cbbdbd5a1a6ef64601dee9144c7c20494452a7|4320       |127       |
+----------------------------------------+-----------+----------+
only showing top 20 rows
"""


max_plays = user_play_count.agg({"plays_count": "max"}).collect()[0]['max(plays_count)']


songs_most_active_user = user_play_count.select('song_count').where(F.col('plays_count') == max_plays)

songs_most_active_user.show()

""" Output
+----------+
|song_count|
+----------+
|       195|
+----------+
"""


### Question 1 c ###


# Saving the above as a csv

user_play_count.write.mode("overwrite").csv('./Assignment2/user_play_count.csv')


# Saving the csv to local in HDFS

# In HDFS
hdfs dfs -copyToLocal ./Assignment2/user_play_count.csv ./
#


# Copying to local machine and combining to one csv

# In local Windows Command Prompt
cd "C:\Users\coach\OneDrive - University of Canterbury\MADS\DATA420 Assignment 2 Code\user_play_count.csv"
copy *.csv user_play_count.csv
#

""" Output
part-00000-e849c5ef-23ce-4c07-b2d8-b76f84159641-c000.csv
part-00001-e849c5ef-23ce-4c07-b2d8-b76f84159641-c000.csv
part-00002-e849c5ef-23ce-4c07-b2d8-b76f84159641-c000.csv
part-00003-e849c5ef-23ce-4c07-b2d8-b76f84159641-c000.csv
part-00004-e849c5ef-23ce-4c07-b2d8-b76f84159641-c000.csv
part-00005-e849c5ef-23ce-4c07-b2d8-b76f84159641-c000.csv
part-00006-e849c5ef-23ce-4c07-b2d8-b76f84159641-c000.csv
part-00007-e849c5ef-23ce-4c07-b2d8-b76f84159641-c000.csv
part-00008-e849c5ef-23ce-4c07-b2d8-b76f84159641-c000.csv
part-00009-e849c5ef-23ce-4c07-b2d8-b76f84159641-c000.csv
part-00010-e849c5ef-23ce-4c07-b2d8-b76f84159641-c000.csv
part-00011-e849c5ef-23ce-4c07-b2d8-b76f84159641-c000.csv
part-00012-e849c5ef-23ce-4c07-b2d8-b76f84159641-c000.csv
part-00013-e849c5ef-23ce-4c07-b2d8-b76f84159641-c000.csv
part-00014-e849c5ef-23ce-4c07-b2d8-b76f84159641-c000.csv
part-00015-e849c5ef-23ce-4c07-b2d8-b76f84159641-c000.csv
        1 file(s) copied.
"""



song_play_count = (
    triplets_not_mismatched
    .groupBy("song_id")
    .agg(
        F.count(col("user_id")).alias("user_count"),
        F.sum(col("plays")).alias("plays_count"),
    )
    .orderBy(col("plays_count").desc())
    .cache()
)


# Saving the above as a csv

song_play_count.write.mode("overwrite").csv('./Assignment2/song_play_count.csv')


# Saving the csv to local in HDFS

# In HDFS
hdfs dfs -copyToLocal ./Assignment2/song_play_count.csv ./
#


# Copying to local machine and combining to one csv

# In local Windows Command Prompt
cd "C:\Users\coach\OneDrive - University of Canterbury\MADS\DATA420 Assignment 2 Code\song_play_count.csv"
copy *.csv song_play_count.csv
#

""" Output
part-00000-22a8ee1e-51ce-41ff-8095-fba60977e3ab-c000.csv
part-00001-22a8ee1e-51ce-41ff-8095-fba60977e3ab-c000.csv
part-00002-22a8ee1e-51ce-41ff-8095-fba60977e3ab-c000.csv
part-00003-22a8ee1e-51ce-41ff-8095-fba60977e3ab-c000.csv
part-00004-22a8ee1e-51ce-41ff-8095-fba60977e3ab-c000.csv
part-00005-22a8ee1e-51ce-41ff-8095-fba60977e3ab-c000.csv
part-00006-22a8ee1e-51ce-41ff-8095-fba60977e3ab-c000.csv
part-00007-22a8ee1e-51ce-41ff-8095-fba60977e3ab-c000.csv
part-00008-22a8ee1e-51ce-41ff-8095-fba60977e3ab-c000.csv
part-00009-22a8ee1e-51ce-41ff-8095-fba60977e3ab-c000.csv
part-00010-22a8ee1e-51ce-41ff-8095-fba60977e3ab-c000.csv
part-00011-22a8ee1e-51ce-41ff-8095-fba60977e3ab-c000.csv
part-00012-22a8ee1e-51ce-41ff-8095-fba60977e3ab-c000.csv
part-00013-22a8ee1e-51ce-41ff-8095-fba60977e3ab-c000.csv
part-00014-22a8ee1e-51ce-41ff-8095-fba60977e3ab-c000.csv
part-00015-22a8ee1e-51ce-41ff-8095-fba60977e3ab-c000.csv
        1 file(s) copied.
"""




### Question 1 d ###


song_play_count.approxQuantile("plays_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)

""" Output
[1.0, 6.0, 28.0, 95.0, 726885.0]
"""


user_play_count.approxQuantile("song_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)

""" Output
[3.0, 14.0, 25.0, 48.0, 4316.0]
"""

# ------------------------
# Limiting Using Threshold
# ------------------------

song_play_count_threshold = 32
user_play_count_threshold = 26
 
# first filter with song_play_count_threshold 
triplets_limited = triplets_not_mismatched
triplets_limited = (
    triplets_limited
    .join(triplets_limited.groupBy("user_id").count().where(col("count") > song_play_count_threshold).select("user_id"), on="user_id", how = "inner")
)

# then filter with user_play_count_threshold 
triplets_limited = (
    triplets_limited
    .join(triplets_limited.groupBy("song_id").count().where(col("count") > user_play_count_threshold).select("song_id"), on = "song_id", how = "inner")
    .cache()
)
triplets_limited.count()

""" Output
33163595
"""
 

### Question 1 d ###

 
user_id_indexer = StringIndexer(inputCol = "user_id", outputCol = "user_id_encoded")
song_id_indexer = StringIndexer(inputCol = "song_id", outputCol = "song_id_encoded")
 
user_id_indexer_model = user_id_indexer.fit(triplets_limited)
song_id_indexer_model = song_id_indexer.fit(triplets_limited)

triplets_limited = user_id_indexer_model.transform(triplets_limited)
triplets_limited = song_id_indexer_model.transform(triplets_limited)

training, test = triplets_limited.randomSplit([0.7, 0.3])
test_not_training = test.join(training, on = "user_id", how = "left_anti")

training.cache()
test.cache()
test_not_training.cache()

test_not_training = test.join(training, on = "user_id", how = "left_anti")
test_in_train = test.join(test_not_training, on = "user_id", how = "left_anti")
test_in_train_clean = test_in_train.join(training, on = "user_id", how ="left_anti")

print(f"training:          {training.count()}")
print(f"test:              {test.count()}")
print(f"test_not_training: {test_not_training.count()}")
print(f"test_in_training: {test_in_train.count()}")
print(f"test_in_train_clean: {test_in_train_clean.count()}")

""" Output
training:          23215738
test:              9947857
test_not_training: 1
test_in_training: 9947856
test_in_train_clean: 0
""" 



test.count() / (training.count() + test.count())

""" Output
0.29996316744309537
"""
