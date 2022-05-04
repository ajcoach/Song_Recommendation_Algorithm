# -*- coding: utf-8 -*-
"""
Created on Thurs Oct 27 12:41:07 2021

@author: ajc364
"""

from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics


### Question 2 a ###


# Modeling

als = ALS(maxIter = 5, regParam = 0.01, userCol = "user_id_encoded", itemCol = "song_id_encoded", ratingCol = "plays", implicitPrefs = True)
als_model = als.fit(training)
predictions = als_model.transform(test)

predictions = predictions.orderBy(col("user_id"), col("song_id"), col("prediction").desc())
predictions.cache()

predictions.show(50, False)

""" Output
+------------------+----------------------------------------+-----+---------------+---------------+------------+
|song_id           |user_id                                 |plays|user_id_encoded|song_id_encoded|prediction  |
+------------------+----------------------------------------+-----+---------------+---------------+------------+
|SOBZFSZ12A8C13F2CA|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |303023.0       |330.0          |0.08651016  |
|SODHJHX12A58A7D24C|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|2    |303023.0       |462.0          |0.12126012  |
|SOGIDSA12A8C142829|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|3    |303023.0       |674.0          |0.08228835  |
|SOIXKRK12A8C140BD1|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |303023.0       |717.0          |0.05746509  |
|SOKUECJ12A6D4F6129|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |303023.0       |175.0          |0.12478107  |
|SOOLKLP12AF729D959|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|2    |303023.0       |980.0          |0.07161901  |
|SOPDRWC12A8C141DDE|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |303023.0       |287.0          |0.12963712  |
|SOPHBRE12A8C142825|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|2    |303023.0       |917.0          |0.075198606 |
|SOQFXDQ12AF72AD0EE|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |303023.0       |1997.0         |0.02191177  |
|SOSNTSY12AF72A7B43|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |303023.0       |533.0          |0.076976895 |
|SOTEFFR12A8C144765|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |303023.0       |375.0          |0.08605776  |
|SOUGLUN12A8C14282A|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |303023.0       |4527.0         |0.030778922 |
|SOWRMTT12A8C137064|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |303023.0       |781.0          |0.08418861  |
|SOYEQLD12AB017C713|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |303023.0       |1210.0         |0.047819152 |
|SOZORGY12A8C140382|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |303023.0       |1063.0         |0.03695368  |
|SOBJYFB12AB018372D|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |224086.0       |6074.0         |9.410853E-4 |
|SOIITTN12A6D4FD74D|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |224086.0       |25279.0        |2.204965E-4 |
|SOKHEEY12A8C1418FE|00007ed2509128dcdd74ea3aac2363e24e9dc06b|2    |224086.0       |5221.0         |0.0015751481|
|SOLJSMV12A8C13B2D9|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |224086.0       |44012.0        |1.2881214E-4|
|SOMUZHL12A8C130AFE|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |224086.0       |25702.0        |2.344354E-4 |
|SOOKJWB12A6D4FD4F8|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |224086.0       |25801.0        |4.3351E-4   |
|SOPIROE12A6D4FD4EB|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |224086.0       |54660.0        |1.4007799E-4|
|SOQPGMT12AF72A0865|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |224086.0       |47645.0        |8.5603344E-5|
|SORPSOF12AB0188C39|00007ed2509128dcdd74ea3aac2363e24e9dc06b|2    |224086.0       |47933.0        |1.0235231E-4|
|SOWYFRZ12A6D4FD507|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |224086.0       |62218.0        |4.7998452E-5|
|SOEWPBR12A58A79271|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |303024.0       |36716.0        |1.3001625E-4|
|SOJAXPH12AB017FC6F|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |303024.0       |16114.0        |4.6647133E-4|
|SOLLIRG12B35055B4C|00009d93dc719d1dbaf13507725a03b9fdeebebb|2    |303024.0       |7037.0         |6.631243E-4 |
|SONLHMZ12A58A7B141|00009d93dc719d1dbaf13507725a03b9fdeebebb|3    |303024.0       |88160.0        |2.999117E-5 |
|SOUKCML12A67020120|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |303024.0       |13003.0        |8.5876544E-4|
|SOVIYDJ12A8C13BFE2|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |303024.0       |29960.0        |1.3157935E-4|
|SOXQGZZ12AB0187A96|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |303024.0       |42371.0        |2.9216626E-5|
|SOYBKUE12A8C13BFEA|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |303024.0       |41321.0        |6.3987885E-5|
|SOYXCKN12AB018058C|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |303024.0       |86391.0        |2.2046272E-5|
|SOZGYIQ12AB01834BF|00009d93dc719d1dbaf13507725a03b9fdeebebb|5    |303024.0       |79587.0        |4.01862E-5  |
|SOCRUVF12A6D4F5906|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |73320.0        |3401.0         |0.0060139913|
|SOEGJGK12A8C143C5A|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |73320.0        |96561.0        |8.836728E-5 |
|SOEHDIY12A58A78EE6|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |73320.0        |12821.0        |0.002591215 |
|SOEIXYS12A6D4F8109|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|2    |73320.0        |36705.0        |6.125644E-4 |
|SOEWMIM12A6D4F7982|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |73320.0        |27192.0        |0.0017445799|
|SOGDTQS12A6310D7D1|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |73320.0        |4130.0         |0.0073519098|
|SOGUQCQ12A58A78C1E|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |73320.0        |58399.0        |3.1608052E-4|
|SOHPIUC12AB018046E|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|3    |73320.0        |61471.0        |1.4085621E-4|
|SOHTWNJ12A6701D0EB|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|8    |73320.0        |56056.0        |6.332155E-4 |
|SOIMBGJ12A6D4F828D|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|2    |73320.0        |55701.0        |2.4461802E-4|
|SOJAYEY12AB0185304|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |73320.0        |45702.0        |3.7472532E-4|
|SOJHEQO12A670203C4|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |73320.0        |69573.0        |2.963182E-4 |
|SOKBDRO12A67020ED9|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |73320.0        |13171.0        |0.0036668468|
|SOKZBJA12AB018B10B|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |73320.0        |89182.0        |1.4951336E-4|
|SOLIZKC12A67ADA232|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |73320.0        |8513.0         |0.0023506726|
+------------------+----------------------------------------+-----+---------------+---------------+------------+
only showing top 50 rows
"""


### Question 2 b ###


# Metrics

k = 100

def extract_songs_top_k(x, k):
    x = sorted(x, key=lambda x: -x[1])
    return [x[0] for x in x][0:k]

extract_songs_top_k_udf = udf(lambda x: extract_songs_top_k(x, k), ArrayType(IntegerType()))

def extract_songs(x):
    x = sorted(x, key=lambda x: -x[1])
    return [x[0] for x in x]

extract_songs_udf = udf(lambda x: extract_songs(x), ArrayType(IntegerType()))

users = test.select(als.getUserCol()).distinct().limit(10)
users.cache()
userSubsetRecs = als_model.recommendForUserSubset(users, k)

recommended_songs = (
    userSubsetRecs
    .withColumn("recommended_songs", extract_songs_top_k_udf(col("recommendations")))
    .select("user_id_encoded", "recommended_songs")
)

recommended_songs.cache()
recommended_songs.count()

""" Output
10
"""


recommended_songs.show(10, 100)

""" Output
+---------------+----------------------------------------------------------------------------------------------------+
|user_id_encoded|                                                                                   recommended_songs|
+---------------+----------------------------------------------------------------------------------------------------+
|         216316|[15, 63, 13, 59, 154, 92, 10, 190, 46, 41, 30, 75, 131, 39, 120, 8, 277, 38, 278, 102, 252, 109, ...|
|         152553|[11, 0, 7, 184, 16, 35, 52, 582, 87, 388, 47, 48, 2, 414, 3, 34, 25, 144, 23, 309, 126, 269, 1213...|
|           8563|[11, 91, 184, 89, 282, 203, 93, 344, 136, 226, 82, 71, 7, 48, 744, 37, 144, 582, 904, 140, 16, 20...|
|         169513|[28, 21, 20, 52, 15, 16, 140, 23, 334, 50, 63, 30, 229, 59, 95, 124, 473, 308, 92, 41, 190, 53, 4...|
|          64181|[15, 63, 190, 13, 277, 153, 154, 59, 75, 252, 10, 92, 131, 334, 41, 165, 30, 189, 213, 8, 194, 20...|
|          95827|[56, 71, 100, 188, 136, 89, 48, 449, 128, 227, 93, 222, 217, 0, 234, 260, 203, 82, 257, 481, 572,...|
|         304342|[221, 233, 237, 207, 241, 245, 249, 255, 317, 261, 319, 336, 307, 340, 281, 143, 357, 56, 72, 136...|
|          99912|[56, 100, 136, 71, 89, 93, 188, 82, 227, 222, 143, 248, 260, 61, 4513, 217, 203, 192, 449, 234, 3...|
|          74739|[11, 0, 7, 25, 2, 5, 18, 48, 37, 47, 3, 34, 87, 49, 35, 32, 23, 147, 16, 184, 1, 106, 126, 64, 11...|
|          75408|[11, 37, 145, 72, 7, 184, 2, 91, 5, 73, 25, 3, 353, 64, 89, 207, 249, 135, 130, 233, 93, 221, 237...|
+---------------+----------------------------------------------------------------------------------------------------+
"""

relevant_songs = (
    test
    .select(
        col("user_id_encoded").cast(IntegerType()),
        col("song_id_encoded").cast(IntegerType()),
        col("plays").cast(IntegerType())
    )
    .groupBy('user_id_encoded')
    .agg(
        collect_list(
            array(
                col("song_id_encoded"),
                col("plays")
            )
        ).alias('relevance')
    )
    .withColumn("relevant_songs", extract_songs_udf(col("relevance")))
    .select("user_id_encoded", "relevant_songs")
    .join(users, ['user_id_encoded'], 'inner')
)

relevant_songs = relevant_songs.join(users, ['user_id_encoded'], 'inner')
relevant_songs.cache()
relevant_songs.count()

""" Output
10
"""


relevant_songs.show(10, 100)

""" Output
+---------------+----------------------------------------------------------------------------------------------------+
|user_id_encoded|                                                                                      relevant_songs|
+---------------+----------------------------------------------------------------------------------------------------+
|         216316|[5842, 31483, 3821, 36221, 5484, 8272, 24405, 1958, 43358, 23307, 12971, 74564, 69283, 7650, 8053...|
|         152553|[64372, 5381, 388, 7274, 2028, 19940, 930, 17459, 83368, 14032, 1491, 26872, 19313, 3115, 109138,...|
|           8563|[2480, 13849, 6836, 1442, 2462, 128, 899, 591, 3571, 808, 4465, 17583, 33959, 14804, 189, 2359, 7...|
|         169513|[29515, 65980, 10662, 108550, 21165, 57470, 39401, 72837, 1508, 108695, 100198, 20144, 86420, 113...|
|          64181|[60583, 42834, 75, 2874, 35858, 257, 3101, 15559, 6593, 13581, 79739, 66173, 3907, 298, 1513, 140...|
|          95827|[4421, 2713, 11680, 21219, 8652, 33476, 23064, 10287, 21214, 10479, 56583, 36490, 12749, 27622, 6...|
|         304342|[28324, 13686, 336, 49968, 35276, 60115, 909, 1133, 17679, 2793, 11838, 13991, 56904, 117826, 805...|
|          99912|[5323, 22431, 869, 9178, 2901, 22466, 43060, 715, 17903, 24869, 12108, 14839, 4889, 2243, 3568, 2...|
|          74739|[44008, 26567, 11245, 29049, 737, 57788, 15878, 8276, 7, 7758, 9477, 46411, 2349, 11200, 35104, 4...|
|          75408|[42255, 69584, 88723, 26193, 2080, 1040, 22905, 99257, 221, 85402, 9689, 8822, 1972, 91454, 7907,...|
+---------------+----------------------------------------------------------------------------------------------------+
"""


combined = (
    recommended_songs.join(relevant_songs, on = 'user_id_encoded', how = 'inner').rdd
    .map(lambda row: (row[1], row[2]))
)
combined.cache()
combined.count()

""" Output
10
"""


combined.take(1)

""" Output
[([15,
   63,
   13,
   59,
   154,
   92,
   10,
   190,
   46,
   41,
   30,
   75,
   131,
   39,
   120,
   8,
   277,
   38,
   278,
   102,
   252,
   109,
   36,
   66,
   113,
   247,
   254,
   177,
   57,
   105,
   156,
   114,
   29,
   76,
   176,
   94,
   101,
   366,
   51,
   27,
   189,
   153,
   211,
   213,
   32,
   193,
   69,
   139,
   166,
   194,
   168,
   123,
   127,
   200,
   86,
   163,
   159,
   26,
   558,
   165,
   199,
   152,
   214,
   250,
   140,
   807,
   334,
   268,
   28,
   170,
   235,
   352,
   62,
   315,
   151,
   141,
   229,
   308,
   201,
   58,
   313,
   205,
   65,
   306,
   367,
   376,
   53,
   160,
   271,
   181,
   262,
   129,
   473,
   794,
   526,
   251,
   708,
   436,
   2229,
   463],
  [5842,
   31483,
   3821,
   36221,
   5484,
   8272,
   24405,
   1958,
   43358,
   23307,
   12971,
   74564,
   69283,
   7650,
   80539,
   26140,
   67984,
   46411,
   22050,
   14099,
   58095,
   53488,
   89626,
   24658])]
"""



### Question 2 c ###


ranking_metrics = RankingMetrics(combined)
precision_at_10 = ranking_metrics.precisionAt(10)
print(precision_at_10)

""" Output
0.05
"""


ndcg_at_10 = ranking_metrics.ndcgAt(10)
print(ndcg_at_10)

""" Output
0.045718160765461056
"""


mean_average_precision = ranking_metrics.meanAveragePrecision
print(mean_average_precision)

""" Output
0.010113885122600193
"""


