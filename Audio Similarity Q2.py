# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:10:36 2021

@author: ajc364
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

### Question 2 a ###

# Removing feature 4, 7 and 9 as they are highly correlated with 3, 6 and 8 respectively

audio_features_genre = (
	audio_features_genre.drop("feature_0004")
	.drop("feature_0007")
	.drop("feature_0009")
)

audio_features_genre.show()

""" Output
+------------------+------------+------------+------------+------------+------------+------------+------------+-------------+
|          track_id|feature_0000|feature_0001|feature_0002|feature_0003|feature_0005|feature_0006|feature_0008|   genre_type|
+------------------+------------+------------+------------+------------+------------+------------+------------+-------------+
|TRAAABD128F429CF47|      0.1308|       9.587|       459.9|     27280.0|      0.2474|       26.02|     67790.0|     Pop_Rock|
|TRAABPK128F424CFDB|      0.1208|       6.738|       215.1|     11890.0|      0.4882|       41.76|    220400.0|     Pop_Rock|
|TRAACER128F4290F96|      0.2838|       8.995|       429.5|     31990.0|      0.5388|       28.29|    185100.0|     Pop_Rock|
|TRAADYB128F92D7E73|      0.1346|       7.321|       499.6|     38460.0|      0.2839|       15.75|    116500.0|         Jazz|
|TRAAGHM128EF35CF8E|      0.1563|       9.959|       502.8|     26190.0|      0.3835|       28.24|    180800.0|   Electronic|
|TRAAGRV128F93526C0|      0.1076|       7.401|       389.7|     19350.0|      0.4221|       30.99|    191700.0|     Pop_Rock|
|TRAAGTO128F1497E3C|      0.1069|       8.987|       562.6|     43100.0|      0.1007|        22.9|    157700.0|     Pop_Rock|
|TRAAHAU128F9313A3D|     0.08485|       9.031|       445.9|     23750.0|      0.2583|       35.59|    198400.0|     Pop_Rock|
|TRAAHEG128E07861C3|      0.1699|       17.22|       741.3|     52440.0|      0.2812|       28.83|    160800.0|          Rap|
|TRAAHZP12903CA25F4|      0.1654|       12.31|       565.1|     33100.0|      0.1861|       38.38|    196600.0|          Rap|
|TRAAICW128F1496C68|      0.1104|       7.123|       398.2|     19540.0|      0.2871|       28.53|    189400.0|International|
|TRAAJJW12903CBDDCB|      0.2267|       14.88|       592.7|     37980.0|      0.4219|       36.17|    179400.0|International|
|TRAAJKJ128F92FB44F|     0.03861|        6.87|       407.8|     41310.0|      0.0466|       15.79|    121700.0|         Folk|
|TRAAKLX128F934CEE4|      0.1647|       16.77|       850.0|     64420.0|      0.2823|       26.52|    152000.0|   Electronic|
|TRAAKWR128F931B29F|     0.04881|       9.331|       564.0|     34410.0|     0.08647|        18.1|     57700.0|     Pop_Rock|
|TRAALQN128E07931A4|      0.1989|       12.83|       578.7|     30690.0|      0.5452|       33.37|    188700.0|   Electronic|
|TRAAMFF12903CE8107|      0.1385|       9.699|       581.6|     31590.0|      0.3706|       23.63|    163800.0|     Pop_Rock|
|TRAAMHG128F92ED7B2|      0.1799|       10.52|       551.4|     29170.0|      0.4046|       30.78|    183200.0|International|
|TRAAROH128F42604B0|      0.1192|        16.4|       737.3|     41670.0|      0.2284|       31.04|    169100.0|   Electronic|
|TRAARQN128E07894DF|      0.2559|       15.23|       757.1|     61750.0|      0.5417|       40.96|    189000.0|     Pop_Rock|
+------------------+------------+------------+------------+------------+------------+------------+------------+-------------+
only showing top 20 rows
"""


### Question 2 b ###

audio_features_genre_class = audio_features_genre.withColumn('genre', when(F.col('genre_type') == 'Electronic', 1).otherwise(0))

audio_features_genre_class.show()

""" Output
+------------------+------------+------------+------------+------------+------------+------------+------------+-------------+-----+
|          track_id|feature_0000|feature_0001|feature_0002|feature_0003|feature_0005|feature_0006|feature_0008|   genre_type|genre|
+------------------+------------+------------+------------+------------+------------+------------+------------+-------------+-----+
|TRAAABD128F429CF47|      0.1308|       9.587|       459.9|     27280.0|      0.2474|       26.02|     67790.0|     Pop_Rock|    0|
|TRAABPK128F424CFDB|      0.1208|       6.738|       215.1|     11890.0|      0.4882|       41.76|    220400.0|     Pop_Rock|    0|
|TRAACER128F4290F96|      0.2838|       8.995|       429.5|     31990.0|      0.5388|       28.29|    185100.0|     Pop_Rock|    0|
|TRAADYB128F92D7E73|      0.1346|       7.321|       499.6|     38460.0|      0.2839|       15.75|    116500.0|         Jazz|    0|
|TRAAGHM128EF35CF8E|      0.1563|       9.959|       502.8|     26190.0|      0.3835|       28.24|    180800.0|   Electronic|    1|
|TRAAGRV128F93526C0|      0.1076|       7.401|       389.7|     19350.0|      0.4221|       30.99|    191700.0|     Pop_Rock|    0|
|TRAAGTO128F1497E3C|      0.1069|       8.987|       562.6|     43100.0|      0.1007|        22.9|    157700.0|     Pop_Rock|    0|
|TRAAHAU128F9313A3D|     0.08485|       9.031|       445.9|     23750.0|      0.2583|       35.59|    198400.0|     Pop_Rock|    0|
|TRAAHEG128E07861C3|      0.1699|       17.22|       741.3|     52440.0|      0.2812|       28.83|    160800.0|          Rap|    0|
|TRAAHZP12903CA25F4|      0.1654|       12.31|       565.1|     33100.0|      0.1861|       38.38|    196600.0|          Rap|    0|
|TRAAICW128F1496C68|      0.1104|       7.123|       398.2|     19540.0|      0.2871|       28.53|    189400.0|International|    0|
|TRAAJJW12903CBDDCB|      0.2267|       14.88|       592.7|     37980.0|      0.4219|       36.17|    179400.0|International|    0|
|TRAAJKJ128F92FB44F|     0.03861|        6.87|       407.8|     41310.0|      0.0466|       15.79|    121700.0|         Folk|    0|
|TRAAKLX128F934CEE4|      0.1647|       16.77|       850.0|     64420.0|      0.2823|       26.52|    152000.0|   Electronic|    1|
|TRAAKWR128F931B29F|     0.04881|       9.331|       564.0|     34410.0|     0.08647|        18.1|     57700.0|     Pop_Rock|    0|
|TRAALQN128E07931A4|      0.1989|       12.83|       578.7|     30690.0|      0.5452|       33.37|    188700.0|   Electronic|    1|
|TRAAMFF12903CE8107|      0.1385|       9.699|       581.6|     31590.0|      0.3706|       23.63|    163800.0|     Pop_Rock|    0|
|TRAAMHG128F92ED7B2|      0.1799|       10.52|       551.4|     29170.0|      0.4046|       30.78|    183200.0|International|    0|
|TRAAROH128F42604B0|      0.1192|        16.4|       737.3|     41670.0|      0.2284|       31.04|    169100.0|   Electronic|    1|
|TRAARQN128E07894DF|      0.2559|       15.23|       757.1|     61750.0|      0.5417|       40.96|    189000.0|     Pop_Rock|    0|
+------------------+------------+------------+------------+------------+------------+------------+------------+-------------+-----+
only showing top 20 rows
"""


audio_features_genre_class.groupby('genre').count().show()

""" Output
+-----+------+
|genre| count|
+-----+------+
|    1| 40666|
|    0|379954|
+-----+------+
"""


### Question 2 c ###

from pyspark.sql.window import *
from pyspark.ml.feature import VectorAssembler
import numpy as np


# Creates function that displays class balance

def show_class_balance(data, name):
    N = data.count()
    counts = data.groupBy("genre").count().toPandas()
    counts["ratio"] = counts["count"] / N
    print(name)
    print(N)
    print(counts)
    print("")

# Assemble features

assembler = VectorAssembler(
    inputCols=[col for col in audio_features_genre_class.columns if col.startswith("feature_")],
    outputCol="Features"
)

features = assembler.transform(audio_features_genre_class).select(["Features", "genre"])
features.cache()

features.count()

""" Output
420620
"""


# Performing a random split (not stratified)

temp = (
    features
    .withColumn("id", monotonically_increasing_id())
    .withColumn("Random", rand())
    .withColumn(
        "Row",
        row_number()
        .over(
            Window
            .partitionBy("genre")
            .orderBy("Random")
        )
    )
)

class0 = audio_features_genre_class.groupby('genre').count().select(F.col('count')).where(F.col('genre') == 0).first()[0]
class1 = audio_features_genre_class.groupby('genre').count().select(F.col('count')).where(F.col('genre') == 1).first()[0]

training = temp.where(
    ((col("genre") == 0) & (col("Row") <  class0 * 0.8)) |
    ((col("genre") == 1) & (col("Row") < class1 * 0.8))
)

training.cache()

training.show()

""" Output
+--------------------+-----+------------+--------------------+---+
|            Features|genre|          id|              Random|Row|
+--------------------+-----+------------+--------------------+---+
|[0.2856,16.35,817...|    1| 25769824335|6.960488762153272E-5|  1|
|[0.096,2.279,122....|    1| 60129554201| 7.45234797490113E-5|  2|
|[0.07426,22.82,96...|    1| 25769819595|8.999153673228122E-5|  3|
|[0.2139,18.51,107...|    1| 42949698933|9.364815130996629E-5|  4|
|[0.1638,10.17,456...|    1|       14987|1.020963217740478...|  5|
|[0.1672,6.582,460...|    1|  8589959412|1.295492828656819E-4|  6|
|[0.1736,10.72,681...|    1|  8589942887|2.037033651823838...|  7|
|[0.2301,16.8,807....|    1|  8589934737|2.348191324749171...|  8|
|[0.178,11.76,509....|    1| 17179894373|2.350807874452742...|  9|
|[0.1703,11.33,492...|    1| 51539615756|2.431979739226930...| 10|
|[0.1895,15.25,714...|    1|128849035571|2.642064088216322...| 11|
|[0.1809,15.65,809...|    1|128849037125|2.798204138186877E-4| 12|
|[0.2954,18.51,572...|    1| 68719477451| 2.93227647228389E-4| 13|
|[0.1006,3.658,197...|    1| 34359752691|3.010540063006495E-4| 14|
|[0.2755,14.34,761...|    1| 17179883262|3.095610802327231E-4| 15|
|[0.2198,10.76,490...|    1| 51539622843|3.323396618140073E-4| 16|
|[0.1447,7.543,508...|    1| 25769817707| 3.42585706617049E-4| 17|
|[0.333,25.78,1019...|    1| 34359752726|3.500807946811290...| 18|
|[0.1614,7.339,406...|    1| 34359757125|3.554317618221780...| 19|
|[0.2795,12.48,720...|    1| 68719495508|3.675730395330534E-4| 20|
+--------------------+-----+------------+--------------------+---+
only showing top 20 rows
"""


test = temp.join(training, on = "id", how = "left_anti")
test.cache()
test.show(truncate = False)


""" Output
+------------+--------------------------------------------------+-----+------------------+-----+
|id          |Features                                          |genre|Random            |Row  |
+------------+--------------------------------------------------+-----+------------------+-----+
|25769810129 |[0.2309,14.27,718.1,40540.0,0.6899,34.4,178600.0] |1    |0.8000864257557913|32533|
|51539620046 |[0.1969,10.6,532.6,29620.0,0.3761,29.56,184200.0] |1    |0.8001277419530735|32534|
|42949695314 |[0.1501,15.38,828.8,60670.0,0.1668,18.03,107200.0]|1    |0.8001469603243452|32535|
|77309428078 |[0.2016,16.74,702.4,39590.0,0.2899,34.47,173200.0]|1    |0.8001791509687367|32536|
|111669155744|[0.1398,12.8,663.5,41510.0,0.2881,24.51,160100.0] |1    |0.8001828285946755|32537|
|68719499550 |[0.2226,10.79,491.2,24600.0,0.5721,34.61,193300.0]|1    |0.8001847863140418|32538|
|17179893230 |[0.1451,7.771,454.9,32560.0,0.3004,23.52,164100.0]|1    |0.800250564244354 |32539|
|68719480219 |[0.08715,4.006,261.7,27050.0,0.2624,13.25,93520.0]|1    |0.800272209251333 |32540|
|8589941769  |[0.1647,7.838,522.3,49380.0,0.1443,6.983,49390.0] |1    |0.8002904796095947|32541|
|42949698948 |[0.2092,11.33,476.0,23890.0,0.4462,38.62,190500.0]|1    |0.8002924300741143|32542|
|94489298241 |[0.1474,9.366,396.5,19110.0,0.3288,38.66,200900.0]|1    |0.8003201965591269|32543|
|34359740461 |[0.1372,13.97,541.4,31160.0,0.3023,41.54,182600.0]|1    |0.8003234886072554|32544|
|94489293271 |[0.1837,12.86,603.9,29660.0,0.2876,31.29,181100.0]|1    |0.8003554043831415|32545|
|4217        |[0.1745,13.48,669.2,37350.0,0.3318,29.99,171500.0]|1    |0.8003750298100523|32546|
|51539608495 |[0.3163,19.72,717.4,54670.0,0.5008,52.16,90730.0] |1    |0.8004206811835922|32547|
|25769812130 |[0.08446,14.36,754.1,52930.0,0.1231,24.2,152200.0]|1    |0.8004489387365882|32548|
|42949678933 |[0.1736,10.24,631.1,37270.0,0.4535,28.36,178700.0]|1    |0.800458865623856 |32549|
|103079235469|[0.06625,14.7,402.6,37020.0,0.1023,61.55,4464.0]  |1    |0.8005120322953309|32550|
|85899365557 |[0.05545,6.13,316.6,40040.0,0.1193,15.29,121300.0]|1    |0.8005188067821406|32551|
|77309434713 |[0.1412,13.01,797.2,44780.0,0.3206,24.27,155400.0]|1    |0.8005317948324262|32552|
+------------+--------------------------------------------------+-----+------------------+-----+
only showing top 20 rows
"""

training = training.drop("id", "Random", "Row")

test = test.drop("id", "Random", "Row")


show_class_balance(features, "features")

""" Output
features
420620
   genre   count     ratio
0      1   40666  0.096681
1      0  379954  0.903319
"""


show_class_balance(training, "training")

""" Output
training
336495
   genre   count     ratio
0      1   32532  0.096679
1      0  303963  0.903321
"""


show_class_balance(test, "test")

""" Output
test
84125
   genre  count     ratio
0      1   8134  0.096689
1      0  75991  0.903311
"""


# Upsampling

ratio = 5
n = 10
p = ratio / n  # ratio < n such that probability < 1

def random_resample(x, n, p):
    # Can implement custom sampling logic per class,
    if x == 0:
        return [0]  # no sampling
    if x == 1:
        return list(range((np.sum(np.random.random(n) > p))))  # upsampling
    return []  # drop

random_resample_udf = udf(lambda x: random_resample(x, n, p), ArrayType(IntegerType()))

training_upsampled = (
    training
    .withColumn("Sample", random_resample_udf(col("genre")))
    .select(
        col("Features"),
        col("genre"),
        explode(col("Sample")).alias("Sample")
    )
    .drop("Sample")
)


show_class_balance(features, "features")

""" Output
features
420620
   genre   count     ratio
0      1   40666  0.096681
1      0  379954  0.903319
"""


show_class_balance(training_upsampled, "training_upsampled")

""" Output
training_upsampled
466282
   genre   count     ratio
0      1  162167  0.347787
1      0  303963  0.651887
"""


# Downsampling

training_downsampled = (
    training
    .withColumn("Random", rand())
    .where((col("genre") != 0) | ((col("genre") == 0) & (col("Random") < 2 * (class1 / class0))))
)

training_downsampled.cache()


show_class_balance(features, "features")

""" Output
features
420620
   genre   count     ratio
0      1   40666  0.096681
1      0  379954  0.903319
"""


show_class_balance(training_downsampled, "training_downsampled")

""" Output
training_downsampled
98063
   genre  count     ratio
0      1  32532  0.331746
1      0  65531  0.668254
"""


# Up then Downsampling

training_updownsampled = (
    training_upsampled
    .withColumn("Random", rand())
    .where((col("genre") != 0) | ((col("genre") == 0) & (col("Random") < 5 * (class1 / class0))))
)

training_updownsampled.cache()

show_class_balance(features, "features")

""" Output
features
420620
   genre   count     ratio
0      1   40666  0.096681
1      0  379954  0.903319
"""


show_class_balance(training_updownsampled, "training_updownsampled")

""" Output
training_updownsampled
325108
   genre   count     ratio
0      1  162319  0.499277
1      0  162789  0.500723
"""


# Reweighting

training_weighted = (
    training
    .withColumn(
        "Weight",
        when(col("genre") == 0, 1.0)
        .when(col("genre") == 1, 5.0)
        .otherwise(1.0)
    )
)


weights = (
    training_weighted
    .groupBy("genre")
    .agg(
        collect_set(col("Weight")).alias("Weights")
    )
    .toPandas()
)


print(weights)

""" Output
   genre Weights
0      1   [5.0]
1      0   [1.0]
"""


### Question 2 d ###

def show_binary_metrics(predictions, predictionsName, labelCol, predictionCol = "prediction", rawPredictionCol = "rawPrediction"):

    total = predictions.count()
    positive = predictions.filter((col(labelCol) == 1)).count()
    negative = predictions.filter((col(labelCol) == 0)).count()
    nP = predictions.filter((col(predictionCol) == 1)).count()
    nN = predictions.filter((col(predictionCol) == 0)).count()
    TP = predictions.filter((col(predictionCol) == 1) & (col(labelCol) == 1)).count()
    FP = predictions.filter((col(predictionCol) == 1) & (col(labelCol) == 0)).count()
    FN = predictions.filter((col(predictionCol) == 0) & (col(labelCol) == 1)).count()
    TN = predictions.filter((col(predictionCol) == 0) & (col(labelCol) == 0)).count()

    binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol = rawPredictionCol, labelCol = labelCol, metricName = "areaUnderROC")
    auroc = binary_evaluator.evaluate(predictions)
    print(predictionsName)
    print('----------------------------')
    print('actual total:    {}'.format(total))
    print('actual positive: {}'.format(positive))
    print('actual negative: {}'.format(negative))
    print('nP:              {}'.format(nP))
    print('nN:              {}'.format(nN))
    print('TP:              {}'.format(TP))
    print('FP:              {}'.format(FP))
    print('FN:              {}'.format(FN))
    print('TN:              {}'.format(TN))
    print('precision:       {}'.format(TP / (TP + FP)))
    print('recall:          {}'.format(TP / (TP + FN)))
    print('accuracy:        {}'.format((TP + TN) / total))
    print('auroc:           {}'.format(auroc))


def with_custom_prediction(predictions, threshold, probabilityCol = "probability", customPredictionCol = "customPrediction"):

    def apply_custom_threshold(probability, threshold):
        return int(probability[1] > threshold)

    apply_custom_threshold_udf = udf(lambda x: apply_custom_threshold(x, threshold), IntegerType())

    return predictions.withColumn(customPredictionCol, apply_custom_threshold_udf(col(probabilityCol)))





# Loads RandomForestClassifier library

from pyspark.ml.classification import RandomForestClassifier


# Random Forest (No sampling)

rf = RandomForestClassifier(featuresCol = 'Features', labelCol = 'genre')
nosampling_rf_model = rf.fit(training)



# Random Forest (Up- then downsampling)

rf = RandomForestClassifier(featuresCol = 'Features', labelCol = 'genre')
updownsampling_rf_model = rf.fit(training_updownsampled)



# Loads DecisionTreeClassifier library

from pyspark.ml.classification import DecisionTreeClassifier


# Decision Tree Classifier (No sampling)

dt = DecisionTreeClassifier(featuresCol = 'Features', labelCol = 'genre', maxDepth = 5)
nosampling_dt_model = dt.fit(training)


# Decision Tree (Up- then downsampling)

dt = DecisionTreeClassifier(featuresCol = 'Features', labelCol = 'genre', maxDepth = 5)
updownsampling_dt_model = dt.fit(training_updownsampled)



# Logistic Regression (no sampling)

lr = LogisticRegression(featuresCol = 'Features', labelCol = 'genre')
nosampling_lr_model = lr.fit(training)


# Logistic Regression (Up- then downsampling)

lr = LogisticRegression(featuresCol = 'Features', labelCol = 'genre')
updownsampling_lr_model = lr.fit(training_updownsampled)



### Question 2 e ###

# Random Forest (No sampling)

nosampling_rf_predictions = nosampling_rf_model.transform(test)

nosampling_rf_predictions.cache()


show_binary_metrics(nosampling_rf_predictions, 'nosampling_rf_predictions', labelCol = 'genre')

""" Output
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
<ipython-input-132-bde4f368842b> in <module>()
----> 1 show_binary_metrics(nosampling_rf_predictions, 'nosampling_rf_predictions', labelCol = 'genre')

<ipython-input-88-7ed8f02578bd> in show_binary_metrics(predictions, predictionsName, labelCol, predictionCol, rawPredictionCol)
     24     print('FN:              {}'.format(FN))
     25     print('TN:              {}'.format(TN))
---> 26     print('precision:       {}'.format(TP / (TP + FP)))
     27     print('recall:          {}'.format(TP / (TP + FN)))
     28     print('accuracy:        {}'.format((TP + TN) / total))

ZeroDivisionError: division by zero
"""



# Random Forest (Up- then downsampling)

updownsampling_rf_predictions = updownsampling_rf_model.transform(test)

updownsampling_rf_predictions.cache()


show_binary_metrics(updownsampling_rf_predictions, 'updownsampling_rf_predictions', labelCol = 'genre')

""" Output
updownsampling_rf_predictions
----------------------------
actual total:    84125
actual positive: 8134
actual negative: 75991
nP:              25478
nN:              58647
TP:              5362
FP:              20116
FN:              2772
TN:              55875
precision:       0.21045607975508282
recall:          0.6592082616179001
accuracy:        0.72792867756315
auroc:           0.7667092042401706
"""


# Decision Tree (No sampling)

nosampling_dt_predictions = nosampling_dt_model.transform(test)

nosampling_dt_predictions.cache()


show_binary_metrics(nosampling_dt_predictions, 'nosampling_dt_predictions', labelCol = 'genre')

""" Output
nosampling_dt_predictions
----------------------------
actual total:    84125
actual positive: 8134
actual negative: 75991
nP:              1246
nN:              82879
TP:              610
FP:              636
FN:              7524
TN:              75355
precision:       0.4895666131621188
recall:          0.0749938529628719
accuracy:        0.903001485884101
auroc:           0.4099737724366612
"""



# Decision Tree (Up- then downsampling)

updownsampling_dt_predictions = updownsampling_dt_model.transform(test)

updownsampling_dt_predictions.cache()


show_binary_metrics(updownsampling_dt_predictions, 'updownsampling_dt_predictions', labelCol = 'genre')

""" Output
updownsampling_dt_predictions
----------------------------
actual total:    84125
actual positive: 8134
actual negative: 75991
nP:              28318
nN:              55807
TP:              5545
FP:              22773
FN:              2589
TN:              53218
precision:       0.19581185111942934
recall:          0.6817064175067618
accuracy:        0.698520059435364
auroc:           0.6442017116756578
"""


# Logistic Regression (No sampling)

nosampling_lr_predictions = nosampling_lr_model.transform(test)

nosampling_lr_predictions.cache()


show_binary_metrics(nosampling_lr_predictions, 'nosampling_lr_predictions', labelCol = 'genre')

""" Output
nosampling_lr_predictions
----------------------------
actual total:    84125
actual positive: 8134
actual negative: 75991
nP:              203
nN:              83922
TP:              96
FP:              107
FN:              8038
TN:              75884
precision:       0.4729064039408867
recall:          0.011802311285960166
accuracy:        0.9031797919762259
auroc:           0.7079666683510464
"""


# Logistic Regression (Up- then downsampling)

updownsampling_lr_predictions = updownsampling_lr_model.transform(test)

updownsampling_lr_predictions.cache()


show_binary_metrics(updownsampling_lr_predictions, 'updownsampling_lr_predictions', labelCol = 'genre')

""" Output
updownsampling_lr_predictions
----------------------------
actual total:    84125
actual positive: 8134
actual negative: 75991
nP:              30578
nN:              53547
TP:              5246
FP:              25332
FN:              2888
TN:              50659
precision:       0.17156125318856694
recall:          0.6449471354806983
accuracy:        0.6645468053491828
auroc:           0.7082848677772808
"""













