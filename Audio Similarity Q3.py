# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 17:17:19 2021

@author: ajc364
"""

import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier


### Question 3 b ###


rf_cv = RandomForestClassifier(labelCol = 'genre', featuresCol = 'Features')
pipeline = Pipeline(stages = [assembler, rf_cv])

paramGrid = ParamGridBuilder() \
    .addGrid(rf_cv.numTrees, [10, 50, 100]) \
    .addGrid(rf_cv.maxDepth, [3, 6, 9]) \
    .build()

cv = CrossValidator(estimator = rf_cv,
                          estimatorParamMaps = paramGrid,
                          evaluator = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction", labelCol = 'genre', metricName = "areaUnderROC"),
                          numFolds = 5)
                          

cv_model = cv.fit(training_updownsampled)

best_cv_model = cv_model.bestModel


print("Best Depth: ", best_cv_model._java_obj.getMaxDepth())
print("Good Number of Trees: ", best_cv_model._java_obj.getNumTrees())

""" Output
Best Depth:  9
Good Number of Trees:  100
"""


rf_best = RandomForestClassifier(featuresCol = 'Features', labelCol = 'genre', numTrees = 100, maxDepth = 9)
updownsampled_rf_best_model = rf_best.fit(training_updownsampled)
updownsampled_rf_best_predictions = updownsampled_rf_best_model.transform(test)
updownsampled_rf_best_predictions.cache()


show_binary_metrics(updownsampled_rf_best_predictions, 'updownsampled_rf_best_predictions', labelCol = 'genre')

""" Output
2021-10-27 18:45:04,314 WARN scheduler.DAGScheduler: Broadcasting large task binary with size 4.7 MiB
2021-10-27 18:45:09,225 WARN scheduler.DAGScheduler: Broadcasting large task binary with size 4.7 MiB
2021-10-27 18:45:09,863 WARN scheduler.DAGScheduler: Broadcasting large task binary with size 4.7 MiB
2021-10-27 18:45:10,596 WARN scheduler.DAGScheduler: Broadcasting large task binary with size 4.7 MiB
2021-10-27 18:45:11,251 WARN scheduler.DAGScheduler: Broadcasting large task binary with size 4.7 MiB
2021-10-27 18:45:11,971 WARN scheduler.DAGScheduler: Broadcasting large task binary with size 4.7 MiB
2021-10-27 18:45:12,735 WARN scheduler.DAGScheduler: Broadcasting large task binary with size 4.7 MiB
2021-10-27 18:45:13,624 WARN scheduler.DAGScheduler: Broadcasting large task binary with size 4.7 MiB
2021-10-27 18:45:14,447 WARN scheduler.DAGScheduler: Broadcasting large task binary with size 4.7 MiB
2021-10-27 18:45:15,238 WARN scheduler.DAGScheduler: Broadcasting large task binary with size 4.8 MiB
updownsampled_rf_best_predictions
----------------------------
actual total:    84125
actual positive: 8134
actual negative: 75991
nP:              24852
nN:              59273
TP:              5675
FP:              19177
FN:              2459
TN:              56814
precision:       0.22835184291002736
recall:          0.6976887140398328
accuracy:        0.7428112927191679
auroc:           0.7940936952801377
"""