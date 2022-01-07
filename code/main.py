#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from pyspark.sql import SparkSession,Row
import pyspark.ml.feature
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer,StopWordsRemover,CountVectorizer,IDF
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
from pyspark.sql.types import StringType


# ----------------------------> 数据读取
spark = SparkSession.builder.appName('text_classification').getOrCreate()
df = spark.read.csv('../data/IMDB.csv', header = True, inferSchema = True)
df.printSchema()
df.show()

# 由于数据为逗号分隔的 csv 格式，在文本列出现混淆
# 我们使用 pandas 进行读取后再转换为 spark DataFrame 格式
pandasDF = pd.read_csv('../data/IMDB.csv')
pandasDF.isnull().sum() # 缺失值检查
df = spark.createDataFrame(pandasDF)
df.printSchema()
df.show()
df.groupBy('sentiment').count().show()
# ----------------------------------------------------------------------


# ----------------------------> 文本清洁
# ----------------------------------------------------------------------


# ----------------------------> 标签数字转换
labelEncoder = StringIndexer(inputCol='sentiment', outputCol='label').fit(df)
labelEncoder.transform(df).show(5)
df = labelEncoder.transform(df)
# ----------------------------------------------------------------------


# ----------------------------> 数据集划分
(trainDF,testDF) = df.randomSplit((0.7,0.3), seed=1)
# ----------------------------------------------------------------------


# ----------------------------> 文本特征化
tokenizer = Tokenizer(inputCol='review', outputCol='tokens')
add_stopwords = ["<br />","amp"]
stopwords_remover = StopWordsRemover(inputCol='tokens', outputCol='filtered_tokens').setStopWords(add_stopwords)
vectorizer = CountVectorizer(inputCol='filtered_tokens', outputCol='rawFeatures')
# or use hashingTF = HashingTF(inputCol="tokens", outputCol="rawFeatures")
idf = IDF(inputCol='rawFeatures', outputCol='vectorizedFeatures')
pipeline = Pipeline(stages=[tokenizer,stopwords_remover,vectorizer,idf])
preprocessModel = pipeline.fit(trainDF)
trainDF = preprocessModel.transform(trainDF)
testDF = preprocessModel.transform(testDF)
# ----------------------------------------------------------------------


# ----------------------------> 训练
lr = LogisticRegression(featuresCol='vectorizedFeatures',labelCol='label')
lr_model = lr.fit(trainDF)
predictions = lr_model.transform(testDF)
predictions.select(['label', 'prediction']).show()
evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
accuracy
# precision recall .... here 
# ----------------------------------------------------------------------


# ----------------------------> 预测
inputText = spark.createDataFrame([("I like this movie",StringType()),
                                   ("Back of Beyond",StringType())],
                                  ["review"])
inputText.show(truncate=False)
inputText = preprocessModel.transform(inputText)
inputPrediction = lr_model.transform(inputText)
inputPrediction.show()
inputPrediction.select(['review', 'prediction']).show()
# ----------------------------------------------------------------------


# ----------------------------> 模型调参
def logisticCV(trainDF, testDF):
    lr = LogisticRegression(featuresCol='vectorizedFeatures',labelCol='label')
    pipeline = Pipeline(stages=[lr])
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.5, 2.0]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .addGrid(lr.maxIter, [1, 5, 10]) \
        .build() 
    evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')
    crossValidator = CrossValidator(estimator=pipeline, 
                                    evaluator=evaluator,
                                    estimatorParamMaps=paramGrid,
                                    numFolds=5)
    cv = crossValidator.fit(trainDF)
    best_model = cv.bestModel.stages[0]
    prediction = best_model.transform(test_df)
    accuracy = evaluator.evaluate(predictions)
    print('Accuracy in Cross Validation of logistic regression: %g' % accuracy)

logisticCV(trainDF, testDF)

