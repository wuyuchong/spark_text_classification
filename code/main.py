#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pyspark.ml.feature
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Tokenizer,StopWordsRemover,CountVectorizer,IDF,StringIndexer,Word2Vec
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.sql import SparkSession,Row
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf, col

# ----------------------------> 数据读取
spark = SparkSession.builder.appName('text_classification').getOrCreate()
try:
    df = spark.read.csv('../data/IMDB.csv', header = True, inferSchema = True)
except:
    # location on server
    df = spark.read.csv('file:///home1/cufe/students/wuyuchong/spark_text_classification/data/IMDB.csv', header = True, inferSchema = True)
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
try:
    # 在服务器上的分布式模式中，需要使用 --py-files 将 gensim 包传到每个子节点
    # 若该过程失败则跳过文本清洁过程
    import gensim.parsing.preprocessing as gsp
    from gensim import utils
    filters = [
        gsp.strip_tags,
        gsp.strip_punctuation,
        gsp.strip_multiple_whitespaces,
        gsp.strip_numeric,
        gsp.remove_stopwords,
        gsp.strip_short,
        gsp.stem_text
    ]
    def clean_text(x):
        x = x.lower()
        x = utils.to_unicode(x)
        for f in filters:
            x = f(x)
        return x

    cleanTextUDF = udf(lambda x: clean_text(x), StringType())
    df = df.withColumn("clean_text", cleanTextUDF(col("review")))
except:
    df = df.withColumn("clean_text", df.review)
# ----------------------------------------------------------------------


# ----------------------------> 标签数字转换
labelEncoder = StringIndexer(inputCol='sentiment', outputCol='label').fit(df)
labelEncoder.transform(df).show(5)
df = labelEncoder.transform(df)
# ----------------------------------------------------------------------


# ----------------------------> 数据集划分
(trainDF,testDF) = df.randomSplit((0.7,0.3), seed=1)
# ----------------------------------------------------------------------


# ----------------------------> 文本特征工程
tokenizer = Tokenizer(inputCol='clean_text', outputCol='tokens')
add_stopwords = ["<br />","amp"]
stopwords_remover = StopWordsRemover(inputCol='tokens', outputCol='filtered_tokens').setStopWords(add_stopwords)
vectorizer = CountVectorizer(inputCol='filtered_tokens', outputCol='rawFeatures')
# or use hashingTF = HashingTF(inputCol="tokens", outputCol="rawFeatures")
idf = IDF(inputCol='rawFeatures', outputCol='vectorizedFeatures')
word2Vec = Word2Vec(vectorSize=5, minCount=2, inputCol="filtered_tokens", outputCol="vectorizedFeatures")
#  pipeline = Pipeline(stages=[tokenizer,stopwords_remover,vectorizer,idf])
pipeline = Pipeline(stages=[tokenizer,stopwords_remover,word2Vec])
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
                                   ("It is so bad",StringType())],
                                  ["clean_text"])
inputText.show(truncate=False)
inputText = preprocessModel.transform(inputText)
inputPrediction = lr_model.transform(inputText)
inputPrediction.show()
inputPrediction.select(['clean_text', 'prediction']).show()
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
    prediction = best_model.transform(testDF)
    accuracy = evaluator.evaluate(predictions)
    print('Accuracy in Cross Validation of logistic regression: %g' % accuracy)

logisticCV(trainDF, testDF)


def RandomForestCV(trainDF, testDF):
    rf = RandomForestClassifier(featuresCol='vectorizedFeatures',labelCol='label')
    pipeline = Pipeline(stages=[rf])
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [5, 10]) \
        .addGrid(rf.maxBins, [25, 31]) \
        .addGrid(rf.minInfoGain, [0.01, 0.001]) \
        .addGrid(rf.numTrees, [20, 60]) \
        .addGrid(rf.impurity, ['gini', 'entropy']) \
        .build() 
    evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')
    crossValidator = CrossValidator(estimator=pipeline, 
                                    evaluator=evaluator,
                                    estimatorParamMaps=paramGrid,
                                    numFolds=5)
    cv = crossValidator.fit(trainDF)
    best_model = cv.bestModel.stages[0]
    prediction = best_model.transform(testDF)
    accuracy = evaluator.evaluate(predictions)
    print('Accuracy in Cross Validation of random forest: %g' % accuracy)

RandomForestCV(trainDF, testDF)




# https://spark.apache.org/docs/latest/api/python/user_guide/python_packaging.html
