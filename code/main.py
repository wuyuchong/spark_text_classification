#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from pyspark.sql import SparkSession,Row
import pyspark.ml.feature
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer,StopWordsRemover,CountVectorizer,IDF
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression


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
df.count()
# ----------------------------------------------------------------------


# ----------------------------> 文本特征化和标签转换
tokenizer = Tokenizer(inputCol='review', outputCol='tokens')
stopwords_remover = StopWordsRemover(inputCol='tokens', outputCol='filtered_tokens')
vectorizer = CountVectorizer(inputCol='filtered_tokens', outputCol='rawFeatures')
idf = IDF(inputCol='rawFeatures', outputCol='vectorizedFeatures')
labelEncoder = StringIndexer(inputCol='sentiment', outputCol='label').fit(df)
labelEncoder.transform(df).show(5)
df = labelEncoder.transform(df)
# ----------------------------------------------------------------------


# ----------------------------> 数据集划分
(trainDF,testDF) = df.randomSplit((0.7,0.3), seed=1)
# ----------------------------------------------------------------------


# ----------------------------> 模型
lr = LogisticRegression(featuresCol='vectorizedFeatures',labelCol='label')
pipeline = Pipeline(stages=[tokenizer,stopwords_remover,vectorizer,idf,lr])
lr_model = pipeline.fit(trainDF)
predictions = lr_model.transform(testDF)
predictions.show()
predictions.select(['label', 'prediction']).show()
# ----------------------------------------------------------------------


# ----------------------------> reference
#  https://www.section.io/engineering-education/multiclass-text-classification-with-pyspark/
