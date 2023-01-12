#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------> 特征工程方法选项
processType = 'word2vec'
#  processType = 'vectorize-idf'
#  processType = 'tf-idf'
# ----------------------------------------------------------------------


# ----------------------------> 文本清洁选项
cleaning = True
#  cleaning = True # 需要上传第三方包 gensim
# ----------------------------------------------------------------------


# ----------------------------> 包及环境启动
import pandas as pd
import pyspark.ml.feature
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Tokenizer,StopWordsRemover,CountVectorizer,IDF,StringIndexer,Word2Vec,HashingTF
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier,GBTClassifier,DecisionTreeClassifier
from pyspark.sql import SparkSession,Row
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf, col

spark = SparkSession.builder.appName('text_classification').getOrCreate()
# ----------------------------------------------------------------------


# ----------------------------> 数据读取
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
if cleaning == False:
    df = df.withColumn("clean_text", df.review)
else:
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
hashingTF = HashingTF(inputCol="filtered_tokens", outputCol="rawFeatures")
idf = IDF(inputCol='rawFeatures', outputCol='vectorizedFeatures')
word2Vec = Word2Vec(vectorSize=30, minCount=2, inputCol="filtered_tokens", outputCol="vectorizedFeatures")
if processType == 'word2vec':
    pipeline = Pipeline(stages=[tokenizer,stopwords_remover,word2Vec])
if processType == 'vectorize-idf':
    pipeline = Pipeline(stages=[tokenizer,stopwords_remover,vectorizer,idf])
if processType == 'tf-idf':
    pipeline = Pipeline(stages=[tokenizer,stopwords_remover,hashingTF,idf])
preprocessModel = pipeline.fit(trainDF)
trainDF = preprocessModel.transform(trainDF)
testDF = preprocessModel.transform(testDF)
# ----------------------------------------------------------------------


# ----------------------------> 训练
lr = LogisticRegression(featuresCol='vectorizedFeatures',labelCol='label')
lr_model = lr.fit(trainDF)
prediction = lr_model.transform(testDF)
prediction.select(['label', 'prediction']).show()
evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')
accuracy = evaluator.evaluate(prediction)
print(accuracy)
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


# ----------------------------> 模型比较
def logisticCV(trainDF, testDF):
    lr = LogisticRegression(featuresCol='vectorizedFeatures',labelCol='label')
    model = lr.fit(trainDF)
    prediction = model.transform(testDF)
    evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')
    accuracy = evaluator.evaluate(prediction)
    print('Accuracy of logistic regression: %g' % accuracy)

def RandomForest(trainDF, testDF):
    rf = RandomForestClassifier(featuresCol='vectorizedFeatures',labelCol='label')
    model = rf.fit(trainDF)
    prdiction = model.transform(testDF)
    evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')
    accuracy = evaluator.evaluate(prediction)
    print('Accuracy of random forest: %g' % accuracy)

def GBT(trainDF, testDF):
    gbt = GBTClassifier(featuresCol='vectorizedFeatures',labelCol='label')
    model = gbt.fit(trainDF)
    prdiction = model.transform(testDF)
    evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')
    accuracy = evaluator.evaluate(prediction)
    print('Accuracy of gbt: %g' % accuracy)

def DecisionTree(trainDF, testDF):
    dt = DecisionTreeClassifier(featuresCol='vectorizedFeatures',labelCol='label')
    model = dt.fit(trainDF)
    prdiction = model.transform(testDF)
    evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')
    accuracy = evaluator.evaluate(prediction)
    print('Accuracy of decision tree: %g' % accuracy)

logisticCV(trainDF, testDF)
RandomForest(trainDF, testDF)
GBT(trainDF, testDF)
DecisionTree(trainDF, testDF)
# ----------------------------------------------------------------------


# ----------------------------> 模型调参
def logisticCV(trainDF, testDF):
    lr = LogisticRegression(featuresCol='vectorizedFeatures',labelCol='label')
    pipeline = Pipeline(stages=[lr])
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0, 0.5, 2.0]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .addGrid(lr.maxIter, [50, 100, 200]) \
        .build() 
    evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')
    crossValidator = CrossValidator(estimator=pipeline, 
                                    evaluator=evaluator,
                                    estimatorParamMaps=paramGrid,
                                    numFolds=5)
    cv = crossValidator.fit(trainDF)
    best_model = cv.bestModel.stages[0]
    prediction = best_model.transform(testDF)
    accuracy = evaluator.evaluate(prediction)
    print('Accuracy in Cross Validation of logistic regression: %g' % accuracy)

def RandomForestCV(trainDF, testDF):
    rf = RandomForestClassifier(featuresCol='vectorizedFeatures',labelCol='label')
    pipeline = Pipeline(stages=[rf])
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [5, 10]) \
        .addGrid(rf.maxBins, [16, 32]) \
        .addGrid(rf.minInfoGain, [0, 0.01]) \
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
    accuracy = evaluator.evaluate(prediction)
    print('Accuracy in Cross Validation of random forest: %g' % accuracy)

def GBTClassifierCV(trainDF, testDF):
    gbt = GBTClassifier(featuresCol='vectorizedFeatures',labelCol='label')
    pipeline = Pipeline(stages=[gbt])
    paramGrid = ParamGridBuilder() \
        .addGrid(gbt.maxDepth, [5, 10]) \
        .addGrid(gbt.maxBins, [16, 32]) \
        .addGrid(gbt.minInfoGain, [0, 0.01]) \
        .addGrid(gbt.maxIter, [10, 20]) \
        .addGrid(gbt.stepSize, [0.1, 0.2]) \
        .build() 
    evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')
    crossValidator = CrossValidator(estimator=pipeline, 
                                    evaluator=evaluator,
                                    estimatorParamMaps=paramGrid,
                                    numFolds=5)
    cv = crossValidator.fit(trainDF)
    best_model = cv.bestModel.stages[0]
    prediction = best_model.transform(testDF)
    accuracy = evaluator.evaluate(prediction)
    print('Accuracy in Cross Validation of GBT: %g' % accuracy)

def DecisionTreeCV(trainDF, testDF):
    dt = DecisionTreeClassifier(featuresCol='vectorizedFeatures',labelCol='label')
    pipeline = Pipeline(stages=[dt])
    paramGrid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [5, 10]) \
        .addGrid(dt.maxBins, [16, 32]) \
        .addGrid(dt.minInfoGain, [0, 0.01]) \
        .addGrid(dt.minWeightFractionPerNode, [0, 0.5]) \
        .addGrid(dt.impurity, ['gini', 'entropy']) \
        .build() 
    evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')
    crossValidator = CrossValidator(estimator=pipeline, 
                                    evaluator=evaluator,
                                    estimatorParamMaps=paramGrid,
                                    numFolds=5)
    cv = crossValidator.fit(trainDF)
    best_model = cv.bestModel.stages[0]
    prediction = best_model.transform(testDF)
    accuracy = evaluator.evaluate(prediction)
    print('Accuracy in Cross Validation of GBT: %g' % accuracy)

logisticCV(trainDF, testDF)
RandomForestCV(trainDF, testDF)
GBTClassifierCV(trainDF, testDF)
GBTClassifierCV(trainDF, testDF)
# ----------------------------------------------------------------------

