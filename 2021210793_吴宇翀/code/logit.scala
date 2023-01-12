// -----------------------> 环境载入
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().appName("text classfication").getOrCreate()

// -----------------------> 数据加载
val df = spark.read.option("header", "true").option("inferSchema", "true").option("sep", "\t").csv("../data/convert.csv")
df.show()

// -----------------------> 标签数字转换
import org.apache.spark.ml.feature.StringIndexer
val indexer = new StringIndexer()
  .setInputCol("sentiment")
  .setOutputCol("label")
val indexed = indexer.fit(df).transform(df)
val df = indexed
df.show()

// -----------------------> 数据集划分
val Array(trainDF, testDF) = df.randomSplit(Array(0.75, 0.25))

// -----------------------> 文本特征工程
import org.apache.spark.ml.feature.RegexTokenizer
val tokenizer = new RegexTokenizer()
  .setInputCol("review")
  .setOutputCol("words")
import org.apache.spark.ml.feature.HashingTF
val hashingTF = new HashingTF()
  .setInputCol(tokenizer.getOutputCol)  // it does not wire transformers -- it's just a column name
  .setOutputCol("features")
  .setNumFeatures(5000)

// -----------------------> logit 回归
import org.apache.spark.ml.classification.LogisticRegression
val lr = new LogisticRegression().setMaxIter(20).setRegParam(0.01)
import org.apache.spark.ml.Pipeline
val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))
val model = pipeline.fit(trainDF)

// -----------------------> 测试集预测
val trainPredictions = model.transform(trainDF)
val testPredictions = model.transform(testDF)
testPredictions.select('review, 'label, 'prediction).show
