#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import findspark
findspark.init('/usr/lib/spark-current/')
import pyspark
from pyspark import SparkConf,SparkContext
spark = pyspark.sql.SparkSession.builder.appName("My Spark App").getOrCreate()
sc = spark.sparkContext
sc
