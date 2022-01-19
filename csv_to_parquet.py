from __future__ import print_function

import sys
from random import random
from operator import add

from pyspark.sql import SparkSession
# from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import time
from datetime import datetime





spark = SparkSession.builder.appName("Q1_parquet").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

#creating manually schemas so as to avoid double reading of the files with inferSchema=True
schema_tripdata = StructType([StructField("ID", StringType(), False),
                          StructField("Start_Datetime", StringType(), False),
                          StructField("End_Datetime", StringType(), False),
                          StructField("Start_Longitude", FloatType(), False),
                          StructField("Start_Latitude", FloatType(), False),
                          StructField("End_Longitude", FloatType(), False),
                          StructField("End_Latitude", FloatType(), False),
                          StructField("Cost", FloatType(), False)
                            ])

schema_tripvendors = StructType([StructField("ID", StringType()),
                                 StructField("Vendor", StringType())
                        ]) 


# read first dataset
yellow_tripdata_1m = spark.read.format('csv').schema(schema_tripdata)\
                          .options(header='false', inferSchema='false')\
                          .load("hdfs://master:9000/data/yellow_tripdata_1m.csv")

# save first dataset as parquet
start_time_write_parquet = time.time()
yellow_tripdata_1m.write.parquet("hdfs://master:9000/data/yellow_tripdata_1m.parquet")
print("Time to write first dataset as parquet is: %s seconds" % (time.time() - start_time_write_parquet))



yellow_tripvendors_1m = spark.read.format("csv").schema(schema_tripvendors)\
                             .options(header='false', inferSchema='false')\
                             .load("hdfs://master:9000/data/yellow_tripvendors_1m.csv")

start_time_write_parquet = time.time()
yellow_tripvendors_1m.write.parquet("hdfs://master:9000/data/yellow_tripvendors_1m.parquet")
print("Time to write second dataset as parquet is: %s seconds" % (time.time() - start_time_write_parquet))


"""
    Usage: pi [partitions]
"""
# spark = SparkSession\
#     .builder\
#     .appName("TaxiDriver_test")\
#     .getOrCreate()

# partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2
# n = 100000 * partitions

# def f(_):
#     x = random() * 2 - 1
#     y = random() * 2 - 1
#     return 1 if x ** 2 + y ** 2 <= 1 else 0
# file1 =  spark.sparkContext.textFile("hdfs://master:9000/data/yellow_tripdata_1m.csv")
# file2 =  spark.sparkContext.textFile("hdfs://master:9000/data/yellow_tripvendors_1m.csv")


# file1.write.parquet("hdfs://master:9000/data/yellow_tripdata_1m.parquet")
# file2.write.parquet("hdfs://master:9000/data/yellow_tripvendors_1m.parquet")
# totalcount = lines.filter(lambda s: ",1" in s).count()
# count = spark.sparkContext.parallelize(range(1, n + 1), partitions).map(f).reduce(add)
# print("Total info counted: %f" % (totalcount))

# spark.stop()
