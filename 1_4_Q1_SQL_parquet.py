from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import time
from datetime import datetime

spark = SparkSession.builder.appName("Query_1_parquet").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

start_time_parquet = time.time()
# Read in the Parquet file
yellow_tripdata_1m = sqlContext.read.parquet("hdfs://master:9000/data/yellow_tripdata_1m.parquet")
 
yellow_tripdata_1m.createOrReplaceTempView("yellow_tripdata_1m")

# execute query 1 using SQL with parquet data
res = spark.sql("""SELECT hour(to_timestamp(Start_Datetime)) AS Hour, avg(Start_Longitude) AS Longitude, avg(Start_Latitude) AS Latitude 
                   FROM yellow_tripdata_1m 
                   WHERE ((to_timestamp(Start_Datetime) < to_timestamp(End_Datetime)) AND ((Start_Longitude != End_Longitude) AND 
                   (Start_Latitude != End_Latitude)) AND Cost > 0 AND
                   (Start_Longitude > -80) AND (Start_Longitude < -70) AND (Start_Latitude > 40) AND (Start_Latitude < 46) AND
                   (End_Longitude > -80) AND (End_Longitude < -70) AND (End_Latitude > 40) AND (End_Latitude < 46))
                   GROUP BY Hour 
                   ORDER BY Hour ASC""") 

#execute lazy and show results
res.show(24)                 
print("Time of Query 1 using SQL with parquet is: %s seconds" % (time.time() - start_time_parquet))
