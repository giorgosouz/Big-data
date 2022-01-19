from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import csv
import time
from datetime import datetime



spark = SparkSession.builder.appName("Query_1_RDD").getOrCreate()
sc = spark.sparkContext

## parsing the data per line from the csv
def parse_Data(arg):
	##split csv line
    line = arg.split(",")
    ##extract the time
    start_time = datetime.strptime(line[1], "%Y-%m-%d %H:%M:%S")
    ##and then the hours
    hour = "{:02d}".format(start_time.hour)
    ##extract coordinates
    longitude = float(line[3])
    latitude = float(line[4])
    ## get hour and a tupple with coordinates and a 1 so as to compute the mean later        
    return hour, (longitude, latitude, 1)

## detect outlier data so as to ignore them
def filter_Data(arg):
    ##split csv line
    line = arg.split(",")
    ##extract the time
    start_time = datetime.strptime(line[1], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(line[2], "%Y-%m-%d %H:%M:%S")
    ##extract coordinates
    start_longitude, start_latitude = float(line[3]), float(line[4])
    end_longitude, end_latitude = float(line[5]), float(line[6])
    ##extract cost
    cost = float(line[7])
    ##filtering for dirty data:
    ## 1) start time later than end time
    ## 2) location must be in new york and the area around the city and not really far away
    ## 3) starting point must be different than the ending point
    ## 4) cost greater than zero, rides are not done for free
    return ((start_time < end_time) and 
	    	(start_longitude > -80) and 
	    	(end_longitude > -80) and 
	    	(start_longitude < -70) and 
	        (end_longitude < -70) and 
	        (start_latitude > 40) and 
	        (end_latitude > 40) and 
	        (start_latitude < 46) and 
	        (end_latitude < 46) and
	        ((start_longitude != end_longitude) and 
	       	(start_latitude != end_latitude)) and 
	       	cost > 0)


start_time_mapreduce = time.time()

## execute query 1 with MapReduce
result = sc.textFile("hdfs://master:9000/data/yellow_tripdata_1m.csv")\
            .filter(lambda x: (filter_Data(x)))\
            .map(lambda x: (parse_Data(x)))\
            .reduceByKey(lambda x,y : (x[0]+y[0], x[1]+y[1], x[2]+y[2]))\
            .map(lambda x: (x[0], x[1][0]/x[1][2], x[1][1]/x[1][2]))\
            .sortBy(lambda x: x[0])\
            .collect()
          
print("HourOfDay    |   Longitude    |   Latitude")
for x in result:
    print(x)
print("Time of Query 1 using Map-Reduce with csv is: %s seconds" % (time.time() - start_time_mapreduce)) 
