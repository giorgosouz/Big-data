from pyspark import SparkConf, SparkContext
import os
import csv
from datetime import datetime
import math
from math import radians, cos, sin, asin, sqrt
from operator import itemgetter
from functools import partial
import time
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Q2_rdd").getOrCreate()
sc = spark.sparkContext

## calculate Haversine distance
def haversine(long1, lat1, long2, lat2):
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((long2-long1)/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = 6371 * c
    return d


## parsing the data per line from the csv
def parse_Data(arg):
    ##split csv line
    line = arg.split(",")
    ##get taxii's id
    id_ = line[0]
    ##extract starting and ending time
    datetime_start = datetime.strptime(line[1], "%Y-%m-%d %H:%M:%S")
    datetime_end = datetime.strptime(line[2], "%Y-%m-%d %H:%M:%S")
    ## calculate duration of the ride
    duration = (datetime_end - datetime_start).total_seconds() / 60.0
    ##calculate haversine distance
    distance = haversine(float(line[3]), float(line[4]), float(line[5]), float(line[6]))
    ## return taxis's id with the total distance and the duration of the ride
    return id_, [distance, duration]



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
    return ((start_time < end_time) and (start_longitude > -80) and (end_longitude > -80) and (start_longitude < -70) and 
            (end_longitude < -70) and (start_latitude > 40) and (end_latitude > 40) and (start_latitude < 46) and (end_latitude < 46)
            and ((start_longitude != end_longitude) and (start_latitude != end_latitude)) and cost > 0)
    

start_time_mapreduce = time.time()

yellow_tripvendors_1m = sc.textFile("hdfs://master:9000/data/yellow_tripvendors_1m.csv")\
                          .map(lambda x: x.split(","))

result = sc.textFile("hdfs://master:9000/data/yellow_tripdata_1m.csv")\
           .filter(lambda x: (filter_Data(x)))\
           .map(lambda x: (parse_Data(x)))\
           .join(yellow_tripvendors_1m)\
           .map(lambda x: (x[1][1], x[1][0][0], x[1][0][1]))\
           .map(lambda x: (x[0], x))\
           .reduceByKey(lambda x, y: x if x[1] > y[1] else y)\
           .sortByKey()\
           .values().collect()

print("Vendor    |   Distance    |   Duration")
for x in result:
    print(x)
print("Time of Query 2 using Map-Reduce with csv is: %s seconds" % (time.time() - start_time_mapreduce))
