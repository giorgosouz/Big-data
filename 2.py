from __future__ import print_function
import os
from datetime import datetime
import math
from math import radians, cos, sin, asin, sqrt, log
from operator import itemgetter
from functools import partial
import time

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from pyspark.ml.linalg import SparseVector
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import MultilayerPerceptronClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import re
import nltk
## utilize english stopwords from nltk to clean our data
nltk.download('stopwords')
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words('english'))
eng_stopwords.add('xxxx')
eng_stopwords.add('xx')
eng_stopwords.add('')
eng_stopwords.add('xxxxxxxx')
eng_stopwords.add('xxxxxxxxxxxx')



spark = SparkSession.builder.appName("ex_2_ML").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)


## check each row if it has a valid complaint
## must have 3 columns, date must start with 201 and the complaint part must not be empty
def filter_Data(arg):
    line = arg.split(",")
    if len(line) == 3:
        date = line[0]
        productCategory = line[1]
        complaints = line[2]
        return (date.startswith("201") and bool(complaints and complaints.strip()))
    else:
        return False


## parse the cleaned data, droping the date and keeping product category and complain
def get_Data(arg):
    line = arg.split(",")
    date = line[0]
    productCategory = line[1]
    complaints = line[2].lower()
    return productCategory, complaints


## read the data, clean them and cache it
customer_complaints = sc.textFile("hdfs://master:9000/data/customer_complaints.csv")\
                        .filter(filter_Data)\
                        .map(get_Data)\
                        .cache()


## create full lexicon of all the words in our cleaned data
## split each sentence with the space character, delete all non latin alphabetical characters, convert all letters to lowercase
## count all the words, sort them in descending order and produce only the descending list of the words
full_lexicon = customer_complaints.flatMap(lambda x : x[1].split(" "))\
                                  .map(lambda x: re.sub('[^a-zA-Z]+', '', x))\
                                  .filter(lambda x: x.lower() not in eng_stopwords)\
                                  .map(lambda x : (x, 1))\
                                  .reduceByKey(lambda x, y: x + y)\
                                  .sortBy(lambda x : x[1], ascending = False)\
                                  .map(lambda x : x[0])


## keep the 200 most frequent words of the lexicon created before
## and broadcast them on all nodes
used_lexicon_size = 200
print(full_lexicon.take(used_lexicon_size))
used_lexicon = full_lexicon.take(used_lexicon_size)
broad_com_words = sc.broadcast(used_lexicon)


## filter all complaints with the most frequent words we selected before.
## for each complaint keep only those words that are frequent and if the complaint
## is empty after the filtering meaning it has no grequent words then discard it
## attach an index at the data to enumerate the sentences
customer_complaints = customer_complaints.map(lambda x : (x[0], x[1].split(" ")))\
                        .map(lambda x : (x[0], [y for y in x[1] if y in broad_com_words.value]))\
                        .filter(lambda x : len(x[1]) != 0)\
                        .zipWithIndex()
                        # Output Tuple : ((string_label, list_of_sentence_words_in_lexicon), sentence_index)       
# print(customer_complaints.take(5))

number_of_complaints = customer_complaints.count()
#print(number_of_complaints)


## calculate IDF
## using set() to remove duplicate words
## emit (word,1) and add them in order to find in how many documents exists each word
## apply the idf formula
idf = customer_complaints.flatMap(lambda x : [(y, 1) for y in set(x[0][1])])\
                         .reduceByKey(lambda x, y : x + y)\
                         .map(lambda x : (x[0], math.log(number_of_complaints/x[1])))
                         # Output Tuple : (word, idf)
#print(idf.take(5))


## emit for each word in a sentence an 1 and get length of each sentence as a value inside key
## sum for each word in a sentence the 1s to find its count
## calculate tf by dividing word_count and length of its sentence, attach info about the word from the lexicon
## attach idf
## calcualte tfidf and rearrange key and value
## reorder the data inside both key and value, make value a list
## create for each sentence a list with elements the tfidf for each word it has and its corresponding index in lexicon
## sort value data based on the index of the lexicon
## create the final form of the value data as (lexicon_size,sorted words of lexicon, corresponding tfidf for the specific sentence)
customer_complaints = customer_complaints.flatMap(lambda x : [((y, x[0][0], x[1], len(x[0][1])), 1) for y in x[0][1]]
                                         # Output Tuple : ((word, string_label, sentence_index, sentence_length), 1)
                                         

                                         ).reduceByKey(lambda x, y : x + y
                                         # Output Tuple : ((word, string_label, sentence_index, sentence_length), word_count_in_sentence)
                                         

                                         ).map(lambda x : ((x[0][0], (x[0][1], x[0][2], x[1]/x[0][3], broad_com_words.value.index(x[0][0]))))
                                         # Output Tuple : (word, (string_label, sentence_index, tf, word_index_in_lexicon))
                                         

                                         ).join(idf
                                         # Output Tuple : (word, ((string_label, sentence_index, tf, word_index_in_lexicon), idf))
                                         

                                         ).map(lambda x : ((x[0], x[1][0][0], x[1][0][1]), (x[1][0][2]*x[1][1], x[1][0][3]))
                                         # Output Tuple : ((word, string_label, sentence_index), (tf*idf, word_index_in_lexicon))
                                         

                                         ).map(lambda x : ((x[0][2], x[0][1]), [(x[1][1], x[1][0])])
                                         # Output Tuple : ((sentence_index, string_label), [(word_index_in_lexicon, tfidf_in_sentence)])
                                         

                                         ).reduceByKey(lambda x, y : x + y)\
                                         .map(lambda x : (x[0][1], sorted(x[1], key = lambda y : y[0])))\
                                         .map(lambda x : (x[0], SparseVector(used_lexicon_size, [y[0] for y in x[1]], [y[1] for y in x[1]])))
                                         # Output Tuple : (string_label, SparseVector(used_lexicon_size, list_of(word_index_in_lexicon), list_of(tfidf_in_sentence)))
print(customer_complaints.take(5))

## our final output has as key the label of the complaint and as value the (lexicon size, the lexicon indices of the words in the sentence, the corresponding tfidf value)
## we have dropped the sentence index as we dont need it anymore


# Convert RDD to dataframe in order to train the model with SparkML and we name appropriately the columns
customer_complaints_DF = customer_complaints.toDF(["string_label", "features"])

# Convert categorical labels to numerical
stringIndexer = StringIndexer(inputCol="string_label", outputCol="label")
stringIndexer.setHandleInvalid("skip")
stringIndexerModel = stringIndexer.fit(customer_complaints_DF)
customer_complaints_DF = stringIndexerModel.transform(customer_complaints_DF)

customer_complaints_DF.groupBy("label").count().show(10)

# Train-Test split at 70-30 %
train = customer_complaints_DF.sampleBy("label", fractions={0: 0.70, 1: 0.70, 2: 0.70, 3: 0.70, 4: 0.70, 5: 0.70, 6: 0.70, 7: 0.70, 8: 0.70, 9: 0.70, 
                                10: 0.70, 11: 0.70, 12: 0.70, 13: 0.70, 14: 0.70, 15: 0.70, 16: 0.70, 17: 0.70}, seed = 0)

# Subtracting 'train' from original 'customer_complaints_DF' to create test set
test = customer_complaints_DF.subtract(train)

# Checking distributions of all labels in train and test sets after sampling 
train.groupBy("label").count().show()
test.groupBy("label").count().show()

# specify layers for the neural network:
# input layer of size used_lexicon_size (features), one intermediate of size (used_lexicon_size+18)//2
# and output of size 18 (classes)
layers = [ (used_lexicon_size//2), 18]



# Execute ml part twice, one for not cached trainset and one for cached trainset
for i in range(0,2):

    # Orismos montelou
    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=0)
    
    if i == 0:
        print("No cached trainset")
        start_time = time.time()
        # Ekpaideusi sto train set kai aksiologisi sto test set
        # Fit the model
        model = trainer.fit(train)
        time_no_cache = time.time() - start_time
        print("Time of fit: %s seconds" %time_no_cache)
        # compute accuracy on the test set
        # Kanoume transform panw sto montelo to test set kai pairnoume mia nea stili sto test dataframe pou perilambanei ta predictions
        result = model.transform(test)
        # Kratame ta pragmatika labels kai ta predictions
        predictionAndLabels = result.select("prediction", "label") 
        # Orizoume enan evaluator pou mporei na upologisei to accuracy
        evaluator = MulticlassClassificationEvaluator(metricName="accuracy") 
        # Ypologizoume kai tupwnoume to accuracy score me vash ta predictions/labels pou apomonwsame nwritera
        print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
    if i == 1:
        spark.catalog.clearCache()
        print("Cached trainset")
        train = train.cache()
        start_time = time.time()
        model = trainer.fit(train)
        time_cache = time.time() - start_time
        print("Time of fit: %s seconds" %time_cache)
        result = model.transform(test)
        predictionAndLabels = result.select("prediction", "label")
        evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
        print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
