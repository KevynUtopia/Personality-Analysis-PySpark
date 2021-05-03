# Databricks notebook source
# MAGIC %md
# MAGIC ## Part 1: Go through teh dataset

# COMMAND ----------

# Load data
file_location = "/FileStore/tables/data_final-1.csv"
df = spark.read.load(file_location, format="csv", sep="\t", inferSchema="true", header="true")
display(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC Process the value into float type for later processing

# COMMAND ----------

features = ["EXT1", "EXT2" ,"EXT3" ,"EXT4" ,"EXT5" ,"EXT6" ,"EXT7" ,"EXT8" ,"EXT9" ,"EXT10",
            "EST1" ,"EST2" ,"EST3" ,"EST4" ,"EST5" ,"EST6" ,"EST7" ,"EST8" ,"EST9" ,"EST10",
            "AGR1" ,"AGR2" ,"AGR3" ,"AGR4" ,"AGR5" ,"AGR6" ,"AGR7" ,"AGR8" ,"AGR9" ,"AGR10",
            "CSN1" ,"CSN2" ,"CSN3" ,"CSN4" ,"CSN5" ,"CSN6" ,"CSN7" ,"CSN8" ,"CSN9" ,"CSN10",
            "OPN1" ,"OPN2" ,"OPN3" ,"OPN4" ,"OPN5" ,"OPN6" ,"OPN7" ,"OPN8" ,"OPN9" ,"OPN10"
           ]


for each_feature in features:
    df = df.withColumn(each_feature, df[each_feature].cast(FloatType()))
    
df.printSchema()

# COMMAND ----------

numeric_feats = [item[0] for item in df.dtypes if item[1].startswith('float')]
print(numeric_feats)

# COMMAND ----------

# MAGIC %md
# MAGIC Fill the missing values with the mean value

# COMMAND ----------

from pyspark.ml.feature import Imputer

imputer = Imputer(strategy='mean', inputCols=numeric_feats, outputCols=numeric_feats)
df = imputer.fit(df).transform(df)

# COMMAND ----------

# MAGIC %md  
# MAGIC Solve the problem of data skewness

# COMMAND ----------

from pyspark.sql.functions import skewness

skewed_feats = []
for col in numeric_feats:
    s = df.select(skewness(df[col]).alias(col)).collect()
    skew_value = float(s[0][col])
    if (skew_value > 0.75):
        print (col, "\t", skew_value)
        skewed_feats.append(col)

print(skewed_feats)

# COMMAND ----------

from pyspark.sql.functions import log1p

for feat in skewed_feats:
   df = df.withColumn(feat, log1p(df[feat]))

# COMMAND ----------

# MAGIC %md
# MAGIC Encode nationality with onehot encoder

# COMMAND ----------

categorical_feats = ['country']
from pyspark.ml.feature import OneHotEncoder, StringIndexer

td = df
for feat in categorical_feats:
    stringIndexer = StringIndexer(inputCol=feat, outputCol=feat+"_indexed")
    model = stringIndexer.fit(td)
    td = model.transform(td)

# COMMAND ----------

encoder = OneHotEncoder(inputCols=[feat+"_indexed" for feat in categorical_feats], outputCols=[feat+"_ohe" for feat in categorical_feats])
model = encoder.fit(td)
encoded = model.transform(td)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

updated_feats = [feat+"_ohe" for feat in categorical_feats]
updated_feats += numeric_feats

assembler = VectorAssembler(inputCols=updated_feats, outputCol='features')
output = assembler.transform(encoded)

output.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Convert the target variable back to integer 1~5.

# COMMAND ----------

from pyspark.sql.functions import when, lit
import math
# output = output.withColumn('EXT1', int(df['EXT1'])

output = output.withColumn('EXT1', df['EXT1'].cast('int'))
output.select('EXT1').show()

# COMMAND ----------

(trainingData, testData) = output.randomSplit([0.7, 0.3])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression  
# MAGIC By using all the numerical data, together with encoded nationality, predict the the answer of the first question "I am the life of the party." in the questionnaire with logistic regression model 

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol='features',labelCol='EXT1')

lrmodel = lr.fit(trainingData)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
predictions = lrmodel.transform(testData)
evaluator = MulticlassClassificationEvaluator(
    labelCol="EXT1", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC Testing accuracy reaches 97.90%, which is very good

# COMMAND ----------


