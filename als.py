from modifyInput import splitCSVBasedOnNumReviewPerUser
from pyspark.mllib.recommendation import MatrixFactorizationModel
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Output manyDf to csv, use as input for ALS: only need to run once!
# lessDf, manyDf = splitCSVBasedOnNumReviewPerUser('data/ml-latest-small/ratings.csv')
# manyDf.to_csv('data/als-input.csv', index=False)

# Create spark context
sc = SparkContext("local", "ALS")
spark = SparkSession.builder.appName('Recommendations').getOrCreate()

# Load and parse the data
# data = sc.textFile("data/als-input.csv")
ratings = spark.read.csv("data/als-input.csv", inferSchema=True, header=True)
ratings.show()

# Create test and training set
(train, test) = ratings.randomSplit([0.8, 0.2], seed = 2020)

# # Build the recommendation model using Alternating Least Squares
# rank = 10
# numIterations = 10 # default 5
# lambda_val = 0.01 # default 0.01
# model = ALS.train(ratings, rank, iterations=numIterations, lambda_=lambda_val, nonnegative=True)
als = ALS(
         userCol="userId", 
         itemCol="movieId",
         ratingCol="rating", 
         nonnegative = True, 
         implicitPrefs = False,
         coldStartStrategy="drop"
)

# hyperparameter grid
param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [10, 50]) \
            .addGrid(als.regParam, [.01, .05, .1, .15]) \
            .build()

# evaluator (rmse)
evaluator = RegressionEvaluator(
           metricName="rmse", 
           labelCol="rating", 
           predictionCol="prediction") 
print ("Num models to be tested: ", len(param_grid))

# cross validator: 5-fold verification
cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

#Fit cross validator to the 'train' dataset
model = cv.fit(train)
#Extract best model from the cv model above
best_model = model.bestModel
# View the predictions
test_predictions = best_model.transform(test)
RMSE = evaluator.evaluate(test_predictions)
print(RMSE)

print("**Best Model**")
# Print "Rank"
print("  Rank:", best_model._java_obj.parent().getRank())
# Print "MaxIter"
print("  MaxIter:", best_model._java_obj.parent().getMaxIter())
# Print "RegParam"
print("  RegParam:", best_model._java_obj.parent().getRegParam())

best_model.save("models/als-best")

# Generate n Recommendations for all users
recommendations = best_model.recommendForAllUsers(5)
recommendations.show()

# nrecommendations = recommendations\
#     .withColumn("rec_exp", explode("recommendations"))\
#     .select('userId', col("rec_exp.movieId"), col("rec_exp.rating"))
# nrecommendations.limit(10).show()

# load the best model (for future reference)
# sameModel = MatrixFactorizationModel.load(sc, "models/als-best")