from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
import sys 
from pyspark.sql import functions as F
from contentbased import user_topN_related_movies

if len(sys.argv) < 2:
    print('python recommender.py listOfUserIds*')
    print('Example: python recommender.py 1 2 3 4')
    exit()

listOfUsers = sys.argv[1:]
users_str = ','.join(listOfUsers)
print(f'Recommendations for {users_str}.' )

# Create spark context
spark = SparkSession.builder.appName('Recommender').getOrCreate()

movies = spark.read.csv("data/ml-latest-small/movies.csv", inferSchema=True, header=True)
ratings = spark.read.csv("data/ml-latest-small/ratings.csv", inferSchema=True, header=True)
ratings.createOrReplaceTempView("ratings")
users = spark.sql("SELECT DISTINCT userId FROM ratings WHERE userId IN (" + users_str +")")
users.show()

# load the best ALS model
als_model = ALSModel.load("models/als-best")

# number of recommendations to generate per user
n = 5

# Generate n Recommendations for all users
# recommendations = als_model.recommendForAllUsers(n)

# Generate n Recommendations for a subset of users
recommendations = als_model.recommendForUserSubset(users, n)

# Explode recommendations column into multiple rows per userId 
recommendations = recommendations.withColumn("exploded", F.explode(F.col("recommendations")))\
                                .select('userId', F.col("exploded.movieId"), F.col("exploded.rating"))\
                                .drop("recommendations", "exploded")

# Join with movies on movieId
recommendations = recommendations.join(movies, "movieId")

# Print ALS Recommendations to console
print("ALS Recommendations type:\n")
recommendations.show()

# Output ALS Recommendations to csv
recommendations.select("userId", "movieId", "rating", "title", "genres")\
               .coalesce(1)\
               .write.option("header","true")\
               .csv('als_recommendations.csv')

print("Content-Based Recommendations:\n")
recommendations2 = user_topN_related_movies(listOfUsers, n)
