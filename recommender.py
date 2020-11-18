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
als_model = ALSModel.load("models/final")

# number of recommendations to generate per user
n = 5

# Generate n Recommendations for all users
# top_n_als_recs = als_model.recommendForAllUsers(n)

# Generate n Recommendations for a subset of users
top_n_als_recs = als_model.recommendForUserSubset(users, n)

# Explode recommendations column into multiple rows per userId 
top_n_als_recs = top_n_als_recs.withColumn("exploded", F.explode(F.col("recommendations")))\
                                .select('userId', F.col("exploded.movieId"), F.col("exploded.rating"))\
                                .drop("recommendations", "exploded")

# Join with movies on movieId
top_n_als_recs = top_n_als_recs.join(movies, "movieId")

# Print ALS Recommendations to console
print("ALS Recommendations:\n")
top_n_als_recs = top_n_als_recs.select("*").toPandas()
print(top_n_als_recs)

# Print CB Recommendations to console
print("Content-Based Recommendations:\n")
top_n_cb_recs = user_topN_related_movies(listOfUsers, n)

# add columns
top_n_als_recs['algo'] = 'ALS'
top_n_cb_recs['algo'] = 'CB'
top_n_cb_recs['rating'] = None

# Output Recommendations to csv
top_n_als_recs.to_csv(
    "top_n_recommendations.csv", 
    columns=['userId','movieId','rating','title','genres','algo'], 
    index=False
)
top_n_cb_recs.to_csv(
    'top_n_recommendations.csv', 
    columns=['userId','movieId','rating','title','genres','algo'],
    mode='a', 
    index=False,
    header=False
)
