from modifyInput import splitCSVBasedOnNumReviewPerUser
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
from pyspark import SparkContext
from pyspark.sql import SparkSession

# Create spark context
sc = SparkContext("local", "CB")
spark = SparkSession.builder.appName('Recommendations2').getOrCreate()

#Output manyDf to csv, use as input for ALS: only need to run once!
#lessDf, manyDf = splitCSVBasedOnNumReviewPerUser('data/ml-latest-small/ratings.csv')
#lessDf['userId'] = lessDf['userId'].transform(lambda x: int(x))
#lessDf.to_csv('data/cb-input.csv', index=False)

# Load data
movies = pd.read_csv("data/ml-latest-small/movies.csv")
ratings = pd.read_csv("data/cb-input.csv").sort_values("rating", ascending = False, na_position ='last') 

# Left join rating & movies
ratings_movie = pd.merge(ratings, movies, on='movieId', how='left')
ratings_movie.sort_values("userId", ascending = True, na_position ='last')
ratings_movie.to_csv('data/ratings_movie.csv', index=False)

user_preference = ratings_movie[['userId', 'genres']]
user_preference['genres'] = user_preference.groupby(['userId'])['genres'].transform(lambda x: "|".join(x))

tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
movies['genres'] = movies['genres'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute consine
consine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
#print("consine_similarity\n", consine_similarity)

# Construct a map with movie titles
mapping = pd.Series(movies.index, index=movies['title'])
#print("Map with movie title example row0 \n:", mapping[0])

def content_based_recommendations(input_title, top_N):
    # Get the index of the input movie
    input_index = mapping[input_title]
    if isinstance(input_index, pd.Series):
        consine_list = []
        for index in input_index:
            tmp = list(enumerate(consine_similarity[index]))
            consine_list.append(tmp)
        similarity_values = list(enumerate(consine_list))
    else:
        # Get similarity values of all movies with the input movie
        similarity_values = list(enumerate(consine_similarity[input_index]))
    
    similarity_values = sorted(similarity_values, key=lambda x: x[1], reverse=True)
    
    # Get the top N score
    similarity_values = similarity_values[1:top_N]
    # Get the movie indices
    movie_indices = []
    for i in similarity_values:
        movie_indices.append(i[0])
    return (movies.iloc[movie_indices])

def user_topN_related_movies(userIds, n):
    user_topN_movies = ratings_movie.loc[ratings_movie['userId'].isin(userIds)]
    user_topN_movies.sort_values("rating", ascending = False, na_position ='last')
    user_topN_movies = user_topN_movies.groupby("userId").head(n)
    user_topN = user_topN_movies["userId"].unique()
    #print("user_topN_movies_list:\n", user_topN, "\n")
    user_topN_movies_list = user_topN_movies["title"].values.tolist()  
    recommendations = pd.DataFrame() 
    #print("user_topN_movies_list:\n", user_topN_movies, "\n")
    j = 0
    i = n
    for m in user_topN_movies_list:
        if i <= 0 :
            i = n
            j = j+1
        
        single = content_based_recommendations(m, 2)
        single["userId"] = user_topN[j]
        recommendations = recommendations.append(single)
        i = i-1
    recommendations = recommendations[['userId','movieId', 'title','genres']]
    print("recommendations:\n", recommendations, "\n")
    #recommendations.to_csv('top10_recs_content_based.csv', mode='a', index=False) // to generate top10 recommendations for lessDf
    return recommendations

#all = ratings["userId"].unique()
#user_topN_related_movies(all, 10) 
