# Recommender System

Our project is a recommender system to recommend movies to users based on their ratings on other movies. The project uses a dataset of movie ratings by many users to generate a model that is used to create the recommendation list. We use two methods to generate the recommendation list: Content Based recommendation, and Collaborative Filtering recommendation. From the input dataset, any users that provided less than or equal to a certain threshold of ratings would be used as input for Content Based recommendation, where users that provided more than a certain threshold of ratings would be used as input for Collaborative Filtering recommendation.

The inputs to generate a recommendation list are the userIds of users that we want to generate recommendations for. The output is a csv file that contains a list of movies recommended to each userId. 

# Required Libraries
`pip install pandas`

`pip install numpy`

`pip install scikit-learn`

`pip install pyspark`

# How to run the code
1. To generate recommendations using the trained ALS model, run `python3 recommender.py [userId*]`

   Example: `python3 recommender.py 1 2 3 4 5`

2. To get pregenerated recommendations, run `python3 recommender_pregenerated.py [userId*]`

   Example: `python3 recommender_pregenerated.py 1 2 3 4 5`

Both options generate an output file `top_n_recommendations.csv` containing the top `n = 5` recommendations for the specified userIds.

# Example Output
|userId|movieId|rating|title|genres|algo|
|------|-------|------|-----|------|----|
|1|3379|5.6596465|On the Beach (1959)|Drama|ALS|
|1|171495|5.5848684|Cosmos|(no genres listed)|ALS|
|1|5490|5.582294|The Big Bus (1976)|Action|Comedy|ALS|
|1|179135|5.5318403|Blue Planet II (2017)|Documentary|ALS|
|1|86237|5.5318403|Connections (1978)|Documentary|ALS|
|4|3851|4.851076|I'm the One That I Want (2000)|Comedy|ALS|
|4|1046|4.729497|Beautiful Thing (1996)|Drama|Romance|ALS|
|4|1262|4.7247959999999996|"Great Escape, The (1963)"|Action|Adventure|Drama|War|ALS|
|4|25825|4.7231793|Fury (1936)|Drama|Film-Noir|ALS|
|4|4765|4.710532|L.I.E. (2001)|Drama|ALS|
|3|1208||Apocalypse Now (1979)|Action|Drama|War|CB|
|3|293||Léon: The Professional (a.k.a. The Professional) (Léon) (1994)|Action|Crime|Drama|Thriller|CB|
|3|30||Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)|Crime|Drama|CB|
|3|76||Screamers (1995)|Action|Sci-Fi|Thriller|CB|
|3|6849||Scrooge (1970)|Drama|Fantasy|Musical|CB|
|2|2159||Henry: Portrait of a Serial Killer (1986)|Crime|Horror|Thriller|CB||
|2|25||Leaving Las Vegas (1995)|Drama|Romance|CB|
|2|2||Jumanji (1995)|Adventure|Children|Fantasy|CB|
|2|257||Just Cause (1995)|Mystery|Thriller|CB|
|2|435||Coneheads (1993)|Comedy|Sci-Fi|CB|
