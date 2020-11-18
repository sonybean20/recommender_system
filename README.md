# Recommender System

Our project is a recommender system to recommend movies to users based on their ratings on other movies. The project uses a dataset of movie ratings by many users to generate a model that is used to create the recommendation list. We use two methods to generate the recommendation list: Content Based recommendation, and Collaborative Filtering recommendation. From the input dataset, any users that provided less than or equal to a certain threshold of ratings would be used as input for Content Based recommendation, where users that provided more than a certain threshold of ratings would be used as input for Collaborative Filtering recommendation.

The inputs to generate a recommendation list are the userIds of users that we want to generate recommendations for. The output is a csv file that contains a list of movies recommended to each userId. 

# Required Libraries
`pip install pandas`
`pip install numpy`
`pip install scikit=learn`
`pip install pyspark`

# How to run the code
To generate recommendations using the trained ALS model, run
`python3 recommender.py`

To print pregenerated recommendations, run
`python3 recommender_pregenerated.py`

# Example Output
