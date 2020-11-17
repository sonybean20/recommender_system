import sys 
import pandas as pd

if len(sys.argv) < 2:
    print('python recommender.py listOfUserIds*')
    print('Example: python recommender.py 1 2 3 4')
    exit()

listOfUsers = sys.argv[1:]
users_str = ','.join(listOfUsers)
print(f'Recommendations for {users_str}.' )

# read top 10 recs for all users (ALS)
als_recs = pd.read_csv("top10_recs_all_users.csv")

# read top 10 recs for all users (Content Based)
cb_recs = pd.read_csv("top10_recs_content_based.csv")

# number of recommendations to generate per user
n = 5

# select top n recommendations per user
top_n_als_recs = als_recs.loc[als_recs['userId'].isin(listOfUsers)].groupby("userId").head(n)
top_n_cb_recs = cb_recs.loc[cb_recs['userId'].isin(listOfUsers)].groupby("userId").head(n)

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
