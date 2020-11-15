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

# number of recommendations to generate per user
n = 5

# select top n recommendations per user
top_n_als_recs = als_recs.loc[als_recs['userId'].isin(listOfUsers)].groupby("userId").head(n)

# Output Recommendations to csv
top_n_als_recs.to_csv("top_n_recommendations.csv", index=False)
