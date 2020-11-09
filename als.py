from modifyInput import splitCSVBasedOnNumReviewPerUser
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext

# lessDf, manyDf = splitCSVBasedOnNumReviewPerUser('data/ml-latest-small/ratings.csv')
# manyDf.to_csv('data/als-input.csv', index=False, header=False)

sc = SparkContext("local", "ALS")

# Load and parse the data
data = sc.textFile("data/als-input.csv")
ratings = data.map(lambda l: l.split(','))\
    .map(lambda l: Rating(int(float(l[0])), int(float(l[1])), float(l[2])))

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10 # default 5
lambda_val = 0.01 # default 0.01
model = ALS.train(ratings, rank, iterations=numIterations, lambda_=lambda_val, nonnegative=True)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

# Save and load model
model.save(sc, "models/als")
sameModel = MatrixFactorizationModel.load(sc, "models/als")