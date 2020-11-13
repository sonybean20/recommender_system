import pandas as pd
import numpy as np

def splitCSVBasedOnNumReviewPerUser(file_path, threshHold=200):	
	# csvFileName = sys.argv[1]
	df = pd.read_csv(file_path)

	manyReview = []
	lessReview = []

	curUserReviewList = []
	curUserReview = []

	for column in df.columns:
		curUserReview.append(df.loc[0][column])

	curUserReviewList.append(curUserReview)

	curUserReview = []

	beforeUserId = df.loc[0]['userId']
	curUserId = df.loc[0]['userId']

	reviewCount = 1

	# print(len(df.index))

	for x in df.index[1:]:
		curUserId = df.loc[x]['userId']
		if curUserId != beforeUserId:
			if reviewCount > threshHold:
				for y in range(len(curUserReviewList)):
					manyReview.append(curUserReviewList[y])
			else:
				for y in range(len(curUserReviewList)):
					lessReview.append(curUserReviewList[y])
			curUserReviewList = []
			reviewCount = 0
			beforeUserId = curUserId

		for column in df.columns:
			curUserReview.append(df.loc[x][column])

		curUserReviewList.append(curUserReview)
		curUserReview = []
		reviewCount += 1

	if reviewCount > threshHold:
		for y in range(len(curUserReviewList)):
			manyReview.append(curUserReviewList[y])
	else:
		for y in range(len(curUserReviewList)):
			lessReview.append(curUserReviewList[y])

	manyDf = pd.DataFrame(manyReview, columns=df.columns.values)
	# print(len(manyDf.index))


	lessDf = pd.DataFrame(lessReview, columns=df.columns.values)
	# print(len(lessDf.index))

	return [lessDf, manyDf]

