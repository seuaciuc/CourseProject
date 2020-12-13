'''
INPUTS
'''
# provide the processed MP3 review file name (pkl file). Must be in the data folder
inputFile = 'HotelReviews.pkl'


###############################################################################
###############################################################################
#### END OF INPUTS ############################################################
#### DO NOT EDIT CODE BELOW ###################################################
###############################################################################
###############################################################################

'''
Do not edit any of the code below. Only provide the inputs above
'''

import os, pickle, math
import numpy as np

# folder management
codefolder = os.getcwd()
basefolder = os.path.dirname(codefolder)
datafolder = os.path.join(basefolder,'data')

# load reviews
pklfile = os.path.join(datafolder,inputFile)
fid = open(pklfile,'rb')
reviews = pickle.load(fid)
fid.close()

# get stats
noReviews = len(reviews)
rating = np.zeros(noReviews)
txtlen = np.zeros(noReviews)
productList = []
avgLength = 0
for idx, review in enumerate(reviews):
    txtlen[idx] = len(review['ReviewText'])
    rating[idx] = review['Rating'][0]
    productList.append(review['Product'])

productList = list(set(productList))
noItems = len(productList)    
avgLength = np.mean(txtlen)
stdLength = np.std(txtlen)
avgLengthStd= stdLength/math.sqrt(noReviews)
avgRating = np.mean(rating)
stdRating = np.std(rating)
avgRatingStd = stdRating/math.sqrt(noReviews)

# print results
print('      File: ' + inputFile)
print('   # Items: ' + str(noItems))
print(' # Reviews: ' + str(noReviews))
print('Avg Length: ' + "{:.2f}".format(avgLength) + ' (' + "{:.2f}".format(stdLength) + ')')
print('    Rating: ' + "{:.2f}".format(avgRating) + ' (' + "{:.2f}".format(stdRating) + ')')