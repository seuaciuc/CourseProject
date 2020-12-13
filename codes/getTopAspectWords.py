'''
INPUTS
'''

dataFile = 'MP3reviews_low_100.pkl' # processed data file. Needs to be in the data folder
modelFile = 'MP3model_low_100_3.pkl' # model file. Needs to be in the model folder
N = 10 # number of top words to retrieve for each aspect


###############################################################################
###############################################################################
#### END OF INPUTS ############################################################
#### DO NOT EDIT CODE BELOW ###################################################
###############################################################################
###############################################################################

'''
Do not edit any of the code below. Only provide the inputs above
'''

import os, pickle
import numpy as np

# folder management
codefolder = os.getcwd()
basefolder = os.path.dirname(codefolder)
datafolder = os.path.join(basefolder,'data')
modelfolder = os.path.join(basefolder,'models')

# load reviews & vocabulary
pklfile = os.path.join(datafolder,dataFile)
fid = open(pklfile,'rb')
reviews = pickle.load(fid)
voc = pickle.load(fid)
fid.close()

# load model
pklfile = os.path.join(modelfolder,modelFile)
fid = open(pklfile,'rb')
modelParams = pickle.load(fid)
reviewParams = pickle.load(fid)
fid.close()


eps = modelParams['eps'] # word distributions for each aspect
(K,V) = eps.shape # number of aspects and vocabulary size
voc = np.array(voc) # turn vocabulary into numpy array

words = []
for i in range(K):
    idx = np.argsort(eps[i,:]) # sort indices
    top = idx[-N:] # retrieve top N indices
    words.append(voc[top])

for i,w in enumerate(words):
    print('Top ' + str(N) + ' words in aspect #' + str(i) + ': ' + str(w.tolist()))