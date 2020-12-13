'''
INPUTS
'''
# provide the processed MP3 review file name (pkl file). Must be in the data folder
inputFile = 'MP3reviews_high.pkl'
outputFile = 'MP3model_high_3.pkl'


###############################################################################
###############################################################################
#### END OF INPUTS ############################################################
#### DO NOT EDIT CODE BELOW ###################################################
###############################################################################
###############################################################################

'''
Do not edit any of the code below. Only provide the inputs above
'''

NUM_ASPECTS = 3 # number of aspects
MAX_REVIEW_ITER = 50 # maximum number of iterarions for review parameters
MAX_MODEL_ITER = 50 # maximum number of iterations for model parameters
EPS = 0.01 # minimum change in likelihood for convergence

import os, pickle, utils, time
import numpy as np

# folder management
codefolder = os.getcwd()
basefolder = os.path.dirname(codefolder)
datafolder = os.path.join(basefolder,'data')
modelfolder = os.path.join(basefolder,'models')

# load reviews
pklfile = os.path.join(datafolder,inputFile)
fid = open(pklfile,'rb')
reviews = pickle.load(fid)
voc = pickle.load(fid)
tdm = pickle.load(fid)
fid.close()

# initialize model parameters
V = len(voc) # vocabulary size
K = NUM_ASPECTS # number of aspects
M = len(reviews) # number of reviews (documents) in corpus
eps = np.ones((K,V)) / V # word (col) distribution for each aspect (row)
gamma = np.ones(K) / K
beta = np.random.randn(K,V)
mu = np.random.randn(K)
SIG = np.random.randn(K,K)
SIG = SIG.transpose()*SIG
delta2 = abs(np.random.randn())
# group model parameters in dictionary
modelParams = {'eps':eps, 'gamma':gamma, 'beta':beta, 'mu':mu, 'SIG':SIG, 'delta2':delta2}

oldL = 0
for iterModel in range(MAX_MODEL_ITER):
    t = time.strftime("%D %H:%M:%S", time.localtime())
    print(t, "Iteration #" + str(iterModel + 1) + "...")
    # compute review parameters
    print(t, " ---> Computing review parameters...")
    reviewParams = utils.computeReviewParams(reviews, tdm, modelParams, MAX_REVIEW_ITER, EPS)
    # update model parameters
    print(t, " ---> Updating model parameters...")
    modelParams = utils.updateModelParams(reviews, modelParams, reviewParams, tdm)
    # compute new L
    Ld = np.zeros(M)
    for idx, review in enumerate(reviews):
        Ld[idx] = utils.computeReviewLog(review, reviewParams[idx], modelParams, tdm[idx,:])
    newL = Ld.sum()
    print(" ---> L="+ "{:.2f}".format(newL))
    # assess change
    if oldL==0:
        oldL = newL
        continue
    if abs((newL-oldL)/newL)<EPS:
            converged = True
            L = newL
            break
    else:
        # update
         oldL = newL
if not converged:
        print('Max model iteration reached.')

### SAVE FILE
t = time.strftime("%D %H:%M:%S", time.localtime())
print(t,'Saving file...')
fullOutputFile = os.path.join(modelfolder,outputFile)
fout = open(fullOutputFile,'wb')
pickle.dump(modelParams, fout,-1)
pickle.dump(reviewParams, fout,-1)
fout.close()
t = time.strftime("%D %H:%M:%S", time.localtime())
print(t, 'Done!')





    
    
 

