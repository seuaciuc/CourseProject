import string, math
import numpy as np
from scipy import special as sc
from scipy import optimize as opt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# HELPER FUNCTIONS



###############################################################################
# this functions performs basic text processing to turn a input text into a series of tokenized words
# It performs the following actions, in order:
#   - set all text to lower case
#   - removes punctuation
#   - splits into words
#   - removes non-alphabetic terms and stop words
def processText(text):
    # set to lower case
    text = text.lower()
    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = text.translate(table)
    # split into words
    tokens = word_tokenize(stripped)
    # remove non-alphabetic terms and stopwords
    stop_words = [w.translate(table) for w in set(stopwords.words('english'))]
    words = [word for word in tokens if ((word.isalpha()) and (word not in stop_words))]
    return words
###############################################################################




###############################################################################
# this function filters the reviews:
#   - eliminates entries with fewer than min_review_length (default=50)
#   - eliminates entries with missing ratings (value of -1)
def filterReviews(reviews, min_review_length=50):
    elim = []
    for idx, entry in enumerate(reviews):
        if -1 in entry['Rating']:
            elim.append(idx)
            continue
        if len(entry['ReviewText'])<min_review_length:
            elim.append(idx)
            continue
    filt_reviews = [reviews[i] for i in range(len(reviews)) if i not in elim]
    return filt_reviews
###############################################################################




###############################################################################
# this function builds the vocabulary
def buildVocabulary(reviews):
    vocabulary = []
    for entry in reviews:
        vocabulary = vocabulary + list(set(entry['ReviewText']))
        vocabulary = list(set(vocabulary))
    vocabulary.sort()
    return vocabulary
###############################################################################




###############################################################################
# this function returns the term-doc matrix: each row is a document (review),
# each column is a vocabulary term (word). matrix[w][d] then is the count of 
# term (word) w in document (review) d
# In the process, it applies the condition that a term needs to appear in at least
# min_doc_count reviews, or else it is removed.
# it returns the term-doc matrix and the updated vocabulary
def buildTermDocMatrix(reviews, vocabulary, min_doc_count=0):
    TermDocMatrix = np.empty((len(reviews),0),int)
    elim = []
    for w, word in enumerate(vocabulary):
        col = [0 for i in range(len(reviews))]
        for d, entry in enumerate(reviews):
            col[d] = entry['ReviewText'].count(word)
        col = np.array([col]) # turn into np array
        ispresent = col>0 # boolean flag for term presence in review
        # if it meets criteria, add to TermDocMatrix
        if ispresent.sum() >= min_doc_count:
            TermDocMatrix = np.append(TermDocMatrix, col.transpose(), axis=1)
        #  if it doesn't, mark for elimination from vocabulary
        else:
            elim.append(w)
    # eliminate terms (words) in voabulary
    new_vocab = [vocabulary[idx] for idx in range(len(vocabulary)) if idx not in elim]
    return TermDocMatrix, new_vocab
###############################################################################




###############################################################################
# this function computes the review parameters
def computeReviewParams(reviews, tdm, modelParams, MAX_REVIEW_ITER, EPS):
# compute review parameters
    reviewParams = []
    M = len(reviews)
    Ld = np.zeros(M)
    for idx, review in enumerate(reviews):
        converged = False
        # initialize review parameters
        params = initializeReviewParams(review,modelParams, tdm[idx,:])
        # compute initial Ld
        oldLd = computeReviewLog(review,params,modelParams, tdm[idx,:])
        print('----------> Review #'+str(idx+1)+': '+str(oldLd))
        # update until convergence
        for iterReview in range(MAX_REVIEW_ITER):
            # update review params
            params = updateReviewParams(review,params, modelParams,tdm[idx,:])
            # compute new Ld
            newLd = computeReviewLog(review,params,modelParams,tdm[idx,:])
            #print(newLd)
            # assess change
            if abs((newLd-oldLd)/newLd)<EPS:
                converged = True
                reviewParams.append(params)
                Ld[idx] = newLd
                break
            else:
                # update
                oldLd = newLd
        if not converged:
            print('Max review iteration reached for review '+str(idx))
    return reviewParams
###############################################################################







###############################################################################
# this function computes the likelihood for a given review 
def computeReviewLog(review, reviewParams, modelParams, termvec):
    K = len(modelParams['gamma'])
    gamma = modelParams['gamma']
    mu = modelParams['mu']
    beta = modelParams['beta']
    eps = modelParams['eps']
    delta2 = modelParams['delta2']
    SIG = modelParams['SIG']
    lamb = reviewParams['lamb']
    sigma = reviewParams['sigma']
    phi = reviewParams['phi']
    eta = reviewParams['eta']
    r = review['Rating'][0]
    SIGinv = np.linalg.inv(SIG)
    sbar = np.zeros(K)
    sVar = np.zeros(K)
    wn = np.where(termvec>0)[0] # nonzero elements
    sumgamma = 0
    sumeta = 0
    for i in range(K):
        sumgamma = sumgamma + gamma[i]
        sumeta = sumeta + eta[i]
        for idx, j in enumerate(wn):
            sbar[i] = sbar[i] + termvec[j]*beta[i,j]*phi[i,idx]
            sVar[i] = sVar[i] + termvec[j]*(beta[i,j]**2)*phi[i,idx]*(1-phi[i,idx])
    # rating portion
    Ld = -(
            (np.matmul(lamb.transpose(),sbar) - r)**2/(2*delta2) +
            0.5*np.matmul( (lamb-mu).transpose(), np.matmul(SIGinv,lamb-mu) ) +
            0.5*np.trace( np.matmul(np.diag(sigma),SIGinv) ) + 
            0.5*math.log(delta2) +
            0.5*math.log(abs(np.linalg.det(SIG))) # what if det is negative?
          )
    for i in range(K):
        Ld = Ld - ( (lamb[i]**2+sigma[i])*sVar[i] + sigma[i]*sbar[i]**2 )/(2*delta2)
    # review portion (LDA paper, eqn 15)
    Ld = Ld + ( 
                  sc.loggamma(sumgamma) #math.log(sc.gamma(sumgamma))
                - sc.loggamma(sumeta) #math.log(sc.gamma(sumeta))
              )
    for i in range(K):
        Ld = Ld + (
                    - sc.loggamma(gamma[i]) #math.log(sc.gamma(gamma[i])) 
                    + (gamma[i]-1) * (sc.digamma(eta[i]) - sc.digamma(sumeta) )
                    + sc.loggamma(eta[i])#math.log(sc.gamma(eta[i]))
                    - (eta[i]-1) * (sc.digamma(eta[i]) - sc.digamma(sumeta))
                  )
    for idx, j in enumerate(wn):
        for i in range(K):
            Ld = Ld + (
                          phi[i,idx]*( sc.digamma(eta[i]) - sc.digamma(sumeta) )
                        - phi[i,idx]*math.log(phi[i,idx])
                        + phi[i,idx]*math.log(eps[i,j])*termvec[j]
                      )
    return Ld
    
###############################################################################




###############################################################################
# this function initializes the review parameters for a single document
def initializeReviewParams(review,modelParams,termvec):
    (K,V) = modelParams['beta'].shape
    N = sum(termvec>0) # number of (unique) words in this review
    eta = modelParams['gamma'] + np.ones(K)*N/K
    phi = np.ones((K,N))/K
    lamb = modelParams['mu']
    sigma = np.diag(modelParams['SIG'])
    reviewParams = {'eta':eta,'phi':phi,'lamb':lamb,'sigma':sigma}
    return reviewParams
###############################################################################


def phifunction(phi,n,wn,review,modelParams,reviewParams, termvec):
    K = len(modelParams['gamma'])
    eps = modelParams['eps']
    beta = modelParams['beta']
    delta2 = modelParams['delta2']
    lamb = reviewParams['lamb']
    sigma = reviewParams['sigma']
    eta = reviewParams['eta']
    oldphi = reviewParams['phi']
    r = review['Rating'][0]
    sbar = np.zeros(K)
    sVar = np.zeros(K)
    #wn = np.where(termvec>0)[0] # nonzero elements
    sumeta = 0
    for i in range(K):
        sumeta = sumeta + eta[i]
        for idx, j in enumerate(wn):
            sbar[i] = sbar[i] + termvec[j]*beta[i,j]*oldphi[i,idx]
            sVar[i] = sVar[i] + termvec[j]*(beta[i,j]**2)*oldphi[i,idx]*(1-oldphi[i,idx])
    phifunction = - (np.matmul(lamb.transpose(),sbar)-r)**2/(2*delta2)
    for i in range(K):
        phifunction = phifunction + (
                                      - ( (lamb[i]**2+sigma[i])*sVar[i] + sigma[i]*sbar[i]**2 )/(2*delta2)
                                      + phi[i]*(sc.digamma(eta[i]) - sc.digamma(sumeta) + math.log(eps[i,wn[n]]) - math.log(phi[i]) )
                                    )
    return phifunction
###############################################################################    



def lambdafunction(lamb,review,modelParams,reviewParams, termvec):
    K = len(modelParams['gamma'])
    beta = modelParams['beta']
    phi = reviewParams['phi']
    mu = modelParams['mu']
    SIG = modelParams['SIG']
    SIGinv = np.linalg.inv(SIG)
    delta2 = modelParams['delta2']
    r = review['Rating'][0]
    wn = np.where(termvec>0)[0] # nonzero elements
    sbar = np.zeros(K)
    sVar = np.zeros(K)
    for i in range(K):
        for idx, j in enumerate(wn):
            sbar[i] = sbar[i] + termvec[j]*beta[i,j]*phi[i,idx]
            sVar[i] = sVar[i] + termvec[j]*(beta[i,j]**2)*phi[i,idx]*(1-phi[i,idx])
    
    lambdafunction = 0.5*np.matmul((lamb-mu).transpose(),np.matmul(SIGinv,(lamb-mu)))
    for i in range(K):
        lambdafunction = lambdafunction + (
                                            ( lamb[i]**2*sVar[i] + ( np.matmul(lamb.transpose(),sbar) - r)**2 )/(2*delta2)
                                          )
    
    
    return lambdafunction



###############################################################################
# this function updates the review parameters (E-step)
def updateReviewParams(review,params,modelParams,termvec):
    # maximization to find phi
    K = len(modelParams['gamma'])
    wn = np.where(termvec>0)[0] # nonzero elements
    gamma = modelParams['gamma']
    phi = params['phi']
    newphi = phi
    bounds = opt.Bounds([1e-10 for i in range(K)],[1 for i in range(K)])
    constraints = opt.LinearConstraint([1 for i in range(K)], 1, 1)
    for idx, j in enumerate(wn):
        x0 = phi[:,idx]
        x0 = [max(val,1e-10) for val in x0]
        x0 = [min(val,1) for val in x0]
        res = opt.minimize(
                        lambda x: -phifunction(x,idx,wn,review,modelParams,params, termvec), x0,
                           bounds = bounds, constraints = constraints)
        newphi[:,idx] = res.x

    params['phi']=newphi
    
    # update eta
    neweta = gamma
    for idx, n in enumerate(wn):
        neweta = neweta + phi[:,idx]
    params['eta']=neweta
    
    # update lambda
    lamb = params['lamb']
    beta = modelParams['beta']
    sbar = np.zeros(K)
    sVar = np.zeros(K)
    for i in range(K):
        for idx, j in enumerate(wn):
            sbar[i] = sbar[i] + termvec[j]*beta[i,j]*phi[i,idx]
            sVar[i] = sVar[i] + termvec[j]*(beta[i,j]**2)*phi[i,idx]*(1-phi[i,idx])
    bounds = opt.Bounds([0 for i in range(K)],[1 for i in range(K)])
    x0 = lamb
    x0 = [max(val,0) for val in x0]
    x0 = [min(val,1) for val in x0]
    res = opt.minimize(
                lambda x: lambdafunction(x,review,modelParams,params, termvec),
                            x0, bounds = bounds, constraints = constraints
                        )
    newlamb= res.x
    params['lamb']=newlamb
    
    # update sigma
    delta2 = modelParams['delta2']
    SIG = modelParams['SIG']
    SIGinv = np.linalg.inv(SIG)
    newsigma = np.zeros(params['sigma'].shape)
    for i in range(K):
        newsigma[i] = delta2/(sVar[i]+sbar[i]**2+delta2/SIGinv[i,i])
    
    reviewParams = {'eta':neweta,'phi':newphi,'lamb':newlamb,'sigma':newsigma}
    return reviewParams




###############################################################################
# this function updates the model parameters (M-step)
def updateModelParams(reviews, modelParams, reviewParams, tdm):
    D = len(reviews)
    K = len(reviewParams[0]['lamb'])
    
    # update eps
    eps = np.zeros(modelParams['eps'].shape)
    norm = np.zeros(K)
    for d in range(D):
        termvec = tdm[d,:]
        wn = np.where(termvec>0)[0] # nonzero elements
        phi = reviewParams[d]['phi']
        for i in range(K):
            for idx, j in enumerate(wn):
                eps[i,j] = eps[i,j] + termvec[j]*phi[i,idx]
            norm[i] = eps[i,:].sum()
    for i in range(K):
        eps[i,:] = eps[i,:]/norm[i]
    modelParams['eps'] = eps
    
    # update gamma
    gamma = modelParams['gamma']
    res = opt.minimize(lambda x: -gammafunction(x,reviews,modelParams,reviewParams,tdm),
                        gamma
                       )
    gamma = res.x
    modelParams['gamma']=gamma
    
    # update mu
    mu = np.zeros(K)
    for params in reviewParams:
        lamb = params['lamb']
        mu = mu + lamb
    mu = mu/D
    modelParams['mu']=mu
    
    # update SIG and delta2
    SIG = np.zeros((K,K))
    delta2 = 0
    beta = modelParams['beta']
    for d, params in enumerate(reviewParams):
        sigma = params['sigma']
        lamb = params['lamb']
        phi = params['phi']
        r = reviews[d]['Rating'][0]
        SIG = SIG + np.matmul( lamb-mu , (lamb-mu).transpose() ) + np.diag(sigma)
        
        termvec = tdm[d,:]
        wn = np.where(termvec>0)[0] # nonzero elements
        sbar = np.zeros(K)
        sVar = np.zeros(K)
        for i in range(K):
            for idx, j in enumerate(wn):
                sbar[i] = sbar[i] + termvec[j]*beta[i,j]*phi[i,idx]
                sVar[i] = sVar[i] + termvec[j]*(beta[i,j]**2)*phi[i,idx]*(1-phi[i,idx])
        
        delta2 = delta2 + (r - np.matmul( lamb.transpose() , sbar ) )**2
        for i in range(K):
            delta2 = delta2 + (lamb[i]**2 + sigma[i])*sVar[i] + sigma[i]*sbar[i]**2
    SIG = SIG/D
    delta2 = delta2/D
    modelParams['SIG']=SIG
    modelParams['delta2']=delta2
    
    # update beta
    (K,V) = beta.shape
    betaFlat = np.reshape(beta,K*V)
    res = opt.minimize(lambda x: betafunction(x,reviews,modelParams,reviewParams,tdm),
                        betaFlat
                       )
    beta = np.reshape(res.x,(K,V))
    
    
    modelParams = {'eps':eps, 'gamma':gamma, 'beta':beta, 'mu':mu, 'SIG':SIG, 'delta2':delta2}
    return modelParams
    
###############################################################################
    

def betafunction(beta,reviews,modelParams,reviewParams,tdm):
    K = len(reviewParams[0]['lamb'])
    (K,V) = modelParams['beta'].shape
    fun = 0
    beta = np.reshape(beta,(K,V))
    for d,review in enumerate(reviews):
        lamb = reviewParams[d]['lamb']
        phi = reviewParams[d]['phi']
        sigma = reviewParams[d]['sigma']
        r = review['Rating'][0]
        termvec = tdm[d,:]
        wn = np.where(termvec>0)[0] # nonzero elements
        sbar = np.zeros(K)
        sVar = np.zeros(K)
        for i in range(K):
            for idx, j in enumerate(wn):
                sbar[i] = sbar[i] + termvec[j]*beta[i,j]*phi[i,idx]
                sVar[i] = sVar[i] + termvec[j]*(beta[i,j]**2)*phi[i,idx]*(1-phi[i,idx])
        fun = (np.matmul(lamb.transpose(),sbar)-r)**2
        for i in range(K):
            fun = fun + (lamb[i]**2+sigma[i])*sVar[i] + sigma[i]*sbar[i]**2
        return fun


def gammafunction(gamma,reviews,modelParams,reviewParams,tdm):
    K = len(reviewParams[0]['lamb'])
    D = len(reviews)
    fun = 0
    for d in range(D):
        eta = reviewParams[d]['eta']
        fun = fun + sc.loggamma(gamma.sum())
        for i in range(K):
            fun = fun - sc.loggamma(gamma[i]) + (gamma[i]-1)*(sc.digamma(eta[i])-sc.digamma(eta.sum()))
    '''
    deriv = []
    for i in range(K):
        aux = D*(sc.digamma(gamma.sum()) -  sc.digamma(gamma[i]) )
        for d in range(D):
            eta = reviewParams[d]['eta']
            aux = aux + sc.digamma(eta[i])-sc.digamma(eta.sum())
        deriv.append(aux)
        for j in range(K):
            aux2 = D*( sc.polygamma(1, gamma.sum()) )
            if i==j:
                aux2 = aux2 - D * sc.polygamma(1,gamma[i]) 
            deriv.append(aux2)
    deriv = np.array(deriv)
    fun = (deriv**2).sum()
    '''
    return fun
    
        
        


'''
###############################################################################
# this function removes terms that appear in less than MIN_DOC_COUNT reviews from
# the vocabulary. It updates the vocabulary and term-doc matrix
def removeTerms(TermDocMatrix, vocabulary, min_doc_count=10):
    # identify which terms to eliminate
    elim = []
    for w in range(len(vocabulary)):
        col = TermDocMatrix[:,w]>0
        if col.sum() < min_doc_count:
            elim.append(w)
    # eliminate terms (words)
    new_vocab = [vocabulary[idx] for idx in range(len(vocabulary)) if idx not in elim]
    new_tdm = np.delete(TermDocMatrix,elim,axis=1)
    return new_vocab, new_tdm
###############################################################################        
'''        
        
