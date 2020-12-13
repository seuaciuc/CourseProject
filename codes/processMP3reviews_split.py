'''
INPUTS
'''
# provide the MP3 review file name (amazon_mp3.txt). Must be in the data folder
inputFile = 'amazon_mp3_redux.txt'
# provide output file names. These will be pkl files saved in the data folder
outputFile_low = 'MP3reviews_low_redux.pkl'
outputFile_high = 'MP3reviews_high_redux.pkl'

###############################################################################
###############################################################################
#### END OF INPUTS ############################################################
#### DO NOT EDIT CODE BELOW ###################################################
###############################################################################
###############################################################################

'''
Do not edit any of the code below. Only provide the inputs above
'''

MIN_REVIEW_LENGTH = 50 # minimum review length
MIN_DOC_COUNT = 10 # minimum number of documents for any term
minRating = 3 # review with rating>minRating will be saved in high_reviews
              # review with rating<=minRating will be saved in low_reviews


import pickle, utils, time, os

# folder management
codefolder = os.getcwd()
basefolder = os.path.dirname(codefolder)
datafolder = os.path.join(basefolder,'data')
# get full file paths
fullInputFile = os.path.join(datafolder,inputFile)
fullOutputFile_low = os.path.join(datafolder,outputFile_low)
fullOutputFile_high = os.path.join(datafolder,outputFile_high)

reviews_low = []
reviews_high = []
entry = None

### READ AND PROCESS FILE
t = time.strftime("%D %H:%M:%S", time.localtime())
print(t,'Processing files...')
fin = open(fullInputFile, "r",encoding='utf-8')
for line in fin:
    line = line.rstrip('\n')
    # check for new record
    field = '#####'
    if line.startswith(field):
        if entry != None:
            if entry['Rating'][0]>minRating:
                reviews_high.append(entry)
            else:
                reviews_low.append(entry)
        entry = dict()
        continue
    # check for product name
    field = '[productName]:'
    if line.startswith(field):
        entry['Product'] = line[len(field):]
        continue
    # check for author
    field = '[author]:'
    if line.startswith(field):
        entry['Author'] = line[len(field):]
        continue
    # check for date
    field = '[createDate]:'
    if line.startswith(field):
        entry['Date'] = line[len(field):]
        continue
    # check for rating
    field = '[rating]:'
    if line.startswith(field):
        entry['Rating'] = [float(line[len(field):])]
        continue
    # check for content
    field = '[fullText]:'
    if line.startswith(field):
        entry['ReviewText'] = utils.processText(line[len(field):])
        continue
fin.close()

### FILTER REVIEWS:
t = time.strftime("%D %H:%M:%S", time.localtime())
print(t,'Filtering reviews...')
reviews_low = utils.filterReviews(reviews_low, min_review_length=MIN_REVIEW_LENGTH)
reviews_high = utils.filterReviews(reviews_high, min_review_length=MIN_REVIEW_LENGTH)

### CREATE VOCABULARY AND TERM-DOC MATRIX:
t = time.strftime("%D %H:%M:%S", time.localtime())
print(t,'Building vocabulary...')
voc_low = utils.buildVocabulary(reviews_low)
voc_high = utils.buildVocabulary(reviews_high)
t = time.strftime("%D %H:%M:%S", time.localtime())
print(t,'Building Term-Document Matrix...')
tdm_low, voc_low = utils.buildTermDocMatrix(reviews_low, voc_low, min_doc_count=MIN_DOC_COUNT)
tdm_high, voc_high = utils.buildTermDocMatrix(reviews_high, voc_high, min_doc_count=MIN_DOC_COUNT)


### SAVE FILE
t = time.strftime("%D %H:%M:%S", time.localtime())
print(t,'Saving files...')
fout = open(fullOutputFile_low,'wb')
pickle.dump(reviews_low, fout,-1)
pickle.dump(voc_low, fout,-1)
pickle.dump(tdm_low, fout,-1)
fout.close()
fout = open(fullOutputFile_high,'wb')
pickle.dump(reviews_high, fout,-1)
pickle.dump(voc_high, fout,-1)
pickle.dump(tdm_high, fout,-1)
fout.close()

t = time.strftime("%D %H:%M:%S", time.localtime())
print(t,'Complete.')
