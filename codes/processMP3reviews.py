'''
INPUTS
'''
# provide the MP3 review file name (amazon_mp3.txt). Must be in the data folder
inputFile = 'amazon_mp3_redux.txt'
# provide output file name. This will be a pkl file saved in the data folder
outputFile = 'MP3reviews_redux.pkl'

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


import pickle, utils, time, os

# folder management
codefolder = os.getcwd()
basefolder = os.path.dirname(codefolder)
datafolder = os.path.join(basefolder,'data')
# get full file paths
fullInputFile = os.path.join(datafolder,inputFile)
fullOutputFile = os.path.join(datafolder,outputFile)

reviews = []
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
            reviews.append(entry)
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
reviews = utils.filterReviews(reviews, min_review_length=MIN_REVIEW_LENGTH)

### CREATE VOCABULARY AND TERM-DOC MATRIX:
t = time.strftime("%D %H:%M:%S", time.localtime())
print(t,'Building vocabulary...')
voc = utils.buildVocabulary(reviews)
t = time.strftime("%D %H:%M:%S", time.localtime())
print(t,'Building Term-Document Matrix...')
tdm, voc = utils.buildTermDocMatrix(reviews, voc, min_doc_count=MIN_DOC_COUNT)


### SAVE FILE
t = time.strftime("%D %H:%M:%S", time.localtime())
print(t,'Saving file...')
fout = open(fullOutputFile,'wb')
pickle.dump(reviews, fout,-1)
pickle.dump(voc, fout,-1)
pickle.dump(tdm, fout,-1)
fout.close()
t = time.strftime("%D %H:%M:%S", time.localtime())
print(t,'Complete.')
