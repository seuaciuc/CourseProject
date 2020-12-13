'''
INPUTS
'''
# provide the folder path to the parsed Hotel review files (must be in the data folder)
inputFolder = 'Texts_redux'
# provide output file name. This will be a pkl file saved in the data folder
outputFile = 'HotelReviews_redux.pkl'

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


import pickle, os, time
import utils

# folder management
codefolder = os.getcwd()
basefolder = os.path.dirname(codefolder)
datafolder = os.path.join(basefolder,'data')
# get full file paths
fullInputFolder = os.path.join(datafolder,inputFolder)
fullOutputFile = os.path.join(datafolder,outputFile)

reviews = []
entry = None

## GET LIST OF FILES
fileList = os.listdir(fullInputFolder)

# loop over files
t = time.strftime("%D %H:%M:%S", time.localtime())
print(t,'Processing files...')
for file in fileList:
    inputFile = os.path.join(fullInputFolder,file)
    if not os.path.isfile(inputFile): # not a file
        continue
    # get hotel ID
    idx1 = file.find('_')
    idx2 = file.find('_',idx1+1)
    hotelID = file[idx1+1:idx2]
    ### READ AND PROCESS FILE
    fin = open(inputFile, "r", encoding='utf-8')
    for line in fin:
        line = line.rstrip('\n')
        # check for new record/Author
        field = '<Author>'
        if line.startswith(field):
            if entry != None:
                reviews.append(entry) # append last entry
            entry = {'Product':hotelID} # reset record
            entry['Author'] = line[len(field):]
        # check for date
        field = '<Date>'
        if line.startswith(field):
            entry['Date'] = line[len(field):]
            continue    
        # check for date
        field = '<Date>'
        if line.startswith(field):
            entry['Date'] = line[len(field):]
            continue        
        # check for rating
        field = '<Rating>'
        if line.startswith(field):
            entry['Rating'] = [float(val) for val in line[len(field):].split()]
            continue
        # check for review content
        field = '<Content>'
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
print(t,'Buildining Term-Document Matrix...')
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
