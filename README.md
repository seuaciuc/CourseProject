# CourseProject

Please fork this repository and paste the github link of your fork on Microsoft CMT. Detailed instructions are on Coursera under Week 1: Course Project Overview/Week 9 Activities.

# General Info

Project material is structured in 3 folders:
- *Data*: contains all data, both raw and processed. Smaller sub-samples of the datasets are provided.
- *Models*: contains all models (pickle files with estimated model paramters). 
- *Codes*: all scripts are here.

Each folder contains a README.md with specific information about their contents and structure.
All code is done in Python 3.8. Data (information from one step to another) is shared through pickle files, so typically the codes take one file as input and generate others as output. Codes are set to the folder structure of this repository, so it needs to be maintained to be able to run the scripts without modification.

## Interface
There is no command line interface built for this yet. Inputs are provided directly in the files, by modifying the first few lines of the scripts.
The most common inputs are clearly identified in the beginning of the scripts.
Other "hyper-parameters" (such as maximum number of iterations, minimum review length, etc) will appear immediately after the primary input section of the code. These typically do not need to be changed unless fine-tuning is desired. Whenever the first *import* command is reached, all possible inputs are done.

In general, the scripts are ready to be run on some of the sample files provided if the folder structure of this repository is maintained.

## Suggested Testing Procedure
Because of the time for computation and size of some of the files, not all aspects of the project can be reproduced. Below is a suggested testing procedure that uses all scripts in the project to assess their functionality. The scripts provided are set to run the steps in this procedure without change if the folder structure is maintained.

1. Data Processing Scripts
PRocessing the entire datasets take considerable time and resources. This has been done, but the resulting files are larger than the allowable limite in GitHub.
To test the codes, you can run the data processing scripts on the provided smaller datasets. These two steps will run the scripts provided to process each of the two datasets:
- Run *processMP3reviews.py* to process the MP3 reviews in *amazon_mp3_redux.txt*. This will generate the *MP3reviews_redux.pkl*. All these files are in the *\data* folder.
- Run *processHOTELreviews.py* to process the hotels reviews in the folder *\Texts_redux*. This will generate the *HotelReviews_redux.pkl*. All these files are in the *\data* folder.

2. Model Building
Building the model takes time. Because of this, a pre-model was built on a random sample of 100 reviews. That can be done, but it still takes a little time.
To test the functionality of the model building code, you can build a model on one of the reduced datasets.
- Run *estimateModel.py* on the reduced MP3 dataset (*MP3reviews_redux.pkl*) to generate the model file *MP3model_redux.pkl*. The suggested number of aspects is 3.

3. Analysis
- Run the *getStats.py* on *MP3reviews_high_100.pkl* to obtain the statistics on that smaller set of reviews.
- Run the *getTopAspectWords.py* using the data *MP3reviews_high_100.pkl* and model *MP3model_high_100_3.pkl* to obtain the top 10 words in each of the 3 aspects of this model on these reviews.


