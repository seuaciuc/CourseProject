# CourseProject

Please fork this repository and paste the github link of your fork on Microsoft CMT. Detailed instructions are on Coursera under Week 1: Course Project Overview/Week 9 Activities.

# General Info

Project material is structured in 3 folders:
- *Data*: contains all data, both raw and processed. Smaller sub-samples of the datasets are provided.
- *Models*: contains all models (pickle files with estimated model paramters). 
- *Codes*: all scripts are here.

Each folder contains a README.md with specific information about their contents and structure.
All code is done in Python. Data (information from one step to another) is shared through pickle files, so typically the codes take one file as input and generate others as output. Codes are set to the folder structure of this repository, so it needs to be maintained to be able to run the scripts without modification.

## Interface
There is no command line interface built for this yet. Inputs are provided directly in the files, by modifying the first few lines of the scripts.
The most common inputs are clearly identified in the beginning of the scripts.
Other "hyper-parameters" (such as maximum number of iterations, minimum review length, etc) will appear immediately after the primary input section of the code. These typically do not need to be changed unless fine-tuning is desired. Whenever the first *import* command is reached, all possible inputs are done.
