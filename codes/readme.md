## Data Processing Scripts
The scripts below were used to process the raw review data.
They are set to run on the full dataset, which takes a while and generates files larger than GitHub allows me.
A smaller sub-sample of the datasets is provided to test the codes on, if desired. The scripts are set to run on those.
Each will generate an output .pkl file. These have been run and are available in the *\data* folder.
- *processHOTELreviews.py*:
  - Processes the hotel reviews.
- processMP3reviews.py:
  - Processes the MP3 reviews.
- processMP3reviews_split.py:
  - Processes the MP3 reviews after separating them in low (3 or lower) and high (above 3) ratings.
  
## Data Processing Steps
Data processing was done in the following manner:
1. All text is set to lower case.
2. Punctuation is removed.
3. Text is tokenized by words.
4. Stopwords are removed.
    - NLTK stopwords for English was used, which is slightly different from what was done in the paper.
5. Non-alphabetical terms are removed.

After that, following the approach in the paper, the reviews are filtered:
1. Only reviews with at least 50 words are kept;
2. Any reviews with missing ratings are eliminated.

Lastly, inbuilding the term-document matrix (TDM), only terms that appear in at least 10 documents are kept. A few notes regarding this:
- This means they are not included in the corpus vocabulary and TDM (which is what is used in any further processing steps), but are not deleted from review content (list of terms).
- This is done *after* applying the minimum review term count.
- When processing the example sub-sampled datasets, a minimum document appearance of 3 was used instead of 10.
