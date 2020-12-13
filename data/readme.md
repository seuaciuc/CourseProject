## Contents
This directory contains mainly examples of a sub-sample of the project datasets (raw and processed).
Full datasets are larger than what GitHub permits me. They can be downloaded at the link provided in the paper:
http://timan.cs.uiuc.edu/downloads.html

The Texts_redux folder contains a sub-sample of the hotel review data.

- amazon_mp3_redux.txt: a smaller sub-sample of the amazon review dataset in its raw format.
- MP3reviews_redux.pkl: the processed data from *amazon_mp3_redux.txt*.
- MP3reviews_low_redux.pkl: the processed data from reviews in *amazon_mp3_redux.txt* with low rating (3 or lower).
- MP3reviews_low_100.pkl: processed data of a random sub-sample of 100 reviews with low rating (3 or lower).
- MP3reviews_high_redux.pkl: the processed data from reviews in *amazon_mp3_redux.txt* with high rating (higher than 3).
- MP3reviews_high_100.pkl: processed data of a random sub-sample of 100 reviews with high rating (higher than 3).
- HotelReviews_redux.pkl: the processed data from the sub-sample of hotel reviews in the *\Texts_redux* folder.
- HotelReviews_100.pkl: processed data of a random the sub-sample of 100 hotel reviews.


## Processed File Content
The raw file format varies. The download from the source site contains a description of each of them.
The processed files (.pkl) have the same content and format for both MP3 and hotel reviews.
Each has three objects, in this order:

1. Reviews: a list of dictionaries. Each dictionary has the following key:value pairs:
   - 'ReviewText': the content of the review, as a list of words, after all processing.
   - 'Author': name of the author, if available
   - 'Product': ID of the product
   - 'Date': date of review
   - 'Rating': list of floats, with the first entry being the overall rating. MP3 reviews only have the overall rating; hotel reviews have up to 7 aspects.
2. Vocabulary: a list of unique words that make up the corpus vocabulary
3. Term-Document Matrix: a 2-D matrix with one document per row, one vocabulary term per column. The entries are the counts of each term in each document.
