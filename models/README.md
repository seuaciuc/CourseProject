# Contents

Available models:
- *MP3model_low_100_3.pkl*
  - Model built with 3 aspects on dataset with 100 MP3 reviews with low rating
  - Associated processed data file: *MP3reviews_low_100.pkl* (see *\data* folder)
- *MP3model_high_100_3.pkl*
  - Model built with 3 aspects on dataset with 100 MP3 reviews with high rating
  - Associated processed data file: *MP3reviews_high_100.pkl* (see *\data* folder)
- *HotelModel_100_7.pkl*
  - Model built with 7 aspects on dataset with 100 hotel reviews
  - Associated processed data file: *HotelReviews_100.pkl* (see *\data* folder)
  - **NOTE:** this will be uploaded when the code finishes.

## File contents
Each model file contains two objects, in this order:
- The corpus-level model paramters, as a dictionary.
- The review-level model parameters, as a list of dictionaries (one entry for each review).

The nomenclature for the variables in each dictionary follows that adopted in the subject paper.
