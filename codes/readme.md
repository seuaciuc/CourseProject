## Data Processing Scripts
The scripts below were used to process the raw review data.
They are set to run on the full dataset, which takes a while. This has been done and the results are saved as noted.
A smaller sub-sample of the datasets is provided to test the codes on, if desired.
- *processHOTELreviews.py*:
  - Processes the hotel reviews.
  - This has been run; results are in *\data\HotelReview.pkl*.
  - You can change the input folder to *\data\Texts_redux* for a smaller sample to test the code.
- processMP3reviews.py:
  - Processes the MP3 reviews.
  - This has been run; results are in *\data\MP3reviews.pkl*.
  - You can change the input file to *amazon_mp3_redux* for a smaller sample to test the code.
- processMP3reviews_split.py:
  - Processes the MP3 reviews after separating them in low (3 or lower) and high (above 3) ratings.
  - This has been run; results are in *\data\MP3reviews_low.pkl* and *\data\MP3reviews_high.pkl*.
  - You can change the input files to *amazon_mp3_redux* for a smaller sample to test the code.
