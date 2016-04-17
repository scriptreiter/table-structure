Usage: python get_dimensions.py img_dir output_dir

Requires: Python 3.0+, opencv, jpg images in img_dir, with scaled versions in img_dir/3x for zoom level of 3 (adjustable in the main script), must configure environmental variable (GOOGLE_APPLICATION_CREDENTIALS) as per instructions at: https://developers.google.com/identity/protocols/application-default-credentials

Outputs: .json files in output_dir/json_out/, .xlsx files in output_dir/xlsx/, cache files for the api calls into output_dir/google_cache and output_dir/json_cache

Can also change settings for oxford api sleep delay and google cloud vision sleep delay (default 5s and 0s). These are delays after any request to the API, to limit request frequency.

Avoid images with extreme numbers of potential detected words or grid cells, such as bad_example.jpg, as this code will consider all combinations of images, and will run out of memory doing so.

In order to train a classifier, you should dump box combination files, by setting the should_record_features variable to True in boxer.py. This will create files at OUTPUT/combos/features/, which will have box combinations, stored with combo features. Corresponding files must be created in OUTPUT/combos/labels/ with a single 0 or 1 per line, with 0 marking the combination as not-to-merge, and 1 as to-merge. The path base must be updated in trainer, and then it can be run. It will generate classifier.pkl, as well as a list of the test/train set division, and will output precision, recall, and accuracy. The split into train/test can be removed fairly easily to train on the entire set. The file, combo_labeler.html, has an example labeling application, that allows box combinations to be labeled. This file reads the image from url parameter (image=), fetches a combo file via AJAX (thus requiring this to be on a server), and allows binary decisions on combinations. It requires 'combo' files with the combos listed as the first 8 numbers on a line, separated by spaces. This can be changed to commas, or the files generated can have commas replaced by spaces to work together.

groundtruth_labeler.html is an html file that allows labeling of image groundtruth.
