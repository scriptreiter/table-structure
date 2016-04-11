Usage: python get_dimensions.py img_dir output_dir

Requires: Python 3.0+, opencv, jpg images in img_dir, with scaled versions in img_dir/3x for zoom level of 3 (adjustable in the main script)

Outputs: .json files in output_dir/json_out/, .xlsx files in output_dir/xlsx/, cache files for the api calls into output_dir/google_cache and output_dir/json_cache

Can also change settings for oxford api sleep delay and google cloud vision sleep delay (default 5s and 0s). These are delays after any request to the API, to limit request frequency.
