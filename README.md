# Fine-grained Museum Localisation
 Fine-grained localisation within a museum using feature detection on images

## Setup
The following steps are required before running any other code:

1. Please ensure that training and test images are located in `/Dataset/train` and `/Dataset/test` respectively.
2. Run `pip install -r requirements.txt` to ensure necessary packages are installed and up to date
3. To construct the training feature set, run the first three code chunks in `Closest image matcher.ipynb`

Please note that for `mpi4py` to work, additional software may be required.

## Feature matcher
To run, use the command:
`mpirun -n <n_cores> python3 feature_matcher.py`

## Pose recovery
The following command must be run once before running the main script:
`mpirun -n <n_cores> python3 extract_scale_image.py`

The command above can also be run if the training dataset changes.

The main script can be run with:
`python3 scale_estimate.py`
