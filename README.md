# Intent-BERT
## Requirements
Python 3.8

*All others in requirements.txt*

Of note:
> tensorflow 2.13
> 
> keras_nlp
> 
> keras_core
> 
> ultralytics

## Dataset
We base our results on the test set of the [InHARD dataset](https://github.com/vhavard/InHARD). To perform the necessary preprocessing of this dataset for our model:
1. Clone this repo and its submodules
1. Download the original dataset from [here](https://zenodo.org/record/4003541) and place it in the InHARD folder in our repo.
2. Run the `correct_action_list.py` script, passing the path to the Action-Meta-action-list.xlsx file (by default under the rsc directory in the InHARD dataset).
3. In the `Prepared_Data` folder of our directory, please run `python clean_dataset.py [PATH_TO_ONLINE_DATA] [MODE]` where [PATH_TO_ONLINE_DATA] is the path to the 'online' portion of InHARD and [MODE] is `testing` for the test set or `training` for the training set. Note this processing may take a long time, depending on your computer's capabilities, as it runs Yolov8 across each frame to get 2D human pose data. Also, note that it results in 500+GB of data in the training set and 200+GB in the test set as each frame is extracted and processed. To reduce this, you can:
 1.  Remove some sessions from the Online directory of InHARD
 2.  Modify `dataset_params.py` in the `Prepared_Data` directory
 3.  Use our provided `Demo` data when running the evaluation.
 
## Training 
To train on the prepare data, simply run `train.py [PATH_TO_TRAINING_DATA]` from the root directory of this repo. By default, [PATH_TO_TRAINING_DATA] is Prepared_Data\training.

## Evaluation
To replicate our results, run `eval.py [PATH_TO_TESTING_DATA]` from the root directory of this repo. By default, [PATH_TO_TESTING_DATA] is Prepared_Data\testing.

To replicate our results comparing GPT to Intent-BERT please see the GPT_Results folder
