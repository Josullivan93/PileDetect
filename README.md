# Automated detection of piling behaviour in UK laying hen flocks

This repository contains everything required to replicate the method described in "Automated detection of piling behaviour in UK laying hen flocks". The paper outlines the use of transfer learning to train a RESNET18 model to detect piling. Piling is a behaviour where chickens are observed to group closely together and can escalate to the point of smothering, wherein mortalities occur.

Briefly this repository comprises: 
- The Python script to train the model
- The Python script to apply the model to novel data
- The model weightings produced and used in the paper
- A reduced sample of video frames on which to train a model from scratch
- This ReadMe

## Set Up

Use your chosen package manager to install the required packages.
It is recommended to use a dedicated virtual environment with Conda. A YAML file with all dependencies required is included. To create a new environment in Conda using the YAML file use:
```conda env create -f PileDetect.yaml```

If you do not wish to use this environment file the packages required are:
- Pytorch
- Torchvision
- Scikit-Learn
- Numpy
- Pillow
- tqdm
- MatPlotLib

A CUDA compatible GPU is suggested. If CUDA is not available the script will default to use the available CPU instead.

## Usage

Images on which the model is to be run should be of JPG format and saved in subdirectories named either PILING or NONPILING. An example of this can be seen in the provided SampleImg directory.

### Training

To train the model run the script ".py" from your terminal. The script requires both a directory containing a sample of labelled images upon which to train the model, and a path to a directory for the model output are specified. In addition the number of training epochs and samples used for each class can be provided (defaulting to 20 epoch and 5000 samples is respectively not provided).

For example the following command will locate the images stored in the "SampleImg" directory, train the model over 20 epochs, with 10000 Piling samples and 10000 Non-Piling samples, and save the result in a directory named "OutDir":

```python pile-detect-training.py "SampleImg" "OutDir" 20 10000```

Figures displaying the training accuracy and loss across epochs will be produced and saved to the working directory.

### Inference

To use trained model weightings to perform inference, either that provided in the "TrainedModel.dat" file or one trained yourself using the "pile-detect-training.py" script, run the "pile-detect-inference.py" script. When run from the terminal the script requires a path to a model weighting .dat file (or simply the file name if held in the current working directory) and the path to a directory containing the images to be classified. To use this script to validate a generated model on pre-labelled data, and generate appropriate performance metrics, a "Validate" argument can be added when running from the terminal. If validating then images passed to the script should be organised in subdirectories as with Training and shown in the SampleImg directory of this repository. An optional Validation Sample Size argument can also be included to specificy the number of each class to include in the validation inference. If validation is not required omit these arguments and all images within the passed directory, and its subdirectories, will be classified. For example to run the script to validate using 5000 prelabelled images (2500 of each class) stored in a directory named "ValidationImg", with weightings from "TrainedModel.dat": 

```python pile-detect-inference.py "ValidationImg" "TrainedModel.dat" Validate 2500```

Figures showing the ROC curve, Precision-Recall curve, and selections of misclassified frames will be produced for inspecition and saved to the working directory if the Validate argument is used alongside a CSV of performance metrics (Accuracy, F1 Measure, Precision, Recall, Specificity, ROC curve AUC, Precision-Recall Curve AUC, and Numbers of True Positives, False Positives, True Negatives, and False Negatives). A list of filenames and classifications will be generated both with and without the Validate argument.

## Credits
A  full description of Author contributions to this code and accompanying manuscript can be found in the manuscript.

The code given here is based on initial exploration and implementation work performed by the Newcastle University Research Software Engineering team, with specific thanks to Drs Mark Turner, Kathryn Garside and Robin Nandi. 
The scripts and methodology were further developed and expanded upon by Genevieve Moat. Final development, implementation and validation was performed by Jack O'Sullivan.