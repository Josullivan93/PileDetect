# Automated detection of piling behaviour in UK laying hen flocks

This repository contains everything required to replicate the method described in "Automated detection of piling behaviour in UK laying hen flocks". The paper outlines the use of transfer learning to train a RESNET18 model to detect piling. Piling is a behaviour where chickens are observed to group closely together and can escalate to the point of smothering, wherein mortalities occur.

Briefly this repository comprises: 
- The Python script to train the model
- The Python script to apply the model to novel data
- The model weightings produced and used in the paper
- A reduced sample of video frames on which to train a model from scratch

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

Images on which the model is to be trained should be of JPG format and saved in subdirectories named either PILING or NONPILING. An example of this can be seen in the provided Sample_Images directory. If attempting Inference from a trained model images should be JPG format but subdirectories with classification labels are only required if validating, and generating performance metrics from, the model.

### Training

To train the model run the script "Pile-Detect-Training.py" from your terminal. The script requires both a directory containing a sample of labelled images upon which to train the model, and a path to a directory for the model output are specified. In addition the number of training epochs and samples used for each class can be provided (defaulting to 20 epoch and 5000 samples respectively if not provided).

For example the following command will locate the images stored in the "Sample_Images" directory (within their label subdirectories), train the model over 20 epochs, with 5000 Piling samples and 5000 Non-Piling samples, and save the result in a directory named "Output":

```python pile-detect-training.py Sample_Images Output 20 5000```

Figures displaying the training accuracy and loss across epochs will be produced and saved to the working directory.

### Inference

To use trained model weightings to perform inference, either those provided in the "TrainedModel.dat" file or one trained yourself using the "pile-detect-training.py" script, run the "pile-detect-inference.py" script. When run from the terminal the script requires a path to a model weighting .dat file (or simply the file name if held in the current working directory) and the path to a directory containing the images to be classified. To use this script to validate a generated model on pre-labelled data, and generate appropriate performance metrics, a "Validate" flag (either -v or --validate) can be set when running from the terminal. If validating then images passed to the script should be organised in subdirectories as with Training and as demonstrated in the Sample_Images directory of this repository. An optional Validation Sample Size argument can also be included (after an -n or --nsamples flag) to specificy the number of each class to include in the validation inference. If wanting to generate samples of misclassified images from the validation the -m (or --misclass) flag can be set alongside the -v flag. If validation is not required omit these flags and all images within the passed directory, and its subdirectories, will be classified. For example to run the script to validate using 5000 prelabelled images (2500 of each class) stored in a directory named "Sample_Images", with weightings from "TrainedModel.dat", and generate the sample images from this validation the command would look like: 

```python pile-detect-inference.py Sample_Images TrainedModel.dat -v -n 2500 -m```

Figures showing the ROC curve, and Precision-Recall curve will be produced whenever the validation flag is set and will be saved to the working directory alongside a CSV of performance metrics (Accuracy, F1 Measure, Precision, Recall, Specificity, ROC curve AUC, Precision-Recall Curve AUC, and Numbers of True Positives, False Positives, True Negatives, and False Negatives). A list of filenames and classifications will be generated both when validating and when performing inference on unclassified images.

## Citation

If you use this repository or dataset in your work, please cite both the paper and the dataset repository:

- **Paper:** Jack O'Sullivan, Helen E. Gray, Genevieve Moat, Lucy Asher; Automated detection of piling behaviour in UK laying hen flocks. R. Soc. Open Sci. 1 May 2026; 13 (5): 252462. https://doi.org/10.1098/rsos.252462

- **Repository:** Jack O'Sullivan, Helen Gray, Genevieve Jiawei Moat, Lucy Asher; Automated detection of piling behaviour in UK laying hen flocks - Python Scripts, Model Weights & Sample Images. *Figshare*. https://doi.org/10.25405/data.ncl.28654664

## License
This repository uses dual-licensing to distinguish between software/model assets and image dataset assets:

- **Source Code, Environment & Model Weights (GPL 3.0+):** 
  The Python scripts (`pile-detect-training.py`, `pile-detect-inference.py`), environment setup (`PileDetect.yaml`), and trained model weightings (`TrainedModel.dat`) are licensed under the **GNU General Public License v3.0**.

- **Sample Image Dataset (CC BY 4.0):** 
  All `.jpg` sample image frames contained within the `Sample_Images/` directory (including `Sample_Images/PILING/` and `Sample_Images/NONPILING/`) are licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**, matching the open access license of the published paper (*R. Soc. Open Sci.*, doi: [10.1098/rsos.252462](https://doi.org/10.1098/rsos.252462)).

## Credits
A  full description of Author contributions to this code and accompanying manuscript can be found in the manuscript.

The code given here is based on initial exploration and implementation work performed by the Newcastle University Research Software Engineering team, with specific thanks to Drs Mark Turner, Kathryn Garside and Robin Nandi. 
The scripts and methodology were further developed and expanded upon by Genevieve Moat. Final development, implementation and validation was performed by Jack O'Sullivan.****
