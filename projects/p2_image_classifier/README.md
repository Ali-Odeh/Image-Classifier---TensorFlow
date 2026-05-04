# Image Classifier Project

This repository contains the completed Udacity flower image classifier project. It includes the reference notebook, the saved model, and a command-line prediction script that loads an image, preprocesses it, and prints the most likely flower classes.

## Overview

The project is centered around a trained neural network for flower classification. You can explore the notebook for the full workflow and use the CLI script to run inference on your own images.

The main goals are:

1. Load a saved TensorFlow/Keras model.
2. Preprocess an input image in the same way the model expects.
3. Predict the most likely flower classes.
4. Optionally map numeric class IDs to human-readable flower names.

## Repository Layout

- [Project_Image_Classifier_Project_Sol.ipynb](Project_Image_Classifier_Project_Sol.ipynb): the full project notebook and solution walkthrough.
- [predict.py](predict.py): command-line script used for inference.
- [my_model.h5](my_model.h5): the saved trained model used by the script.
- [label_map.json](label_map.json): JSON mapping from class IDs to flower names.
- [test_images/](test_images/): sample images for quick testing.
- [assets/](assets/): supporting project assets.

## Notebook

The notebook [Project_Image_Classifier_Project_Sol.ipynb](Project_Image_Classifier_Project_Sol.ipynb) contains the project solution in notebook form. It is the best place to review the step-by-step workflow, inspect the model development process, and compare the notebook-based implementation with the final prediction script.

If you want to run the notebook locally, open it from this folder in Jupyter or VS Code and make sure the Python environment has the required packages installed.

## Installation

Install the project dependencies from the repository root:

```bash
pip install -r ../../requirements.txt
```

The dependency list includes TensorFlow Hub, tf-keras, TensorFlow datasets, and SciPy. The script also relies on standard image-processing packages such as NumPy and Pillow.

## Running The Prediction Script

Run the CLI from inside [projects/p2_image_classifier](.). The simplest invocation predicts the top class for one image:

```bash
python predict.py test_images/orange_dahlia.jpg my_model.h5
```

Show the top 5 predictions instead of only the best one:

```bash
python predict.py test_images/orange_dahlia.jpg my_model.h5 --top_k 5
```

Include flower names by passing the label map:

```bash
python predict.py test_images/orange_dahlia.jpg my_model.h5 --top_k 5 --category_names label_map.json
```

## Script Behavior

The script in [predict.py](predict.py) performs the following steps:

1. Validates that the input files exist.
2. Loads the saved model with TensorFlow Hub support.
3. Opens the image and converts it to RGB.
4. Resizes the image to 224 x 224 pixels.
5. Normalizes pixel values to the range 0 to 1.
6. Runs inference and extracts the top K predictions.
7. Prints the prediction rank, label, and probability.

If `--category_names` is provided, the script converts predicted class indices into flower names using [label_map.json](label_map.json). If the option is omitted, the script prints the raw class indices.

## Input And Output

The expected command-line arguments are:

- `image_path`: path to the input image.
- `saved_model`: path to the `.h5` model file.
- `--top_k`: number of predictions to display, with a minimum value of 1.
- `--category_names`: optional JSON file that maps class IDs to names.

Typical output looks like this:

```text
Prediction Results
------------------------------
1. orange dahlia - Probability: 0.8665 (86.65%)
2. gazania - Probability: 0.0317 (3.17%)
3. english marigold - Probability: 0.0263 (2.63%)
4. mexican aster - Probability: 0.0106 (1.06%)
5. osteospermum - Probability: 0.0071 (0.71%)
```

## Notes

- `--top_k` must be at least 1.
- The script raises a file-not-found error if the image, model, or label map path is invalid.
- The model is loaded with `compile=False` because the project only needs inference.
- The notebook and the script are intended to work together: the notebook documents the project, and the script provides a repeatable CLI entry point.

## Troubleshooting

If the script does not run correctly, check the following:

1. The virtual environment is activated.
2. The dependencies from [requirements.txt](../../requirements.txt) are installed.
3. You are running the command from [projects/p2_image_classifier](.).
4. The image file, model file, and optional label map all exist at the paths you passed in.

If you want to verify the model quickly, try one of the files in [test_images/](test_images/) with the provided [my_model.h5](my_model.h5) model.
