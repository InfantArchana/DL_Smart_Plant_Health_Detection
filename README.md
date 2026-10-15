# Smart Plant Health Detection

A deep learning project for tomato leaf disease classification using a TensorFlow CNN.

## Overview

This repository trains and evaluates a model that identifies tomato leaf health conditions from images. It is designed for 10 tomato classes, including healthy leaves and several common diseases.

The project includes:

- a notebook for dataset preparation, training, validation, and evaluation
- saved model checkpoints in the `models/` directory
- evaluation metrics in `results/evaluation_results.json`
- a simple prediction script for new leaf images

## Project status

The model is already trained and saved in the repository. The final recorded evaluation metric is:

- Test accuracy: 84.47%
- Test loss: 0.4521

## Included classes

| Class index | Disease / Condition |
| --- | --- |
| 0 | Bacterial spot |
| 1 | Early blight |
| 2 | Healthy |
| 3 | Late blight |
| 4 | Leaf mold |
| 5 | Septoria leaf spot |
| 6 | Spider mites |
| 7 | Target spot |
| 8 | Tomato mosaic virus |
| 9 | Tomato yellow leaf curl virus |

## Repository structure

```text
DL_Smart_Plant_Health_Detection/
├── README.md
├── requirements.txt
├── predict.py
├── notebooks/
│   └── Tomato_Leaf_Disease_Classification.ipynb
├── models/
│   ├── baseline_5epochs.keras
│   ├── best_tomato_baseline.keras
│   ├── continued_best.keras
│   ├── tomato_cnn_selected.keras
│   └── class_names.json
├── results/
│   ├── evaluation_results.json
│   └── split_indices.npz
├── test_images/
└── .git/
```

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Run prediction on a new image

Place an image in the `test_images/` folder, then run:

```bash
python3 predict.py test_images/your_leaf.jpg
```

The script loads the trained model, preprocesses the input image, and prints the predicted tomato class and confidence score.

## Model training and notebook workflow

The full training workflow is available in:

- `notebooks/Tomato_Leaf_Disease_Classification.ipynb`

This notebook covers:

- PlantVillage dataset filtering
- remapping the original labels to the 10 tomato classes
- stratified train/validation/test splitting
- CNN training and checkpointing
- evaluation and saving final results

## Performance summary

The saved evaluation file is located at `results/evaluation_results.json` and contains:

- `selected_validation_accuracy`: 0.8421
- `selected_test_accuracy`: 0.8447
- `selected_test_loss`: 0.4521
- `validation_macro_f1_rounded`: 0.80

## Notes

- This project is specifically for tomato leaf disease classification.
- The repository already contains trained model weights, so you can use inference immediately without retraining.
- If you want to reproduce the notebook training process, additional exploratory packages may be useful, but the minimal runtime for inference is already captured in `requirements.txt`.

## License

This project does not currently include a license file. If you plan to publish it publicly on GitHub, add an appropriate license before distribution.

## GitHub upload

This repository is already initialized as a Git repository, but there is no remote configured yet. To upload it to GitHub, add your repository URL:

```bash
git remote add origin <your-github-repository-url>
git branch -M main
git add .
git commit -m "Initial project commit"
git push -u origin main
```
