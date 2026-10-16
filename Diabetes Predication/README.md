# Diabetes Prediction

This project predicts whether a person is diabetic or non-diabetic using a machine learning classifier. The workflow is implemented in the notebook `Diabetes Prediction.ipynb` and uses the dataset stored in `Dataset/diabetes.csv`.

## Project Goal

The objective is to classify patients into two categories:

- `0` = Non-Diabetic
- `1` = Diabetic

This is a binary classification problem, and the model is trained to detect diabetes based on medical feature values such as glucose level, BMI, insulin, and age.

## Dataset

The project uses the PIMA Indians Diabetes dataset, which contains clinical records for patients. Each row represents one patient, and the final column is the target label.

Dataset details:

- 768 rows
- 9 columns total
- 8 feature columns
- 1 target column: `Outcome`

The dataset file is located here:

- `Dataset/diabetes.csv`

## Workflow

The project follows this workflow:

1. Load the dataset using pandas.
2. Explore the dataset shape and class distribution.
3. Separate the features (`X`) and target labels (`Y`).
4. Standardize the feature values using `StandardScaler`.
5. Split the data into training and testing sets with `train_test_split`.
6. Train a `Support Vector Machine (SVM)` classifier with a linear kernel.
7. Evaluate the model accuracy on both training and test data.
8. Use the trained model to make a prediction for a new patient input.

## Model and Evaluation

The notebook uses a linear SVM model from scikit-learn.

Verified model accuracy from the project implementation:

- Training Accuracy: `0.7866449511400652`
- Testing Accuracy: `0.7727272727272727`

These results show the model is able to classify diabetic and non-diabetic patients with a reasonable accuracy level.

## Setup Instructions

This project uses `uv` for environment creation and dependency installation.

1. Create a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

2. Install the required dependencies:

```bash
uv pip install -r requrments.txt
```

3. Open the notebook:

```bash
uv run jupyter notebook "Diabetes Prediction.ipynb"
```

## File Structure

```text
Diabetes Predication/
├── README.md
├── requrments.txt
├── Diabetes Prediction.ipynb
├── Dataset/
│   └── diabetes.csv
└── .venv/
```

## Prediction Example

After training, the model can be used to predict a new patient by passing a feature vector in the same order as the dataset columns.

```python
import numpy as np

input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
```

## Technologies Used

- Python
- Pandas
- NumPy
- scikit-learn
- Jupyter Notebook
- uv

## Summary

This project demonstrates a practical machine learning workflow for diabetes prediction using the PIMA dataset, feature scaling, and a linear SVM classifier. It covers data loading, preprocessing, model training, performance evaluation, and prediction for new patient records.
