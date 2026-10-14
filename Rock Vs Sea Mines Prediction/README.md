# Rock vs Mine Prediction

This project predicts whether an object detected by sonar is a rock or a sea mine using a machine learning classifier. The workflow is implemented in the notebook `Rock vs mine Prediction.ipynb` and uses the dataset stored in `Dataset/sonar data.csv`.

## Project Goal

The objective is to classify sonar readings into two classes:

- `R` = Rock
- `M` = Mine

This is a binary classification problem, and the model is trained to distinguish between underwater rock formations and mines based on sensor features.

## Dataset

The dataset is a sonar dataset with multiple frequency-based measurements. Each row represents a sample, and the final column contains the target label:

- 60 feature columns
- 1 target column
- Label values: `R` and `M`

The data file is located here:

- `Dataset/sonar data.csv`

## Workflow

The project follows this workflow:

1. Load the dataset using pandas.
2. Explore the dataset shape and class distribution.
3. Separate features (`X`) and target labels (`y`).
4. Split the data into training and testing sets using train-test split.
5. Train a `LogisticRegression` model on the training data.
6. Evaluate the model using accuracy scores.
7. Use the trained model to predict whether a new sonar sample is a rock or a mine.

## Model and Evaluation

The notebook uses the `LogisticRegression` algorithm from scikit-learn.

Training and testing accuracy were evaluated as follows:

- Training Accuracy: `0.8465909090909091`
- Testing Accuracy: `0.84375`

These results indicate the model performs reasonably well for this binary classification task.

## Setup Instructions

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requrments.txt
```

3. Open the notebook and run all cells:

```bash
jupyter notebook "Rock vs mine Prediction.ipynb"
```

## File Structure

```text
Rock vs mine Prediction/
├── README.md
├── requrments.txt
├── Rock vs mine Prediction.ipynb
├── Dataset/
│   └── sonar data.csv
```

## Prediction Example

After training, the model can be used to predict a new sonar sample by reshaping the input into a 2D array and passing it to the classifier.

```python
input_data = [0.02, 0.03, 0.05, ...]
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 'R':
    print('The object is a Rock')
else:
    print('The object is a mine')
```

## Technologies Used

- Python
- Pandas
- NumPy
- scikit-learn
- Jupyter Notebook

## Summary

This project demonstrates a simple but effective machine learning workflow for solving a sonar-based rock-vs-mine classification problem using logistic regression. It covers data loading, preprocessing, training, evaluation, and real-time prediction.
