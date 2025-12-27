# Crime-Report-Risk-Classifier
NLP classifier that predicts police report areas from crime descriptions using TF-IDF and Logistic Regression in Python

# Description

This project is a multi-class NLP classifier that predicts the police report area from crime descriptions. The model uses TF-IDF vectorization and Logistic Regression to process and classify crime data. It includes dataset preprocessing, class balancing, training, evaluation, and a saved pipeline for reproducible predictions.

Features
- Preprocesses and balances crime report data
- Converts text descriptions into numerical features using TF-IDF
- Trains a Logistic Regression classifier to predict report areas
- Evaluates model performance with accuracy, confusion matrix, and classification report
- Saves the trained pipeline for reuse without retraining
- Installation

# Usage
import joblib

(Load trained model)
model = joblib.load("crime_area_classifier.pkl")

(Predict new crime descriptions)
sample_texts = [
    "BURGLARY AT JEWELRY STORE",
    "ASSAULT IN HOLLYWOOD"
]
predictions = model.predict(sample_texts)

for text, pred in zip(sample_texts, predictions):
    print(f"Crime: '{text}' --> Predicted Area: {pred}")

# Dataset

- Crime data used in this project comes from the Los Angeles Police Department (2020–present).
   
- You can download it here: [data.lacity.org](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8)


# Model Performance

-Accuracy: ~40–55% (depending on data sample and preprocessing)

-Evaluated using a confusion matrix and classification report
