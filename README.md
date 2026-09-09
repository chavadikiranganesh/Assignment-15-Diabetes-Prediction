Assignment 15 — Diabetes Prediction using LightGBM and XGBoost

📌 Project Overview

This project implements a machine learning based Diabetes Prediction System using the LightGBM and XGBoost classification algorithms.

The project is based on the Diabetes dataset and includes:

Exploratory Data Analysis (EDA)

Data preprocessing

Missing-value handling

LightGBM classification

XGBoost classification

Model evaluation

Cross-validation

Hyperparameter tuning

Model comparison

Streamlit web application

Model deployment

The original assignment focuses on comparing the performance of LightGBM and XGBoost on the diabetes dataset.

🎯 Objective

The objective of this assignment is to compare the performance of LightGBM and XGBoost algorithms for predicting whether a person is likely to have diabetes.

The project also converts the trained machine learning models into an interactive Streamlit web application for real-time prediction.

📊 Dataset

The project uses a diabetes classification dataset.

The target variable is:

Outcome = 0 → No Diabetes

Outcome = 1 → Diabetes

Features

Feature

Description

Pregnancies

Number of pregnancies

Glucose

Plasma glucose concentration

BloodPressure

Diastolic blood pressure

SkinThickness

Triceps skin fold thickness

Insulin

2-Hour serum insulin

BMI

Body Mass Index

DiabetesPedigreeFunction

Diabetes pedigree function

Age

Age of the person

Outcome

Target variable

🔍 Exploratory Data Analysis

The notebook performs exploratory analysis to understand the dataset.

The analysis includes:

Dataset shape

Data types

Statistical summary

Missing-value analysis

Feature distributions

Histograms

Box plots

Correlation analysis

Relationship between features and diabetes outcome

🧹 Data Preprocessing

The following preprocessing steps are performed:

Separate input features and target variable.

Identify invalid zero values in medical measurements.

Treat zero values in selected columns as missing values.

Replace missing values using median imputation.

Split the dataset into training and testing sets.

Use stratification to maintain the class distribution.

The following features are treated for missing-value handling:

Glucose

BloodPressure

SkinThickness

Insulin

BMI

🤖 Machine Learning Models

Two gradient-boosting classification algorithms are implemented.

1. LightGBM

LightGBM is a gradient boosting framework designed for efficient and high-performance machine learning.

It is trained using a preprocessing pipeline that includes median imputation.

2. XGBoost

XGBoost is another powerful gradient boosting algorithm widely used for classification and regression problems.

The XGBoost model is also trained using the preprocessing pipeline.

⚙️ Model Optimization

Cross-validation and hyperparameter tuning are used to improve model performance.

Important hyperparameters include:

LightGBM

n_estimators

max_depth

learning_rate

XGBoost

n_estimators

max_depth

learning_rate

The models are compared using appropriate classification metrics.

📈 Model Evaluation

The models are evaluated using:

Accuracy

Precision

Recall

F1-Score

The results are compared to identify which algorithm performs better on the diabetes dataset.

🖥️ Streamlit Application

The trained models are integrated into an interactive Streamlit web application.

The application allows users to enter patient information such as:

Pregnancies

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age

The user can select either:

LightGBM

XGBoost

and click the Predict Diabetes button.

The application then displays:

Prediction result

Estimated diabetes probability

🏗️ Project Architecture

Diabetes Dataset
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Data Preprocessing
       │
       ▼
Train/Test Split
       │
       ├───────────────┐
       ▼               ▼
   LightGBM         XGBoost
       │               │
       └───────┬───────┘
               ▼
       Model Evaluation
               │
               ▼
       Save Trained Models
               │
               ▼
        Streamlit App
               │
               ▼
       Diabetes Prediction
               │
               ▼
          Deployment

🌐 Live Application

The deployed Streamlit application is available here: 
https://assignment-15-diabetes-prediction-4zo5vb6uaoggypw2vybrwm.streamlit.app/

Open the Live Diabetes Prediction Application

📁 Project Structure

Assignment_15_Diabetes_Deployment/
│
├── Assignment_15_LGBM_XGBM_Diabetes.ipynb
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── diabetes.csv
│
├── models/
│   ├── lightgbm_model.pkl
│   └── xgboost_model.pkl
│
└── screenshots/
    ├── home_page.png
    ├── prediction_result.png
    └── deployed_app.png

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

LightGBM

XGBoost

Joblib

Streamlit

Jupyter Notebook

💻 Installation

Clone the repository:

git clone REPOSITORY_URL

Move into the project directory:

cd Assignment_15_Diabetes_Deployment

Install the required libraries:

pip install -r requirements.txt

▶️ Run the Project Locally

First train and save the machine learning models:

python train_models.py

Then start the Streamlit application:

streamlit run app.py

The application will be available at:

http://localhost:8501

📋 Assignment Submission

The final submission contains:

Jupyter Notebook

EDA

Data preprocessing

LightGBM

XGBoost

Model evaluation

Hyperparameter tuning

Comparative analysis

Source Code

app.py

train_models.py

Dataset

diabetes.csv

Trained Models

LightGBM model

XGBoost model

Requirements File

requirements.txt

Project Documentation

README.md
Live Streamlit application URL
https://assignment-15-diabetes-prediction-4zo5vb6uaoggypw2vybrwm.streamlit.app/
