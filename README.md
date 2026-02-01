# 🫀 Heart Disease Detection Model — Python & Machine Learning
📌 Project Overview

This project presents a machine learning-based heart disease detection system developed in Python using a publicly available Kaggle dataset derived from the UCI Heart Disease repository. The system predicts whether a patient has heart disease based on a comprehensive set of clinical and demographic attributes.

The objective is to assist healthcare professionals and researchers by providing a data-driven, explainable, and reliable predictive model that supports early diagnosis and risk assessment of cardiovascular disease.

📊 Dataset Description

Source:
Kaggle — UCI Heart Disease Dataset
🔗 https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

Kaggle Contributor:
Redwan Sony

Original Dataset Contributors:

Hungarian Institute of Cardiology, Budapest — Andras Janosi, M.D.

University Hospital, Zurich — William Steinbrunn, M.D.

University Hospital, Basel — Matthias Pfisterer, M.D.

V.A. Medical Center, Long Beach & Cleveland Clinic Foundation — Robert Detrano, M.D., Ph.D.

🧾 System Inputs: Patient Attributes

The model uses the following structured clinical features:

Feature	Description
id	Unique patient identifier
age	Age of the patient (years)
origin	Place of study
sex	Patient sex (Male/Female)
cp	Chest pain type: typical angina, atypical angina, non-anginal, asymptomatic
trestbps	Resting blood pressure (mm Hg)
chol	Serum cholesterol (mg/dl)
fbs	Fasting blood sugar > 120 mg/dl (True/False)
restecg	Resting ECG results: normal, ST-T abnormality, left ventricular hypertrophy
thalach	Maximum heart rate achieved
exang	Exercise-induced angina (True/False)
oldpeak	ST depression induced by exercise relative to rest
slope	Slope of peak exercise ST segment
ca	Number of major vessels colored by fluoroscopy (0–3)
thal	Thalassemia status: normal, fixed defect, reversible defect
num	Target variable (diagnosis outcome)
🎯 System Output

Binary Classification Output:

Label	Meaning
0	No Heart Disease
1	Heart Disease Present
⚙️ Model Architecture & Workflow

The system follows a standard machine learning pipeline:

Data Ingestion
Load and inspect the Kaggle dataset using Pandas.

Data Preprocessing

Handle missing values.

Encode categorical variables.

Normalize or scale numerical features.

Remove irrelevant identifiers (e.g., id).

Exploratory Data Analysis (EDA)

Visualize distributions and correlations.

Identify key predictors of heart disease.

Model Training
Multiple supervised learning algorithms may be evaluated, such as:

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Model Evaluation
Performance is assessed using:

Accuracy

Precision, Recall, F1-score

Confusion Matrix

ROC-AUC Curve

Model Selection & Optimization
The best-performing model is selected and fine-tuned using cross-validation and hyperparameter optimization techniques.

🧠 Use Case & Applications

This model can be used for:

Clinical decision support systems.

Early risk screening tools.

Educational and research purposes.

Data science portfolio projects.

While this system is not intended for direct clinical diagnosis, it demonstrates how machine learning can support medical professionals by identifying high-risk patients using structured health data.

🛠️ Technology Stack

Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Environment: Jupyter Notebook / Python scripts

Dataset Platform: Kaggle
