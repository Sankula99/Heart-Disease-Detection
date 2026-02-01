#Used for importing dataset from kaggle using API
!pip install opendatasets

!pip install xgboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Used for importing dataset from kaggle using API
import opendatasets as od
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
## Data Exploration
#Reveals data types number of rows filled in each column, column name, number of columns
data.info()
#Checking that there are no duplicate entries
print("Unique ID",data["id"].unique())
#Checking Age Range by using min and max function (28-77)
print("Unique Ages",data["age"].unique(),'.min() or .max()')
#Ensure these rows are filled with either male or female
print(data["sex"].unique())
#Dataset Location ('Cleveland' 'Hungary' 'Switzerland' 'VA Long Beach) only 4 unique locations
print(data["dataset"].unique())
#Chest Pain Types ('typical angina' 'asymptomatic' 'non-anginal' 'atypical angina')
print(data["cp"].unique())
#resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital))-nothing jumps out, range(0-200)
print("tresbps",data["trestbps"].unique())
#serum cholesterol in mg/dl- nothing jumps out here, range(0-603)
print("chol",data["chol"].unique())
#(if fasting blood sugar > 120 mg/dl), options (true,false)
print(data["fbs"].unique())
#(resting electrocardiographic results) Values: [normal, stt abnormality, lv hypertrophy]
print(data["restecg"].unique())
# maximum heart rate achieved, range 60-202
print("thalch",data["thalch"].unique())
#Exercise-induced angina (True/ False)
print(data["exang"].unique())
#ST depression induced by exercise relative to rest, range(-2.6-6.2)
print(data["oldpeak"].unique())
#the slope of the peak exercise ST segment, options('downsloping' 'flat' 'upsloping')
print(data["slope"].unique())
#number of major vessels (0-3) colored by fluoroscopy, options(0.  3.  2.  1.)
print(data["ca"].unique())
#options[normal; fixed defect; reversible defect]
print(data["thal"].unique())
#Severity of heart disease (1-4, with 4 being the max) and 0 meaning that there is no presence of the condition
print(data["num"].unique())
#Missing values and percentages
missing_values = data.isnull().sum()
print("Number of Missing Values",missing_values)
missing_percent = (missing_values/len(data))*100
print("Missing Percentages",missing_percent)
#Visually checking outliers for chol column
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['chol'],color='red')
plt.title('Boxplot of chol')
plt.xlabel('chol')
plt.show()

#Outliers for trestbps
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['trestbps'],color='blue')
plt.title('Boxplot of trestbps')
plt.xlabel('trestbps')
plt.show()

#Outliers for fbs
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['thalch'],color='green')
plt.title('Boxplot of thalch')
plt.xlabel('thalch')
plt.show()

#Outliers for OldPeak
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['oldpeak'],color='hotpink')
plt.title('Boxplot of oldpeak')
plt.xlabel('oldpeak')
plt.show()
data['heart_disease'] = data['num'].apply(lambda x: 0 if x == 0 else 1)
plt.figure(figsize=(8, 5))
sns.histplot(data['chol'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Cholesterol')
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data['trestbps'], bins=30, kde=True, color='red')
plt.title('Distribution of trestbps')
plt.xlabel('trestbps')
plt.ylabel('Frequency')
plt.show()
## Encoding for XGBoost, since it doesnt work very well with categorical features.
from sklearn.preprocessing import LabelEncoder
categorical_cols = ["sex","cp","fbs","restecg","exang","slope","thal","dataset"]
label_encoders = {}
for col in categorical_cols:
  le = LabelEncoder()
  data[col] = le.fit_transform(data[col].astype(str))
  label_encoders[col]=le
print(data.dtypes)

print(data.head())
# Extra Preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
#Assign the dataframe/dataset to X and drop/remove columns id, num and heart_disease
X = data.drop(['id', 'num', 'heart_disease'], axis=1, errors='ignore')
#Assign the heart disease column to
y = data['heart_disease']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test1 = train_test_split(
    X_temp, y_temp, test_size=0.3333, random_state=42, stratify=y_temp
)

cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]

X_train[['chol','trestbps']]= X_train[['chol','trestbps']].replace(0,np.nan)
X_val[['chol','trestbps']]= X_val[['chol','trestbps']].replace(0,np.nan)
X_test[['chol','trestbps']]= X_test[['chol','trestbps']].replace(0,np.nan)

median_values = X_train[cols].median()

X_train[cols] = X_train[cols].fillna(median_values)
X_val[cols] = X_val[cols].fillna(median_values)
X_test[cols] = X_test[cols].fillna(median_values)
#Optimisation 1:Hyper Param
from sklearn.model_selection import GridSearchCV

xgb_param_grid = {
    'n_estimators': [100,200,300],
    'max_depth': [3,4,5],
    'learning_rate': [0.01,0.1,0.2],
    'subsample': [0.7,0.8,1],
    'colsample_bytree': [0.7,0.8,1],
    'gamma': [0,0.1,0.2,0.5,1]
}

xgb_op1 = xgb.XGBClassifier(
    objective= 'binary:logistic',
    eval_metric= 'logloss',
    use_label_encoder=False,
    random_state=42
)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=xgb_op1,
    param_grid=xgb_param_grid,
    scoring='f1',
    cv=kf,
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train,
         eval_set=[(X_val, y_val)])

print("Best hyperparameters:", grid.best_params_)
print("Best CV F1 Score:", round(grid.best_score_, 4))

best_model1 = grid.best_estimator_
y_pred2 = best_model1.predict(X_test)

print("Test Accuracy:", round(accuracy_score(y_test1, y_pred2), 4))
print("Test F1 Score:", round(f1_score(y_test1, y_pred2), 4))
print("\nClassification Report:\n", classification_report(y_test1, y_pred2))

cm = confusion_matrix(y_test1, y_pred2)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Heart Disease Presence)")
plt.show()

xgb.plot_importance(best_model1)
plt.show()
# Moved Model 1 under model 2 due to affecting model 2 performance becuase of outlier removal- this before optimisation**
iqr_bounds = {}
cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
for col in cols:
  Q1 = X_train[col].quantile(0.25)
  Q3 = X_train[col].quantile(0.75)
  IQR = Q3 - Q1
  lower = Q1 - 1.5 * IQR
  upper = Q3 + 1.5 * IQR
  iqr_bounds[col] = (lower, upper)
  mask = (X_train[col] >= lower) & (X_train[col] <= upper)
  X_train = X_train[mask].reset_index(drop=True)
  y_train = y_train[mask].reset_index(drop=True)

for col in cols:
  lower, upper = iqr_bounds[col]
  X_val[col] = X_val[col].clip(lower, upper)
  X_test[col] = X_test[col].clip(lower, upper)


base = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    learning_rate=0.1,
    max_depth=4,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False  # Avoid label encoder warning

)

base.fit(X_train, y_train)

y_pred1 = base.predict(X_test)

print("Accuracy:", accuracy_score(y_test1, y_pred1))
print("F1 Score:", f1_score(y_test1, y_pred1))
print("\nClassification Report:\n", classification_report(y_test1, y_pred1))

cm = confusion_matrix(y_test1, y_pred1)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Heart Disease Presence)XGB")
plt.show()
#Optimisation 2: HyperParam and removal of Outliers
xgb_param_grid = {
    'n_estimators': [100,200,300],
    'max_depth': [3,4,5],
    'learning_rate': [0.01,0.1,0.2],
    'subsample': [0.7,0.8,1],
    'colsample_bytree': [0.7,0.8,1],
    'gamma': [0,0.1,0.2,0.5,1]
}

xgb_op2 = xgb.XGBClassifier(
    objective= 'binary:logistic',
    eval_metric= 'logloss',
    use_label_encoder=False,
    random_state=42
)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=xgb_op2,
    param_grid=xgb_param_grid,
    scoring='f1',
    cv=kf,
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train,
         eval_set=[(X_val, y_val)])

print("Best hyperparameters:", grid.best_params_)
print("Best CV F1 Score:", round(grid.best_score_, 4))

best_model2 = grid.best_estimator_
y_pred3 = best_model2.predict(X_test)

print("Test Accuracy:", round(accuracy_score(y_test1, y_pred3), 4))
print("Test F1 Score:", round(f1_score(y_test1, y_pred3), 4))
print("\nClassification Report:\n", classification_report(y_test1, y_pred3))

cm = confusion_matrix(y_test1, y_pred3)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Heart Disease Presence)")
plt.show()

xgb.plot_importance(best_model2)
plt.show()
#ROC CURVE
from sklearn.metrics import roc_curve, auc
y_pred_prob = base.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test1, y_pred_prob)
roc_auc = auc(fpr,tpr)

models = {
    "Base": base,
    "OP1": best_model1,
    "OP2": best_model2
}

for model_name, model in models.items():
  y_pred_prob = model.predict_proba(X_test)[:, 1]
  fpr, tpr, thresholds = roc_curve(y_test1, y_pred_prob)
  roc_auc = auc(fpr,tpr)
  plt.figure(figsize=(8,6))
  plt.plot(fpr,tpr,color='darkorange',lw=2,label='ROC curve (area = %0.2f)' % roc_auc)
  plt.legend(loc="lower right")
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()
