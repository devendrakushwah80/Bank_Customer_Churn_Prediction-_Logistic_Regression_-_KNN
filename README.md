# 🏦 Bank Customer Churn Prediction  
### Logistic Regression & K-Nearest Neighbors (KNN)

---

## 📌 Project Overview

Customer churn prediction is a critical problem in the banking industry.  
This project builds an **end-to-end machine learning pipeline** to predict whether a customer will **leave the bank or not**, using **two classification algorithms**:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**

The notebook focuses strongly on **class imbalance handling**, **proper preprocessing**, and **model comparison**.

---

## 🎯 Problem Statement

Given customer demographic and financial information, predict:

- `Exited = 1` → Customer will leave the bank  
- `Exited = 0` → Customer will stay  

This is a **binary classification problem** with **imbalanced classes**.

---

## 📊 Dataset Description

Key features used in the dataset:

- CreditScore  
- Geography  
- Gender  
- Age  
- Tenure  
- Balance  
- NumOfProducts  
- HasCrCard  
- IsActiveMember  
- EstimatedSalary  

🎯 **Target Variable**
- `Exited`

---

## 🔍 Exploratory Data Analysis (EDA)

### ✔ Steps Performed
- Checked dataset shape and data types  
- Verified missing values and duplicates  
- Analyzed target class distribution  
- Performed basic numerical & categorical analysis  

### ⚠️ Class Imbalance
The dataset is highly imbalanced:
- Majority class → `Exited = 0`
- Minority class → `Exited = 1`

This imbalance negatively impacts model learning if not handled.

---

## ⚖️ Handling Class Imbalance

To fix imbalance:
- Applied **manual random undersampling**
- Made both classes (`0` and `1`) equal in count
- Shuffled the dataset

✅ This ensures fair learning for both classes.

---

## 🛠 Feature Engineering & Preprocessing

### 🧹 Dropped Irrelevant Columns
- `RowNumber`
- `CustomerId`
- `Surname`

### 🔄 Encoding & Scaling (Using ColumnTransformer)
- **StandardScaler** → Numerical features  
- **OneHotEncoder** → Categorical features (`Geography`, `Gender`)

📌 Preprocessing was applied **after train-test split** to avoid data leakage.

---

## 🤖 Models Implemented

### 1️⃣ Logistic Regression
- Used as a **baseline classification model**
- Works well with scaled and encoded data
- Evaluated using:
  - Precision
  - Recall
  - F1-score
  - Accuracy

📌 Observations:
- Recall for churned customers improved after balancing
- Provides good interpretability

---

### 2️⃣ K-Nearest Neighbors (KNN)
- Distance-based, non-parametric algorithm
- Value of **K selected using heuristic**:
K = √(number of training samples)

- Requires proper feature scaling (handled in preprocessing)

📌 Observations:
- Sensitive to scaling and data distribution
- Captures non-linear decision boundaries better than Logistic Regression

---

## 📈 Model Evaluation & Comparison

- Models evaluated on **unseen test data**
- Used **classification report**:
  - Precision
  - Recall
  - F1-score
  - Accuracy

### Key Insight
> In churn prediction, **recall is more important than accuracy**, because missing a churned customer is costlier than a false alarm.

---

## 🧠 Key Learnings

- Accuracy alone is misleading for imbalanced datasets  
- Logistic Regression:
  - Stable
  - Interpretable
- KNN:
  - Flexible
  - Sensitive to preprocessing
- **Data quality & imbalance handling matter more than model choice**

---

## 🧪 Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---

## 🚀 Future Improvements

- Use **SMOTE / ADASYN** instead of undersampling  
- Add **ROC–AUC & Precision–Recall curves**  
- Hyperparameter tuning:
  - `C` for Logistic Regression  
  - `K` and distance metric for KNN  
- Compare with:
  - Random Forest  
  - XGBoost  

---

## 📂 Project Structure
📦 Bank-Churn-Prediction
┣ 📜 Bank_Churn_prediction.ipynb
┣ 📜 churn.csv
┗ 📜 README.md

---

## ✅ Conclusion

This project demonstrates a **real-world machine learning workflow**:

- Handling imbalanced data  
- Proper preprocessing pipelines  
- Training **multiple classification algorithms**  
- Business-oriented model evaluation  

A strong foundation for **industry-level churn prediction problems** 🚀
