# Breast Cancer Classification - M.Tech Assignment 2

### 1. Problem Statement
The goal of this project is to build an end-to-end Machine Learning classification pipeline to predict whether a breast tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)** based on diagnostic measurements. The solution involves training six different classification models, evaluating their performance using standard metrics, and deploying the best-performing solution via an interactive Streamlit web application.

### 2. Dataset Description
* **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
* **Source:** Scikit-Learn (originally from UCI Machine Learning Repository)
* **Sample Size:** 569 instances
* **Feature Count:** 30 numeric features (e.g., mean radius, mean texture, mean smoothness) plus 1 target variable.
* **Target Classes:**
    * `0`: Malignant
    * `1`: Benign

### 3. Models Used & Comparison
The following classification models were trained and evaluated on the test set (20% split). The performance metrics are recorded below:

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **98.25%** | **0.9954** | **0.9861** | **0.9861** | **0.9861** | **0.9623** |
| **Decision Tree** | 91.23% | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| **KNN** | 95.61% | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 93.86% | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| **Random Forest** | 95.61% | 0.9937 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **XGBoost** | 94.74% | 0.9917 | 0.9459 | 0.9722 | 0.9589 | 0.8864 |

### 4. Observations
Analysis of model performance on the Breast Cancer dataset:

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | **Best Performer.** Achieved the highest Accuracy (98.25%) and MCC (0.96), indicating the dataset is linearly separable. Feature scaling was critical for this result. |
| **Decision Tree** | Lowest performance (91.23%) among all models. Likely prone to overfitting compared to the ensemble methods (Random Forest/XGBoost). |
| **KNN** | Strong performance (95.61%), tying with Random Forest. High recall (0.9722) makes it reliable for medical diagnosis, though inference is slower. |
| **Naive Bayes** | Solid baseline (93.86%) with high AUC (0.9878), performing well despite the assumption of feature independence. |
| **Random Forest** | Excellent generalization (95.61%) and very high AUC (0.9937). It matched KNN in accuracy but provides better interpretability than KNN. |
| **XGBoost** | Very competitive (94.74%) with near-perfect AUC (0.9917). Slightly lower accuracy than Logistic Regression suggests a simpler linear model fits this specific data better than complex boosting. |