# Heart Disease Prediction using Machine Learning

Heart Disease Prediction project — a machine learning workflow for data preprocessing, model training, evaluation, and interactive dashboard deployment. Built for learning and experimentation in data science and machine learning.

**Link To Dashboard**:
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nfp-05-heart-disease-prediction-dashboarddash-7u1akx.streamlit.app/)

## Dataset Information

The Dataset used in this project is sourced from Kaggle:

- **Dataset Link**: [https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Source**: Created by fedesoriano, combining 5 heart datasets (Cleveland, Hungary, Switzerland, Long Beach, and Stalog).
- **Data Size**: 918 observations with 12 clinical features.

## 📂 Project Structure

```text
Heart-Disease-Prediction/
├── Data/                  # Raw and cleaned datasets
│   ├── heart.csv
│   ├── heart_cleaned.csv
│   └── heart_processed.csv
├── src/                   #
│   └── app_api.py
├── dashboard/             # Streamlit dashboard for prediction
│   └── dash.py
├── outputs/               # Model and evaluation artifacts
│   ├── best_LRmodel.pkl
│   ├── best_LRmodel_predictions.csv
│   ├── LRconfusion_matrix.png
│   ├── encoders.pkl
│   ├── LRfeature_importance.png
│   ├── hr_ratio_correlation.png
│   ├── LRmetrics.png
│   ├── LRmodel_eval_data.pkl
│   ├── LRmodel_results.csv
│   ├── original_correlation.png
│   ├── LRroc_curve.png
│   ├── scaler.pkl
│   ├── train_columns.pkl
│   ├── logistic_regression_odds_ratio.csv
│   └── LRgridsearch_info.pkl
├── notebooks/            #
│   ├── Preprocess_Heart_Disease.ipynb
│   ├── Modeling_Heart_Disease.ipynb
│   └── EDA_Heart_Disease.ipynb
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.streamlit
├── .gitignore
└── README.md
```

---

## Features

- **Data Preprocessing**: Imputation of missing values, feature engineering, categorical encoding, and feature scaling.
- **Feature Engineering**: Create `HR_Ratio` from `MaxHR` and `Age`, then compare feature correlations before and after engineering.
- **Model Training**: Train Logistic Regression using GridSearchCV and a held-out test set.
- **Evaluation**: Measure accuracy, precision, recall, specificity, F1-score, ROC-AUC, kappa, and G-mean.
- **Dashboard**: Interactive Streamlit app for patient risk prediction.

---

## 🛠️ Tech Tools

- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib
- **Tools:** VS Code, Git/GitHub

---

## Workflow

1. **Checking Dataset Quality**: Inspect missing values, duplicate rows, and zero-value entries.
2. **Feature Engineering**: Create `HR_Ratio` and inspect correlation before and after creating the new feature.
3. **Preprocessing**: Impute missing values, encode categorical variables, scale numeric features, and save preprocessing artifacts.
4. **Modeling**: Train a Logistic Regression model, tune hyperparameters with GridSearchCV, and evaluate on a held-out test set.
5. **Evaluation**: Generate metrics, confusion matrices, ROC curves, and save results.
6. **Deployment**: Save the best model and artifacts, then integrate with Streamlit dashboard for prediction.

---

## Model Performance

The modeling notebook trains a single classifier:

- **Logistic Regression**

The model is tuned with GridSearchCV and evaluated on a held-out test set. The script reports and saves metrics such as accuracy, precision, recall, specificity, F1-score, ROC-AUC, kappa, and G-mean.

The main evaluation artifacts are stored in the outputs folder:

- [outputs/LRmodel_results.csv](outputs/LRmodel_results.csv) for the summarized metrics
- [outputs/best_LRmodel_predictions.csv](outputs/best_LRmodel_predictions.csv) for prediction results
- [outputs/best_LRmodel.pkl](outputs/best_LRmodel.pkl) for the trained model

---

## 📊 Detailed Evaluation Results

### Metrics Comparison

The following table shows the main evaluation metrics for the model on the CV and test sets.

#### Logistic Regression

| Dataset      | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC | G-Mean | Kappa  |
| ------------ | -------- | --------- | ------ | ----------- | -------- | ------- | ------ | ------ |
| CV (Train)   | 0.8501   | 0.8491    | 0.8607 | 0.8049      | 0.8548   | 0.9255  | 0.8448 | 0.6952 |
| Test (Blind) | 0.8913   | 0.8942    | 0.9118 | 0.8659      | 0.9029   | 0.9280  | 0.8885 | 0.7795 |

### 📉 ROC Curve

![ROC Curve](outputs/LRroc_curve.png)

### 🔲 Confusion Matrix

![Confusion Matrix](outputs/LRconfusion_matrix.png)

### 📊 Metrics Visualization

![Metrics](outputs/LRmetrics.png)

### 🎯 Feature Interpretation

![Feature Importance](outputs/LRfeature_importance.png)

This section shows the model explanation for the classifier:

- Logistic Regression coefficient magnitudes

These visuals make it easier to see which clinical features drive the predictions.

---

## 💡 Key Insights

1. **Strong Performance of Logistic Regression**: The Logistic Regression model delivers the highest ROC-AUC on test data (0.9280).

2. **Very High Recall**: Logistic Regression achieves 91.18% recall on the test set, which is important for reducing missed heart disease cases.

3. **Good Specificity**: Logistic Regression also scores 86.59% specificity on the test set, showing a balanced trade-off between catching disease and limiting false alarms.

4. **Feature Engineering Value**: The engineered HR_Ratio feature continues to add predictive power and is visible in the model interpretation plot.

---

## Conclusion

The Logistic Regression classifier demonstrates **strong and reliable performance** for heart disease prediction with:

### Strengths:

- **Good Balance of Metrics**: Strong performance across accuracy, recall, specificity, and ROC-AUC
- **Interpretability**: Coefficients are easier to interpret than tree-based models
- **Stable Generalization**: Good performance on the held-out test set
- **Practical Use**: Suitable as a baseline and competitive model for this classification task

### Limitations:

- Some false positives and false negatives are still present, as is common in medical prediction tasks
- The model may still benefit from further tuning, threshold adjustment, or comparison with additional algorithms

---

## Goals

- This project demonstrates an end-to-end machine learning pipeline for heart disease risk prediction, from preprocessing to deployment.
- Build an interactive dashboard for predictions.

---

This project is for **educational purposes** and part of my journey in Data Science & Machine Learning.

> **⚠️ Medical Disclaimer**: This application is for **educational purposes only**. The predictions generated are based on statistical patterns and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any medical concerns.
