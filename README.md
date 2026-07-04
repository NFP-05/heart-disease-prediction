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
├── src/                   # Python scripts for preprocessing & modeling
│   ├── preprocess.py
│   └── modeling.py
├── dashboard/             # Streamlit dashboard for prediction
│   └── dash.py
├── outputs/               # Model and evaluation artifacts
│   ├── best_model.pkl
│   ├── best_logistic_regression.pkl
│   ├── best_model_predictions.csv
│   ├── confusion_matrices.png
│   ├── encoders.pkl
│   ├── feature_comparison.png
│   ├── hr_ratio_correlation.png
│   ├── metrics_comparison.png
│   ├── model_eval_data.pkl
│   ├── model_results.csv
│   ├── original_correlation.png
│   ├── roc_curves.png
│   ├── scaler.pkl
│   ├── train_columns.pkl
│   └── gridsearch_info.pkl
├── notebooks/            # Workflow in the project
│   └── project-walkthrough.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Features

- **Data Preprocessing**: Imputation of missing values, feature engineering, categorical encoding, and feature scaling.
- **Feature Engineering**: Create `HR_Ratio` from `MaxHR` and `Age`, then compare feature correlations before and after engineering.
- **Model Training**: Compare Random Forest and Logistic Regression using GridSearchCV and a held-out test set.
- **Evaluation**: Measure accuracy, precision, recall, specificity, F1-score, ROC-AUC, kappa, and G-mean.
- **Visualization**: Generate confusion matrices, ROC curves, metric comparison plots, and feature/coefficients comparison charts.
- **Dashboard**: Interactive Streamlit app for patient risk prediction.

---

## 🛠️ Tech Tools

- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib
- **Tools:** VS Code, Git/GitHub

---

## 📊 Workflow

1. **Checking Dataset Quality**: Inspect missing values, duplicate rows, and zero-value entries.
2. **Feature Engineering**: Create `HR_Ratio` and inspect correlation before and after creating the new feature.
3. **Preprocessing**: Impute missing values, encode categorical variables, scale numeric features, and save preprocessing artifacts.
4. **Modeling**: Train a Random Forest and Logistic Regression model, tune hyperparameters with GridSearchCV, and evaluate on a held-out test set.
5. **Evaluation**: Generate metrics, confusion matrices, ROC curves, and save results.
6. **Deployment**: Save the best model and artifacts, then integrate with Streamlit dashboard for prediction.

---

## 📈 Model Performance

The modeling script now compares two classifiers:

- **Random Forest**
- **Logistic Regression**

Both models are tuned with GridSearchCV and evaluated on a held-out test set. The script reports and saves metrics such as accuracy, precision, recall, specificity, F1-score, ROC-AUC, kappa, and G-mean.

The main evaluation artifacts are stored in the outputs folder:

- [outputs/model_results.csv](outputs/model_results.csv) for the summarized metrics
- [outputs/best_model_predictions.csv](outputs/best_model_predictions.csv) for prediction results
- [outputs/best_model.pkl](outputs/best_model.pkl) and [outputs/best_logistic_regression.pkl](outputs/best_logistic_regression.pkl) for the trained models

---

## 📊 Detailed Evaluation Results

### Metrics Comparison

The script generates a comparison table and plots for both models so you can assess which one performs better for your priority metric, such as recall or specificity.

### 📉 ROC Curves

The ROC comparison plot is saved as [outputs/roc_curves.png](outputs/roc_curves.png).

### 🔲 Confusion Matrices

The confusion matrices for the model comparison are saved as [outputs/confusion_matrices.png](outputs/confusion_matrices.png).

### 📊 Metrics Comparison Visualization

The bar chart for model comparison is saved as [outputs/metrics_comparison.png](outputs/metrics_comparison.png).

### 🎯 Feature Interpretation

The feature comparison plot is saved as [outputs/feature_comparison.png](outputs/feature_comparison.png).

This plot shows:

- Random Forest feature importance values
- Logistic Regression coefficient magnitudes

These visuals help interpret which clinical variables contribute most to the predictions.

---

## 💡 Key Insights

1. **Strong Performance of Logistic Regression**: The Logistic Regression model performs best overall on the test set, making it the preferred model for this project.

2. **Excellent Discrimination**: The model shows strong ability to separate positive and negative cases, supported by a high ROC-AUC score.

3. **Good Generalization**: The evaluation results suggest the model generalizes well to unseen data with minimal overfitting.

4. **Balanced Classification**: The model provides a good trade-off between recall and specificity, which is important for medical prediction tasks.

5. **Feature Engineering Value**: The engineered HR_Ratio feature contributes useful information to the predictive models.

---

## ✅ Conclusion

The Logistic Regression classifier demonstrates **strong and reliable performance** for heart disease prediction with:

### ✨ Strengths:

- **Good Balance of Metrics**: Strong performance across accuracy, recall, specificity, and ROC-AUC
- **Interpretability**: Coefficients are easier to interpret than tree-based models
- **Stable Generalization**: Good performance on the held-out test set
- **Practical Use**: Suitable as a baseline and competitive model for this classification task

### ⚠️ Limitations:

- Some false positives and false negatives are still present, as is common in medical prediction tasks
- The model may still benefit from further tuning, threshold adjustment, or comparison with additional algorithms

---

## 🎯 Goals

- This project demonstrates an end-to-end machine learning pipeline for heart disease risk prediction, from preprocessing to deployment.
- Build an interactive dashboard for predictions.

---

⭐️ This project is for **educational purposes** and part of my journey in Data Science & Machine Learning.

> **⚠️ Medical Disclaimer**: This application is for **educational purposes only**. The predictions generated are based on statistical patterns and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any medical concerns.
