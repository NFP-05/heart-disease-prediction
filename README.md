# heart-disease-prediction

Heart Disease Prediction project — a machine learning workflow for data preprocessing, model training, evaluation, and interactive dashboard deployment. Built for learning and experimentation in data science and machine learning.

**Link To Dashboard**:
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nfp-05-heart-disease-prediction-dashboarddash-7u1akx.streamlit.app/)

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
├── outputs/               # Model and evaluation results
│   ├── best_model.pkl
│   ├── best_model_predictions.pkl
│   ├── encoders.pkl
│   ├── metrics_comparison.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   ├── gridsearch_info.pkl
│   ├── model_results.csv
│   ├── roc_curves.png
│   ├── scaler.pkl
│   └── train_columns.pkl
│   requirements.txt
│   .gitignore
└── README.md
```

---

## 🚀 Features

- **Data Preprocessing**: Imputations, encoding categorical features, scaling numeric features.
- **Model Training**: Random Forest.
- **Visualization**: Confusion matrices, ROC curves, metrics comparisons.
- **Dashboard**: Interactive Streamlit app for patient risk prediction.

---

## 🛠️ Tech Stack

- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib
- **Tools:** VS Code, Git/GitHub

---

## 📊 Workflow

1. **Checking Dataset Quality**: Missing Value, Duplicate and Zero
2. **Preprocessing**: Imputations, encode categorical variables.
3. **Modeling**: Train&Validation(CV)/test split, tuning models, evaluate performance.
4. **Evaluation**: Generate metrics, confusion matrices, ROC curves.
5. **Deployment**: Save best model (`best_model.pkl`) and integrate with Streamlit dashboard.

---

## 🎯 Goals

- Learn end-to-end ML workflow with a real dataset.
- Practice reproducible project structure.
- Build an interactive dashboard for predictions.

---

⭐️ This project is for **educational purposes** and part of my journey in Data Science & Machine Learning. Feedback and suggestions are welcome!
