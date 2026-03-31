# heart-disease-prediction
Heart Disease Prediction project — a machine learning workflow for data preprocessing, model training, evaluation, and interactive dashboard deployment. Built for learning and experimentation in data science and machine learning.

## 📂 Project Structure

```text
Heart-Disease-Prediction/
├── Data/                  # Raw and cleaned datasets
│   ├── heart.csv
│   ├── heart_cleaned.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── src/                   # Python scripts for preprocessing & modeling
│   ├── preprocess.py
│   ├── modeling.py
│   └── Profiling_Dataset.py
├── dashboard/             # Streamlit dashboard for prediction
│   └── dash.py
├── outputs/               # Model and evaluation results
│   ├── best_model.pkl
│   ├── metrics_comparison.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── report.html
└── README.md
```

---

## 🚀 Features
- **Data Preprocessing**: Cleaning missing/zero values, encoding categorical features, scaling numeric features.
- **Model Training**: Logistic Regression and other ML algorithms with evaluation metrics.
- **Visualization**: Confusion matrices, ROC curves, feature importance.
- **Dashboard**: Interactive Streamlit app for patient risk prediction.

---

## 🛠️ Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib  
- **Tools:** VS Code, Git/GitHub

---

## 📊 Workflow
1. **Preprocessing**: Clean dataset, encode categorical variables, scale numeric features.
2. **Modeling**: Train/test split, fit models, evaluate performance.
3. **Evaluation**: Generate metrics, confusion matrices, ROC curves.
4. **Deployment**: Save best model (`best_model.pkl`) and integrate with Streamlit dashboard.

---

## 🎯 Goals
- Learn end-to-end ML workflow with a real dataset.
- Practice reproducible project structure.
- Build an interactive dashboard for predictions.

---

⭐️ This project is for **educational purposes** and part of my journey in Data Science & Machine Learning. Feedback and suggestions are welcome!
