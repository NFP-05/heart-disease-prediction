# Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, cohen_kappa_score,
    classification_report
)
from scipy.stats import gmean
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load Processed Data
df = pd.read_csv('Data/heart_processed.csv')
print(f"Data shape: {df.shape}")

# Separate features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}\n")

# Splitting Dataset 80:20
print("="*50)
print("Creating Train Set (80%) & Test Set (20%)")
print("="*50)
X_cv, X_test, y_cv, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"CV Data (80%): {X_cv.shape}")
print(f"Blind Test Data (20%): {X_test.shape}")
print(f"  - Class 0: {(y_test == 0).sum()}")
print(f"  - Class 1: {(y_test == 1).sum()}\n")

# RF Modeling
print("="*50)
print("RANDOM FOREST MODEL")
print("="*50)

param_grid = {
    'n_estimators': [90, 100, 200, 300],
    'max_depth': [8, 10, 15],
    'min_samples_split': [25, 30]
}

print("RF Parameter Grid:")
for key, value in param_grid.items():
    print(f"  {key}: {value}")

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# Setup GridSearchCV with 5-fold cross-validation
rf_grid_search = GridSearchCV(
    rf_base,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("\nTraining with GridSearchCV (5-Fold CV)...")
rf_grid_search.fit(X_cv, y_cv)

# Get best model
best_rf_model = rf_grid_search.best_estimator_
best_params = rf_grid_search.best_params_

print(f"\nBest RF Parameters Found:")
for key, value in best_params.items():
    print(f"  {key}: {value}")
print(f"Best RF CV ROC-AUC Score: {rf_grid_search.best_score_:.4f}\n")

# Train Logistic Regression as a baseline comparison model
print("="*50)
print("LOGISTIC REGRESSION MODEL")
print("="*50)

logreg_param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear']
}

print("Logistic Regression Parameter Grid:")
for key, value in logreg_param_grid.items():
    print(f"  {key}: {value}")

logreg_base = LogisticRegression(random_state=42, max_iter=1000)
logreg_grid_search = GridSearchCV(
    logreg_base,
    logreg_param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("\nTraining Logistic Regression with GridSearchCV (5-Fold CV)...")
logreg_grid_search.fit(X_cv, y_cv)

best_logreg_model = logreg_grid_search.best_estimator_
best_logreg_params = logreg_grid_search.best_params_

print(f"\nBest Logistic Regression Parameters Found:")
for key, value in best_logreg_params.items():
    print(f"  {key}: {value}")
print(f"Best CV ROC-AUC Score: {logreg_grid_search.best_score_:.4f}\n")

# EVALUATE: Make predictions on CV and test data
print("="*50)
print("EVALUATE: Making Predictions & Computing Metrics")
print("="*50)

# Random Forest Predictions
y_pred_cv_rf = best_rf_model.predict(X_cv)
y_pred_proba_cv_rf = best_rf_model.predict_proba(X_cv)[:, 1]

y_pred_test_rf = best_rf_model.predict(X_test)
y_pred_proba_test_rf = best_rf_model.predict_proba(X_test)[:, 1]

# Logistic Regression predictions
y_pred_cv_lr = best_logreg_model.predict(X_cv)
y_pred_proba_cv_lr = best_logreg_model.predict_proba(X_cv)[:, 1]

y_pred_test_lr = best_logreg_model.predict(X_test)
y_pred_proba_test_lr = best_logreg_model.predict_proba(X_test)[:, 1]

# Function to calculate metrics
def calculate_metrics(y_true, y_pred, y_pred_proba, dataset_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Calculate Specificity and G-Mean
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/Sensitivity
    tnr = specificity  # Specificity
    g_mean = gmean([tpr, tnr])
    
    cm = confusion_matrix(y_true, y_pred)
    
    metrics_dict = {
        'Dataset': dataset_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Kappa': kappa,
        'G-Mean': g_mean,
        'Confusion_Matrix': cm
    }
    
    return metrics_dict

# Calculate metrics for both datasets
metrics_cv = calculate_metrics(y_cv, y_pred_cv_rf, y_pred_proba_cv_rf, 'CV (Train)')
metrics_test = calculate_metrics(y_test, y_pred_test_rf, y_pred_proba_test_rf, 'Test (Blind)')

metrics_cv_lr = calculate_metrics(y_cv, y_pred_cv_lr, y_pred_proba_cv_lr, 'CV (Train)')
metrics_test_lr = calculate_metrics(y_test, y_pred_test_lr, y_pred_proba_test_lr, 'Test (Blind)')

# Display metrics
print("\n" + "="*50)
print("METRICS SUMMARY")
print("="*50)

rf_metrics_summary = pd.DataFrame([
    {
        'Dataset': 'CV (Train)',
        'Accuracy': f"{metrics_cv['Accuracy']:.4f}",
        'Precision': f"{metrics_cv['Precision']:.4f}",
        'Recall': f"{metrics_cv['Recall']:.4f}",
        'Specificity': f"{metrics_cv['Specificity']:.4f}",
        'F1-Score': f"{metrics_cv['F1-Score']:.4f}",
        'Kappa': f"{metrics_cv['Kappa']:.4f}",
        'G-Mean': f"{metrics_cv['G-Mean']:.4f}",
        'ROC-AUC': f"{metrics_cv['ROC-AUC']:.4f}"
    },
    {
        'Dataset': 'Test (Blind)',
        'Accuracy': f"{metrics_test['Accuracy']:.4f}",
        'Precision': f"{metrics_test['Precision']:.4f}",
        'Recall': f"{metrics_test['Recall']:.4f}",
        'Specificity': f"{metrics_test['Specificity']:.4f}",
        'F1-Score': f"{metrics_test['F1-Score']:.4f}",
        'Kappa': f"{metrics_test['Kappa']:.4f}",
        'G-Mean': f"{metrics_test['G-Mean']:.4f}",
        'ROC-AUC': f"{metrics_test['ROC-AUC']:.4f}"
    }
])

lr_metrics_summary = pd.DataFrame([
    {
        'Dataset': 'CV (Train)',
        'Accuracy': f"{metrics_cv_lr['Accuracy']:.4f}",
        'Precision': f"{metrics_cv_lr['Precision']:.4f}",
        'Recall': f"{metrics_cv_lr['Recall']:.4f}",
        'Specificity': f"{metrics_cv_lr['Specificity']:.4f}",
        'F1-Score': f"{metrics_cv_lr['F1-Score']:.4f}",
        'Kappa': f"{metrics_cv_lr['Kappa']:.4f}",
        'G-Mean': f"{metrics_cv_lr['G-Mean']:.4f}",
        'ROC-AUC': f"{metrics_cv_lr['ROC-AUC']:.4f}"
    },
    {
        'Dataset': 'Test (Blind)',
        'Accuracy': f"{metrics_test_lr['Accuracy']:.4f}",
        'Precision': f"{metrics_test_lr['Precision']:.4f}",
        'Recall': f"{metrics_test_lr['Recall']:.4f}",
        'Specificity': f"{metrics_test_lr['Specificity']:.4f}",
        'F1-Score': f"{metrics_test_lr['F1-Score']:.4f}",
        'Kappa': f"{metrics_test_lr['Kappa']:.4f}",
        'G-Mean': f"{metrics_test_lr['G-Mean']:.4f}",
        'ROC-AUC': f"{metrics_test_lr['ROC-AUC']:.4f}"
    }
])

print("\nRandom Forest Metrics")
print(rf_metrics_summary.to_string(index=False))
print("\nLogistic Regression Metrics")
print(lr_metrics_summary.to_string(index=False))

# Classification Reports
print("\n" + "="*50)
print("CLASSIFICATION REPORT - RANDOM FOREST - CV (TRAIN)")
print("="*50)
print(classification_report(y_cv, y_pred_cv_rf, target_names=['No Disease', 'Has Disease']))

print("\n" + "="*50)
print("CLASSIFICATION REPORT - LOGISTIC REGRESSION - CV (TRAIN)")
print("="*50)
print(classification_report(y_cv, y_pred_cv_lr, target_names=['No Disease', 'Has Disease']))

print("\n" + "="*50)
print("CLASSIFICATION REPORT - RANDOM FOREST - TEST (BLIND)")
print("="*50)
print(classification_report(y_test, y_pred_test_rf, target_names=['No Disease', 'Has Disease']))

print("\n" + "="*50)
print("CLASSIFICATION REPORT - LOGISTIC REGRESSION - TEST (BLIND)")
print("="*50)
print(classification_report(y_test, y_pred_test_lr, target_names=['No Disease', 'Has Disease']))

# VISUALIZATIONS
print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

# 1. Confusion Matrices for RF and Logistic Regression (Test Set)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Confusion Matrices - Model Comparison', fontsize=16, fontweight='bold')

cm_rf_test = confusion_matrix(y_test, y_pred_test_rf)
cm_lr_test = confusion_matrix(y_test, y_pred_test_lr)

sns.heatmap(cm_rf_test, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Random Forest - Test Set')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(cm_lr_test, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Logistic Regression - Test Set')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('outputs/RFLRconfusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: outputs/RFLRconfusion_matrices.png")

# 2. ROC Curves for RF and Logistic Regression (Test Set)
fig, ax = plt.subplots(figsize=(10, 8))

fpr_rf_test, tpr_rf_test, _ = roc_curve(y_test, y_pred_proba_test_rf)
roc_auc_rf_test = roc_auc_score(y_test, y_pred_proba_test_rf)

fpr_lr_test, tpr_lr_test, _ = roc_curve(y_test, y_pred_proba_test_lr)
roc_auc_lr_test = roc_auc_score(y_test, y_pred_proba_test_lr)

ax.plot(fpr_rf_test, tpr_rf_test, label=f'Random Forest (AUC = {roc_auc_rf_test:.4f})', linewidth=2, color='blue')
ax.plot(fpr_lr_test, tpr_lr_test, label=f'Logistic Regression (AUC = {roc_auc_lr_test:.4f})', linewidth=2, color='green')
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Random Forest vs Logistic Regression', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/RFLRroc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: outputs/RFLRroc_curves.png")

# 3. Metrics Comparison Bar Plot for RF and Logistic Regression
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Performance Metrics Comparison - RF vs Logistic Regression', fontsize=16, fontweight='bold')

metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
datasets = ['Random Forest', 'Logistic Regression']
colors_bars = ['#3498db', '#2ecc71']

for idx, metric_name in enumerate(metrics_names):
    ax = axes[idx // 3, idx % 3]
    
    metric_key = metric_name.replace(' ', '_').replace('-', '_')
    if metric_key == 'F1_Score':
        metric_key = 'F1-Score'
    
    rf_value = metrics_test[metric_key]
    lr_value = metrics_test_lr[metric_key]
    
    bars = ax.bar(datasets, [rf_value, lr_value], color=colors_bars, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', alpha=0.3)
    
    for bar, value in zip(bars, [rf_value, lr_value]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/RFLRmetrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: outputs/RFLRmetrics_comparison.png")

# 4. Feature Importance / Coefficients for RF and Logistic Regression
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Feature Importance / Coefficients', fontsize=16, fontweight='bold')

feature_names = X.columns

# Random Forest feature importance
rf_importances = best_rf_model.feature_importances_
rf_indices = np.argsort(rf_importances)[-15:]
axes[0].barh(range(len(rf_indices)), rf_importances[rf_indices], color='steelblue', alpha=0.8, edgecolor='black')
axes[0].set_yticks(range(len(rf_indices)))
axes[0].set_yticklabels([feature_names[i] for i in rf_indices], fontsize=10)
axes[0].set_xlabel('Importance Score', fontsize=11)
axes[0].set_title('Random Forest - Top 15 Features', fontsize=13, fontweight='bold')
axes[0].grid(True, axis='x', alpha=0.3)

# Logistic Regression coefficients
logreg_coefficients = np.abs(best_logreg_model.coef_[0])
logreg_indices = np.argsort(logreg_coefficients)[-15:]
axes[1].barh(range(len(logreg_indices)), logreg_coefficients[logreg_indices], color='salmon', alpha=0.8, edgecolor='black')
axes[1].set_yticks(range(len(logreg_indices)))
axes[1].set_yticklabels([feature_names[i] for i in logreg_indices], fontsize=10)
axes[1].set_xlabel('Absolute Coefficient', fontsize=11)
axes[1].set_title('Logistic Regression - Top 15 Coefficients', fontsize=13, fontweight='bold')
axes[1].grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/RFLRfeature_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: outputs/RFLRfeature_comparison.png")

# SAVE RESULTS AND MODEL
print("\n" + "="*50)
print("SAVING RESULTS & MODEL")
print("="*50)

# Save comprehensive metrics report
metrics_report = {
    'RandomForest': {
        'CV (Train)': {
            'Accuracy': metrics_cv['Accuracy'],
            'Precision': metrics_cv['Precision'],
            'Recall': metrics_cv['Recall'],
            'Specificity': metrics_cv['Specificity'],
            'F1-Score': metrics_cv['F1-Score'],
            'Kappa': metrics_cv['Kappa'],
            'G-Mean': metrics_cv['G-Mean'],
            'ROC-AUC': metrics_cv['ROC-AUC']
        },
        'Test (Blind)': {
            'Accuracy': metrics_test['Accuracy'],
            'Precision': metrics_test['Precision'],
            'Recall': metrics_test['Recall'],
            'Specificity': metrics_test['Specificity'],
            'F1-Score': metrics_test['F1-Score'],
            'Kappa': metrics_test['Kappa'],
            'G-Mean': metrics_test['G-Mean'],
            'ROC-AUC': metrics_test['ROC-AUC']
        }
    },
    'LogisticRegression': {
        'CV (Train)': {
            'Accuracy': metrics_cv_lr['Accuracy'],
            'Precision': metrics_cv_lr['Precision'],
            'Recall': metrics_cv_lr['Recall'],
            'Specificity': metrics_cv_lr['Specificity'],
            'F1-Score': metrics_cv_lr['F1-Score'],
            'Kappa': metrics_cv_lr['Kappa'],
            'G-Mean': metrics_cv_lr['G-Mean'],
            'ROC-AUC': metrics_cv_lr['ROC-AUC']
        },
        'Test (Blind)': {
            'Accuracy': metrics_test_lr['Accuracy'],
            'Precision': metrics_test_lr['Precision'],
            'Recall': metrics_test_lr['Recall'],
            'Specificity': metrics_test_lr['Specificity'],
            'F1-Score': metrics_test_lr['F1-Score'],
            'Kappa': metrics_test_lr['Kappa'],
            'G-Mean': metrics_test_lr['G-Mean'],
            'ROC-AUC': metrics_test_lr['ROC-AUC']
        }
    }
}

metrics_df = pd.DataFrame(metrics_report).T
metrics_df.to_csv('outputs/RFLRmodels_results.csv')
print("[OK] Saved: outputs/RFLRmodels_results.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'RandomForest_Predicted': y_pred_test_rf,
    'RandomForest_Probability_Class_0': 1 - y_pred_proba_test_rf,
    'RandomForest_Probability_Class_1': y_pred_proba_test_rf,
    'LogisticRegression_Predicted': y_pred_test_lr,
    'LogisticRegression_Probability_Class_0': 1 - y_pred_proba_test_lr,
    'LogisticRegression_Probability_Class_1': y_pred_proba_test_lr
})
predictions_df.to_csv('outputs/best_RFLRmodels_predictions.csv', index=False)
print("[OK] Saved: outputs/best_RFLRmodels_predictions.csv")

# Save best RandomForest model
joblib.dump(best_rf_model, 'outputs/best_RFmodel.pkl')
print("[OK] Saved: outputs/best_RFmodel.pkl")

# Save best Logistic Regression model
joblib.dump(best_logreg_model, 'outputs/best_LRmodel.pkl')
print("[OK] Saved: outputs/best_LRmodel.pkl")

# Save best parameters and GridSearch info
gridsearch_info = {
    'random_forest': {
        'best_params': best_params,
        'best_cv_score': rf_grid_search.best_score_,
        'cv_results': rf_grid_search.cv_results_
    },
    'logistic_regression': {
        'best_params': best_logreg_params,
        'best_cv_score': logreg_grid_search.best_score_,
        'cv_results': logreg_grid_search.cv_results_
    }
}
joblib.dump(gridsearch_info, 'outputs/RFLRgridsearch_info.pkl')
print("[OK] Saved: outputs/RFLRgridsearch_info.pkl")

# Save y_test and y_probs for ROC Curve
evaluation_data = {
    'y_test': y_test,
    'random_forest_probs': y_pred_proba_test_rf,
    'logistic_regression_probs': y_pred_proba_test_lr
}
joblib.dump(evaluation_data, 'outputs/RFLRmodel_eval_data.pkl')

print("\n" + "="*50)
print("MODELING COMPLETE!")
print("="*50)
print(f"\nBest Random Forest Model: RandomForestClassifier")
print(f"Best Random Forest Parameters: {best_params}")
print(f"Best Logistic Regression Parameters: {best_logreg_params}")
print(f"\nTest Set Performance:")
print(f"  Random Forest Accuracy: {metrics_test['Accuracy']:.4f}")
print(f"  Random Forest ROC-AUC: {metrics_test['ROC-AUC']:.4f}")
print(f"  Random Forest Recall: {metrics_test['Recall']:.4f}")
print(f"  Random Forest Specificity: {metrics_test['Specificity']:.4f}")
print(f"  Random Forest F1-Score: {metrics_test['F1-Score']:.4f}")
print(f"  Random Forest G-Mean: {metrics_test['G-Mean']:.4f}")
print(f"\n  Logistic Regression Accuracy: {metrics_test_lr['Accuracy']:.4f}")
print(f"  Logistic Regression ROC-AUC: {metrics_test_lr['ROC-AUC']:.4f}")
print(f"  Logistic Regression Recall: {metrics_test_lr['Recall']:.4f}")
print(f"  Logistic Regression Specificity: {metrics_test_lr['Specificity']:.4f}")
print(f"  Logistic Regression F1-Score: {metrics_test_lr['F1-Score']:.4f}")
print(f"  Logistic Regression G-Mean: {metrics_test_lr['G-Mean']:.4f}")
