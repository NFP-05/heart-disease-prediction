# Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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
print("Loading processed data...")
df = pd.read_csv('Data/heart_processed.csv')
print(f"Data shape: {df.shape}")

# Separate features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}\n")

# SPLIT 1: Create blind test set (20%)
print("="*50)
print("SPLIT 1: Creating Blind Test Set (20%)")
print("="*50)
X_cv, X_test, y_cv, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"CV Data (80%): {X_cv.shape}")
print(f"Blind Test Data (20%): {X_test.shape}")
print(f"  - Class 0: {(y_test == 0).sum()}")
print(f"  - Class 1: {(y_test == 1).sum()}\n")

# SPLIT 2: GridSearchCV with Cross-Validation (5-fold)
print("="*50)
print("SPLIT 2: Setup GridSearchCV with 5-Fold CV")
print("="*50)

# Define parameter grid for RandomForest
param_grid = {
    'n_estimators': [90, 100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [25, 30],
    'class_weight': [None]
}

print("Parameter Grid:")
for key, value in param_grid.items():
    print(f"  {key}: {value}")

# Initialize RandomForestClassifier
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# Setup GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    rf_base,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("\nTraining with GridSearchCV (5-Fold CV)...")
grid_search.fit(X_cv, y_cv)

# Get best model
best_rf_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"\nBest Parameters Found:")
for key, value in best_params.items():
    print(f"  {key}: {value}")
print(f"Best CV ROC-AUC Score: {grid_search.best_score_:.4f}\n")

# EVALUATE: Make predictions on CV and test data
print("="*50)
print("EVALUATE: Making Predictions & Computing Metrics")
print("="*50)

# Predictions on CV data (training performance)
y_pred_cv = best_rf_model.predict(X_cv)
y_pred_proba_cv = best_rf_model.predict_proba(X_cv)[:, 1]

# Predictions on blind test data
y_pred_test = best_rf_model.predict(X_test)
y_pred_proba_test = best_rf_model.predict_proba(X_test)[:, 1]

# Function to calculate metrics
def calculate_metrics(y_true, y_pred, y_pred_proba, dataset_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Calculate G-Mean (geometric mean of sensitivities)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/Sensitivity
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    g_mean = gmean([tpr, tnr])
    
    cm = confusion_matrix(y_true, y_pred)
    
    metrics_dict = {
        'Dataset': dataset_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Kappa': kappa,
        'G-Mean': g_mean,
        'Confusion_Matrix': cm
    }
    
    return metrics_dict

# Calculate metrics for both datasets
metrics_cv = calculate_metrics(y_cv, y_pred_cv, y_pred_proba_cv, 'CV (Train)')
metrics_test = calculate_metrics(y_test, y_pred_test, y_pred_proba_test, 'Test (Blind)')

# Display metrics
print("\n" + "="*50)
print("METRICS SUMMARY")
print("="*50)
metrics_summary = pd.DataFrame([
    {
        'Dataset': 'CV (Train)',
        'Accuracy': f"{metrics_cv['Accuracy']:.4f}",
        'Precision': f"{metrics_cv['Precision']:.4f}",
        'Recall': f"{metrics_cv['Recall']:.4f}",
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
        'F1-Score': f"{metrics_test['F1-Score']:.4f}",
        'Kappa': f"{metrics_test['Kappa']:.4f}",
        'G-Mean': f"{metrics_test['G-Mean']:.4f}",
        'ROC-AUC': f"{metrics_test['ROC-AUC']:.4f}"
    }
])
print("\n", metrics_summary.to_string(index=False))

# Classification Reports
print("\n" + "="*50)
print("CLASSIFICATION REPORT - CV (TRAIN)")
print("="*50)
print(classification_report(y_cv, y_pred_cv, target_names=['No Disease', 'Has Disease']))

print("\n" + "="*50)
print("CLASSIFICATION REPORT - TEST (BLIND)")
print("="*50)
print(classification_report(y_test, y_pred_test, target_names=['No Disease', 'Has Disease']))

# VISUALIZATIONS
print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

# 1. Confusion Matrices (CV and Test)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Confusion Matrices - RandomForest Classifier', fontsize=16, fontweight='bold')

cm_cv = confusion_matrix(y_cv, y_pred_cv)
cm_test = confusion_matrix(y_test, y_pred_test)

sns.heatmap(cm_cv, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('CV Data (Train)')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Test Data (Blind)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('outputs/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/confusion_matrices.png")

# 2. ROC Curves (CV and Test)
fig, ax = plt.subplots(figsize=(10, 8))

fpr_cv, tpr_cv, _ = roc_curve(y_cv, y_pred_proba_cv)
roc_auc_cv = roc_auc_score(y_cv, y_pred_proba_cv)

fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)
roc_auc_test = roc_auc_score(y_test, y_pred_proba_test)

ax.plot(fpr_cv, tpr_cv, label=f'CV Data (AUC = {roc_auc_cv:.4f})', linewidth=2, color='blue')
ax.plot(fpr_test, tpr_test, label=f'Test Data (AUC = {roc_auc_test:.4f})', linewidth=2, color='green')
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - RandomForest Classifier', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/roc_curves.png")

# 3. Metrics Comparison Bar Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Performance Metrics Comparison - CV vs Test', fontsize=16, fontweight='bold')

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
datasets = ['CV (Train)', 'Test (Blind)']
colors_bars = ['#3498db', '#2ecc71']

for idx, metric_name in enumerate(metrics_names):
    ax = axes[idx // 2, idx % 2]
    
    metric_key = metric_name.replace(' ', '_').replace('-', '_')
    if metric_key == 'F1_Score':
        metric_key = 'F1-Score'
    
    cv_value = metrics_cv[metric_key]
    test_value = metrics_test[metric_key]
    
    bars = ax.bar(datasets, [cv_value, test_value], color=colors_bars, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, [cv_value, test_value]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/metrics_comparison.png")

# 4. Feature Importance (Top 15)
fig, ax = plt.subplots(figsize=(11, 6))

feature_names = X.columns
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[-15:]  # Top 15 features

ax.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
ax.set_xlabel('Importance Score', fontsize=11)
ax.set_title('RandomForest - Top 15 Most Important Features', fontsize=13, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/feature_importance.png")

# SAVE RESULTS AND MODEL
print("\n" + "="*50)
print("SAVING RESULTS & MODEL")
print("="*50)

# Save comprehensive metrics report
metrics_report = {
    'CV (Train)': {
        'Accuracy': metrics_cv['Accuracy'],
        'Precision': metrics_cv['Precision'],
        'Recall': metrics_cv['Recall'],
        'F1-Score': metrics_cv['F1-Score'],
        'Kappa': metrics_cv['Kappa'],
        'G-Mean': metrics_cv['G-Mean'],
        'ROC-AUC': metrics_cv['ROC-AUC']
    },
    'Test (Blind)': {
        'Accuracy': metrics_test['Accuracy'],
        'Precision': metrics_test['Precision'],
        'Recall': metrics_test['Recall'],
        'F1-Score': metrics_test['F1-Score'],
        'Kappa': metrics_test['Kappa'],
        'G-Mean': metrics_test['G-Mean'],
        'ROC-AUC': metrics_test['ROC-AUC']
    }
}

metrics_df = pd.DataFrame(metrics_report).T
metrics_df.to_csv('outputs/model_results.csv')
print("✓ Saved: outputs/model_results.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_test,
    'Probability_Class_0': 1 - y_pred_proba_test,
    'Probability_Class_1': y_pred_proba_test
})
predictions_df.to_csv('outputs/best_model_predictions.csv', index=False)
print("✓ Saved: outputs/best_model_predictions.csv")

# Save best RandomForest model
joblib.dump(best_rf_model, 'outputs/best_model.pkl')
print("✓ Saved: outputs/best_model.pkl")

# Save best parameters and GridSearch info
gridsearch_info = {
    'best_params': best_params,
    'best_cv_score': grid_search.best_score_,
    'cv_results': grid_search.cv_results_
}
joblib.dump(gridsearch_info, 'outputs/gridsearch_info.pkl')
print("✓ Saved: outputs/gridsearch_info.pkl")

print("\n" + "="*50)
print("MODELING COMPLETE!")
print("="*50)
print(f"\nBest Model: RandomForestClassifier")
print(f"Best Parameters: {best_params}")
print(f"\nTest Set Performance:")
print(f"  Accuracy: {metrics_test['Accuracy']:.4f}")
print(f"  ROC-AUC: {metrics_test['ROC-AUC']:.4f}")
print(f"  F1-Score: {metrics_test['F1-Score']:.4f}")
print(f"  G-Mean: {metrics_test['G-Mean']:.4f}")
