# Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    classification_report
)
import joblib

# Load Training & Test Data
print("Loading training and test data...")
X_train = pd.read_csv('Data/X_train.csv')
X_test = pd.read_csv('Data/X_test.csv')
y_train = pd.read_csv('Data/y_train.csv').values.ravel()
y_test = pd.read_csv('Data/y_test.csv').values.ravel()

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Dictionary to store models and results
models = {}
results = {}

# 1. LOGISTIC REGRESSION
print("\n" + "="*50)
print("Training Logistic Regression")
print("="*50)
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

models['Logistic Regression'] = lr_model
results['Logistic Regression'] = {
    'y_pred': y_pred_lr,
    'y_pred_proba': y_pred_proba_lr,
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr),
    'recall': recall_score(y_test, y_pred_lr),
    'f1': f1_score(y_test, y_pred_lr),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_lr)
}

# 2. K-NEAREST NEIGHBORS
print("\nTraining K-Nearest Neighbors")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
y_pred_proba_knn = knn_model.predict_proba(X_test)[:, 1]

models['KNN'] = knn_model
results['KNN'] = {
    'y_pred': y_pred_knn,
    'y_pred_proba': y_pred_proba_knn,
    'accuracy': accuracy_score(y_test, y_pred_knn),
    'precision': precision_score(y_test, y_pred_knn),
    'recall': recall_score(y_test, y_pred_knn),
    'f1': f1_score(y_test, y_pred_knn),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_knn)
}

# 3. DECISION TREE
print("\nTraining Decision Tree")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]

models['Decision Tree'] = dt_model
results['Decision Tree'] = {
    'y_pred': y_pred_dt,
    'y_pred_proba': y_pred_proba_dt,
    'accuracy': accuracy_score(y_test, y_pred_dt),
    'precision': precision_score(y_test, y_pred_dt),
    'recall': recall_score(y_test, y_pred_dt),
    'f1': f1_score(y_test, y_pred_dt),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_dt)
}

# MODEL COMPARISON
print("\n" + "="*50)
print("MODEL PERFORMANCE COMPARISON")
print("="*50)

results_df = pd.DataFrame(results).T
print("\n", results_df)

# Find best model
best_model_name = results_df['roc_auc'].idxmax()
print(f"\n✓ Best Model (by ROC-AUC): {best_model_name}")
print(f"  Accuracy: {results_df.loc[best_model_name, 'accuracy']:.4f}")
print(f"  Precision: {results_df.loc[best_model_name, 'precision']:.4f}")
print(f"  Recall: {results_df.loc[best_model_name, 'recall']:.4f}")
print(f"  F1-Score: {results_df.loc[best_model_name, 'f1']:.4f}")
print(f"  ROC-AUC: {results_df.loc[best_model_name, 'roc_auc']:.4f}")

# DETAILED CLASSIFICATION REPORT FOR BEST MODEL
print("\n" + "="*50)
print(f"DETAILED CLASSIFICATION REPORT - {best_model_name.upper()}")
print("="*50)
best_predictions = results[best_model_name]['y_pred']
print(classification_report(y_test, best_predictions, target_names=['No Disease', 'Has Disease']))

# CONFUSION MATRICES
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')

for idx, (model_name, pred) in enumerate([(name, results[name]['y_pred']) for name in results]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(model_name)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('outputs/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# ROC CURVES
fig, ax = plt.subplots(figsize=(10, 8))

for model_name in results:
    y_pred_proba = results[model_name]['y_pred_proba']
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = results[model_name]['roc_auc']
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# METRICS COMPARISON BAR PLOT
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

metrics = ['accuracy', 'precision', 'recall', 'f1']
colors = plt.cm.Set3(np.linspace(0, 1, len(results)))

for idx, metric in enumerate(metrics):
    values = [results[model][metric] for model in results]
    axes[idx].bar(results.keys(), values, color=colors)
    axes[idx].set_ylabel(metric.capitalize(), fontsize=11)
    axes[idx].set_title(f'{metric.capitalize()} Comparison', fontsize=12, fontweight='bold')
    axes[idx].set_ylim([0, 1])
    axes[idx].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# FEATURE IMPORTANCE (Decision Tree)
fig, ax = plt.subplots(figsize=(10, 6))

model = models['Decision Tree']
feature_names = X_train.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]  # Top 10 features

ax.barh(range(len(indices)), importances[indices], color='steelblue')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_names[i] for i in indices])
ax.set_xlabel('Importance', fontsize=11)
ax.set_title('Decision Tree - Top 10 Features', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# SAVE RESULTS
results_summary = results_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].copy()
results_summary.to_csv('outputs/model_results.csv')

# Save best model predictions
best_predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': results[best_model_name]['y_pred'],
    'Probability': results[best_model_name]['y_pred_proba']
})
best_predictions_df.to_csv('outputs/best_model_predictions.csv', index=False)

print("\n")
print("MODELING COMPLETE!\n")

best_model = models[best_model_name]
joblib.dump(best_model, f"outputs/best_model.pkl")
