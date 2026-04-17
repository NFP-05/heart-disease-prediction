# Load Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Calling Dataset
df: pd.DataFrame = pd.read_csv("Data/heart.csv")
print(df.head())

# Data Structure
print("\n" + "="*50)
print("Data Structure and Feat. Type")
print("="*50)
print(f"\nDimension dataset: {df.shape}")
print("\nData Structure:")
print(df.info())
print("\nFeature Type:")
print(df.dtypes)
print("\nStatistics Desc.:")
print(df.describe())
print("\nUnique values per feat.:")
for col in df.columns:
    print(f"  {col}: {df[col].nunique()} unique values")

# Checking Dataset (Missing Value, Zero Value, & Duplicate Value)
print(f"Duplicate rows: {df.duplicated().sum()}")

print("Missing values per column:\n", df.isna().sum())

num_cols = df.select_dtypes(include="number").columns
table_zeros = (df[num_cols] == 0).sum()
print("\nZero counts in numeric columns:\n", table_zeros)

# Correlation plot for feature engineering before preprocessing
os.makedirs('outputs', exist_ok=True)
print(f"\nCreating correlation plot for original numeric features...")
correlation_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'HeartDisease']
corr_matrix = df[correlation_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Correlation Matrix of Original Features')
fig.tight_layout()
fig.savefig('outputs/original_correlation.png')
plt.close(fig)
print("Saved: outputs/original_correlation.png")

# Feature Engineering: HR Ratio after correlation
print(f"\n" + "="*50)
print("FEATURE ENGINEERING: HR_Ratio")
print("="*50)
df['HR_Ratio'] = df['MaxHR'] / (220 - df['Age'])
print(f"Created feature HR_Ratio. Sample values:")
print(df[['Age', 'MaxHR', 'HR_Ratio']].head())

# Remove original MaxHR feature since it's now represented in HR_Ratio
df = df.drop(columns=['MaxHR'])
print(f"\nRemoved original MaxHR feature. New shape: {df.shape}")
print(f"Remaining numeric columns: {df.select_dtypes(include='number').columns.tolist()}")

# Correlation plot for feature engineering after HR_Ratio creation
print(f"\nCreating correlation plot including HR_Ratio...")
correlation_cols = ['Age', 'RestingBP', 'Cholesterol', 'Oldpeak', 'HR_Ratio', 'HeartDisease']
corr_matrix = df[correlation_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Correlation Matrix Including HR_Ratio')
fig.tight_layout()
fig.savefig('outputs/hr_ratio_correlation.png')
plt.close(fig)
print("Saved: outputs/hr_ratio_correlation.png")

# Imputation
print(f"\n" + "="*50)
print("IMPUTASI NILAI NOL")
print("="*50)
print(f"\nOriginal dataset shape: {df.shape}")

cholesterol_median = df[df['Cholesterol'] != 0]['Cholesterol'].median()
resting_bp_median = df[df['RestingBP'] != 0]['RestingBP'].median()

print(f"\nNilai nol sebelum imputasi:")
print(f"  Cholesterol: {(df['Cholesterol'] == 0).sum()} nol values")
print(f"  RestingBP: {(df['RestingBP'] == 0).sum()} nol values")
print(f"  Oldpeak: {(df['Oldpeak'] == 0).sum()} nol values")

df_cleaned = df.copy()
df_cleaned.loc[df_cleaned['Cholesterol'] == 0, 'Cholesterol'] = cholesterol_median
df_cleaned.loc[df_cleaned['RestingBP'] == 0, 'RestingBP'] = resting_bp_median

print(f"\nMedian values:")
print(f"  Cholesterol median: {cholesterol_median}")
print(f"  RestingBP median: {resting_bp_median}")
print(f"\nCleaned dataset shape: {df_cleaned.shape}")
print(f"\nNilai nol setelah imputasi:")
print(f"  Cholesterol: {(df_cleaned['Cholesterol'] == 0).sum()} nol values")
print(f"  RestingBP: {(df_cleaned['RestingBP'] == 0).sum()} nol values")
print(f"  Oldpeak: {(df_cleaned['Oldpeak'] == 0).sum()} nol values")

# Save cleaned data
df_cleaned.to_csv('Data/heart_cleaned.csv', index=False)

# ENCODING CATEGORICAL FEATURES
print(f"\n" + "="*50)
print("ENCODING CATEGORICAL FEATURES")
print("="*50)

ml_df = df_cleaned.copy()

# Identificate Categorical Feat.
categorical_cols = ml_df.select_dtypes(include=['object']).columns.tolist()
print(f"\nFeat. categorical: {categorical_cols}")

# Encoding (Label Encoding & One-Hot Encoding)
binary_cols = []
multiclass_cols = []

for col in categorical_cols:
    n_unique = ml_df[col].nunique()
    print(f"  {col}: {n_unique} Category")
    if n_unique == 2:
        binary_cols.append(col)
    elif n_unique > 2:
        multiclass_cols.append(col)

print(f"\nFeat. binary (label encoding): {binary_cols}")
print(f"Feat. multiclass (one-hot encoding): {multiclass_cols}")

# Dictionary untuk menyimpan encoder objects
encoders_dict = {}

# Label Encoding
if binary_cols:
    print(f"\nDoing label encoding at: {binary_cols}")
    encoders_dict['label_encoders'] = {}
    for col in binary_cols:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col])
        encoders_dict['label_encoders'][col] = le
        print(f"  {col}: {le.classes_} -> {list(range(len(le.classes_)))}")

# One-Hot Encoding
if multiclass_cols:
    print(f"\nDoing one-hot encoding at: {multiclass_cols}")
    ml_df_before_ohe = ml_df.copy()
    ml_df = pd.get_dummies(ml_df, columns=multiclass_cols, drop_first=False)
    encoders_dict['onehot_encoders'] = {
        'columns': multiclass_cols,
        'encoded_columns': [col for col in ml_df.columns if col not in ml_df_before_ohe.columns]
    }
    print(f"\nFeat. after one-hot encoding:")
    print(ml_df.columns.tolist())
else:
    print(f"\nTheres no feat. with >2 category for one-hot encoding")
    encoders_dict['onehot_encoders'] = None

print(f"\nFinal dataset shape: {ml_df.shape}")
print(f"\nFinal columns:")
print(ml_df.columns.tolist())
print(f"\nData types:")
print(ml_df.dtypes)

# Scale numeric features for modeling
print(f"\n" + "="*50)
print("SCALING NUMERIC FEATURES")
print("="*50)
scale_cols = ['Age', 'RestingBP', 'Cholesterol', 'Oldpeak', 'HR_Ratio']
scaler = StandardScaler()
ml_df[scale_cols] = scaler.fit_transform(ml_df[scale_cols])
print(f"Scaled columns: {scale_cols}")
print(ml_df[scale_cols].describe())

# Save processed data
ml_df.to_csv('Data/heart_processed.csv', index=False)
print(f"\nSaved: Data/heart_processed.csv")

# Save scaler and training columns
joblib.dump(scaler, 'outputs/scaler.pkl')
print(f"\nScaler saved to: outputs/scaler.pkl")
train_columns = ml_df.drop(columns=['HeartDisease']).columns.tolist()
joblib.dump(train_columns, 'outputs/train_columns.pkl')
print(f"Train columns saved to: outputs/train_columns.pkl")

# Save encoder objects
joblib.dump(encoders_dict, 'outputs/encoders.pkl')
print(f"\nEncoder objects saved to: outputs/encoders.pkl")
print(f"\nEncoder dictionary structure:")
print(f"  - label_encoders: {list(encoders_dict.get('label_encoders', {}).keys())}")
if encoders_dict.get('onehot_encoders'):
    print(f"  - onehot_encoders:")
    print(f"      - columns: {encoders_dict['onehot_encoders']['columns']}")
    print(f"      - encoded_columns: {len(encoders_dict['onehot_encoders']['encoded_columns'])} columns")