# Load Libraries
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Calling Dataset
df: pd.DataFrame = pd.read_csv("Data/heart.csv")
print(df.head())

# Checking Dataset (Missing Value, Zero Value, & Duplicate Value)
print(f"Duplicate rows: {df.duplicated().sum()}")

print("Missing values per column:\n", df.isna().sum())

num_cols = df.select_dtypes(include="number").columns
table_zeros = (df[num_cols] == 0).sum()
print("\nZero counts in numeric columns:\n", table_zeros)

# Remove rows with zero values in key medical columns
zero_columns = ['Cholesterol', 'RestingBP', 'Oldpeak']

print(f"\nOriginal dataset shape: {df.shape}")
print(f"Removing rows where {zero_columns} contain zeros...")

df_cleaned = df[(df[zero_columns] != 0).all(axis=1)].copy()

print(f"Cleaned dataset shape: {df_cleaned.shape}")
print(f"Rows removed: {df.shape[0] - df_cleaned.shape[0]}")

# Save cleaned data for hypothesis testing in R
df_cleaned.to_csv('Data/heart_cleaned.csv', index=False)

# FEATURE SELECTION, ENCODING, SCALING, SPLIT FOR ML
# Feature Selection: drop non-significant based on hypothesis results
selected_features = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'MaxHR',
    'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'HeartDisease'
]
ml_df = df_cleaned[selected_features].copy()
print(f"Selected features: {selected_features}")

# Encoding categorical features
ml_df = pd.get_dummies(
    ml_df,
    columns=['Sex', 'ChestPainType', 'ExerciseAngina', 'ST_Slope'],
    drop_first=True
)
print("One-hot encoding done. Columns now:")
print(ml_df.columns.tolist())

# Scaling numeric features
scaler = StandardScaler()
scale_cols = ['Age', 'RestingBP', 'MaxHR', 'Oldpeak']
ml_df[scale_cols] = scaler.fit_transform(ml_df[scale_cols])

print("Numeric scaling done (StandardScaler).")

# Train/test split
X = ml_df.drop('HeartDisease', axis=1)
y = ml_df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train/test split done: X_train={X_train.shape}, X_test={X_test.shape}")

# Save split files for modeling
X_train.to_csv('Data/X_train.csv', index=False)
X_test.to_csv('Data/X_test.csv', index=False)
y_train.to_csv('Data/y_train.csv', index=False)
y_test.to_csv('Data/y_test.csv', index=False)
print("Saved: Data/X_train.csv, Data/X_test.csv, Data/y_train.csv, Data/y_test.csv")

joblib.dump(X_train.columns.tolist(), "outputs/train_columns.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")