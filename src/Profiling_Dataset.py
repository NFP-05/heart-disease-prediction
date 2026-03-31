# Load libraries
import pandas as pd
from ydata_profiling import ProfileReport

# Panggil Dataset
df = pd.read_csv("Data/heart_cleaned.csv")

# Profiling Data
profiledf = ProfileReport(df, title = "Profiling Data Heart Disease")
profiledf.to_file("outputs/report.html")
