import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

# Load original and preprocessed data
original_data_path = 'C:\\Users\\SALIL HIREMATH\\Downloads\\Fraud data FY 2023-24 for B&CC.xlsx'
preprocessed_data_path = 'C:\\Users\\SALIL HIREMATH\\Downloads\\preprocessed_fraud_data.csv'

# Load datasets
original_data = pd.ExcelFile(original_data_path).parse('Fraud data')
preprocessed_data = pd.read_csv(preprocessed_data_path)

# ==========================
# Compare Numerical Columns
# ==========================
columns_to_compare = ['ASSURED_AGE', 'Premium', 'Annual Income']

for col in columns_to_compare:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(original_data[col].dropna(), bins=20, alpha=0.7, color='blue', label='Original')
    plt.title(f"Original {col} Distribution")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(preprocessed_data[col].dropna(), bins=20, alpha=0.7, color='green', label='Preprocessed')
    plt.title(f"Preprocessed {col} Distribution")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

# ==========================
# Compare Categorical Columns
# ==========================
# Example: NOMINEE_RELATION
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
original_data['NOMINEE_RELATION'].value_counts().plot(kind='bar', color='blue', alpha=0.7)
plt.title('Original NOMINEE_RELATION Distribution')
plt.ylabel('Frequency')
plt.xlabel('Category')

plt.subplot(1, 2, 2)
preprocessed_data['NOMINEE_RELATION'].value_counts().plot(kind='bar', color='green', alpha=0.7)
plt.title('Preprocessed NOMINEE_RELATION Distribution')
plt.ylabel('Frequency')
plt.xlabel('Encoded Category')

plt.tight_layout()
plt.show()

print("Visualization complete. Compare the graphs to analyze preprocessing effects.")
