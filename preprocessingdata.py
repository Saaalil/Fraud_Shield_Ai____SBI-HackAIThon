import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

# Load original data
original_data_path = 'C:\\Users\\SALIL HIREMATH\\Downloads\\Fraud data FY 2023-24 for B&CC.xlsx'
original_data = pd.ExcelFile(original_data_path).parse('Fraud data')

# ==========================
# Step 1: Data Validation & Cleaning
# ==========================
# Handle missing values
original_data['Bank code'] = original_data['Bank code'].fillna('Unknown')
numerical_columns = ['ASSURED_AGE', 'Premium', 'Annual Income', 'POLICY SUMASSURED']

# Fill missing values in numerical columns
numerical_imputer = SimpleImputer(strategy='mean')
original_data[numerical_columns] = numerical_imputer.fit_transform(original_data[numerical_columns])

# ==========================
# Step 2: Feature Engineering
# ==========================
# Financial Features
original_data['Premium/Income Ratio'] = original_data['Premium'] / original_data['Annual Income']
original_data['Policy Value/Income Ratio'] = original_data['POLICY SUMASSURED'] / original_data['Annual Income']

# Temporal Features
original_data['POLICYRISKCOMMENCEMENTDATE'] = pd.to_datetime(original_data['POLICYRISKCOMMENCEMENTDATE'], errors='coerce')
original_data['Date of Death'] = pd.to_datetime(original_data['Date of Death'], errors='coerce')
original_data['INTIMATIONDATE'] = pd.to_datetime(original_data['INTIMATIONDATE'], errors='coerce')

original_data['Policy to Death Timeline'] = (
    original_data['Date of Death'] - original_data['POLICYRISKCOMMENCEMENTDATE']
).dt.days
original_data['Death to Intimation Gap'] = (
    original_data['INTIMATIONDATE'] - original_data['Date of Death']
).dt.days

# Categorical Features
categorical_columns = [
    'NOMINEE_RELATION', 'OCCUPATION', 'PREMIUMPAYMENTMODE', 'HOLDERMARITALSTATUS',
    'INDIV_REQUIREMENTFLAG', 'CORRESPONDENCECITY', 'CORRESPONDENCESTATE',
    'Product Type', 'CHANNEL', 'STATUS', 'SUB_STATUS', 'Fraud Category'
]

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    original_data[col] = le.fit_transform(original_data[col].astype(str))
    label_encoders[col] = le

# Location Data
# Combine city and state into a single feature
original_data['Location Risk'] = original_data['CORRESPONDENCECITY'].astype(str) + "_" + original_data['CORRESPONDENCESTATE'].astype(str)
original_data['Location-Based Risk'] = LabelEncoder().fit_transform(original_data['Location Risk'])

# ==========================
# Step 3: Risk Indicators
# ==========================
# Create aggregated risk indicators
original_data['Occupation Risk Score'] = original_data['OCCUPATION'] * original_data['Premium/Income Ratio']
original_data['Product Type Risk'] = original_data['Product Type'] * original_data['Policy Value/Income Ratio']

# ==========================
# Save Engineered Dataset
# ==========================
preprocessed_data_path = 'C:\\Users\\SALIL HIREMATH\\Downloads\\preprocessed_fraud_data.csv'
original_data.to_csv(preprocessed_data_path, index=False)

print(f"Feature engineering complete. Preprocessed data saved to {preprocessed_data_path}")
