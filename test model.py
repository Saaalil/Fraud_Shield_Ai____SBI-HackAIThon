import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter

# Define fraud category mappings
FRAUD_CATEGORIES = {
    0: "No Fraud",
    1: "Identity Theft",
    2: "Document Forgery",
    3: "Medical Fraud",
    4: "Premium Fraud",
    5: "Agent Fraud",
    6: "Claim Fraud",
    7: "Beneficiary Fraud",
    8: "Legitimate Claim",
    9: "Policy Fraud",
    10: "Multiple Policy Fraud",
    11: "Other Fraud"
}

# Load data
file_path = r'C:\Users\SALIL HIREMATH\Downloads\preprocessed_fraud_data_final (1).csv'
data = pd.read_csv(file_path)

# Drop irrelevant columns
columns_to_drop = [
    "Dummy Policy No", "POLICYRISKCOMMENCEMENTDATE", "Date of Death", "INTIMATIONDATE", 
    "CORRESPONDENCECITY", "CORRESPONDENCESTATE", "CORRESPONDENCEPOSTCODE"
]
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

# Define features
features = [
    "ASSURED_AGE", "NOMINEE_RELATION", "OCCUPATION", "POLICY SUMASSURED", "Premium",
    "PREMIUMPAYMENTMODE", "Annual Income", "Policy Term", "Policy Payment Term", "CHANNEL",
    "Bank code", "Premium/Income Ratio", "Policy Value/Income Ratio", "Policy to Death Timeline",
    "Death to Intimation Gap", "Location Risk", "Occupation Risk Score", "Product Type Risk"
]

X = data[features].copy()
y = data["Fraud Category"].copy()

# Identify categorical and numerical columns
categorical_columns = ["NOMINEE_RELATION", "OCCUPATION", "PREMIUMPAYMENTMODE", "CHANNEL", "Bank code"]
numerical_columns = [col for col in features if col not in categorical_columns]

# Create separate imputers for numerical and categorical data
numerical_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Impute numerical columns
X[numerical_columns] = numerical_imputer.fit_transform(X[numerical_columns])

# Impute categorical columns
X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])

# Create and store label encoders
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Handle rare classes
class_counts = Counter(y)
print("\nClass Distribution Before Processing:", class_counts)

min_samples_per_class = 6
rare_classes = [cls for cls, count in class_counts.items() if count < min_samples_per_class]
if rare_classes:
    print(f"Rare classes detected: {rare_classes}. These classes will be removed.")
    mask = ~y.isin(rare_classes)
    X = X[mask]
    y = y[mask]

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nClass Distribution After SMOTE:", Counter(y_resampled))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Print metrics
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def predict_fraud_category(custom_input):
    """Predict fraud category with description."""
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([custom_input], columns=features)

        # Handle numerical features
        input_df[numerical_columns] = numerical_imputer.transform(input_df[numerical_columns])

        # Handle categorical features
        input_df[categorical_columns] = categorical_imputer.transform(input_df[categorical_columns])

        # Transform categorical variables
        for col in categorical_columns:
            le = label_encoders[col]
            unseen_labels = set(input_df[col]) - set(le.classes_)
            if unseen_labels:
                print(f"Unseen labels in column '{col}': {unseen_labels}. Replacing with 'unknown'.")
                input_df[col] = input_df[col].replace(unseen_labels, "unknown")
                le.classes_ = np.append(le.classes_, "unknown")
            input_df[col] = le.transform(input_df[col].astype(str))

        # Verify no NaN values
        if input_df.isna().sum().sum() > 0:
            raise ValueError("Input contains NaN values after preprocessing.")

        # Make prediction
        prediction = rf.predict(input_df)[0]
        category_description = FRAUD_CATEGORIES.get(prediction, "Unknown Category")

        return prediction, category_description

    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

# Example usage
if __name__ == "__main__":
    custom_input = {
    "ASSURED_AGE": 35,  # Mid-range age
    "NOMINEE_RELATION": "2",  # Common relation
    "OCCUPATION": "Self-Employed",  # Moderate risk occupation
    "POLICY SUMASSURED": 1_000_000,  # Moderate sum assured
    "Premium": 100_000,  # Moderate premium
    "PREMIUMPAYMENTMODE": "Annual",  # Standard payment mode
    "Annual Income": 1_200_000,  # Moderate income
    "Policy Term": 15,  # Common policy term
    "Policy Payment Term": 15,  # Matches policy term
    "CHANNEL": "Agent",  # Standard channel
    "Bank code": "3",  # Neutral bank code
    "Premium/Income Ratio": 0.0833,  # Reasonable ratio
    "Policy Value/Income Ratio": 0.8333,  # Reasonable ratio
    "Policy to Death Timeline": 20,  # Moderate timeline
    "Death to Intimation Gap": 10,  # Short reporting gap
    "Location Risk": 500,  # Neutral location risk
    "Occupation Risk Score": 5.0,  # Neutral risk score
    "Product Type Risk": 10.0  # Neutral product type risk
}



    try:
        category_id, category_description = predict_fraud_category(custom_input)
        print(f"\nPredicted Fraud Category {category_id}: {category_description}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
