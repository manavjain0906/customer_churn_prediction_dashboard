import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# -----------------------------
# STEP 17: DATA PREPARATION
# -----------------------------

df = pd.read_csv("data/telo_churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove missing values
df = df.dropna()

# Remove customer ID
df = df.drop("customerID", axis=1)

# Convert Churn into numbers
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\n--- DATASET AFTER PREPROCESSING ---")
print(df.head())

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- CHURN VALUES ---")
print(df["Churn"].value_counts())


# -----------------------------
# STEP 18: TRAIN / TEST SPLIT
# -----------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- TRAINING DATA ---")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\n--- TESTING DATA ---")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# -----------------------------
# STEP 19: PREPROCESSING
# -----------------------------

numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

print("\n--- PREPROCESSING PIPELINE CREATED ---")


# -----------------------------
# STEP 20: LOGISTIC REGRESSION
# -----------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

# Train the model
model.fit(X_train, y_train)

print("\n--- MODEL TRAINING COMPLETED ---")
print("Logistic Regression model trained successfully!")

# -----------------------------
# STEP 21: MODEL EVALUATION
# -----------------------------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- MODEL EVALUATION ---")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

# -----------------------------
# STEP 23: RANDOM FOREST MODEL
# -----------------------------

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest pipeline
rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

# Train Random Forest
rf_model.fit(X_train, y_train)

print("\n--- RANDOM FOREST TRAINING COMPLETED ---")

# Make predictions
rf_pred = rf_model.predict(X_test)

# Evaluate Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

# ROC-AUC
rf_probability = rf_model.predict_proba(X_test)[:, 1]
rf_roc_auc = roc_auc_score(y_test, rf_probability)

print("\n--- RANDOM FOREST RESULTS ---")
print("Accuracy :", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall   :", round(rf_recall, 4))
print("F1 Score :", round(rf_f1, 4))
print("ROC-AUC  :", round(rf_roc_auc, 4))

# -----------------------------
# STEP 24: SAVE BEST MODEL
# -----------------------------

import joblib

# Save Logistic Regression model
joblib.dump(
    model,
    "models/churn_model.pkl"
)

print("\n--- MODEL SAVED ---")
print("Model saved to: models/churn_model.pkl")