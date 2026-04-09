import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load data
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_FD001.csv"))
rul_df = pd.read_csv(os.path.join(DATA_DIR, "RUL_FD001.csv"))

# -----------------------------
# Feature selection
# -----------------------------
X = train_df.iloc[:, 2:]   # sensor features
y_cycles = train_df.iloc[:, 1]  # cycle count proxy

# -----------------------------
# Convert cycles → risk classes
# -----------------------------
max_cycles = y_cycles.max()

def risk_label(cycle):
    if cycle < 0.33 * max_cycles:
        return 2  # High Risk
    elif cycle < 0.66 * max_cycles:
        return 1  # Medium Risk
    else:
        return 0  # Low Risk

y = y_cycles.apply(risk_label)

print("Risk class distribution:")
print(y.value_counts())

# -----------------------------
# Train / Validation split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Scaling (important for LR & SVM)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel="rbf"),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )
}

# -----------------------------
# Training & Evaluation
# -----------------------------
for name, model in models.items():
    print("\n" + "="*50)
    print(f"Model: {name}")

    if name == "Random Forest":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

    acc = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")

    print("Classification Report:")
    print(classification_report(
        y_val,
        y_pred,
        target_names=["Low Risk", "Medium Risk", "High Risk"]
    ))