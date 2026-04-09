import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score

print("Loading dataset...")

# correct path to dataset
data_path = "android app/data/train_FD001.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully")

# -----------------------------
# Assign column names
# -----------------------------

columns = ['unit','cycle','op_setting_1','op_setting_2','op_setting_3']

for i in range(1,22):
    columns.append(f"sensor_{i}")

df.columns = columns


# -----------------------------
# Calculate RUL
# -----------------------------

max_cycle = df.groupby('unit')['cycle'].max()

df = df.merge(max_cycle.rename("max_cycle"), on='unit')

df['RUL'] = df['max_cycle'] - df['cycle']

df.drop("max_cycle", axis=1, inplace=True)

print("RUL calculated")


# -----------------------------
# Create rolling mean features
# -----------------------------

for i in range(1,22):
    
    sensor = f"sensor_{i}"
    
    df[f"{sensor}_mean"] = df.groupby("unit")[sensor].rolling(5).mean().reset_index(0,drop=True)


df.fillna(method="bfill", inplace=True)

print("Sensor mean features created")


# -----------------------------
# Load trained model
# -----------------------------

print("Loading trained model...")

model = joblib.load("android app/models/final_rul_model.pkl")

print("Model loaded successfully")


# -----------------------------
# Select model features
# -----------------------------

model_features = model.feature_names_in_

print("Model expects features:")
print(model_features)


X = df[model_features]

y = df["RUL"]


# -----------------------------
# Predictions
# -----------------------------

print("Generating predictions...")

y_pred = model.predict(X)


# -----------------------------
# Evaluation metrics
# -----------------------------

rmse = np.sqrt(mean_squared_error(y, y_pred))

r2 = r2_score(y, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)


# -----------------------------
# 1️⃣ Actual vs Predicted Plot
# -----------------------------

plt.figure(figsize=(8,6))

plt.scatter(y, y_pred, alpha=0.3)

plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')

plt.xlabel("Actual RUL")

plt.ylabel("Predicted RUL")

plt.title("Actual vs Predicted RUL")

plt.show()


# -----------------------------
# 2️⃣ Feature Importance
# -----------------------------

importances = model.feature_importances_

indices = np.argsort(importances)[-10:]

plt.figure(figsize=(8,6))

plt.barh(range(len(indices)), importances[indices])

plt.yticks(range(len(indices)), X.columns[indices])

plt.xlabel("Feature Importance")

plt.title("Top Features for RUL Prediction")

plt.show()


# -----------------------------
# 3️⃣ RUL Degradation Curve
# -----------------------------

engine_id = df.iloc[0]["unit"]

engine_data = df[df["unit"] == engine_id]

plt.figure(figsize=(8,5))

plt.plot(engine_data["cycle"], engine_data["RUL"])

plt.xlabel("Cycle")

plt.ylabel("Remaining Useful Life")

plt.title("Engine Degradation Curve")

plt.show()


print("All plots generated successfully")