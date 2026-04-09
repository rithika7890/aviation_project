import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# ==============================
# PATH SETUP
# ==============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

data_path = os.path.join(BASE_DIR, "android app", "data", "train_FD001.csv")
model_path = os.path.join(BASE_DIR, "android app", "models", "final_rul_model.pkl")

print("Resolved Data Path:", data_path)
print("Resolved Model Path:", model_path)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(data_path)

df.columns = ["unit", "cycle",
              "op_setting_1", "op_setting_2", "op_setting_3"] + \
             [f"sensor_{i}" for i in range(1, 22)]

# ==============================
# RUL CALCULATION
# ==============================
max_cycle = df.groupby("unit")["cycle"].max()
df["RUL"] = df.apply(lambda row: max_cycle[row["unit"]] - row["cycle"], axis=1)

CAP_VALUE = 125
df["RUL"] = df["RUL"].clip(upper=CAP_VALUE)

print("RUL calculated and capped at", CAP_VALUE)

# ==============================
# FEATURE ENGINEERING
# ==============================
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

for col in sensor_cols:
    df[col + "_mean"] = df.groupby("unit")[col].rolling(5, min_periods=1)\
                         .mean().reset_index(level=0, drop=True)

for col in sensor_cols:
    df[col + "_slope"] = df.groupby("unit")[col].diff().fillna(0)

print("Feature engineering completed")

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load(model_path)

X = df[model.feature_names_in_]
y = df["RUL"]

print("Feature alignment successful")

# ==============================
# PREDICTION
# ==============================
y_pred = model.predict(X)

# ==============================
# 1️⃣ ACTUAL vs PREDICTED (CLEAR)
# ==============================
plt.figure()

plt.scatter(y, y_pred, label="Predictions")

# Ideal reference line
plt.plot([0, CAP_VALUE], [0, CAP_VALUE], 'r--', label="Ideal Prediction Line")

plt.xlim(0, CAP_VALUE)
plt.ylim(0, CAP_VALUE)

plt.xlabel("Actual Remaining Useful Life (Cycles)")
plt.ylabel("Predicted Remaining Useful Life (Cycles)")
plt.title("Actual vs Predicted RUL (Capped at 125 Cycles)")
plt.legend()

plt.savefig(os.path.join(BASE_DIR, "actual_vs_predicted.png"))
plt.show()
plt.close()

# ==============================
# 2️⃣ RESIDUAL DISTRIBUTION
# ==============================
residuals = y - y_pred

plt.figure()

plt.hist(residuals, bins=40, label="Residuals")

plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Residual Distribution of RUL Prediction")
plt.legend()

plt.savefig(os.path.join(BASE_DIR, "residual_distribution.png"))
plt.show()
plt.close()

# ==============================
# 3️⃣ CLEAN DEGRADATION CURVES
# ==============================
# ===============================
# GRAPH 3: CLEAN DEGRADATION CURVE
# ===============================

# ===============================
# DEGRADATION CURVE (FINAL CLEAN)
# ===============================

plt.figure(figsize=(10, 6))

# Select only 3 engines for clarity
engine_ids = df['unit'].unique()[:3]

# Consistent colors for each engine
colors = ['blue', 'green', 'red']

for i, engine_id in enumerate(engine_ids):
    engine_data = df[df['unit'] == engine_id]

    cycles = engine_data['cycle']
    actual_rul = engine_data['RUL']
    predicted_rul = y_pred[engine_data.index]

    # Actual RUL (solid)
    plt.plot(cycles, actual_rul,
             color=colors[i],
             linewidth=2,
             label=f'Engine {engine_id} - Actual')

    # Predicted RUL (dashed)
    plt.plot(cycles, predicted_rul,
             color=colors[i],
             linestyle='--',
             linewidth=2,
             label=f'Engine {engine_id} - Predicted')

# RUL cap line
plt.axhline(y=125,
            color='black',
            linestyle=':',
            linewidth=2,
            label='RUL Cap (125 Cycles)')

# Labels and styling
plt.xlabel("Engine Cycles")
plt.ylabel("Remaining Useful Life (RUL)")
plt.title("Engine Degradation Curves (Actual vs Predicted RUL)")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(BASE_DIR, "degradation_curve_final.png"))
plt.show()

plt.savefig(os.path.join(BASE_DIR, "clean_degradation_curve.png"))
plt.show()
plt.close()

print("\n✅ FINAL LABELED GRAPHS GENERATED SUCCESSFULLY")