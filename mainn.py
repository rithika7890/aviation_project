import pandas as pd
import os

# Correct data folder path
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_FD001.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test_FD001.csv"))
rul_df   = pd.read_csv(os.path.join(DATA_DIR, "RUL_FD001.csv"))

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("RUL shape:", rul_df.shape)