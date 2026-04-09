import pandas as pd
import os

# Folder where TXT files are stored
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# File names (NASA C-MAPSS)
TRAIN_FILE = "train_FD001.txt"
TEST_FILE = "test_FD001.txt"
RUL_FILE = "RUL_FD001.txt"

def convert_txt_to_csv(txt_file, csv_file):
    """
    Converts space-separated TXT to CSV
    """
    txt_path = os.path.join(DATA_DIR, txt_file)
    csv_path = os.path.join(DATA_DIR, csv_file)

    df = pd.read_csv(txt_path, sep=r"\s+", header=None)
    df.to_csv(csv_path, index=False)

    print(f"✅ Converted {txt_file} → {csv_file}")

def main():
    convert_txt_to_csv(TRAIN_FILE, "train_FD001.csv")
    convert_txt_to_csv(TEST_FILE, "test_FD001.csv")
    convert_txt_to_csv(RUL_FILE, "RUL_FD001.csv")

    print("\n🎉 All TXT files converted successfully!")

if __name__ == "__main__":
    main()