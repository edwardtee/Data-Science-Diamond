"""
Shared preprocessing for the Diamond Price Prediction project.

Every model (Gradient Boosting, Extra Trees, Linear Regression, Support Vector Regression)
imports `load_split()` from here, so all four are trained and evaluated on the same rows

The pipeline (agreed by the group):
    1. Load the raw CSV and drop the unnamed index column.
    2. Remove outliers with the IQR rule on: carat, depth, table, x, y, z.
    3. Use features: carat, x, y, z   (target = price).
    4. Split 80/20 with random_state=42 so the split never changes between runs.

Run this file directly to (re)build the data/processed/ CSVs:
    python src/preprocessing/preprocess.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# preprocess.py lives at <project>/src/preprocessing/preprocess.py
# parents[0]=preprocessing, [1]=src, [2]=<project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "Diamonds Prices2022.csv"

FEATURES = ["carat", "x", "y", "z"]
TARGET = "price"
OUTLIER_COLS = ["carat", "depth", "table", "x", "y", "z"]


def remove_outliers_iqr(data: pd.DataFrame, cols) -> pd.DataFrame:
    # Drop rows that fall outside 1.5*IQR on any of `cols` (applied in order).
    data = data.copy()
    for col in cols:
        Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data


def load_test_split():
    df = pd.read_csv(RAW_CSV)
    df_clean = remove_outliers_iqr(df, OUTLIER_COLS)
    print("Cleaned shape (after outlier removal):", df.shape)

    X = df_clean[FEATURES]
    y = df_clean[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Train:", X_train.shape, " Test:", X_test.shape)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    load_test_split()
