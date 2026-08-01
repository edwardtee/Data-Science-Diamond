# Shared preprocessing for the Diamond Price Prediction project.
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
    df = df.drop(columns="Unnamed: 0")
    df_clean = remove_outliers_iqr(df, OUTLIER_COLS)
    print("Cleaned shape (after outlier removal):", df_clean.shape)

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


def load_full_split():
    df = pd.read_csv(RAW_CSV)
    df = df.drop(columns="Unnamed: 0")
    df_clean = remove_outliers_iqr(df, OUTLIER_COLS)
    print("Cleaned shape (after outlier removal):", df_clean.shape)

    encoder = LabelEncoder()
    for col in ["cut", "color", "clarity"]:
        df_clean[col] = encoder.fit_transform(df_clean[col])
    
    X = df_clean.drop(columns=["price"])
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
    load_full_split()
