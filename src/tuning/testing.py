import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_PATH = "Diamonds Prices2022.csv"     # <-- update to your dataset path

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
for col in ["cut", "color", "clarity"]:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])

def remove_outliers_iqr(data, columns):
    data = data.copy()
    for col in columns:
        q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        iqr = q3 - q1
        data = data[(data[col] >= q1 - 1.5 * iqr) & (data[col] <= q3 + 1.5 * iqr)]
    return data

df_clean = remove_outliers_iqr(df, ["carat", "depth", "table", "x", "y", "z"])

FEATURES = ["carat", "x", "y", "z"]
X = df_clean[FEATURES]
y = df_clean["price"]

# Same 80:20 split, same random_state=42 as the rest of the group's models
# (Extra Trees, Linear Regression) -> test set is identical across all models,
# so R2/MAE/RMSE comparisons in Section 6.0 are apples-to-apples.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Split check: 80:20, random_state=42 (matches Extra Trees / Linear Regression)")
print(f"Full training set: {X_train_scaled.shape[0]} rows ({X_train_scaled.shape[0]/len(X):.1%})")
print(f"Full test set     : {X_test_scaled.shape[0]} rows ({X_test_scaled.shape[0]/len(X):.1%})")
print("Every fit below uses the FULL training set above (no subsampling).\n")


def evaluate(model, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_time = time.time() - t0
    pred = model.predict(X_te)
    r2 = r2_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    return r2, mae, rmse, fit_time


def plot_tuning(x_values, r2_values, time_values, xlabel, title, filename, log_x=False):
    fig, ax1 = plt.subplots(figsize=(7, 5))
    color1 = "tab:blue"
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("R² Score", color=color1)
    ax1.plot(x_values, r2_values, marker="o", color=color1, label="R²")
    ax1.tick_params(axis="y", labelcolor=color1)
    if log_x:
        ax1.set_xscale("log")

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Fit Time (s)", color=color2)
    ax2.plot(x_values, time_values, marker="s", linestyle="--", color=color2, label="Fit Time")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title(title)
    fig.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved graph -> {filename}")

# ---------------------------------------------------------------------
# 6. FINAL MODEL - build with the best value found for each parameter.
#    (Already trained on the full training set throughout, so this is
#    really just re-confirming/printing the winning combination cleanly.)
# ---------------------------------------------------------------------
BEST_C = 20000
BEST_EPS = 5
BEST_GAMMA = 2

print("=" * 70)
print("FINAL CHOSEN PARAMETERS (best value found per sweep):")
print(f"  kernel  = rbf   (see caution note above about the kernel scan)")
print(f"  C       = {BEST_C}")
print(f"  epsilon = {BEST_EPS}")
print(f"  gamma   = {BEST_GAMMA}")
print("=" * 70)

final_model = SVR(kernel="rbf", C=BEST_C, epsilon=BEST_EPS, gamma=BEST_GAMMA)
t0 = time.time()
final_model.fit(X_train_scaled, y_train)
final_fit_time = time.time() - t0
final_pred = final_model.predict(X_test_scaled)

final_r2 = r2_score(y_test, final_pred)
final_mae = mean_absolute_error(y_test, final_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))

print(f"\nFINAL MODEL RESULTS (trained on full {X_train_scaled.shape[0]}-row training set,")
print(f"evaluated on the same {X_test_scaled.shape[0]}-row 80:20 test split, random_state=42):")
print(f"  R^2         : {final_r2:.4f}")
print(f"  MAE    a     : {final_mae:.2f}")
print(f"  RMSE        : {final_rmse:.2f}")
print(f"  Fit Time (s): {final_fit_time:.2f}")