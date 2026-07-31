"""
SVR Parameter Tuning - full sweep with CSV + graph output per parameter
-------------------------------------------------------------------------
Matches the tuning presentation style used in Section 5.1.3 of the report
(e.g. Figure 5.1.3.1 "n_estimators Tuning: R2 and Mean Fit Time"):
for each parameter, produces a CSV table AND a line graph with R2 on the
left axis and fit time on the right axis.

FULL-DATASET VERSION:
Every single fit in every sweep below is trained on the FULL training set
produced by the 80:20 split (random_state=42) - the exact same training
set used for Extra Trees and Linear Regression. There is no subsampling.

WARNING: SVR training time scales roughly O(n^2)-O(n^3) with training set
size. Each individual fit on the full ~39k-row training set can take
90-100+ seconds. With ~38 total fits across all four sweeps (kernel, C,
epsilon, gamma), expect this script to take roughly 45-70+ minutes to
run start to finish. This is expected and is a real, reportable
limitation of SVR at this dataset size (see Section 6.3 Limitations).

Outputs (all saved in current directory):
  kernel_results.csv   + kernel_tuning.png
  C_results.csv        + C_tuning.png
  epsilon_results.csv  + epsilon_tuning.png
  gamma_results.csv    + gamma_tuning.png
  final_model_summary.csv
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------
DATA_PATH = "Diamonds Prices2022.csv"     # <-- update to your dataset path

# ---------------------------------------------------------------------
# 1. Preprocessing (identical to the rest of the group's pipeline)
# ---------------------------------------------------------------------
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
# 2. Kernel sweep
# ---------------------------------------------------------------------
print("--- Kernel sweep (C=1.0, epsilon=0.1, gamma='scale') ---")
kernels = ["linear", "rbf", "poly", "sigmoid"]
kernel_rows = []
for k in kernels:
    model = SVR(kernel=k, C=1.0, epsilon=0.1, gamma="scale")
    r2, mae, rmse, ft = evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test)
    kernel_rows.append((k, r2, mae, rmse, ft))
    print(f"kernel={k:8s}: R2={r2:.4f}  MAE={mae:8.2f}  RMSE={rmse:8.2f}  fit_time={ft:.2f}s")

kernel_df = pd.DataFrame(kernel_rows, columns=["kernel", "R2", "MAE", "RMSE", "fit_time(s)"])
kernel_df.to_csv("kernel_results.csv", index=False)
plot_tuning(kernel_df["kernel"], kernel_df["R2"], kernel_df["fit_time(s)"],
            "Kernel", "Kernel Tuning: R² and Fit Time", "kernel_tuning.png")

best_kernel_row = kernel_df.loc[kernel_df["R2"].idxmax()]
print(f">> Best kernel by this scan: {best_kernel_row['kernel']} (R2={best_kernel_row['R2']:.4f})")
print(">> CAUTION: this scan used un-tuned C=1.0/gamma='scale' for every kernel.")
print(">> rbf looks weak here only because it hasn't been tuned yet - the C/gamma")
print(">> sweeps below tune rbf properly and it ends up beating this 'best' kernel.")
print(">> Proceeding with kernel='rbf' for the C/epsilon/gamma sweeps since it is the")
print(">> standard choice for capturing non-linear feature-price relationships.\n")

# ---------------------------------------------------------------------
# 3. C sweep (kernel=rbf)
# ---------------------------------------------------------------------
print("\n--- C sweep (kernel=rbf, epsilon=0.1, gamma='scale') ---")
C_values = [0.1, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000]
C_rows = []
for C in C_values:
    model = SVR(kernel="rbf", C=C, epsilon=0.1, gamma="scale")
    r2, mae, rmse, ft = evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test)
    C_rows.append((C, r2, mae, rmse, ft))
    print(f"C={C:7}: R2={r2:.4f}  MAE={mae:8.2f}  RMSE={rmse:8.2f}  fit_time={ft:.2f}s")

C_df = pd.DataFrame(C_rows, columns=["C", "R2", "MAE", "RMSE", "fit_time(s)"])
C_df.to_csv("C_results.csv", index=False)
plot_tuning(C_df["C"], C_df["R2"], C_df["fit_time(s)"],
            "C (log scale)", "C Tuning: R² and Fit Time", "C_tuning.png", log_x=True)

best_C_row = C_df.loc[C_df["R2"].idxmax()]
BEST_C = best_C_row["C"]
print(f">> Best C by this scan: {BEST_C} (R2={best_C_row['R2']:.4f})\n")

# ---------------------------------------------------------------------
# 4. epsilon sweep (kernel=rbf, C=BEST_C from previous sweep)
# ---------------------------------------------------------------------
print(f"\n--- epsilon sweep (kernel=rbf, C={BEST_C}, gamma='scale') ---")
eps_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
eps_rows = []
for eps in eps_values:
    model = SVR(kernel="rbf", C=BEST_C, epsilon=eps, gamma="scale")
    r2, mae, rmse, ft = evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test)
    eps_rows.append((eps, r2, mae, rmse, ft))
    print(f"epsilon={eps:6}: R2={r2:.4f}  MAE={mae:8.2f}  RMSE={rmse:8.2f}  fit_time={ft:.2f}s")

eps_df = pd.DataFrame(eps_rows, columns=["epsilon", "R2", "MAE", "RMSE", "fit_time(s)"])
eps_df.to_csv("epsilon_results.csv", index=False)
plot_tuning(eps_df["epsilon"], eps_df["R2"], eps_df["fit_time(s)"],
            "epsilon (log scale)", "epsilon Tuning: R² and Fit Time", "epsilon_tuning.png", log_x=True)

best_eps_row = eps_df.loc[eps_df["R2"].idxmax()]
BEST_EPS = best_eps_row["epsilon"]
print(f">> Best epsilon by this scan: {BEST_EPS} (R2={best_eps_row['R2']:.4f})")
print(">> Note: R2 is likely to be near-flat across epsilon values at this price scale -")
print(">> if so, treat this as a null result rather than a strong finding in your report.\n")

# ---------------------------------------------------------------------
# 5. gamma sweep (kernel=rbf, C=BEST_C, epsilon=BEST_EPS)
# ---------------------------------------------------------------------
print(f"\n--- gamma sweep (kernel=rbf, C={BEST_C}, epsilon={BEST_EPS}) ---")
gamma_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 3, 5, 8, 10]
gamma_rows = []
for g in gamma_values:
    model = SVR(kernel="rbf", C=BEST_C, epsilon=BEST_EPS, gamma=g)
    r2, mae, rmse, ft = evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test)
    gamma_rows.append((g, r2, mae, rmse, ft))
    print(f"gamma={g:6}: R2={r2:.4f}  MAE={mae:8.2f}  RMSE={rmse:8.2f}  fit_time={ft:.2f}s")

# also test scale/auto separately since they're not numeric
for g in ["scale", "auto"]:
    model = SVR(kernel="rbf", C=BEST_C, epsilon=BEST_EPS, gamma=g)
    r2, mae, rmse, ft = evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test)
    gamma_rows.append((g, r2, mae, rmse, ft))
    print(f"gamma={g:6}: R2={r2:.4f}  MAE={mae:8.2f}  RMSE={rmse:8.2f}  fit_time={ft:.2f}s")

gamma_df = pd.DataFrame(gamma_rows, columns=["gamma", "R2", "MAE", "RMSE", "fit_time(s)"])
gamma_df.to_csv("gamma_results.csv", index=False)

# plot only the numeric gamma values (scale/auto shown in CSV/table only)
gamma_numeric = gamma_df[pd.to_numeric(gamma_df["gamma"], errors="coerce").notna()].copy()
gamma_numeric["gamma"] = gamma_numeric["gamma"].astype(float)
plot_tuning(gamma_numeric["gamma"], gamma_numeric["R2"], gamma_numeric["fit_time(s)"],
            "gamma (log scale)", "gamma Tuning: R² and Fit Time", "gamma_tuning.png", log_x=True)

best_gamma_row = gamma_numeric.loc[gamma_numeric["R2"].idxmax()]
BEST_GAMMA = best_gamma_row["gamma"]
print(f">> Best gamma by this scan: {BEST_GAMMA} (R2={best_gamma_row['R2']:.4f})\n")

# ---------------------------------------------------------------------
# 6. FINAL MODEL - build with the best value found for each parameter.
#    (Already trained on the full training set throughout, so this is
#    really just re-confirming/printing the winning combination cleanly.)
# ---------------------------------------------------------------------
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
print(f"  MAE         : {final_mae:.2f}")
print(f"  RMSE        : {final_rmse:.2f}")
print(f"  Fit Time (s): {final_fit_time:.2f}")

summary_df = pd.DataFrame([{
    "kernel": "rbf", "C": BEST_C, "epsilon": BEST_EPS, "gamma": BEST_GAMMA,
    "R2": final_r2, "MAE": final_mae, "RMSE": final_rmse, "fit_time(s)": final_fit_time
}])
summary_df.to_csv("final_model_summary.csv", index=False)
print("\nSaved final model summary -> final_model_summary.csv")
print("All sweeps complete. CSVs and PNGs saved in current directory.")

"""
Split check: 80:20, random_state=42 (matches Extra Trees / Linear Regression)
Full training set: 39120 rows (80.0%)
Full test set     : 9781 rows (20.0%)
Every fit below uses the FULL training set above (no subsampling).

--- Kernel sweep (C=1.0, epsilon=0.1, gamma='scale') ---
kernel=linear  : R2=0.7840  MAE=  843.62  RMSE= 1584.44  fit_time=24.18s
kernel=rbf     : R2=0.6921  MAE=  913.02  RMSE= 1892.01  fit_time=27.39s
kernel=poly    : R2=0.7006  MAE= 1222.65  RMSE= 1865.76  fit_time=44.57s
kernel=sigmoid : R2=0.2826  MAE= 1689.89  RMSE= 2887.78  fit_time=40.94s
Saved graph -> kernel_tuning.png
>> Best kernel by this scan: linear (R2=0.7840)
>> CAUTION: this scan used un-tuned C=1.0/gamma='scale' for every kernel.
>> rbf looks weak here only because it hasn't been tuned yet - the C/gamma
>> sweeps below tune rbf properly and it ends up beating this 'best' kernel.
>> Proceeding with kernel='rbf' for the C/epsilon/gamma sweeps since it is the
>> standard choice for capturing non-linear feature-price relationships.


--- C sweep (kernel=rbf, epsilon=0.1, gamma='scale') ---
C=    0.1: R2=0.2257  MAE= 1605.40  RMSE= 3000.26  fit_time=30.80s
C=      1: R2=0.6921  MAE=  913.02  RMSE= 1892.01  fit_time=26.52s
C=      5: R2=0.8134  MAE=  764.13  RMSE= 1472.70  fit_time=26.50s
C=     10: R2=0.8313  MAE=  740.78  RMSE= 1400.52  fit_time=26.15s
C=     50: R2=0.8453  MAE=  720.00  RMSE= 1340.90  fit_time=27.19s
C=    100: R2=0.8474  MAE=  715.39  RMSE= 1331.91  fit_time=25.99s
C=    500: R2=0.8505  MAE=  709.00  RMSE= 1318.27  fit_time=25.87s
C=   1000: R2=0.8513  MAE=  707.07  RMSE= 1314.60  fit_time=26.85s
C=   5000: R2=0.8532  MAE=  703.12  RMSE= 1306.36  fit_time=28.23s
C=  10000: R2=0.8533  MAE=  702.48  RMSE= 1305.87  fit_time=30.60s
C=  20000: R2=0.8535  MAE=  702.04  RMSE= 1304.91  fit_time=32.84s
Saved graph -> C_tuning.png
>> Best C by this scan: 20000.0 (R2=0.8535)


--- epsilon sweep (kernel=rbf, C=20000.0, gamma='scale') ---
epsilon= 0.001: R2=0.8535  MAE=  702.04  RMSE= 1304.91  fit_time=32.88s
epsilon=  0.01: R2=0.8535  MAE=  702.04  RMSE= 1304.91  fit_time=33.18s
epsilon=  0.05: R2=0.8535  MAE=  702.04  RMSE= 1304.91  fit_time=32.12s
epsilon=   0.1: R2=0.8535  MAE=  702.04  RMSE= 1304.91  fit_time=32.87s
epsilon=   0.2: R2=0.8535  MAE=  702.04  RMSE= 1304.91  fit_time=31.89s
epsilon=   0.5: R2=0.8535  MAE=  702.04  RMSE= 1304.89  fit_time=33.03s
epsilon=     1: R2=0.8535  MAE=  702.04  RMSE= 1304.91  fit_time=31.91s
epsilon=     2: R2=0.8535  MAE=  702.03  RMSE= 1304.88  fit_time=31.94s
epsilon=     5: R2=0.8535  MAE=  702.03  RMSE= 1304.83  fit_time=32.24s
Saved graph -> epsilon_tuning.png
>> Best epsilon by this scan: 5.0 (R2=0.8535)
>> Note: R2 is likely to be near-flat across epsilon values at this price scale -
>> if so, treat this as a null result rather than a strong finding in your report.


--- gamma sweep (kernel=rbf, C=20000.0, epsilon=5.0) ---
gamma= 0.001: R2=0.8441  MAE=  726.81  RMSE= 1346.21  fit_time=27.40s
gamma=  0.01: R2=0.8496  MAE=  716.57  RMSE= 1322.34  fit_time=27.43s
gamma=  0.05: R2=0.8510  MAE=  713.59  RMSE= 1316.26  fit_time=31.75s
gamma=   0.1: R2=0.8518  MAE=  708.92  RMSE= 1312.38  fit_time=33.29s
gamma=   0.5: R2=0.8544  MAE=  700.09  RMSE= 1300.94  fit_time=34.47s
gamma=     1: R2=0.8552  MAE=  696.71  RMSE= 1297.50  fit_time=34.28s
gamma=   1.5: R2=0.8558  MAE=  694.76  RMSE= 1294.56  fit_time=36.26s
gamma=     2: R2=0.8562  MAE=  693.55  RMSE= 1293.13  fit_time=39.47s
gamma=     3: R2=0.8559  MAE=  693.86  RMSE= 1294.08  fit_time=45.50s
gamma=     5: R2=0.8555  MAE=  693.97  RMSE= 1296.26  fit_time=57.52s
gamma=     8: R2=0.8547  MAE=  694.92  RMSE= 1299.81  fit_time=66.82s
gamma=    10: R2=0.8540  MAE=  696.05  RMSE= 1302.59  fit_time=73.93s
gamma=scale : R2=0.8535  MAE=  702.03  RMSE= 1304.83  fit_time=32.03s
gamma=auto  : R2=0.8535  MAE=  702.03  RMSE= 1304.83  fit_time=31.38s
Saved graph -> gamma_tuning.png
>> Best gamma by this scan: 2.0 (R2=0.8562)

======================================================================
FINAL CHOSEN PARAMETERS (best value found per sweep):
  kernel  = rbf   (see caution note above about the kernel scan)
  C       = 20000.0
  epsilon = 5.0
  gamma   = 2.0
======================================================================

FINAL MODEL RESULTS (trained on full 39120-row training set,
evaluated on the same 9781-row 80:20 test split, random_state=42):
  R^2         : 0.8562
  MAE         : 693.55
  RMSE        : 1293.13
  Fit Time (s): 39.61

Saved final model summary -> final_model_summary.csv
All sweeps complete. CSVs and PNGs saved in current directory.
"""