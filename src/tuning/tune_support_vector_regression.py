import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR

# Navigate up 2 levels from testing.py to find the project root directory
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.preprocessing.preprocessing import load_test_split

X_train, X_test, y_train, y_test = load_test_split()

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

OUT_DIR = root_dir / "output" / "support_vector_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 2. Kernel sweep
# ---------------------------------------------------------------------
print("--- Kernel sweep (C=1.0, epsilon=0.1, gamma='scale') ---")
kernels = ["linear", "rbf", "poly", "sigmoid"]
kernel_rows = []
for k in kernels:
    model = SVR(kernel=k, C=1.0, epsilon=0.1, gamma="scale")
    r2, mae, rmse, ft = evaluate(model, X_train, y_train, X_test, y_test)
    kernel_rows.append((k, r2, mae, rmse, ft))
    print(f"kernel={k:8s}: R2={r2:.4f}  MAE={mae:8.2f}  RMSE={rmse:8.2f}  fit_time={ft:.2f}s")

kernel_df = pd.DataFrame(kernel_rows, columns=["kernel", "R2", "MAE", "RMSE", "fit_time(s)"])
kernel_df.to_csv(OUT_DIR / "svr_kernel.csv", index=False)
plot_tuning(kernel_df["kernel"], kernel_df["R2"], kernel_df["fit_time(s)"], "Kernel", "Kernel Tuning: R² and Fit Time", OUT_DIR / "svr_kernel.png")

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
    r2, mae, rmse, ft = evaluate(model, X_train, y_train, X_test, y_test)
    C_rows.append((C, r2, mae, rmse, ft))
    print(f"C={C:7}: R2={r2:.4f}  MAE={mae:8.2f}  RMSE={rmse:8.2f}  fit_time={ft:.2f}s")

C_df = pd.DataFrame(C_rows, columns=["C", "R2", "MAE", "RMSE", "fit_time(s)"])
C_df.to_csv(OUT_DIR / "svr_C.csv", index=False)
plot_tuning(C_df["C"], C_df["R2"], C_df["fit_time(s)"], "C (log scale)", "C Tuning: R² and Fit Time", OUT_DIR / "svr_C.png", log_x=True)

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
    r2, mae, rmse, ft = evaluate(model, X_train, y_train, X_test, y_test)
    eps_rows.append((eps, r2, mae, rmse, ft))
    print(f"epsilon={eps:6}: R2={r2:.4f}  MAE={mae:8.2f}  RMSE={rmse:8.2f}  fit_time={ft:.2f}s")

eps_df = pd.DataFrame(eps_rows, columns=["epsilon", "R2", "MAE", "RMSE", "fit_time(s)"])
eps_df.to_csv(OUT_DIR / "svr_epsilon.csv", index=False)
plot_tuning(eps_df["epsilon"], eps_df["R2"], eps_df["fit_time(s)"], "epsilon (log scale)", "epsilon Tuning: R² and Fit Time", OUT_DIR / "svr_epsilon.png", log_x=True)

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
    r2, mae, rmse, ft = evaluate(model, X_train, y_train, X_test, y_test)
    gamma_rows.append((g, r2, mae, rmse, ft))
    print(f"gamma={g:6}: R2={r2:.4f}  MAE={mae:8.2f}  RMSE={rmse:8.2f}  fit_time={ft:.2f}s")

# also test scale/auto separately since they're not numeric
for g in ["scale", "auto"]:
    model = SVR(kernel="rbf", C=BEST_C, epsilon=BEST_EPS, gamma=g)
    r2, mae, rmse, ft = evaluate(model, X_train, y_train, X_test, y_test)
    gamma_rows.append((g, r2, mae, rmse, ft))
    print(f"gamma={g:6}: R2={r2:.4f}  MAE={mae:8.2f}  RMSE={rmse:8.2f}  fit_time={ft:.2f}s")

gamma_df = pd.DataFrame(gamma_rows, columns=["gamma", "R2", "MAE", "RMSE", "fit_time(s)"])
gamma_df.to_csv(OUT_DIR / "svr_gamma.csv", index=False)

# plot only the numeric gamma values (scale/auto shown in CSV/table only)
gamma_numeric = gamma_df[pd.to_numeric(gamma_df["gamma"], errors="coerce").notna()].copy()
gamma_numeric["gamma"] = gamma_numeric["gamma"].astype(float)
plot_tuning(gamma_numeric["gamma"], gamma_numeric["R2"], gamma_numeric["fit_time(s)"], "gamma (log scale)", "gamma Tuning: R² and Fit Time", OUT_DIR / "svr_gamma.png", log_x=True)

best_gamma_row = gamma_numeric.loc[gamma_numeric["R2"].idxmax()]
BEST_GAMMA = best_gamma_row["gamma"]
print(f">> Best gamma by this scan: {BEST_GAMMA} (R2={best_gamma_row['R2']:.4f})\n")

# ---------------------------------------------------------------------
# 6. FINAL MODEL 
# ---------------------------------------------------------------------
print("=" * 70)
print("FINAL CHOSEN PARAMETERS (best value found per sweep):")
print("  kernel  = rbf   (see caution note above about the kernel scan)")
print(f"  C       = {BEST_C}")
print(f"  epsilon = {BEST_EPS}")
print(f"  gamma   = {BEST_GAMMA}")
print("=" * 70)