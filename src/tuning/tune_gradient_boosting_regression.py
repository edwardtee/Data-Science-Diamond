import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Navigate up 2 levels from testing.py to find the project root directory
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing.preprocessing import load_test_split

X_train, X_test, y_train, y_test = load_test_split()

SCORING = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}

def sweep(param_name, values, fixed_params, cv=5):
    base_params = dict(fixed_params)
    base_params["random_state"] = 42
    model = GradientBoostingRegressor(**base_params)

    param_grid = {param_name: values}

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=SCORING,
        refit="r2",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    cv_results = grid.cv_results_

    rows = []
    for i, v in enumerate(values):
        rows.append({
            param_name: v,
            "R2": cv_results["mean_test_r2"][i],
            "MAE": -cv_results["mean_test_mae"][i],
            "RMSE": -cv_results["mean_test_rmse"][i],
            "Time(s)": cv_results["mean_fit_time"][i],   # avg time to fit ONE model at this value
        })
        print(f"  {param_name}={v}: R2={rows[-1]['R2']:.4f} MAE={rows[-1]['MAE']:.2f} "
              f"RMSE={rows[-1]['RMSE']:.2f} avg_fit_time={rows[-1]['Time(s)']:.3f}s")
    print(f"  -> GridSearchCV best {param_name} = {grid.best_params_[param_name]} "
          f"(R2={grid.best_score_:.4f})")
    return pd.DataFrame(rows)

def plot_sweep(df, param_name, out_path, xlabel=None, logx=False):
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    x = df[param_name]
    ax1.plot(x, df["R2"], marker="o", color="#2563eb", label="R2 (mean CV)")
    ax1.set_xlabel(xlabel or param_name)
    ax1.set_ylabel("R2 score", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    if logx:
        ax1.set_xscale("log")
    ax2 = ax1.twinx()
    ax2.plot(x, df["Time(s)"], marker="s", color="#dc2626", linestyle="--", label="Avg fit time (s)")
    ax2.set_ylabel("Avg fit time per model (s)", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    plt.title(f"Effect of {param_name} on R2 and Fit Time")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  saved", out_path)

BASE = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "min_samples_leaf": 1, "subsample": 1.0}
OUT_DIR = ROOT_DIR / "output" / "gradient_boosting_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n--- n_estimators ---")
df_n = sweep("n_estimators", [50, 100, 200, 300, 400, 500], BASE)
plot_sweep(df_n, "n_estimators", OUT_DIR / "gbr_n_estimators.png")

print("\n--- max_depth ---")
df_depth = sweep("max_depth", [2, 3, 4, 5, 6, 7, 8], BASE)
plot_sweep(df_depth, "max_depth", OUT_DIR / "gbr_max_depth.png")

print("\n--- learning_rate ---")
df_lr = sweep("learning_rate", [0.01, 0.05, 0.1, 0.2, 0.3, 0.4], BASE)
plot_sweep(df_lr, "learning_rate", OUT_DIR / "gbr_learning_rate.png", logx=True)

print("\n--- min_samples_leaf ---")
df_leaf = sweep("min_samples_leaf", [1, 2, 4, 8, 16, 32, 64, 128], BASE)
plot_sweep(df_leaf, "min_samples_leaf", OUT_DIR / "gbr_min_samples_leaf.png")

print("\n--- subsample ---")
df_sub = sweep("subsample", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], BASE)
plot_sweep(df_sub, "subsample", OUT_DIR / "gbr_subsample.png")

for name, d in [("n_estimators", df_n), ("max_depth", df_depth), ("learning_rate", df_lr), ("min_samples_leaf", df_leaf), ("subsample", df_sub)]:
    d.to_csv(OUT_DIR / f"gbr_{name}.csv", index=False)