import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
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
    base_params["n_jobs"] = 1  # avoid nested parallelism fighting GridSearchCV's n_jobs
    model = ExtraTreesRegressor(**base_params)

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
            "Time(s)": cv_results["mean_fit_time"][i],
        })
        print(f"  {param_name}={v}: R2={rows[-1]['R2']:.4f} MAE={rows[-1]['MAE']:.2f} "
              f"RMSE={rows[-1]['RMSE']:.2f} avg_fit_time={rows[-1]['Time(s)']:.3f}s")
    print(f"  -> GridSearchCV best {param_name} = {grid.best_params_[param_name]} "
          f"(R2={grid.best_score_:.4f})")
    return pd.DataFrame(rows)

def plot_sweep(df, param_name, out_path, xlabel=None, logx=False):
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    # for mixed-type params (e.g. max_depth with None, max_features with strings)
    # use categorical positions on the x-axis instead of numeric plotting
    x_labels = [str(v) for v in df[param_name]]
    x = range(len(x_labels))

    ax1.plot(x, df["R2"], marker="o", color="#2563eb", label="R2 (mean CV)")
    ax1.set_xlabel(xlabel or param_name)
    ax1.set_ylabel("R2 score", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(x_labels, rotation=45, ha="right")
    if logx:
        ax1.set_xscale("log")  # only meaningful for purely numeric params

    ax2 = ax1.twinx()
    ax2.plot(x, df["Time(s)"], marker="s", color="#dc2626", linestyle="--", label="Avg fit time (s)")
    ax2.set_ylabel("Avg fit time per model (s)", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")

    plt.title(f"Effect of {param_name} on R2 and Fit Time")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  saved", out_path)

BASE = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
}
OUT_DIR = ROOT_DIR / "output" / "extra_trees_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n--- n_estimators ---")
df_n = sweep("n_estimators", [50, 100, 200, 300, 400, 500], BASE)
plot_sweep(df_n, "n_estimators", OUT_DIR / "etr_n_estimators.png")
BASE["n_estimators"] = int(df_n.loc[df_n["R2"].idxmax(), "n_estimators"])

print("\n--- max_depth ---")
df_depth = sweep("max_depth", [None, 10, 20, 30, 40, 50, 60, 70], BASE)
plot_sweep(df_depth, "max_depth", OUT_DIR / "etr_max_depth.png")
best_depth = df_depth.loc[df_depth["R2"].idxmax(), "max_depth"]
BASE["max_depth"] = None if pd.isna(best_depth) else int(best_depth)

print("\n--- min_samples_split ---")
df_split = sweep("min_samples_split", [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], BASE)
plot_sweep(df_split, "min_samples_split", OUT_DIR / "etr_min_samples_split.png")
BASE["min_samples_split"] = int(df_split.loc[df_split["R2"].idxmax(), "min_samples_split"])

print("\n--- min_samples_leaf ---")
df_leaf = sweep("min_samples_leaf", [1, 2, 4, 6, 8, 10], BASE)
plot_sweep(df_leaf, "min_samples_leaf", OUT_DIR / "etr_min_samples_leaf.png")
BASE["min_samples_leaf"] = int(df_leaf.loc[df_leaf["R2"].idxmax(), "min_samples_leaf"])

print("\n--- max_features ---")
df_feat = sweep("max_features", [1.0, 0.7, 0.5, 'sqrt', 'log2'], BASE)
plot_sweep(df_feat, "max_features", OUT_DIR / "etr_max_features.png")
best_feat = df_feat.loc[df_feat["R2"].idxmax(), "max_features"]
BASE["max_features"] = best_feat if isinstance(best_feat, str) else float(best_feat)

print("\nFinal chosen params:", BASE)

for name, d in [
    ("n_estimators", df_n),
    ("max_depth", df_depth),
    ("min_samples_split", df_split),
    ("min_samples_leaf", df_leaf),
    ("max_features", df_feat),
]:
    d.to_csv(OUT_DIR / f"etr_{name}.csv", index=False)