import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# Navigate up 2 levels from testing.py to find the project root directory
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing.preprocessing import load_full_split

X_train, X_test, y_train, y_test = load_full_split()

FINAL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "min_samples_leaf": 64,
    "subsample": 1.0,
    "random_state": 42,
}

final_model = GradientBoostingRegressor(**FINAL_PARAMS)
t0 = time.time()
final_model.fit(X_train, y_train)
fit_time = time.time() - t0
y_pred = final_model.predict(X_test)

print("\n=== Gradient Boosting Regression - Results ===")
print("Config:", FINAL_PARAMS)
print("Fit time:", round(fit_time, 6), "s")
print("R2:  ", round(r2_score(y_test, y_pred), 6))
print("MAE: ", round(mean_absolute_error(y_test, y_pred), 6))
print("RMSE:", round(root_mean_squared_error(y_test, y_pred), 6))

OUT_DIR = ROOT_DIR / "src" / "full_models" / "scatterplots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.4, s=15, color="royalblue", label="Predicted vs Actual")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Perfect Prediction (y = x)")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (Gradient Boosting Regression)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "full_gbr_actual_predicted.png", dpi=300)