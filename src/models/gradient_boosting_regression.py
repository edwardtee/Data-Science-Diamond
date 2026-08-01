import sys
import time
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# Navigate up 2 levels from testing.py to find the project root directory
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.preprocessing.preprocessing import load_split

X_train, X_test, y_train, y_test = load_split()

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
pred = final_model.predict(X_test)

print("\n=== Gradient Boosting Regression - Results ===")
print("Config:", FINAL_PARAMS)
print("Fit time:", round(fit_time, 6), "s")
print("R2:  ", round(r2_score(y_test, pred), 6))
print("MAE: ", round(mean_absolute_error(y_test, pred), 6))
print("RMSE:", round(root_mean_squared_error(y_test, pred), 6))

MODEL_DIR = root_dir / "src" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODEL_DIR / "gbr_model.joblib"
joblib.dump(final_model, model_path)
print(f"Saved model to {model_path}")