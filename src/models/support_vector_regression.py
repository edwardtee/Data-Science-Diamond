import sys
import time
from pathlib import Path

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.svm import SVR

# Navigate up 2 levels from testing.py to find the project root directory
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.preprocessing.preprocessing import load_split

X_train, X_test, y_train, y_test = load_split()

FINAL_PARAMS = {
    "kernel": "rbf",
    "C": 20000,
    "epsilon": 5,
    "gamma": 2
}

final_model = SVR(**FINAL_PARAMS)
t0 = time.time()
final_model.fit(X_train, y_train)
fit_time = time.time() - t0
pred = final_model.predict(X_test)

print("\n=== Support Vector Regression - Results ===")
print("Config:", FINAL_PARAMS)
print("Fit time:", round(fit_time, 6), "s")
print("R2:  ", round(r2_score(y_test, pred), 6))
print("MAE: ", round(mean_absolute_error(y_test, pred), 6))
print("RMSE:", round(root_mean_squared_error(y_test, pred), 6))