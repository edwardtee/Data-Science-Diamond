import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.svm import SVR

# Navigate up 2 levels from testing.py to find the project root directory
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing.preprocessing import load_test_split

X_train, X_test, y_train, y_test = load_test_split()

# --- Train ExtraTreesRegressor Model --- 
base_models = [
    ("etr", ExtraTreesRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_split=40,
        min_samples_leaf=2,
        random_state=42
    )),
    ("gbr", GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=64,
        subsample=1.0,
        loss="squared_error",
        random_state=42
    )),
    ("svr",SVR(
        kernel="rbf",
        C=1000,
        gamma="scale",
        epsilon=500
    ))
]

stack = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression(),
    cv=5,
    n_jobs=-1
)

stack.fit(X_train, y_train)

pred = stack.predict(X_test)
# --- Measure Training Time ---
start_train = time.perf_counter()
stack.fit(X_train, y_train)
end_train = time.perf_counter()
training_time = end_train - start_train

# --- Measure Prediction Time ---
start_pred = time.perf_counter()
y_pred = stack.predict(X_test)
end_pred = time.perf_counter()
prediction_time = end_pred - start_pred

# Print Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² :", r2)
print("MAE :", mae)
print("RMSE:", rmse)
print(f"Total Execution Time: {training_time + prediction_time:.4f} seconds")