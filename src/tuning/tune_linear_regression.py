import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Navigate up 2 levels from testing.py to find the project root directory
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.preprocessing.preprocessing import load_test_split

X_train, X_test, y_train, y_test = load_test_split()

# 1. Evaluation Function
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    start_time = time.time()
    model.fit(X_train, y_train)
    exec_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return {
        "Model": model_name,
        "R2": round(r2, 4),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "Time (s)": round(exec_time, 4)
    }

# 2. Model Tuning 
results = []

# A. Standard Linear Regression (Intercept True vs False)
ols_true = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression(fit_intercept=True))])
ols_false = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression(fit_intercept=False))])

results.append(evaluate_model(ols_true, X_train, y_train, X_test, y_test, "OLS (fit_intercept=True)"))
results.append(evaluate_model(ols_false, X_train, y_train, X_test, y_test, "OLS (fit_intercept=False)"))

# B. Ridge Regression Sweep
alphas_ridge = [0.01, 0.1, 1, 10, 50, 100]
for alpha in alphas_ridge:
    ridge_model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha))])
    results.append(evaluate_model(ridge_model, X_train, y_train, X_test, y_test, f"Ridge (alpha={alpha})"))

# C. Lasso Regression Sweep
alphas_lasso = [0.001, 0.01, 0.1, 1, 10, 50]
for alpha in alphas_lasso:
    lasso_model = Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(alpha=alpha, max_iter=10000))])
    results.append(evaluate_model(lasso_model, X_train, y_train, X_test, y_test, f"Lasso (alpha={alpha})"))

tuning_results = pd.DataFrame(results)
print("\n--- Tuning Results ---")
print(tuning_results.to_string(index=False))


# 3. Final Model & Visualizations
final_model = ols_true
final_model.fit(X_train, y_train)
y_test_pred = final_model.predict(X_test)

# Overfitting check by comparing training and testing performance
y_train_pred = final_model.predict(X_train)
linear_step = final_model.named_steps['lr']
gap = pd.DataFrame([
    {"Set": "Training",
     "R2": round(r2_score(y_train, y_train_pred), 4),
     "MAE": round(mean_absolute_error(y_train, y_train_pred), 2),
     "RMSE": round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 2)},
    {"Set": "Testing",
     "R2": round(r2_score(y_test, y_test_pred), 4),
     "MAE": round(mean_absolute_error(y_test, y_test_pred), 2),
     "RMSE": round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2)},
])

print("\n--- Overfitting Check ---")
print(gap.to_string(index=False))
print("R2 gap (train - test):", round(r2_score(y_train, y_train_pred) - r2_score(y_test, y_test_pred), 4))

print("\nIntercept (beta_0):", round(linear_step.intercept_, 2))
print("Mean training price:", round(y_train.mean(), 2))


OUT_DIR = root_dir / "output" / "linear_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Actual vs Predicted Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_test_pred, alpha=0.5)

lims = [y_test.min(), y_test.max()]

# 45-degree reference: where perfect predictions would lie
plt.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction (slope = 1)')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Diamond Price (Testing Set)")
plt.savefig(OUT_DIR / "lr_actual_vs_predicted_test.png", dpi=150)
plt.close()

# Residual Plot
residuals = y_test - y_test_pred
plt.figure(figsize=(8,6))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Plot (Testing Set)")
plt.savefig(OUT_DIR / "lr_residual_plot.png", dpi=150)
plt.close()

# Residual Distribution
plt.figure(figsize=(8,5))
sns.histplot(residuals, kde=True)
plt.xlabel("Residual")
plt.title("Residual Distribution")
plt.savefig(OUT_DIR / "lr_residual_distribution.png", dpi=150)
plt.close()