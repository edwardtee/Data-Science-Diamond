import time
from pathlib import Path

# For Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib.ticker import FormatStrFormatter

# For Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# # 1. Load Dataset
PROJECT_ROOT = Path(__file__).resolve().parents[0]
RAW_CSV = PROJECT_ROOT / "data" / "Diamonds Prices2022.csv"
data = pd.read_csv(RAW_CSV)


# 2. Data Cleaning & Preprocessing
data = data[(data[['x','y','z']] != 0).all(axis=1)]

numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

X = data[['carat', 'x', 'y', 'z']]
X_full = pd.get_dummies(
    data.drop(columns=["price"]),
    columns=["cut", "color", "clarity"],
    drop_first=True
)
y = data['price']


# 3. Split with 80% training, 20% testing and 42 random seed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
XF_train, XF_test, yF_train, yF_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

# 4. Evaluation Function
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
        "R2_raw": r2,          # unrounded, for plotting
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "Time (s)": round(exec_time, 4)
    }

# 5. Model Tuning 
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


# 6. Final Model & Visualizations
final_model = ols_true
final_model.fit(X_train, y_train)
y_test_pred = final_model.predict(X_test)

final_model_full = Pipeline([ ('scaler', StandardScaler()), ('lr', LinearRegression(fit_intercept=True))])
final_model_full.fit(XF_train, yF_train)
y_test_pred_full = final_model_full.predict(XF_test)

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
print("R2 gap (train - test):",
      round(r2_score(y_train, y_train_pred) - r2_score(y_test, y_test_pred), 4))

print("\nIntercept (beta_0):", round(linear_step.intercept_, 2))
print("Mean training price:", round(y_train.mean(), 2))

# Print Coefficients
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": linear_step.coef_})
print("\n--- Feature Coefficients ---")
print(coefficients)

# To Generate graph
OUT = PROJECT_ROOT / "output" / "linear_regression"
def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=300)

ridge_r2 = [r["R2_raw"] for r in results if r["Model"].startswith("Ridge")]
lasso_r2 = [r["R2_raw"] for r in results if r["Model"].startswith("Lasso")]
ols_r2   = results[0]["R2_raw"]

tuning_results = pd.DataFrame(results).drop(columns=["R2_raw"])

def plot_alpha_sweep(alphas, scores, colour, penalty, filename):
    plt.figure(figsize=(9, 5.5))
    plt.plot(alphas, scores, marker="o", markersize=7,
             color=colour, linewidth=2, label=f"{penalty} Regression",
             zorder=3)
    plt.axhline(y=ols_r2, color="red", linestyle="--", linewidth=2,
                label=f"Unpenalised OLS (R² = {ols_r2:.5f})", zorder=2)

    for a, s in zip(alphas, scores):
        plt.annotate(f"{s:.5f}", (a, s), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=8, color="#333333")

    plt.xscale("log")
    span = max(scores) - min(scores)
    pad  = span * 0.35 if span > 0 else 0.0001
    plt.ylim(min(scores) - pad, max(scores) + pad * 1.4)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.5f"))

    plt.xticks(alphas, [str(a) for a in alphas])
    plt.xlabel("Alpha (log scale)")
    plt.ylabel("R² on Testing Set")
    plt.title(f"Effect of {penalty} Alpha on Model Performance")
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.3, which="both")
    save(filename)


plot_alpha_sweep(alphas_ridge, ridge_r2, "royalblue", "Ridge", "lr_ridge_alpha_tuning.png")
plot_alpha_sweep(alphas_lasso, lasso_r2, "seagreen", "Lasso", "lr_lasso_alpha_tuning.png")


plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_test_pred, alpha=0.4, s=15,
            color="royalblue", label="Predicted vs Actual")
lo = min(y_test.min(), y_test_pred.min())
hi = max(y_test.max(), y_test_pred.max())
plt.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=2,
         label="Perfect Prediction (y = x)")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Diamond Price (Testing Set)")
plt.legend()
plt.grid(alpha=0.3)
save("lr_actual_vs_predicted.png")


residuals = y_test - y_test_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred, residuals, alpha=0.4, s=15,
            color="royalblue", label="Residuals")
plt.axhline(y=0, color="red", linestyle="--", linewidth=2, label="Zero Error")
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Plot (Testing Set)")
plt.legend()
plt.grid(alpha=0.3)
save("lr_residual_plot.png")


plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, color="royalblue",
             edgecolor="none", alpha=0.6)
plt.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero Error")
plt.axvline(x=residuals.mean(), color="darkorange", linewidth=2,
            label=f"Mean = {residuals.mean():.2f}")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.title("Residual Distribution")
plt.legend()
plt.grid(alpha=0.3)
save("lr_residual_distribution.png")
