import pandas as pd
import numpy as np
import os
import time

# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Dataset
file_path = os.path.join(os.path.dirname(__file__), "Diamonds Prices2022.csv")
data = pd.read_csv(file_path)

if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])


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
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "Time (s)": round(exec_time, 4)
    }

# 6. Model Tuning 
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


# 7. Final Model & Visualizations
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

# Actual vs Predicted Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_test_pred, alpha=0.5)

lims = [y_test.min(), y_test.max()]

# 45-degree reference: where perfect predictions would lie
plt.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction (slope = 1)')

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Diamond Price (Testing Set)")
plt.legend()
plt.show()

# Actual vs Predicted Plot (Full with all feature)
plt.figure(figsize=(8,6))
plt.scatter(yF_test, y_test_pred_full, alpha=0.5)

lims = [yF_test.min(), yF_test.max()]

# 45-degree reference: where perfect predictions would lie
plt.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction (slope = 1)')

print("\n--- Extended Model (all features) ---")
print("Features used:", X_full.shape[1])
print("R2  :", round(r2_score(yF_test, y_test_pred_full), 4))
print("MAE :", round(mean_absolute_error(yF_test, y_test_pred_full), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(yF_test, y_test_pred_full)), 2))

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Diamond Price (Full feature)")
plt.legend()
plt.show()

# Residual Plot
residuals = y_test - y_test_pred
plt.figure(figsize=(8,6))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Plot (Testing Set)")
plt.show()

# Residual Distribution
plt.figure(figsize=(8,5))
sns.histplot(residuals, kde=True)
plt.xlabel("Residual")
plt.title("Residual Distribution")
plt.show()