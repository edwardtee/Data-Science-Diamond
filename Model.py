import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
import joblib
import seaborn as sns
import time
import pandas as pd
import os

file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "Diamonds Prices2022.csv"
)

data = pd.read_csv(file_path)
#data = pd.read_csv("D:/Data Science Diamond/Diamonds Prices2022.csv")
df1 = data.copy()
#not empty value
'''print(df1.info())
print(df1.describe())   '''
df1 = df1.drop(columns='Unnamed: 0')
#print(df1.isnull().sum())

encoder = LabelEncoder()

for col in ["cut","color","clarity"]:
    df1[col] = encoder.fit_transform(df1[col])

'''#Generate the Correlation Matrix Image
plt.figure(figsize=(10,8))

sns.heatmap(df1.corr()>0.7,
            annot=True,
            cmap="coolwarm")

plt.savefig('0.7correlation.png', dpi = 300) '''

important_features = ["carat", "x","y","z"]
numeric_features = [
    "carat","depth","table","x","y","z"
]
df_clean = df1.copy()

for col in numeric_features:

    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df_clean[(df_clean[col] < lower) |
                        (df_clean[col] > upper)]

    print(col, len(outliers))

    df_clean = df_clean[
        (df_clean[col] >= lower) &
        (df_clean[col] <= upper)
    ]

#X = df_clean.drop(columns=["price"])
X = df_clean[important_features].copy()
y = df_clean["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize numerical features
scaler = StandardScaler()

# Fit on training data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# --- Train ExtraTreesRegressor Model --- 
extra_model = ExtraTreesRegressor(
    random_state=42,
    n_estimators=400,
    max_depth=10,
    min_samples_split=40,
    min_samples_leaf=2,
    max_features=1.0
)

# --- Measure Training Time ---
start_train = time.perf_counter()
extra_model.fit(X_train_scaled, y_train)
end_train = time.perf_counter()
training_time = end_train - start_train

# --- Measure Prediction Time ---
start_pred = time.perf_counter()
y_pred = extra_model.predict(X_test_scaled)
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

'''
# Save for UI 
# Save trained model
joblib.dump(extra_model, "ExtraTreeR_model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("Model Saved Successfully!")'''
'''
# Parameter Tuning for ExtraTreesRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def tune_parameter_table(param_name, param_values, base_model, X_train, y_train):
    scoring = {
        'r2': 'r2',
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
        'rmse': make_scorer(rmse, greater_is_better=False)
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid={param_name: param_values},
        scoring=scoring,
        refit='r2',
        cv=5,
        n_jobs=-1,
        return_train_score=False
    )

    grid.fit(X_train, y_train)

    results = pd.DataFrame(grid.cv_results_)

    summary = pd.DataFrame({
        param_name: results[f'param_{param_name}'].astype(str),
        'R²': results['mean_test_r2'],
        'MAE': -results['mean_test_mae'],
        'RMSE': -results['mean_test_rmse'],
        'Execution Time (s)': results['mean_fit_time'] + results['mean_score_time']
    })

    summary = summary.sort_values(by='R²', ascending=False).reset_index(drop=True)

    return summary, grid.best_params_[param_name], grid.best_score_

model = ExtraTreesRegressor(random_state=42, n_estimators=400)

n_summary, best_n, best_r2 = tune_parameter_table(
    "max_depth",
    [None,10,20,30,40,50,60,70],
    model,
    X_train_scaled,
    y_train
)

print(n_summary)
print("Best max_depth:", best_n)
print("Best R²:", best_r2)'''

# --- Actual vs Predicted Plot with R² reference line ---
plt.figure(figsize=(8, 8))

plt.scatter(y_test, y_pred, alpha=0.4, s=15, color="royalblue", label="Predicted vs Actual")

# Perfect prediction line (y = x)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Perfect Prediction (y = x)")

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Actual vs Predicted Price")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("FullFeature_actual_vs_predicted.png", dpi=300)
