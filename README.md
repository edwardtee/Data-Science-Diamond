# 💎 Diamond Price Prediction

A machine learning project that predicts diamond prices from a handful of physical measurements (carat, length, width, depth). It trains and compares four regression models — Linear Regression, Support Vector Regression, Extra Trees, and Gradient Boosting, and serves live predictions through a Streamlit app.

## Requirements

Python 3.9+ is recommended. Install the packages used across the project:

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn
```

## Running the Streamlit app

From the project root:

```bash
streamlit run app.py
```

---

## Models included

| Model | Training script | Saved as |
|---|---|---|
| Linear Regression | `linear_regression.py` | `lr_model.joblib` |
| Support Vector Regression | `support_vector_regression.py` | `svr_model.joblib` |
| Extra Trees Regression | `extra_trees_regression.py` | `etr_model.joblib` |
| Gradient Boosting Regression | `gradient_boosting_regression.py` | `gbr_model.joblib` |

Each `tune_*.py` script sweeps hyperparameters for its model, prints results, and saves CSVs + PNG charts to `output/`. The `FINAL_PARAMS` in each training script reflect the best values found during tuning.

## Project structure

```
project-root/
├── app.py                          # Streamlit app — run from here
├── data/
│   └── Diamonds Prices2022.csv     # raw dataset
├── output/                         # auto-created by tune_*.py scripts
│   ├── linear_regression/
│   ├── support_vector_regression/
│   ├── extra_trees_regression/
│   └── gradient_boosting_regression/
├── src/
│   ├── preprocessing/
│   │   └── preprocessing.py        # shared load_test_split() / load_full_split()
│   ├── models/                     # models with feature selection
│   │   ├── linear_regression.py
│   │   ├── support_vector_regression.py
│   │   ├── extra_trees_regression.py
│   │   ├── gradient_boosting_regression.py
│   │   └── hybrid_model.py
│   ├── full_models/                # models without feature selection
│   │   ├── full_linear_regression.py
│   │   ├── full_support_vector_regression.py
│   │   ├── full_extra_trees_regression.py
│   │   └── full_gradient_boosting_regression.py
│   ├── tuning/                     # tuning files to find best hyperparameters
│   │   ├── tune_linear_regression.py
│   │   ├── tune_support_vector_regression.py
│   │   ├── tune_extra_tree_regression.py
│   │   └── tune_gradient_boosting_regression.py
│   └── saved_models/               # auto-created — holds .joblib files + metrics.json
│       ├── lr_model.joblib
│       ├── svr_model.joblib
│       ├── etr_model.joblib
│       ├── gbr_model.joblib
│       └── metrics.json
└── README.md
```
