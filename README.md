# 💎 Diamond Price Prediction

A machine learning project that predicts diamond prices from a handful of physical measurements (carat, length, width, depth). It trains and compares four regression models — Linear Regression, Support Vector Regression, Extra Trees, and Gradient Boosting — plus a stacked hybrid model, and serves live predictions through a Streamlit app.

## Models included

| Model | Training script | Saved as |
|---|---|---|
| Linear Regression | `linear_regression.py` | `lr_model.joblib` |
| Support Vector Regression | `support_vector_regression.py` | `svr_model.joblib` |
| Extra Trees Regression | `extra_trees_regression.py` | `etr_model.joblib` |
| Gradient Boosting Regression | `gradient_boosting_regression.py` | `gbr_model.joblib` |
| Hybrid (Stacked: ETR + GBR + SVR → Linear) | `hybrid_model.py` | not saved (see notes) |

Each `tune_*.py` script sweeps hyperparameters for its model, prints results, and saves CSVs + PNG charts to `output/`. The `FINAL_PARAMS` in each training script reflect the best values found during tuning.

## Project structure

This is the layout the code expects. The exact folder names for the training/tuning scripts (`models/`, `tuning/`) aren't fixed by anything in the code — but they **do** need to sit exactly two folders below the project root, since every script finds the root with `Path(__file__).resolve().parents[2]`. If you place a script one level shallower or deeper, the root-finding (and therefore the `data/` and `src/` imports) will break.

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

## Requirements

Python 3.9+ is recommended. Install the packages used across the project:

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn
```

## Setup

1. Clone/download the project and place the files into the structure above.
2. Put `Diamonds Prices2022.csv` in `data/`. `preprocessing.py` expects an
   `Unnamed: 0` index column in the CSV (it drops it automatically) — that's the
   standard artifact of a CSV exported with `df.to_csv()` without `index=False`.
3. Install the dependencies (above).

## Training a model

Run any training script directly — since path resolution is based on `__file__`,
it works whether you invoke it as a plain script or as a module:

```bash
python src/models/linear_regression.py
```

Each script loads the cleaned train/test split, fits the model, prints
R² / MAE / RMSE / fit time, and saves the trained model to `src/saved_models/`.

## Tuning a model

```bash
python src/tuning/tune_linear_regression.py
```

Tuning scripts run sweeps over one hyperparameter at a time, save a CSV + PNG
chart per sweep to `output/<model_name>/`, and print the best value found.

## Running the Streamlit app

From the project root:

```bash
streamlit run app.py
```

The app auto-discovers any `*_model.joblib` file in `src/saved_models/` and lists
it as a selectable model, loads the shared scaler, and updates the predicted price
live as you move the carat/length/width/depth sliders.

## Notes

- Both `load_test_split()` and `load_full_split()` remove outliers via IQR on
  `carat`, `depth`, `table`, `x`, `y`, `z` before splitting.
- `load_test_split()` returns just the four numeric features (`carat`, `x`, `y`,
  `z`) — this is what all the training/tuning scripts and the app use.
  `load_full_split()` one-hot encodes `cut`, `color`, and `clarity` too, but
  nothing currently consumes it.
- The hybrid stacking model fits twice in `hybrid_model.py` (once before, once
  during timing) — harmless but redundant; feel free to remove the first `.fit()`
  call if you want faster runs.
