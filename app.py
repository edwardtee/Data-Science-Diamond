import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# Paths
# ============================================================
# Adjust MODEL_DIR if your saved_models folder lives elsewhere
# relative to wherever you run `streamlit run app.py` from.
MODEL_DIR = Path("src/saved_models")
SCALER_PATH = MODEL_DIR / "AllFeature_scaler.joblib"
ENCODER_PATH = MODEL_DIR / "AllFeature_label_encoders.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered"
)

st.title("💎 Diamond Price Prediction System")
st.write(
    """
    Pick a model, describe the diamond's characteristics, and get a live
    price estimate. All fields update the prediction instantly — no
    button needed.
    """
)

# ============================================================
# Discover available models
# ============================================================
# Any *_model.joblib file in saved_models is treated as a selectable model.
# The scaler file is explicitly excluded from this list.
if not MODEL_DIR.exists():
    st.error(f"Model directory not found: {MODEL_DIR.resolve()}")
    st.stop()

model_files = sorted(MODEL_DIR.glob("*_model.joblib"))

if not model_files:
    st.error(f"No '*_model.joblib' files found in {MODEL_DIR.resolve()}")
    st.stop()

# Build a friendly display name -> path mapping, e.g. "etr" -> etr_model.joblib
MODEL_OPTIONS = {p.stem.replace("_model", "").upper(): p for p in model_files}

# ============================================================
# Cached loaders (avoid re-reading disk on every slider move)
# ============================================================
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_resource
def load_scaler(path: str):
    return joblib.load(path)


@st.cache_resource
def load_encoder(path: str):
    return joblib.load(path)


@st.cache_data
def load_metrics(path: str):
    """Load precomputed R2 / MAE / RMSE per model, if available.

    Expected format (see testing.py note below for how to produce this):
    {
        "ETR": {"r2": 0.98, "mae": 250.1, "rmse": 400.3, "fit_time": 3.2},
        "GBR": {...},
        ...
    }
    """
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)


try:
    scaler = load_scaler(str(SCALER_PATH))
    encoder = load_encoder(str(ENCODER_PATH))
except FileNotFoundError as e:
    st.error(f"Required file not found: {e.filename}")
    st.stop()

metrics = load_metrics(str(METRICS_PATH))

# ============================================================
# Model selection
# ============================================================
model_choice = st.selectbox("Select a model", list(MODEL_OPTIONS.keys()))

try:
    model = load_model(str(MODEL_OPTIONS[model_choice]))
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_OPTIONS[model_choice].resolve()}")
    st.stop()

# ============================================================
# Metrics expander
# ============================================================
with st.expander("📊 Model Performance"):
    if metrics is None:
        st.info(
            f"No metrics file found at `{METRICS_PATH}`. "
            "Run the updated `testing.py` (see notes) to generate one, "
            "then re-run this app."
        )
    elif model_choice not in metrics:
        st.warning(f"No stored metrics for '{model_choice}' in metrics.json.")
    else:
        m = metrics[model_choice]
        col1, col2, col3 = st.columns(3)
        col1.metric("R²", f"{m['r2']:.4f}")
        col2.metric("MAE", f"${m['mae']:,.2f}")
        col3.metric("RMSE", f"${m['rmse']:,.2f}")
        if "fit_time" in m:
            st.caption(f"Fit time: {m['fit_time']:.4f} s")

# ============================================================
# Training ranges (numerical features) — used for slider bounds
# Pulled from your describe() output. Update these if your split differs.
# ============================================================
CARAT_MIN, CARAT_MAX = 0.20, 2.00
DEPTH_MIN, DEPTH_MAX = 59.0, 64.6
TABLE_MIN, TABLE_MAX = 51.6, 63.5
X_MIN, X_MAX = 3.70, 8.30
Y_MIN, Y_MAX = 3.68, 8.30
Z_MIN, Z_MAX = 1.40, 5.30


# ============================================================
# Categorical dropdown options
# ============================================================
# These are just the label strings shown in the dropdowns (for display and
# for building input_data). The ACTUAL numeric encoding used for prediction
# comes from AllFeature_label_encoders.joblib below — not from any manual
# mapping here — so this list only needs to contain valid category strings,
# spelled exactly as they appear in the original training data.
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]          # worst -> best (display only)
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]  # worst -> best (display only)

# ============================================================
# User inputs — all 9 features
# ============================================================
st.subheader("Diamond Characteristics")

c1, c2 = st.columns(2)

with c1:
    carat = st.slider("Carat", CARAT_MIN, CARAT_MAX, 1.00, 0.01)
    depth = st.slider("Depth %", DEPTH_MIN, DEPTH_MAX, 61.8, 0.10)
    table = st.slider("Table %", TABLE_MIN, TABLE_MAX, 57.0, 0.10)
    x = st.slider("Length (x) mm", X_MIN, X_MAX, 5.50, 0.01)
    y = st.slider("Width (y) mm", Y_MIN, Y_MAX, 5.50, 0.01)
    z = st.slider("Depth (z) mm", Z_MIN, Z_MAX, 3.50, 0.01)

with c2:
    cut = st.selectbox("Cut", CUT_ORDER, index=len(CUT_ORDER) - 1)
    color = st.selectbox("Color (D best, J worst)", COLOR_ORDER, index=COLOR_ORDER.index("G"))
    clarity = st.selectbox("Clarity (IF best, I1 worst)", CLARITY_ORDER, index=CLARITY_ORDER.index("VS2"))

# ============================================================
# Logical (cross-field) sanity checks
# ============================================================
warnings = []
if z > x:
    warnings.append("Depth (z) is currently greater than Length (x) — that's unusual for a real diamond.")
if z > y:
    warnings.append("Depth (z) is currently greater than Width (y) — that's unusual for a real diamond.")

for msg in warnings:
    st.warning(msg)

# ============================================================
# Live prediction
# ============================================================
# Column order MUST match the order used during training in preprocessing.py.
# This order follows the feature table you provided (carat, cut, color,
# clarity, depth, table, x, y, z). Adjust if your training pipeline differs.
input_data = pd.DataFrame({
    "carat": [carat],
    "cut": [cut],          # raw label, e.g. "Ideal" — encoded below
    "color": [color],      # raw label, e.g. "G"
    "clarity": [clarity],  # raw label, e.g. "VS2"
    "depth": [depth],
    "table": [table],
    "x": [x],
    "y": [y],
    "z": [z],
})


def encode_categoricals(df: pd.DataFrame, enc) -> pd.DataFrame:
    """Apply the saved per-column encoders to cut/color/clarity.

    ASSUMPTION: `enc` is a dict shaped like
    {"cut": LabelEncoder(...), "color": LabelEncoder(...), "clarity": LabelEncoder(...)}
    which is the standard output of fitting one LabelEncoder per column and
    saving them together. If AllFeature_label_encoders.joblib is structured
    differently (e.g. a single ColumnTransformer, or keyed by different
    names), update this function to match — run:
        import joblib; e = joblib.load("src/saved_models/AllFeature_label_encoders.joblib")
        print(type(e), e)
    to check its real structure.
    """
    df = df.copy()
    if not isinstance(enc, dict):
        raise TypeError(
            f"Expected a dict of per-column encoders, got {type(enc)}. "
            "Update encode_categoricals() to match the real structure of "
            "AllFeature_label_encoders.joblib."
        )
    for col in ["cut", "color", "clarity"]:
        if col not in enc:
            raise KeyError(f"No encoder found for column '{col}' in the loaded encoder dict.")
        df[col] = enc[col].transform(df[col])  # transform, NOT fit_transform
    return df


try:
    input_encoded = encode_categoricals(input_data, encoder)
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)
    price = prediction[0]
    st.success(f"Estimated Diamond Price ({model_choice}): ${price:,.2f}")
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.caption(
        "This usually means the feature columns/order, the encoder's "
        "expected structure, or the category label spelling doesn't match "
        "what preprocessing.py used for training."
    )

# py -m streamlit run app.py