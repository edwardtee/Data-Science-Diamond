import streamlit as st
import joblib
import pandas as pd

# ----------------------------
# Load Model
# ----------------------------
try:
    model = joblib.load("ExtraTreeR_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Model or scaler file not found.")
    st.stop()

st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered"
)

st.title("💎 Diamond Price Prediction System")

st.write(
    """
    Move the sliders below to describe the diamond's characteristics.
    The trained Extra Trees Regressor will update its price estimate live
    as you adjust the values — no button needed.
    """
)

# Training ranges — used both for slider bounds and for display
CARAT_MIN, CARAT_MAX = 0.20, 2.00
X_MIN, X_MAX = 3.70, 8.30
Y_MIN, Y_MAX = 3.68, 8.30
Z_MIN, Z_MAX = 1.40, 5.30

with st.expander("Valid Input Range"):
    st.write(f"""
    Sliders are already constrained to the range used to train the model.

    - Carat: **{CARAT_MIN} – {CARAT_MAX}**
    - Length (x): **{X_MIN} – {X_MAX} mm**
    - Width (y): **{Y_MIN} – {Y_MAX} mm**
    - Depth (z): **{Z_MIN} – {Z_MAX} mm**
    """)

# ----------------------------
# User Inputs (sliders)
# ----------------------------
# st.slider enforces min/max by construction, so there's no way to enter an
# out-of-range value the way a text/number box would allow mid-typing.
# Because these live outside any st.form, moving any slider triggers an
# immediate rerun — which is what gives us the "live" prediction below.
carat = st.slider(
    "Carat",
    min_value=CARAT_MIN,
    max_value=CARAT_MAX,
    value=1.00,
    step=0.01
)

x = st.slider(
    "Length (x) mm",
    min_value=X_MIN,
    max_value=X_MAX,
    value=5.50,
    step=0.01
)

y = st.slider(
    "Width (y) mm",
    min_value=Y_MIN,
    max_value=Y_MAX,
    value=5.50,
    step=0.01
)

z = st.slider(
    "Depth (z) mm",
    min_value=Z_MIN,
    max_value=Z_MAX,
    value=3.50,
    step=0.01
)

# ----------------------------
# Logical (cross-field) checks
# ----------------------------
# These aren't hard range limits, so a slider can't prevent them — but since
# there's no submit button to gate on anymore, we surface them as warnings
# rather than stopping execution, so the live prediction still updates.
warnings = []

if z > x:
    warnings.append("Depth (z) is currently greater than Length (x) — that's unusual for a real diamond.")

if z > y:
    warnings.append("Depth (z) is currently greater than Width (y) — that's unusual for a real diamond.")

for msg in warnings:
    st.warning(msg)

# ----------------------------
# Live Prediction
# ----------------------------
input_data = pd.DataFrame({
    "carat": [carat],
    "x": [x],
    "y": [y],
    "z": [z]
})

try:
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    price = prediction[0]

    st.success(f"Estimated Diamond Price: ${price:,.2f}")

except Exception as e:
    st.error(f"Prediction failed: {e}")

#py -m streamlit run UI.py