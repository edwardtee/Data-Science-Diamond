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
    Enter the diamond characteristics below.
    The trained Extra Trees Regressor will estimate the diamond price.
    """
)

# Training ranges — used both for display and for server-side validation
CARAT_MIN, CARAT_MAX = 0.10, 10.00
X_MIN, X_MAX = 0.00, 20.00
Y_MIN, Y_MAX = 0.00, 20.00
Z_MIN, Z_MAX = 0.00, 20.00

with st.expander("Valid Input Range"):
    st.write(f"""
    Please enter values within the range used to train the model.

    - Carat: **{CARAT_MIN} – {CARAT_MAX}**
    - Length (x): **{X_MIN} – {X_MAX} mm**
    - Width (y): **{Y_MIN} – {Y_MAX} mm**
    - Depth (z): **{Z_MIN} – {Z_MAX} mm**
    """)

# ----------------------------
# User Inputs (inside a form)
# ----------------------------
# Using st.form means none of the widgets below trigger a rerun or update
# their values in session_state until the user clicks the submit button.
# That closes the gap where a half-typed / out-of-range value (e.g. "-5"
# still showing in the box) could be used by the script before it's been
# corrected or committed.
with st.form("prediction_form"):

    carat = st.number_input(
        "Carat",
        min_value=CARAT_MIN,
        max_value=CARAT_MAX,
        value=1.00,
        step=0.01
    )

    x = st.number_input(
        "Length (x)",
        min_value=X_MIN,
        max_value=X_MAX,
        value=5.50,
        step=0.01
    )

    y = st.number_input(
        "Width (y)",
        min_value=Y_MIN,
        max_value=Y_MAX,
        value=5.50,
        step=0.01
    )

    z = st.number_input(
        "Depth (z)",
        min_value=Z_MIN,
        max_value=Z_MAX,
        value=3.50,
        step=0.01
    )

    submitted = st.form_submit_button("Predict Price")

# ----------------------------
# Prediction
# ----------------------------
if submitted:

    error_message = []

    # Explicit server-side range validation.
    # Even with min_value/max_value set above, Streamlit's widget bounds are
    # only a UI hint — they narrow what you *can* select, but don't replace
    # a real server-side check. We re-validate here as the source of truth.
    if not (CARAT_MIN <= carat <= CARAT_MAX):
        error_message.append(f"Carat must be between {CARAT_MIN} and {CARAT_MAX}.")

    if not (X_MIN <= x <= X_MAX):
        error_message.append(f"Length (x) must be between {X_MIN} and {X_MAX} mm.")

    if not (Y_MIN <= y <= Y_MAX):
        error_message.append(f"Width (y) must be between {Y_MIN} and {Y_MAX} mm.")

    if not (Z_MIN <= z <= Z_MAX):
        error_message.append(f"Depth (z) must be between {Z_MIN} and {Z_MAX} mm.")

    # Logical checks
    if z > x:
        error_message.append("Depth (z) cannot be greater than Length (x).")

    if z > y:
        error_message.append("Depth (z) cannot be greater than Width (y).")

    if error_message:
        for msg in error_message:
            st.error(msg)
        st.stop()

    # Prediction
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

        st.balloons()

    except Exception as e:
        st.error(f"Prediction failed: {e}")

#py -m streamlit run UI.py