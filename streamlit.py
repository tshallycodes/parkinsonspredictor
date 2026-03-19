import streamlit as st
import numpy as np
import joblib
import pandas as pd

from preprocessing import pad_signal, extract_features_from_signal

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Parkinson's Classifier", page_icon="🫀", layout="centered")

st.markdown("""
    <style>
        [data-testid="stMetricValue"] {
            font-size: 1rem;
            white-space: normal;
            word-break: break-word;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🫀 Parkinson's Finger Tapping Classifier")
st.caption("Predicts Parkinson's presence (Positive / Negative) from finger tapping signals")

# ─────────────────────────────────────────────────────────────
# LOAD MODELS — cached so they don't reload on every interaction
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "Left":  joblib.load("models/left_svm.pkl"),
        "Right": joblib.load("models/right_svm.pkl"),
    }

models = load_models()

# ─────────────────────────────────────────────────────────────
# SIDEBAR — inputs
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

hand = st.sidebar.selectbox("Select Hand", ["Left", "Right"])

updrs_raw = st.sidebar.selectbox(
    "UPDRS Finger Tapping Score",
    options=[0, 1, 2, 3, 4]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Signal Data**")

amplitude_input = st.sidebar.text_area("Amplitude values (comma separated)", height=120)
time_input      = st.sidebar.text_area("Time values (comma separated)",      height=120)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def parse_signal(text):
    return np.array([float(x.strip()) for x in text.split(",") if x.strip() != ""])

# ─────────────────────────────────────────────────────────────
# MAIN — prediction on button click
# ─────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True):

    if not amplitude_input or not time_input:
        st.error("Please provide both amplitude and time values.")

    else:
        amplitude = parse_signal(amplitude_input)
        time      = parse_signal(time_input)

        amplitude, time = pad_signal(amplitude, time)
        features        = extract_features_from_signal(amplitude, time)

        if features is None:
            st.error("Not enough valid taps detected. Please check your signal.")

        else:
            # Encode UPDRS as feature
            updrs_encoded      = 1 if updrs_raw >= 3 else 0
            features["updrs"]  = updrs_encoded
            feature_vector     = pd.DataFrame([features])

            model = models[hand]
            prob  = model.predict_proba(feature_vector)[0][1]

            if prob < 0.40:
                prediction = "🟢 Negative (No Parkinson's)"
            elif prob > 0.55:
                prediction = "🔴 Positive (Parkinson's)"
            else:
                prediction = "🟡 Uncertain — further evaluation needed"

            # ── Results ───────────────────────────────────────
            st.markdown("---")
            st.subheader("📊 Prediction Result")

            col1, col2 = st.columns(2)
            col1.metric("Prediction",    prediction)
            col2.metric("Hand Tested",   f"{hand} Hand")

            st.markdown("**Confidence:**")
            st.progress(float(prob), text=f"{prob:.1%} probability of Parkinson's")

            st.markdown("---")
            st.caption(f"UPDRS score {updrs_raw} encoded as {updrs_encoded}")