import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Analyser",
    page_icon="🩺",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .title-block {
    text-align: center;
    padding: 2rem 0 1rem;
  }
  .title-block h1 {
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #f0f0f0;
    margin: 0;
  }
  .title-block p {
    color: #888;
    font-size: 0.9rem;
    margin-top: 0.4rem;
  }

  .section-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #555;
    margin: 1.6rem 0 0.6rem;
  }

  .result-safe {
    background: linear-gradient(135deg, #0d2b1a, #0a1f13);
    border: 1px solid #1a5c30;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
  }
  .result-risk {
    background: linear-gradient(135deg, #2b0d0d, #1f0a0a);
    border: 1px solid #5c1a1a;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
  }
  .result-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
  }
  .result-verdict {
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: -0.02em;
  }
  .result-prob {
    font-size: 0.9rem;
    margin-top: 0.4rem;
    opacity: 0.7;
  }

  .prob-bar-wrap {
    background: #1a1a2e;
    border-radius: 8px;
    height: 10px;
    margin: 0.8rem 0;
    overflow: hidden;
  }
  .prob-bar-fill-safe { background: #22c55e; height: 100%; border-radius: 8px; }
  .prob-bar-fill-risk { background: #ef4444; height: 100%; border-radius: 8px; }

  .factor-box {
    background: #16161e;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: #aaa;
  }
  .factor-box span { color: #f0f0f0; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    for path in ['Model/diabetes_model.pkl', 'diabetes_model.pkl']:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError("Could not find diabetes_model.pkl")

@st.cache_resource
def load_scaler():
    for path in ['Model/diabetes_scaler.pkl', 'diabetes_scaler.pkl']:
        if os.path.exists(path):
            obj = joblib.load(path)
            print(f"Loaded scaler type: {type(obj)}")  # check terminal output
            return obj
    return None

try:
    model  = load_model()
    scaler = load_scaler()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.info("Make sure diabetes_model.pkl is in a Model/ folder next to this file.")
    st.stop()

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <h1>🩺 Diabetes Risk Analyser</h1>
  <p>Enter patient details to assess diabetes risk</p>
  <p>The model is trained on Pima Indians Diabetes dataset with female patients</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Input form ─────────────────────────────────────────────────
st.markdown('<div class="section-label">Personal info</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age",
                           min_value=21, max_value=100, value=30,
                           help="Patient age in years (dataset covers ages 21+)")
with col2:
    pregnancies = st.number_input("Number of pregnancies",
                                   min_value=0, max_value=20, value=1,
                                   help="Number of times pregnant")

st.markdown('<div class="section-label">Blood measurements</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    glucose = st.number_input("Glucose level (mg/dL)",
                               min_value=50, max_value=250, value=110,
                               help="Plasma glucose concentration from 2-hour oral glucose tolerance test. Normal: 70–140")
with col4:
    insulin = st.number_input("Insulin level (µU/mL)",
                               min_value=0, max_value=900, value=80,
                               help="2-hour serum insulin. Normal fasting: 2–25. Leave at 0 if unknown.")

col5, col6 = st.columns(2)
with col5:
    blood_pressure = st.number_input("Diastolic blood pressure (mmHg)",
                                      min_value=30, max_value=140, value=72,
                                      help="The lower number in a blood pressure reading. Normal: 60–80")
with col6:
    skin_thickness = st.number_input("Skin fold thickness (mm)",
                                      min_value=5, max_value=100, value=23,
                                      help="Triceps skin fold thickness. Leave at 20 if unknown.")

st.markdown('<div class="section-label">Body measurements</div>', unsafe_allow_html=True)

col7, col8 = st.columns(2)
with col7:
    bmi = st.number_input("BMI (kg/m²)",
                           min_value=10.0, max_value=70.0, value=28.0, step=0.1,
                           help="Body mass index = weight(kg) / height(m)². Healthy: 18.5–24.9, Overweight: 25–29.9, Obese: 30+")
with col8:
    dpf = st.slider("Diabetes pedigree function",
                     min_value=0.05, max_value=2.5, value=0.35, step=0.01,
                     help="Genetic likelihood score based on family history. Higher = stronger family history of diabetes.")

# ── Derived features (must match training exactly) ──────────────
glucose_bmi = glucose * bmi
age_risk    = 1 if age >= 45 else 0

st.divider()

# ── Predict ────────────────────────────────────────────────────
if st.button("Analyse diabetes risk →", use_container_width=True, type="primary"):

    input_data = pd.DataFrame([{
        'Pregnancies':               pregnancies,
        'Glucose':                   glucose,
        'BloodPressure':             blood_pressure,
        'SkinThickness':             skin_thickness,
        'Insulin':                   insulin,
        'BMI':                       bmi,
        'DiabetesPedigreeFunction':  dpf,
        'Age':                       age,
        'glucose_bmi':               glucose_bmi,
        'age_risk':                  age_risk
    }])

    # Scale if scaler is available (Logistic Regression needs it)
    if scaler is not None:
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0][1]
    else:
        prediction   = model.predict(input_data)[0]
        probability  = model.predict_proba(input_data)[0][1]

    pct = f"{probability * 100:.1f}"

    if prediction == 0:
        st.markdown(f"""
        <div class="result-safe">
          <div class="result-label" style="color:#4ade80">✓ Low Risk</div>
          <div class="result-verdict" style="color:#4ade80">UNLIKELY DIABETIC</div>
          <div class="result-prob">{pct}% probability of diabetes</div>
        </div>
        <div class="prob-bar-wrap">
          <div class="prob-bar-fill-safe" style="width:{pct}%"></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-risk">
          <div class="result-label" style="color:#f87171">⚠ High Risk</div>
          <div class="result-verdict" style="color:#f87171">LIKELY DIABETIC</div>
          <div class="result-prob">{pct}% probability of diabetes</div>
        </div>
        <div class="prob-bar-wrap">
          <div class="prob-bar-fill-risk" style="width:{pct}%"></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Key factors ─────────────────────────────────────────────
    st.markdown('<div class="section-label">Key factors used</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f'<div class="factor-box">Glucose level <span>({glucose} mg/dL)</span> — strongest predictor</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="factor-box">BMI <span>({bmi:.1f} kg/m²)</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="factor-box">Glucose × BMI score <span>({glucose_bmi:.0f})</span></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown(f'<div class="factor-box">Age risk flag <span>({"Yes — 45+" if age_risk else "No — under 45"})</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="factor-box">Diabetes pedigree <span>({dpf:.2f})</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="factor-box">Age <span>({age} years)</span></div>', unsafe_allow_html=True)