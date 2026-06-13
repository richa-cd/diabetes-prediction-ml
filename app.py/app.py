import streamlit as st
import numpy as np
import joblib
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetIQ · Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #070B18 !important;
    color: #E8EEFF !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, #0D2040 0%, #070B18 60%) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Hero Banner ── */
.hero {
    text-align: center;
    padding: 3.5rem 2rem 2rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #00D4FF;
    margin-bottom: 0.9rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 700;
    line-height: 1.08;
    background: linear-gradient(135deg, #FFFFFF 30%, #C084FC 70%, #00D4FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}
.hero-sub {
    font-size: 1rem;
    color: #7A8AAF;
    max-width: 520px;
    margin: 0 auto 0.5rem;
    line-height: 1.6;
}
.hero-rule {
    width: 56px;
    height: 2px;
    background: linear-gradient(90deg, #00D4FF, #C084FC);
    margin: 1.6rem auto 0;
    border-radius: 99px;
}

/* ── Main Card ── */
.main-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 2.5rem 2.8rem;
    margin: 1.6rem auto;
    max-width: 860px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 0 60px rgba(0,212,255,0.05), 0 20px 60px rgba(0,0,0,0.4);
}

.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #00D4FF;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(0,212,255,0.2);
}

/* ── Streamlit number inputs inside the card ── */
[data-testid="stNumberInput"] label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #9BAACF !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #E8EEFF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #00D4FF !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.15) !important;
    outline: none !important;
}

/* ── Predict Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00D4FF 0%, #0099CC 50%, #C084FC 100%) !important;
    color: #030810 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 24px rgba(0,212,255,0.35) !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,212,255,0.5) !important;
    filter: brightness(1.08) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Result Cards ── */
.result-box {
    border-radius: 20px;
    padding: 2.2rem 2rem;
    text-align: center;
    margin: 1.2rem auto 0;
    max-width: 860px;
    animation: fadeSlideUp 0.5s ease forwards;
}
.result-box.danger {
    background: linear-gradient(135deg, rgba(255,71,87,0.12) 0%, rgba(192,36,75,0.08) 100%);
    border: 1px solid rgba(255,71,87,0.35);
    box-shadow: 0 0 40px rgba(255,71,87,0.1);
}
.result-box.safe {
    background: linear-gradient(135deg, rgba(0,212,255,0.10) 0%, rgba(192,132,252,0.07) 100%);
    border: 1px solid rgba(0,212,255,0.3);
    box-shadow: 0 0 40px rgba(0,212,255,0.1);
}
.result-icon { font-size: 3rem; margin-bottom: 0.6rem; }
.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.result-title.danger { color: #FF6B7A; }
.result-title.safe   { color: #00D4FF; }
.result-desc {
    font-size: 0.9rem;
    color: #7A8AAF;
    max-width: 400px;
    margin: 0 auto 1.4rem;
    line-height: 1.55;
}
.model-badges {
    display: flex;
    gap: 0.8rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 0.5rem;
}
.badge {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.35rem 0.9rem;
    border-radius: 99px;
    border: 1px solid rgba(255,255,255,0.12);
    color: #9BAACF;
    background: rgba(255,255,255,0.05);
}
.badge.active-danger { border-color: rgba(255,71,87,0.5); color: #FF6B7A; background: rgba(255,71,87,0.08); }
.badge.active-safe   { border-color: rgba(0,212,255,0.5); color: #00D4FF; background: rgba(0,212,255,0.08); }

/* ── Gauge SVG ── */
.gauge-wrap { margin: 0.2rem auto 1rem; display: flex; justify-content: center; }

/* ── Disclaimer ── */
.disclaimer {
    max-width: 860px;
    margin: 1.2rem auto 2.5rem;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
}
.disclaimer-icon { font-size: 1rem; margin-top: 0.1rem; flex-shrink: 0; color: #FFBB33; }
.disclaimer-text { font-size: 0.78rem; color: #5A6A8A; line-height: 1.55; }

/* ── Stats strip ── */
.stats-strip {
    display: flex;
    gap: 1px;
    max-width: 860px;
    margin: 0 auto 1.6rem;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07);
}
.stat-item {
    flex: 1;
    padding: 1rem 0.8rem;
    text-align: center;
    background: rgba(255,255,255,0.03);
}
.stat-item:not(:last-child) { border-right: 1px solid rgba(255,255,255,0.07); }
.stat-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #00D4FF;
    line-height: 1.1;
}
.stat-lbl {
    font-size: 0.68rem;
    color: #4A5A7A;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* ── Divider ── */
.soft-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 1.8rem 0;
}

/* ── Animation ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Fix Streamlit column gap */
[data-testid="column"] { padding: 0 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    lr_model  = joblib.load("model/lr_model.pkl")
    lda_model = joblib.load("model/lda_model.pkl")
    scaler    = joblib.load("model/scaler.pkl")
    return lr_model, lda_model, scaler

lr_model, lda_model, scaler = load_models()

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-eyebrow">AI-Powered Clinical Tool</p>
    <h1 class="hero-title">DiabetIQ<br>Risk Predictor</h1>
    <p class="hero-sub">Enter patient biomarkers below. Two ML models analyze your data in real-time and return a combined risk assessment.</p>
    <div class="hero-rule"></div>
</div>
""", unsafe_allow_html=True)

# ─── Stats Strip ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-strip">
    <div class="stat-item"><div class="stat-num">2</div><div class="stat-lbl">ML Models</div></div>
    <div class="stat-item"><div class="stat-num">8</div><div class="stat-lbl">Biomarkers</div></div>
    <div class="stat-item"><div class="stat-num">LR + LDA</div><div class="stat-lbl">Ensemble</div></div>
    <div class="stat-item"><div class="stat-num">~0.78</div><div class="stat-lbl">Accuracy</div></div>
</div>
""", unsafe_allow_html=True)

# ─── Input Card ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Patient Biomarkers</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    pregnancies    = st.number_input("Pregnancies",             min_value=0,   max_value=20,  value=1,    step=1)
    glucose        = st.number_input("Glucose (mg/dL)",         min_value=0,   max_value=300, value=110,  step=1)
    blood_pressure = st.number_input("Blood Pressure (mmHg)",   min_value=0,   max_value=200, value=72,   step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)",     min_value=0,   max_value=100, value=23,   step=1)

with col2:
    insulin = st.number_input("Insulin (μU/mL)",                min_value=0,   max_value=900, value=85,   step=1)
    bmi     = st.number_input("BMI (kg/m²)",                    min_value=0.0, max_value=70.0,value=26.5, step=0.1, format="%.1f")
    dpf     = st.number_input("Diabetes Pedigree Function",     min_value=0.0, max_value=3.0, value=0.47, step=0.01,format="%.2f")
    age     = st.number_input("Age (years)",                    min_value=1,   max_value=120, value=30,   step=1)

st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
predict_btn = st.button("⚡  Run Risk Analysis", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)  # close main-card

# ─── Prediction Logic ─────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Analyzing biomarkers…"):
        time.sleep(0.6)  # brief pause for effect

    input_data   = np.array([[pregnancies, glucose, blood_pressure,
                               skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    lr_pred  = lr_model.predict(input_scaled)[0]
    lda_pred = lda_model.predict(input_scaled)[0]
    final    = 1 if (lr_pred == 1 or lda_pred == 1) else 0

    # Risk score (simple heuristic for the gauge)
    risk_factors = sum([
        glucose > 140,
        bmi > 30,
        age > 45,
        dpf > 0.8,
        insulin > 200,
        blood_pressure > 90,
    ])
    gauge_pct = min(95, max(12, (risk_factors / 6) * 100)) if final == 1 else min(40, 10 + risk_factors * 5)
    gauge_color = "#FF4757" if final == 1 else "#00D4FF"
    circumference = 2 * 3.14159 * 54
    dash_val  = (gauge_pct / 100) * circumference
    dash_gap  = circumference - dash_val

    lr_badge  = "active-danger" if lr_pred  == 1 else "active-safe"
    lda_badge = "active-danger" if lda_pred == 1 else "active-safe"
    lr_label  = "LR → High Risk"  if lr_pred  == 1 else "LR → Low Risk"
    lda_label = "LDA → High Risk" if lda_pred == 1 else "LDA → Low Risk"

    if final == 1:
        st.markdown(f"""
        <div class="result-box danger">
            <div class="gauge-wrap">
              <svg width="140" height="140" viewBox="0 0 140 140">
                <circle cx="70" cy="70" r="54" fill="none" stroke="rgba(255,71,87,0.12)" stroke-width="10"/>
                <circle cx="70" cy="70" r="54" fill="none" stroke="{gauge_color}"
                    stroke-width="10" stroke-linecap="round"
                    stroke-dasharray="{dash_val:.1f} {dash_gap:.1f}"
                    transform="rotate(-90 70 70)" style="transition:stroke-dasharray 1s ease;"/>
                <text x="70" y="66" text-anchor="middle"
                    font-family="Space Grotesk,sans-serif" font-size="22" font-weight="700" fill="#FF6B7A">{gauge_pct:.0f}%</text>
                <text x="70" y="83" text-anchor="middle"
                    font-family="Inter,sans-serif" font-size="9" fill="#5A6A8A" letter-spacing="1">RISK SCORE</text>
              </svg>
            </div>
            <div class="result-icon">⚠️</div>
            <div class="result-title danger">Elevated Diabetes Risk</div>
            <p class="result-desc">The ensemble model flags a higher-than-baseline risk. Please consult a qualified healthcare professional for proper evaluation.</p>
            <div class="model-badges">
                <span class="badge {lr_badge}">{lr_label}</span>
                <span class="badge {lda_badge}">{lda_label}</span>
                <span class="badge active-danger">Ensemble → High Risk</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box safe">
            <div class="gauge-wrap">
              <svg width="140" height="140" viewBox="0 0 140 140">
                <circle cx="70" cy="70" r="54" fill="none" stroke="rgba(0,212,255,0.10)" stroke-width="10"/>
                <circle cx="70" cy="70" r="54" fill="none" stroke="{gauge_color}"
                    stroke-width="10" stroke-linecap="round"
                    stroke-dasharray="{dash_val:.1f} {dash_gap:.1f}"
                    transform="rotate(-90 70 70)" style="transition:stroke-dasharray 1s ease;"/>
                <text x="70" y="66" text-anchor="middle"
                    font-family="Space Grotesk,sans-serif" font-size="22" font-weight="700" fill="#00D4FF">{gauge_pct:.0f}%</text>
                <text x="70" y="83" text-anchor="middle"
                    font-family="Inter,sans-serif" font-size="9" fill="#5A6A8A" letter-spacing="1">RISK SCORE</text>
              </svg>
            </div>
            <div class="result-icon">✅</div>
            <div class="result-title safe">Low Diabetes Risk</div>
            <p class="result-desc">Both models indicate low risk based on the entered biomarkers. Maintain a healthy lifestyle and schedule regular checkups.</p>
            <div class="model-badges">
                <span class="badge {lr_badge}">{lr_label}</span>
                <span class="badge {lda_badge}">{lda_label}</span>
                <span class="badge active-safe">Ensemble → Low Risk</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Disclaimer ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    <span class="disclaimer-icon">⚕</span>
    <span class="disclaimer-text"><strong style="color:#9BAACF;">Educational use only.</strong> This tool demonstrates ML classification on the Pima Indians Diabetes Dataset. It is not a medical device and must not replace professional clinical diagnosis. Always consult a licensed physician.</span>
</div>
""", unsafe_allow_html=True)