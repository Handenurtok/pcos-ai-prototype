# app.py - PCOS AI Clinical Decision Support System

import streamlit as st
import joblib
import numpy as np

# Load model and features (Model ve özellikleri yükle)
model = joblib.load("pcos_model.pkl")
features = joblib.load("pcos_features.pkl")

st.set_page_config(page_title="PCOS AI Assistant", layout="centered")
st.title("🩺 PCOS AI - Clinical Decision Support System")
st.markdown("Enter patient data below to get a PCOS risk prediction. *(Hasta verilerini girerek PCOS risk tahmini alın.)*")

st.divider()

# --- Input Form (Giriş Formu) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔬 Clinical Data")
    follicle_r = st.number_input("Follicle No. Right (Sağ folikül sayısı)", 0, 25, 4)
    follicle_l = st.number_input("Follicle No. Left (Sol folikül sayısı)", 0, 25, 4)
    amh = st.number_input("AMH ng/mL", 0.0, 25.0, 2.0)
    lh = st.number_input("LH mIU/mL", 0.0, 50.0, 5.0)
    fsh_lh = st.number_input("FSH/LH Ratio", 0.0, 10.0, 1.0)

with col2:
    st.subheader("📋 Symptoms & Cycle")
    cycle = st.selectbox("Cycle (Adet Düzeni)", [2, 4], format_func=lambda x: "Regular (Düzenli)" if x == 2 else "Irregular (Düzensiz)")
    cycle_len = st.number_input("Cycle length days (Adet süresi gün)", 1, 60, 28)
    skin_dark = st.selectbox("Skin darkening (Cilt koyulaşması)", [0, 1], format_func=lambda x: "No (Hayır)" if x == 0 else "Yes (Evet)")
    weight_gain = st.selectbox("Weight gain (Kilo alımı)", [0, 1], format_func=lambda x: "No (Hayır)" if x == 0 else "Yes (Evet)")
    hair_growth = st.selectbox("Hair growth (Tüylenme)", [0, 1], format_func=lambda x: "No (Hayır)" if x == 0 else "Yes (Evet)")

st.divider()

# --- Prediction (Tahmin) ---
if st.button("🔍 Predict PCOS Risk (PCOS Riskini Tahmin Et)", type="primary", use_container_width=True):
    input_data = np.array([[follicle_r, follicle_l, skin_dark, weight_gain,
                            hair_growth, amh, cycle, lh, fsh_lh, cycle_len]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK - PCOS Detected (PCOS Riski Yüksek) — Confidence: {probability[1]*100:.1f}%")
        st.markdown("**Recommendation (Öneri):** Consult an endocrinologist for further evaluation. *(Detaylı değerlendirme için endokrinoloji uzmanına başvurun.)*")
    else:
        st.success(f"✅ LOW RISK - No PCOS Detected (PCOS Riski Düşük) — Confidence: {probability[0]*100:.1f}%")
        st.markdown("**Recommendation (Öneri):** Continue regular health check-ups. *(Düzenli sağlık kontrollerinize devam edin.)*")

    st.caption("⚕️ Disclaimer: This is a decision-support tool, not a medical diagnosis. Always consult a healthcare professional. *(Bu bir karar destek aracıdır, tıbbi teşhis değildir. Her zaman bir sağlık uzmanına danışın.)*")
