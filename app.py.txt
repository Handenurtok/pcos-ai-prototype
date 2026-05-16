# app.py - PCOS AI Clinical Decision Support System

import streamlit as st
import joblib
import numpy as np

# Load model and features (Model ve özellikleri yükle)
model = joblib.load("pcos_model.pkl")
features = joblib.load("pcos_features.pkl")

st.set_page_config(page_title="PCOS AI Assistant", layout="wide")
st.title("🩺 PCOS AI - Clinical Decision Support System")
st.markdown("Enter patient data below to get a PCOS risk prediction. *(Hasta verilerini girerek PCOS risk tahmini alın.)*")

st.divider()

# --- Input Form (Giriş Formu) ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Basic Info")
    age = st.number_input("Age (Yaş)", 15, 50, 25)
    weight = st.number_input("Weight in kg (Kilo)", 30.0, 150.0, 60.0)
    height = st.number_input("Height in cm (Boy)", 100.0, 200.0, 160.0)
    bmi = round(weight / ((height / 100) ** 2), 2)
    st.metric("BMI", bmi)
    blood_group = st.selectbox("Blood Group (Kan Grubu)", [11, 12, 13, 14, 15, 16, 17, 18],
                               format_func=lambda x: {11:"A+",12:"A-",13:"B+",14:"B-",15:"O+",16:"O-",17:"AB+",18:"AB-"}[x])
    pulse = st.number_input("Pulse rate bpm (Nabız)", 40, 120, 72)
    rr = st.number_input("Resp. rate breaths/min (Solunum)", 10, 40, 20)
    hb = st.number_input("Hb g/dl (Hemoglobin)", 5.0, 20.0, 12.0)

with col2:
    st.subheader("🔬 Hormones & Lab")
    cycle = st.selectbox("Cycle Regular/Irregular (Adet Düzeni)", [2, 4], format_func=lambda x: "Regular (Düzenli)" if x == 2 else "Irregular (Düzensiz)")
    cycle_len = st.number_input("Cycle length days (Adet süresi gün)", 1, 60, 28)
    marriage_yrs = st.number_input("Marriage Status Yrs (Evlilik yılı)", 0.0, 30.0, 0.0)
    pregnant = st.selectbox("Pregnant (Hamile)", [0, 1], format_func=lambda x: "No (Hayır)" if x == 0 else "Yes (Evet)")
    abortions = st.number_input("No. of abortions (Düşük sayısı)", 0, 10, 0)
    beta_hcg1 = st.number_input("I beta-HCG mIU/mL", 0.0, 2000.0, 1.99)
    beta_hcg2 = st.number_input("II beta-HCG mIU/mL", 0.0, 2000.0, 1.99)
    fsh = st.number_input("FSH mIU/mL", 0.0, 50.0, 5.0)
    lh = st.number_input("LH mIU/mL", 0.0, 50.0, 5.0)
    fsh_lh = round(fsh / lh, 2) if lh > 0 else 0
    st.metric("FSH/LH Ratio", fsh_lh)

with col3:
    st.subheader("📊 Symptoms & More")
    hip = st.number_input("Hip inch (Kalça)", 20.0, 60.0, 36.0)
    waist = st.number_input("Waist inch (Bel)", 20.0, 60.0, 30.0)
    waist_hip = round(waist / hip, 2) if hip > 0 else 0
    tsh = st.number_input("TSH mIU/L", 0.0, 30.0, 3.0)
    amh = st.number_input("AMH ng/mL", 0.0, 25.0, 2.0)
    prl = st.number_input("PRL ng/mL", 0.0, 100.0, 15.0)
    vitd3 = st.number_input("Vit D3 ng/mL", 0.0, 100.0, 25.0)
    prg = st.number_input("PRG ng/mL", 0.0, 50.0, 0.5)
    rbs = st.number_input("RBS mg/dl", 50, 300, 100)
    weight_gain = st.selectbox("Weight gain (Kilo alımı)", [0, 1], format_func=lambda x: "No (Hayır)" if x == 0 else "Yes (Evet)")
    hair_growth = st.selectbox("Hair growth (Tüylenme)", [0, 1], format_func=lambda x: "No (Hayır)" if x == 0 else "Yes (Evet)")
    skin_dark = st.selectbox("Skin darkening (Cilt koyulaşması)", [0, 1], format_func=lambda x: "No (Hayır)" if x == 0 else "Yes (Evet)")
    hair_loss = st.selectbox("Hair loss (Saç dökülmesi)", [0, 1], format_func=lambda x: "No (Hayır)" if x == 0 else "Yes (Evet)")
    pimples = st.selectbox("Pimples (Sivilce)", [0, 1], format_func=lambda x: "No (Hayır)" if x == 0 else "Yes (Evet)")
    fast_food = st.selectbox("Fast food", [0, 1], format_func=lambda x: "No (Hayır)" if x == 0 else "Yes (Evet)")
    exercise = st.selectbox("Regular Exercise (Düzenli egzersiz)", [0, 1], format_func=lambda x: "No (Hayır)" if x == 0 else "Yes (Evet)")
    bp_sys = st.number_input("BP Systolic mmHg (Tansiyon üst)", 80, 200, 120)
    bp_dia = st.number_input("BP Diastolic mmHg (Tansiyon alt)", 40, 130, 80)
    fol_l = st.number_input("Follicle No. Left (Sol folikül)", 0, 25, 4)
    fol_r = st.number_input("Follicle No. Right (Sağ folikül)", 0, 25, 4)
    avg_f_l = st.number_input("Avg Follicle size Left mm (Sol ort.)", 0.0, 30.0, 15.0)
    avg_f_r = st.number_input("Avg Follicle size Right mm (Sağ ort.)", 0.0, 30.0, 15.0)
    endo = st.number_input("Endometrium mm", 0.0, 20.0, 8.0)

st.divider()

# --- Prediction (Tahmin) ---
if st.button("🔍 Predict PCOS Risk (PCOS Riskini Tahmin Et)", type="primary", use_container_width=True):
    input_data = np.array([[age, weight, height, bmi, blood_group, pulse, rr, hb,
                            cycle, cycle_len, marriage_yrs, pregnant, abortions,
                            beta_hcg1, beta_hcg2, fsh, lh, fsh_lh, hip, waist, waist_hip,
                            tsh, amh, prl, vitd3, prg, rbs,
                            weight_gain, hair_growth, skin_dark, hair_loss, pimples,
                            fast_food, exercise, bp_sys, bp_dia,
                            fol_l, fol_r, avg_f_l, avg_f_r, endo]])

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