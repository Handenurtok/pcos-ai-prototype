"""PCOS tahmini için ana modül"""

# gerekli kütüphaneler import edildi.
import streamlit as st
import numpy as np
import joblib

# uygulamanın konfigürasyonu.
st.set_page_config(
    page_title = 'PCOS AI - Clinical Decision Support System',
    page_icon = 'random',
    layout = 'centered',
    initial_sidebar_state = 'auto'
)

# model ve özelliklerin yüklenmesi.
@st.cache_resource
def load_model():
    """Bu fonksiyon eğitilmiş modeli yükler"""
    model = joblib.load("pcos_model.pkl")
    features = joblib.load("pcos_features.pkl")
    return model, features

model, features = load_model()

# sayfa başlığı
st.title("🩺 PCOS AI - Clinical Decision Support System")
st.markdown("Enter patient data below to get a PCOS risk prediction. *(Hasta verilerini girerek PCOS risk tahmini alın.)*")

st.divider()

# kullanıcıdan giriş verilerinin alınması.
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔬 Clinical Data (Klinik Veriler)")
    follicle_r = st.number_input("Follicle No. Right (Sağ folikül sayısı)", 0, 25, 4)
    follicle_l = st.number_input("Follicle No. Left (Sol folikül sayısı)", 0, 25, 4)
    amh = st.number_input("AMH ng/mL", 0.0, 25.0, 2.0)
    lh = st.number_input("LH mIU/mL", 0.0, 50.0, 5.0)
    fsh_lh = st.number_input("FSH/LH Ratio", 0.0, 10.0, 1.0)

with col2:
    st.subheader("📋 Symptoms & Cycle (Semptomlar & Adet Döngüsü)")
    cycle = st.selectbox("Cycle (Adet Düzeni)", [2, 4],
        format_func=lambda x: "Regular (Düzenli)" if x == 2 else "Irregular (Düzensiz)")
    cycle_len = st.number_input("Cycle length days (Adet süresi gün)", 1, 60, 28)
    skin_dark = st.selectbox("Skin darkening (Cilt koyulaşması)", [0, 1],
        format_func=lambda x: "No (Hayır)" if x == 0 else "Yes (Evet)")
    weight_gain = st.selectbox("Weight gain (Kilo alımı)", [0, 1],
        format_func=lambda x: "No (Hayır)" if x == 0 else "Yes (Evet)")
    hair_growth = st.selectbox("Hair growth (Tüylenme)", [0, 1],
        format_func=lambda x: "No (Hayır)" if x == 0 else "Yes (Evet)")

st.divider()

# tahmin fonksiyonu
def predict(model, features_input):
    """Bu fonksiyon modeli kullanarak tahmin yapar"""
    prediction = model.predict(np.array(features_input).reshape(1, -1))
    probability = model.predict_proba(np.array(features_input).reshape(1, -1))
    return prediction, probability

# tahmin butonunun oluşturulması.
if st.button("🔍 Predict PCOS Risk (PCOS Riskini Tahmin Et)", type="primary", use_container_width=True):

    # kullanıcı verilerinin modele verilmesi.
    features_input = [follicle_r, follicle_l, skin_dark, weight_gain,
                      hair_growth, amh, cycle, lh, fsh_lh, cycle_len]

    prediction, probability = predict(model, features_input)
    risk_pct = probability[0][1] * 100

    st.divider()

    # sonuçların gösterilmesi.
    if prediction[0] == 1:
        st.error(f"⚠️ HIGH RISK - PCOS Detected (PCOS Riski Yüksek) — Confidence: {risk_pct:.1f}%")
        st.markdown("**Recommendation (Öneri):** Consult an endocrinologist for further evaluation. *(Detaylı değerlendirme için endokrinoloji uzmanına başvurun.)*")
    else:
        st.success(f"✅ LOW RISK - No PCOS Detected (PCOS Riski Düşük) — Confidence: {(100-risk_pct):.1f}%")
        st.markdown("**Recommendation (Öneri):** Continue regular health check-ups. *(Düzenli sağlık kontrollerinize devam edin.)*")

    # uyarı mesajı
    st.caption("⚕️ Disclaimer: This is a decision-support tool, not a medical diagnosis. Always consult a healthcare professional. *(Bu bir karar destek aracıdır, tıbbi teşhis değildir.)*")
