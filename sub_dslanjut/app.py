import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Jaya Jaya Institut - Deteksi Dropout", page_icon="🎓", layout="centered")

@st.cache_resource
def load_model():
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'model.joblib')
    return joblib.load(model_path)

model = load_model()

st.title("🎓 Sistem Prediksi Dropout Mahasiswa")
st.write("Memprediksi potensi dropout berdasarkan 9 fitur utama.")

st.markdown("---")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        grade_1st = st.number_input("Nilai Semester 1 (0-20)", min_value=0.0, max_value=20.0, value=12.0)
        grade_2nd = st.number_input("Nilai Semester 2 (0-20)", min_value=0.0, max_value=20.0, value=12.0)
        app_1st = st.number_input("SKS Lulus Semester 1", min_value=0, max_value=20, value=6)
        app_2nd = st.number_input("SKS Lulus Semester 2", min_value=0, max_value=20, value=6)
        age = st.number_input("Usia saat mendaftar", min_value=15, max_value=60, value=19)
    
    with col2:
        tuition = st.selectbox("Uang Kuliah (SPP) Lunas?", options=["Tidak", "Ya"])
        scholarship = st.selectbox("Memiliki Beasiswa?", options=["Tidak", "Ya"])
        debtor = st.selectbox("Memiliki Tunggakan (Debtor)?", options=["Tidak", "Ya"])
        gender = st.selectbox("Jenis Kelamin", options=["Perempuan (0)", "Laki-laki (1)"])
        
    submit_button = st.form_submit_button(label="🔍 Prediksi Status")

if submit_button:
    # 1. Menyiapkan input data sesuai dengan 9 fitur yang dilatih di Notebook
    input_data = {
        'Curricular_units_1st_sem_grade': grade_1st,
        'Curricular_units_2nd_sem_grade': grade_2nd,
        'Curricular_units_1st_sem_approved': app_1st,
        'Curricular_units_2nd_sem_approved': app_2nd,
        'Age_at_enrollment': age,
        'Tuition_fees_up_to_date': 1 if tuition == "Ya" else 0,
        'Scholarship_holder': 1 if scholarship == "Ya" else 0,
        'Debtor': 1 if debtor == "Ya" else 0,
        'Gender': 1 if gender == "Laki-laki (1)" else 0
    }
    
    input_df = pd.DataFrame([input_data])
    
    # 2. Prediksi
    prediction = model.predict(input_df)[0]
    
    # 3. Tampilkan hasil
    st.markdown("---")
    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.error("⚠️ **BERISIKO TINGGI DROPOUT**")
    else:
        st.success("✅ **AMAN (LULUS/GRADUATE)**")
