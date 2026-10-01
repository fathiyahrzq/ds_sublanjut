import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Jaya Jaya Institut - Deteksi Dropout", page_icon="🎓", layout="centered")

# Menggunakan cache agar model dan data template tidak diload berulang kali
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')

@st.cache_data
def load_data_template():
    # Membaca data untuk mendapatkan struktur kolom
    df = pd.read_csv('data.csv', sep=';')
    # Menghapus kolom target dari fitur
    if 'Status' in df.columns:
        df = df.drop(['Status'], axis=1)
    if 'Status_Binary' in df.columns:
        df = df.drop(['Status_Binary'], axis=1)
    return df

model = load_model()
df_template = load_data_template()

# Judul Aplikasi
st.title("🎓 Sistem Prediksi Dropout Mahasiswa")
st.write("Aplikasi ini digunakan untuk memprediksi apakah seorang mahasiswa berisiko mengalami *dropout* berdasarkan profil dan performa akademiknya di semester awal.")

st.markdown("---")

# Membuat Form Input untuk User
st.subheader("📝 Masukkan Data Mahasiswa")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # Fitur-fitur Akademik (Biasanya paling berpengaruh)
        grade_1st = st.number_input("Nilai Semester 1 (Skala 0-20)", min_value=0.0, max_value=20.0, value=12.0)
        grade_2nd = st.number_input("Nilai Semester 2 (Skala 0-20)", min_value=0.0, max_value=20.0, value=12.0)
        age = st.number_input("Usia saat mendaftar", min_value=15, max_value=60, value=19)
    
    with col2:
        # Fitur Finansial / Administratif
        scholarship = st.selectbox("Memiliki Beasiswa?", options=["Tidak", "Ya"])
        tuition_up_to_date = st.selectbox("Uang Kuliah (SPP) Lunas?", options=["Tidak", "Ya"])
        debtor = st.selectbox("Apakah memiliki tunggakan/hutang (Debtor)?", options=["Tidak", "Ya"])
        
    submit_button = st.form_submit_button(label="🔍 Prediksi Status Mahasiswa")

# Logika Prediksi ketika tombol ditekan
if submit_button:
    # 1. Membuat dictionary dengan nilai default (rata-rata dari dataset asli)
    # Ini bertujuan agar 28 kolom lainnya yang tidak diinput user tidak error
    input_data = {}
    for col in df_template.columns:
        input_data[col] = df_template[col].mean() # Menggunakan nilai rata-rata sebagai default
        
    # 2. Mengubah nilai default dengan inputan dari user
    input_data['Curricular_units_1st_sem_grade'] = grade_1st
    input_data['Curricular_units_2nd_sem_grade'] = grade_2nd
    input_data['Age_at_enrollment'] = age
    input_data['Scholarship_holder'] = 1 if scholarship == "Ya" else 0
    input_data['Tuition_fees_up_to_date'] = 1 if tuition_up_to_date == "Ya" else 0
    input_data['Debtor'] = 1 if debtor == "Ya" else 0
    
    # 3. Konversi dictionary ke DataFrame dengan 1 baris
    input_df = pd.DataFrame([input_data])
    
    # Pastikan urutan kolom sesuai dengan saat model dilatih
    input_df = input_df[df_template.columns]
    
    # 4. Melakukan Prediksi
    prediction = model.predict(input_df)[0]
    
    # 5. Menampilkan Hasil
    st.markdown("---")
    st.subheader("Hasil Prediksi:")
    
    if prediction == 1: # Asumsi 1 adalah Dropout
        st.error("⚠️ **BERISIKO TINGGI DROPOUT**")
        st.write("Siswa ini terdeteksi memiliki probabilitas tinggi untuk tidak menyelesaikan pendidikannya. Disarankan untuk segera menjadwalkan sesi bimbingan konseling dan meninjau bantuan finansial.")
    else:
        st.success("✅ **AMAN (TIDAK DROPOUT)**")
        st.write("Siswa ini diprediksi akan melanjutkan pendidikannya dengan baik.")