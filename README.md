# Dashboard TA \u2014 Analisis Topik & Sentimen Postpartum Depression

Dashboard Streamlit untuk menyajikan hasil TA "Analisis Topik dan Sentimen Postpartum
Depression pada Media Sosial di Indonesia Menggunakan IndoBERTweet dan BERTopic",
mengikuti struktur dashboard contoh (Overview / Ringkasan, Analisis Hasil, Uji Teks Baru).

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`.

## Struktur Halaman

1. **Overview** (`app.py`) \u2014 ringkasan penelitian: funnel data (62.312 \u2192 39.464 \u2192
   33.551 \u2192 33.425), distribusi dokumen per aspek, metrik utama model.
2. **Analisis Hasil** (`pages/1_\U0001F4CA_Analisis_Hasil.py`) \u2014 3 tab:
   - Performa Model Klasifikasi (perbandingan rasio split 80:10:10 / 70:15:15 / 60:20:20)
   - Performa & Sub-Topik BERTopic (Coherence/Diversity/Silhouette per aspek + 5 sub-topik
     dominan per aspek beserta frasa representatif c-TF-IDF)
   - Validasi Klinis (ringkasan skor dari psikolog validator + kutipan komentar klinis)
3. **Uji Teks Baru** (`pages/2_\U0001F50D_Uji_Teks_Baru.py`) \u2014 form untuk memasukkan
   teks baru dan melihat prediksi sentimen + aspek + sub-topik terdekat.

## Sumber Data

Seluruh angka pada `data/*.json` diambil dari hasil penelitian TA kamu (notebook,
naskah revisi, dan sesi validasi klinis bersama psikolog), bukan data dummy.

## Menghubungkan Model Asli (IndoBERTweet & BERTopic)

Secara default halaman **Uji Teks Baru** berjalan dalam **mode simulasi** (heuristik
berbasis kata kunci hasil penelitian), supaya dashboard tetap bisa langsung dijalankan dan
didemokan tanpa perlu file model besar. Untuk memakai model asli, ikuti langkah pada
`models/README.md`. Karena arsitekturnya multi-task learning kustom, bagian loading &
inference model di `utils.py` kemungkinan perlu disesuaikan lagi begitu file model kamu
sudah diunggah \u2014 kirimkan file modelnya dan saya bantu sambungkan.

## Deploy ke Streamlit Community Cloud

1. Push folder ini ke sebuah repository GitHub.
2. Buka https://share.streamlit.io, hubungkan repo tersebut, pilih `app.py` sebagai entry point.
3. Jika memakai model asli yang berukuran besar, pertimbangkan menyimpannya di
   HuggingFace Hub / Google Drive dan mengunduhnya saat start-up (jangan commit file besar
   langsung ke GitHub).
