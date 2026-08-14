import streamlit as st
import pandas as pd
import plotly.express as px

from utils import predict_text, real_model_available, ASPECTS

st.set_page_config(page_title="Uji Teks Baru - Dashboard TA", page_icon="\U0001F50D", layout="wide")

st.title("\U0001F50D Uji Teks Baru")
st.write(
    "Masukkan sebuah kalimat/unggahan baru (Bahasa Indonesia) untuk melihat prediksi "
    "**sentimen** dan **aspek psikososial** (Mental Health / Finansial / Support System), "
    "serta sub-topik BERTopic yang paling mendekati."
)

if real_model_available():
    st.success("\u2705 Mode: Model Asli (ditemukan di folder models/)")
else:
    st.warning(
        "\u26A0\uFE0F Mode: **SIMULASI**. Model IndoBERTweet & BERTopic hasil training kamu "
        "belum ditemukan di folder `models/`, sehingga prediksi di bawah memakai heuristik "
        "pencocokan kata kunci dari hasil penelitian (bukan inferensi model asli). "
        "Lihat `models/README.md` untuk cara menghubungkan model asli."
    )

contoh_list = [
    "-- pilih contoh --",
    "Sejak lahiran aku jadi gampang nangis sendiri, capek banget rasanya",
    "Alhamdulillah suami baik banget bantuin jagain bayi tengah malam",
    "Bingung mau cari uang dari mana buat kebutuhan bayi bulan ini",
    "Susah tidur terus dari malam sampai pagi karena harus nyusuin",
]
contoh = st.selectbox("Atau coba salah satu contoh:", contoh_list)

default_text = "" if contoh == contoh_list[0] else contoh
text_input = st.text_area("Teks yang ingin diuji:", value=default_text, height=120)

if st.button("\U0001F680 Prediksi", type="primary"):
    if not text_input.strip():
        st.error("Mohon masukkan teks terlebih dahulu.")
    else:
        result = predict_text(text_input)
        st.divider()

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Sentimen")
            warna = {"Positif": "green", "Negatif": "red", "Netral": "gray"}
            st.markdown(
                f"### :{warna.get(result['sentimen'], 'gray')}[{result['sentimen']}]"
            )
            st.progress(result["confidence_sentimen"])
            st.caption(f"Skor keyakinan (heuristik): {result['confidence_sentimen']:.2f}")

        with col2:
            st.subheader("Aspek Psikososial Terdeteksi")
            if result["aspek_terdeteksi"]:
                st.write(", ".join(f"**{a}**" for a in result["aspek_terdeteksi"]))
            else:
                st.write("*Tidak ada aspek yang cocok dengan kata kunci hasil penelitian.*")

            score_rows = [
                {"Aspek": a, "Skor Kecocokan Kata Kunci": v["skor"]}
                for a, v in result["detail_aspek"].items()
            ]
            score_df = pd.DataFrame(score_rows)
            fig = px.bar(score_df, x="Aspek", y="Skor Kecocokan Kata Kunci", color="Aspek",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sub-Topik BERTopic Terdekat per Aspek")
        for aspek in ASPECTS:
            detail = result["detail_aspek"][aspek]
            with st.expander(f"{aspek}", expanded=(aspek in result["aspek_terdeteksi"])):
                if detail["sub_topik"]:
                    st.write(f"**Sub-topik terdekat:** {detail['sub_topik']}")
                    st.write("**Frasa representatif yang cocok:** " + ", ".join(detail["frasa_cocok"]))
                else:
                    st.write("Tidak ditemukan kecocokan sub-topik yang signifikan.")

        st.info(
            "Catatan: hasil ini bersifat deskriptif/eksploratif berdasarkan pola bahasa, "
            "**bukan alat diagnosis PPD**. Interpretasi klinis tetap memerlukan penilaian "
            "tenaga profesional kesehatan mental."
        )

st.divider()
with st.expander("\u2699\uFE0F Cara menghubungkan model asli (IndoBERTweet & BERTopic)"):
    st.markdown(
        """
1. Buat folder `models/indobertweet/` berisi hasil `save_pretrained()` dari tokenizer & model IndoBERTweet kamu (`config.json`, `tokenizer.json`/`vocab.txt`, `model.safetensors` atau `pytorch_model.bin`).
2. Buat folder `models/bertopic_mental_health/`, `models/bertopic_finansial/`, `models/bertopic_support_system/` berisi hasil `BERTopic.save(...)` untuk masing-masing aspek.
3. Jalankan ulang dashboard (`streamlit run app.py`) \u2014 badge di atas akan otomatis berubah menjadi "Model Asli" begitu file terbaca.
4. Karena model kamu memakai **multi-task learning** (satu backbone, dua output kepala klasifikasi), fungsi `predict_text()` pada `utils.py` perlu disesuaikan dengan arsitektur/format penyimpanan spesifik model kamu (mis. custom `nn.Module`, state_dict, atau class kepala klasifikasi terpisah). Bagian ini ditandai `TODO` pada kode \u2014 kirim file model & script arsitekturnya, nanti kodenya bisa saya sesuaikan persis.
        """
    )
