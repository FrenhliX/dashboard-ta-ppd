import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import get_overview_stats, real_model_available

st.set_page_config(
    page_title="Dashboard TA - PPD Media Sosial",
    page_icon="\U0001F4CA",
    layout="wide",
)

stats = get_overview_stats()

st.title("\U0001F4CA Dashboard Analisis Topik & Sentimen Postpartum Depression")
st.caption(stats["judul_ta"])

with st.sidebar:
    st.header("Tentang Penelitian")
    st.write(f"**Peneliti:** {stats['peneliti']}")
    st.write(f"**NIM:** {stats['nim']}")
    st.write("**Sumber data:** " + ", ".join(stats["sumber_data"]))
    st.divider()
    st.write("**Arsitektur Model**")
    st.write(f"- Klasifikasi: {stats['arsitektur']['klasifikasi']}")
    st.write(f"- Topic Modeling: {stats['arsitektur']['topic_modeling']}")
    st.divider()
    if real_model_available():
        st.success("Model asli terdeteksi di folder models/")
    else:
        st.warning("Model asli belum ditemukan di folder models/. "
                    "Halaman 'Uji Teks Baru' berjalan dalam mode simulasi.")

st.subheader("Ringkasan Utama")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Data Terkumpul", f"{stats['funnel'][0]['jumlah']:,}".replace(",", "."))
c2.metric("Data Siap Dimodelkan", f"{stats['funnel'][-1]['jumlah']:,}".replace(",", "."))
c3.metric("F1-Score Sentimen", f"{stats['metrik_klasifikasi_terpilih']['F1 Sentimen']:.4f}")
c4.metric("F1-Score Aspek", f"{stats['metrik_klasifikasi_terpilih']['F1 Aspek']:.4f}")

st.divider()

col_left, col_right = st.columns([1.3, 1])

with col_left:
    st.subheader("Alur Penyusutan Data (Data Funnel)")
    funnel_df = pd.DataFrame(stats["funnel"])
    fig = go.Figure(go.Funnel(
        y=funnel_df["tahap"],
        x=funnel_df["jumlah"],
        textinfo="value+percent initial",
        marker={"color": ["#7c3aed", "#8b5cf6", "#a78bfa", "#c4b5fd"]},
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Rincian data yang dibuang saat tahap modeling (\u2212126 data)"):
        rincian = stats["rincian_data_dibuang_modeling"]
        rincian_df = pd.DataFrame(
            {"Alasan Dibuang": list(rincian.keys()), "Jumlah": list(rincian.values())}
        )
        st.dataframe(rincian_df, hide_index=True, use_container_width=True)

with col_right:
    st.subheader("Distribusi Dokumen per Aspek")
    aspek_df = pd.DataFrame(
        {"Aspek": list(stats["distribusi_aspek"].keys()),
         "Jumlah Dokumen": list(stats["distribusi_aspek"].values())}
    )
    fig2 = px.pie(aspek_df, names="Aspek", values="Jumlah Dokumen", hole=0.45,
                  color_discrete_sequence=px.colors.sequential.Purples_r)
    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Konfigurasi Model Terpilih")
m1, m2, m3 = st.columns(3)
m1.metric("Rasio Split Data", stats["split_ratio_terpilih"])
m2.metric("Coherence (C_v) rata-rata", f"{stats['metrik_topic_modeling_rata2']['Coherence (C_v)']:.4f}")
m3.metric("Topic Diversity rata-rata", f"{stats['metrik_topic_modeling_rata2']['Topic Diversity']:.4f}")

st.info(
    "Gunakan menu di sidebar (kiri) untuk membuka halaman **Analisis Hasil** "
    "(performa model & sub-topik per aspek, termasuk hasil validasi klinis) "
    "dan **Uji Teks Baru** (coba masukkan teks baru untuk diprediksi)."
)
