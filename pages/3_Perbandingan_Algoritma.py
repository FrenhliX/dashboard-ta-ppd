import streamlit as st
import pandas as pd
import plotly.express as px
from utils import get_model_comparison

st.set_page_config(page_title="Perbandingan Algoritma - Dashboard TA", page_icon="⚖️", layout="wide")

st.title("⚖️ Perbandingan Performa Algoritma")

# Load Data
comparison_data = get_model_comparison()
df_comp = pd.DataFrame(comparison_data)

# Highlight Model Terbaik
best_model = df_comp.loc[df_comp['avg_f1'].idxmax()]

# --- Ringkasan Metric Card ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Model Terbaik", value=best_model['model'])
with col2:
    st.metric(label="Rata-rata F1-Score Terbaik", value=f"{best_model['avg_f1']:.4f}")
with col3:
    st.metric(label="Akurasi Terbaik", value=f"{best_model['akurasi']:.4f}")

st.divider()

# --- Tabel Ringkasan ---
st.subheader("Tabel Komparasi Model")
df_display = df_comp.copy()
for col in ["akurasi", "f1_sentimen", "f1_aspek", "avg_f1"]:
    df_display[col] = df_display[col].map(lambda v: f"{v:.4f}")

st.dataframe(
    df_display.rename(columns={
        "model": "Algoritma / Model",
        "akurasi": "Akurasi",
        "f1_sentimen": "F1 Sentimen",
        "f1_aspek": "F1 Aspek",
        "avg_f1": "Rata-rata F1",
        "waktu_latih_sec": "Waktu Pelatihan (detik)"
    }),
    hide_index=True,
    use_container_width=True
)

# --- Visualisasi Grafik ---
st.subheader("Visualisasi Metrik Performa")

tab_f1, tab_time = st.tabs(["Skor F1-Score & Akurasi", "Waktu Pelatihan"])

with tab_f1:
    melt_df = df_comp.melt(
        id_vars="model", 
        value_vars=["f1_sentimen", "f1_aspek", "avg_f1"],
        var_name="Metrik", 
        value_name="Skor"
    )
    label_map = {"f1_sentimen": "F1 Sentimen", "f1_aspek": "F1 Aspek", "avg_f1": "Rata-rata F1"}
    melt_df["Metrik"] = melt_df["Metrik"].map(label_map)
    
    fig_bar = px.bar(
        melt_df, 
        x="model", 
        y="Skor", 
        color="Metrik", 
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Bold,
        text_auto=".3f"
    )
    fig_bar.update_layout(height=400, xaxis_title="Algoritma", yaxis_title="Skor Evaluasi", yaxis_range=[0, 1])
    st.plotly_chart(fig_bar, use_container_width=True)

with tab_time:
    fig_time = px.bar(
        df_comp, 
        x="model", 
        y="waktu_latih_sec", 
        color="model",
        title="Waktu Komputasi Pelatihan (Detik)",
        text_auto=".1f",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_time.update_layout(height=400, showlegend=False, xaxis_title="Algoritma", yaxis_title="Waktu (detik)")
    st.plotly_chart(fig_time, use_container_width=True)

st.caption(
    "Hasil pengujian menunjukkan model Transformer (IndoBERTweet Multi-Task) unggul dalam menangkap konteks teks dibandingkan model baseline pembelajaran mesin tradisional."
)