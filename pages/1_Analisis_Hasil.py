import streamlit as st
import pandas as pd
import plotly.express as px

from utils import get_split_ratio, get_topic_metrics, get_topics, get_validation

st.set_page_config(page_title="Analisis Hasil - Dashboard TA", page_icon="\U0001F4CA", layout="wide")

st.title("\U0001F4CA Analisis Hasil Penelitian")

tab1, tab2, tab3 = st.tabs([
    "Performa Model Klasifikasi",
    "Performa & Sub-Topik BERTopic",
    "Validasi Klinis (Psikolog)",
])

# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Perbandingan Rasio Pembagian Data (Data Split)")
    split_df = pd.DataFrame(get_split_ratio())
    display_df = split_df.copy()
    for col in ["akurasi", "f1_sentimen", "f1_aspek", "avg_f1"]:
        display_df[col] = display_df[col].map(lambda v: f"{v:.4f}")
    st.dataframe(
        display_df.rename(columns={
            "rasio": "Rasio (Latih:Val:Uji)", "data_latih": "Data Latih",
            "data_validasi": "Data Validasi", "data_uji": "Data Uji",
            "akurasi": "Akurasi", "f1_sentimen": "F1 Sentimen",
            "f1_aspek": "F1 Aspek", "avg_f1": "Rata-rata F1",
            "best_epoch": "Epoch Terbaik",
        }),
        hide_index=True, use_container_width=True,
    )

    melt_df = split_df.melt(
        id_vars="rasio", value_vars=["f1_sentimen", "f1_aspek", "avg_f1"],
        var_name="metrik", value_name="skor",
    )
    label_map = {"f1_sentimen": "F1 Sentimen", "f1_aspek": "F1 Aspek", "avg_f1": "Rata-rata F1"}
    melt_df["metrik"] = melt_df["metrik"].map(label_map)
    fig = px.bar(melt_df, x="rasio", y="skor", color="metrik", barmode="group",
                 range_y=[0.8, 0.95], color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(height=380, xaxis_title="Rasio Split", yaxis_title="Skor")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Rasio **80:10:10** dipilih sebagai konfigurasi final karena secara **konsisten** "
        "unggul pada seluruh metrik dibanding dua rasio pembanding \u2014 bukan karena "
        "selisihnya besar (semua rasio berada di sekitar Avg F1 0,88). Uji signifikansi "
        "statistik antar-rasio belum dilakukan pada penelitian ini (lihat catatan revisi DTP-21)."
    )

# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Metrik Evaluasi Topic Modeling per Aspek")
    tm_df = pd.DataFrame(get_topic_metrics())
    m1, m2, m3 = st.columns(3)
    with m1:
        fig_c = px.bar(tm_df, x="aspek", y="coherence", color="aspek", title="Coherence (C_v)",
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig_c.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_c, use_container_width=True)
    with m2:
        fig_d = px.bar(tm_df, x="aspek", y="diversity", color="aspek", title="Topic Diversity",
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig_d.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_d, use_container_width=True)
    with m3:
        fig_s = px.bar(tm_df, x="aspek", y="silhouette", color="aspek", title="Silhouette Score",
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig_s.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_s.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_s, use_container_width=True)

    st.caption(
        "Topic Diversity tinggi (0,87\u20130,93) menunjukkan topik tidak redundan. "
        "Silhouette mendekati/di bawah 0 (terutama Mental Health) mengindikasikan klaster "
        "masih tumpang-tindih di ruang embedding \u2014 ini adalah keterbatasan metodologis "
        "yang dibahas pada Bab V.5.5."
    )

    st.divider()
    st.subheader("Sub-Topik Dominan per Aspek (Top-5, BERTopic Bigram)")
    topics = get_topics()
    aspek_pilih = st.selectbox("Pilih aspek:", list(topics.keys()))
    subtopics = topics[aspek_pilih]
    df_sub = pd.DataFrame(subtopics)
    fig_tree = px.bar(
        df_sub, x="persen_segmen", y="tema", orientation="h",
        hover_data=["jumlah_dokumen"], color="persen_segmen",
        color_continuous_scale="Purples",
        labels={"persen_segmen": "% dari segmen aspek", "tema": "Tema Sub-Topik"},
    )
    fig_tree.update_layout(height=350, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_tree, use_container_width=True)

    for st_row in subtopics:
        with st.expander(f"{st_row['tema']} \u2014 {st_row['jumlah_dokumen']:,} dokumen ({st_row['persen_segmen']}%)".replace(",", ".")):
            st.write("**Frasa representatif (c-TF-IDF):** " + "; ".join(st_row["frasa"]))
            if st_row["relevansi_klinis"] is not None:
                st.write(f"**Skor validasi klinis:** Relevansi {st_row['relevansi_klinis']}/5, "
                         f"Kejelasan Label {st_row['kejelasan_label']}/5")
            else:
                st.write("*Kategori residual (zero-shot) \u2014 tidak dinilai dalam sesi validasi klinis.*")

    st.warning(
        "Beberapa tema (mis. *Depresi dan Kesedihan*, *Dukungan Keluarga Besar*) muncul "
        "lebih dari sekali pada aspek yang sama karena BERTopic membentuk beberapa klaster "
        "berbeda dengan label tema yang sama \u2014 menunjukkan sub-dimensi berbeda di dalam "
        "tema besar yang sama, bukan duplikasi (lihat catatan revisi DTP-22)."
    )

# ---------------------------------------------------------------------------
with tab3:
    val = get_validation()
    st.subheader("Hasil Validasi Klinis oleh Psikolog")
    st.write(f"**Validator:** {val['validator']['nama']}")
    st.write(f"**Profesi/Institusi:** {val['validator']['profesi']}")
    st.write(f"**Tanggal sesi:** {val['validator']['tanggal']} \u2014 {val['validator']['metode']}")

    val_df = pd.DataFrame(val["ringkasan_per_aspek"])
    fig_val = px.bar(
        val_df[val_df["aspek"] != "Keseluruhan"], x="aspek",
        y=["rata_relevansi", "rata_kejelasan"], barmode="group",
        labels={"value": "Skor rata-rata (1-5)", "aspek": "Aspek", "variable": "Metrik"},
        color_discrete_sequence=["#7c3aed", "#f59e0b"],
    )
    fig_val.update_layout(height=350, yaxis_range=[0, 5])
    st.plotly_chart(fig_val, use_container_width=True)

    st.dataframe(
        val_df.rename(columns={
            "aspek": "Aspek", "n_subtopik": "N Sub-topik Dinilai",
            "rata_relevansi": "Rata-rata Relevansi Klinis",
            "rata_kejelasan": "Rata-rata Kejelasan Label",
            "persen_skor_ge_4": "% Skor Relevansi \u2265 4",
        }),
        hide_index=True, use_container_width=True,
    )

    st.markdown(f"**Kesimpulan validator:** {val['kesimpulan']}")
    for q in val["kutipan"]:
        st.markdown(f"> {q}")

    st.error(val["disclaimer"])
