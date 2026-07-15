import streamlit as st
import pandas as pd
import numpy as np
import os
from dashboard.utils.data_loader import load_accepted
from dashboard.utils.model_loader import load_metadata
from dashboard.utils.charts import plot_missing_bar, kpi_card

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw")

TARGET_MAP = {
    "Fully Paid": 0, "Current": 0, "Charged Off": 1,
    "Default": 1, "Late (31-120 days)": 1, "Late (16-30 days)": 1,
    "In Grace Period": 1,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

def show():
    st.title("Salud del Pipeline")
    st.markdown("Métricas técnicas sobre la calidad y el volumen de datos procesados.")

    df = load_accepted()
    meta = load_metadata()

    st.subheader("Volumen de Datos")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.plotly_chart(kpi_card(len(df), "Filas Procesadas", fmt=","),
                        use_container_width=True)
    with c2:
        st.plotly_chart(kpi_card(len(df.columns), "Columnas", fmt=","),
                        use_container_width=True)
    with c3:
        raw_acc = os.path.join(RAW_DIR, "accepted_2007_to_2018Q4.csv")
        raw_rows = 2260701 if os.path.exists(raw_acc) else 0
        st.plotly_chart(kpi_card(raw_rows, "Filas Raw", fmt=","),
                        use_container_width=True)
    with c4:
        mem = df.memory_usage(deep=True).sum() / 1024**2
        st.plotly_chart(kpi_card(mem, "Memoria Usada (MB)", fmt=".1f"),
                        use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribución Original de loan_status")
        if "loan_status" in df.columns:
            ls_dist = df["loan_status"].value_counts().reset_index()
            ls_dist.columns = ["loan_status", "count"]
            # add mapped target
            ls_dist["bad_loan"] = ls_dist["loan_status"].map(TARGET_MAP)
            import plotly.express as px
            colors = ls_dist["bad_loan"].map({0: "#2ecc71", 1: "#e74c3c", np.nan: "gray"})
            fig = px.bar(ls_dist, x="loan_status", y="count",
                         color="bad_loan",
                         color_continuous_scale=["#2ecc71", "#e74c3c"],
                         text_auto=",.0f",
                         labels={"loan_status": "Estado Original", "count": "Cantidad"},
                         height=400)
            fig.update_xaxes(tickangle=45)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Columna loan_status no disponible en datos procesados.")

    with c2:
        st.subheader("Columnas Eliminadas (leakage >50% missing)")
        const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        high_null = df.columns[df.isna().mean() > 0.5].tolist()
        all_dropped = list(set(const_cols + high_null))
        st.metric("Columnas constantes", len(const_cols))
        st.metric("Columnas >50% nulos", len(high_null))
        if all_dropped:
            st.dataframe(pd.DataFrame({"columna_eliminada": all_dropped,
                                       "razón": ["constante" if c in const_cols else ">50% nulos" for c in all_dropped]}),
                        use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Valores Faltantes por Variable")
    miss = df.isnull().sum()
    miss_pct = (miss / len(df)) * 100
    miss_report = pd.DataFrame({"pct_missing": miss_pct})
    miss_report = miss_report[miss_report["pct_missing"] > 0].sort_values("pct_missing", ascending=False)
    top_n = st.slider("Mostrar top N variables", 5, 40, 20)
    st.plotly_chart(plot_missing_bar(miss_report, top=top_n), use_container_width=True)

    st.markdown("---")

    if meta:
        st.subheader("Métricas del Modelo (XGBoost Tuned)")
        m = meta["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("AUC-ROC", f"{m['auc_roc']:.4f}")
        with c2:
            st.metric("Precision", f"{m['precision']:.4f}")
        with c3:
            st.metric("Recall", f"{m['recall']:.4f}")
        with c4:
            st.metric("F1-Score", f"{m['f1']:.4f}")
        st.caption(f"Train shape: {meta['train_shape']} | Test shape: {meta['test_shape']} | scale_pos_weight: {meta['scale_pos_weight']:.2f}")

    st.markdown("---")
    st.markdown("*Nota: 17 columnas constantes (mayormente `sec_app_*` y `revol_bal_joint` por ser de aplicaciones conjuntas) y 21 columnas con >50% missing fueron eliminadas en el pipeline de preprocesamiento.*")
