import streamlit as st
import pandas as pd
import numpy as np
from dashboard.utils.data_loader import load_accepted
from dashboard.utils.charts import (
    plot_default_by_purpose, plot_default_rate_by_home,
    plot_default_rate_by_term, plot_state_map
)

def show():
    st.title("Perfil del Solicitante")
    st.markdown("Análisis de las características demográficas y de solicitud asociadas al riesgo de default.")

    df = load_accepted()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Default por Propósito")
        st.markdown("¿Para qué se piden los préstamos? ¿Cuáles son más riesgosos?")
        st.plotly_chart(plot_default_by_purpose(df), use_container_width=True)

    with c2:
        st.subheader("Default por Tipo de Vivienda")
        st.markdown("¿La situación de vivienda se asocia al riesgo?")
        st.plotly_chart(plot_default_rate_by_home(df), use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Default por Plazo (Term)")
        st.markdown("¿El plazo más largo implica mayor riesgo?")
        st.plotly_chart(plot_default_rate_by_term(df), use_container_width=True)

    with c2:
        st.subheader("Default por Rango de Ingreso Anual")
        bins = [0, 30000, 50000, 75000, 100000, 150000, np.inf]
        labels = ["< 30k", "30k-50k", "50k-75k", "75k-100k", "100k-150k", "> 150k"]
        df_temp = df.copy()
        df_temp["income_range"] = pd.cut(df_temp["annual_inc"], bins=bins, labels=labels)
        grp = df_temp.groupby("income_range", observed=True)["bad_loan"].mean().reset_index()
        import plotly.express as px
        fig = px.bar(grp, x="income_range", y="bad_loan",
                     color="bad_loan", color_continuous_scale="RdYlGn_r",
                     text_auto=".1%",
                     labels={"income_range": "Ingreso Anual", "bad_loan": "Tasa de Default"},
                     height=350)
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Default por Antigüedad Laboral")
        emp_order = sorted(df["emp_length"].dropna().unique())
        grp = df.groupby("emp_length", observed=True)["bad_loan"].mean().reset_index()
        import plotly.express as px
        fig = px.bar(grp.sort_values("emp_length"),
                     x="emp_length", y="bad_loan",
                     color="bad_loan", color_continuous_scale="RdYlGn_r",
                     text_auto=".1%",
                     labels={"emp_length": "Antigüedad Laboral (años)",
                             "bad_loan": "Tasa de Default"},
                     height=350)
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Distribución Geográfica del Riesgo")
        st.markdown("¿Cómo se distribuye el riesgo por estado?")
        st.plotly_chart(plot_state_map(df), use_container_width=True)
