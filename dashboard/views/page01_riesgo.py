import streamlit as st
import pandas as pd
from dashboard.utils.data_loader import load_accepted
from dashboard.utils.charts import (
    kpi_card, plot_grade_default_rate, plot_fico_histogram,
    plot_dti_boxplot, plot_time_series, plot_fred_indicators, plot_fred_correlation
)

def show():
    st.title("Riesgo y Calidad de Cartera")
    st.markdown("Monitoreo de los KPIs fundamentales de riesgo de crédito del portafolio LendingClub.")

    df = load_accepted()

    with st.container():
        st.subheader("KPIs Globales")
        c1, c2, c3, c4 = st.columns(4)
        default_rate = df["bad_loan"].mean()
        total = len(df)
        avg_loan = df["loan_amnt"].mean()
        avg_fico = df["fico_score"].mean()

        with c1:
            st.plotly_chart(kpi_card(default_rate * 100, "Default Rate", suffix="%", fmt=".1f"),
                            use_container_width=True)
        with c2:
            st.plotly_chart(kpi_card(total, "Total Préstamos", fmt=","),
                            use_container_width=True)
        with c3:
            st.plotly_chart(kpi_card(avg_loan, "Monto Promedio", prefix="$", fmt=",.0f"),
                            use_container_width=True)
        with c4:
            st.plotly_chart(kpi_card(avg_fico, "FICO Score Promedio", fmt=",.0f"),
                            use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Tasa de Default por Grade")
        st.markdown("¿El grade de LendingClub predice bien el riesgo real?")
        st.plotly_chart(plot_grade_default_rate(df), use_container_width=True)
    with c2:
        st.subheader("FICO Score: Good vs Bad")
        st.markdown("¿Qué tan bien separa el FICO a buenos y malos pagadores?")
        st.plotly_chart(plot_fico_histogram(df), use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("DTI por Outcome")
        st.markdown("¿El nivel de endeudamiento es un buen predictor?")
        st.plotly_chart(plot_dti_boxplot(df), use_container_width=True)
    with c2:
        st.subheader("Evolución Temporal")
        st.markdown("¿Cómo varía la tasa de default en el tiempo?")
        freq = st.radio("Frecuencia", ["M", "Q", "Y"], index=0, horizontal=True, key="ts_freq")
        st.plotly_chart(plot_time_series(df, freq=freq), use_container_width=True)

    st.markdown("---")
    st.subheader("Contexto Macroeconomico (FRED)")
    st.markdown("Indicadores macroeconomicos integrados via API de la Reserva Federal: desempleo, tasa de interes de la Fed e inflacion.")
    st.plotly_chart(plot_fred_indicators(df), use_container_width=True)
    st.dataframe(plot_fred_correlation(df), use_container_width=True, hide_index=True)
