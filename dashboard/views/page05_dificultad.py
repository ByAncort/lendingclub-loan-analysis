import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.utils.data_loader import load_accepted
from dashboard.utils.charts import (
    plot_hardship_by_grade, plot_delinquency_cross, kpi_card
)

def show():
    st.title("Senales de Dificultad Financiera")
    st.markdown("Análisis de programas de hardship, settlement y morosidad previa como señales de alerta temprana.")

    df = load_accepted()

    st.subheader("KPIs de Dificultad Financiera")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.plotly_chart(
            kpi_card(df[df["had_hardship"] == 1]["bad_loan"].mean() * 100,
                     "Default si Hardship=Y", suffix="%", fmt=".1f"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            kpi_card(df[df["had_hardship"] == 0]["bad_loan"].mean() * 100,
                     "Default si Hardship=N", suffix="%", fmt=".1f"),
            use_container_width=True,
        )
    with c3:
        st.plotly_chart(
            kpi_card(df[df["debt_settlement"] == 1]["bad_loan"].mean() * 100,
                     "Default si Settlement=Y", suffix="%", fmt=".1f"),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            kpi_card(df[df["had_delinquency"] == 1]["bad_loan"].mean() * 100,
                     "Default si Delincuencia Previa", suffix="%", fmt=".1f"),
            use_container_width=True,
        )

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Hardship por Grade")
        st.markdown("¿Los hardship se concentran en grades más riesgosos?")
        st.plotly_chart(plot_hardship_by_grade(df), use_container_width=True)

    with c2:
        st.subheader("Delincuencia Previa vs Default")
        st.markdown("¿La morosidad previa predice el default actual?")
        st.plotly_chart(plot_delinquency_cross(df), use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Debt Settlement por Propósito")
        grp = df.groupby("purpose", observed=True)["debt_settlement"].mean().reset_index().sort_values("debt_settlement", ascending=True)
        fig = px.bar(grp, y="purpose", x="debt_settlement", orientation="h",
                     color="debt_settlement", color_continuous_scale="Reds",
                     text_auto=".2%",
                     labels={"purpose": "Propósito", "debt_settlement": "Tasa de Settlement"},
                     height=400)
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Multiplicador de Riesgo")
        risk_mult = pd.DataFrame({
            "Señal": ["Hardship", "Debt Settlement", "Delincuencia Previa"],
            "Default Rate si Señal=Y": [
                df[df["had_hardship"] == 1]["bad_loan"].mean(),
                df[df["debt_settlement"] == 1]["bad_loan"].mean(),
                df[df["had_delinquency"] == 1]["bad_loan"].mean(),
            ],
            "Default Rate si Señal=N": [
                df[df["had_hardship"] == 0]["bad_loan"].mean(),
                df[df["debt_settlement"] == 0]["bad_loan"].mean(),
                df[df["had_delinquency"] == 0]["bad_loan"].mean(),
            ],
        })
        risk_mult["Multiplicador"] = (risk_mult["Default Rate si Señal=Y"] / risk_mult["Default Rate si Señal=N"]).round(1)
        risk_mult["Default Rate si Señal=Y"] = risk_mult["Default Rate si Señal=Y"].map("{:.1%}".format)
        risk_mult["Default Rate si Señal=N"] = risk_mult["Default Rate si Señal=N"].map("{:.1%}".format)
        risk_mult.columns = ["Señal", "Default si Y", "Default si N", "× Riesgo"]
        st.dataframe(risk_mult, use_container_width=True, hide_index=True)
        st.caption("Multiplicador = cuántas veces mayor es el riesgo cuando la señal está presente")
