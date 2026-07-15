import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.utils.data_loader import load_combined, load_rejected
from dashboard.utils.charts import plot_rejected_comparison, kpi_card

def show():
    st.title("Aceptados vs Rechazados")
    st.markdown("¿Qué diferencia a un solicitante aprobado de uno rechazado?")

    combined = load_combined()
    rejected = load_rejected()

    st.subheader("KPIs Comparativos")
    c1, c2, c3, c4 = st.columns(4)
    total = len(combined)
    rej_count = combined["rejected"].sum()
    rej_rate = rej_count / total
    with c1:
        st.plotly_chart(kpi_card(total, "Total Solicitudes", fmt=","),
                        use_container_width=True)
    with c2:
        st.plotly_chart(kpi_card(rej_count, "Rechazados", fmt=","),
                        use_container_width=True)
    with c3:
        st.plotly_chart(kpi_card(rej_rate * 100, "Tasa de Rechazo", suffix="%", fmt=".1f"),
                        use_container_width=True)
    with c4:
        avg_score_acc = combined[combined["rejected"] == 0]["fico_score"].mean()
        avg_score_rej = combined[combined["rejected"] == 1]["fico_score"].mean()
        diff = avg_score_acc - avg_score_rej
        st.plotly_chart(kpi_card(diff, "Diferencia FICO (Acep - Rech)", fmt=".1f"),
                        use_container_width=True)

    st.markdown("---")

    st.subheader("Distribuciones Comparativas")
    st.plotly_chart(plot_rejected_comparison(combined), use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Monto Solicitado")
        acc = combined[combined["rejected"] == 0]
        rej = combined[combined["rejected"] == 1]
        fig = px.histogram(
            pd.DataFrame({
                "Monto": pd.concat([acc["loan_amnt"], rej["loan_amnt"]]),
                "Grupo": ["Aceptados"] * len(acc) + ["Rechazados"] * len(rej),
            }),
            x="Monto", color="Grupo", barmode="overlay", histnorm="percent",
            color_discrete_map={"Aceptados": "#2ecc71", "Rechazados": "#e74c3c"},
            opacity=0.6, nbins=40, height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Distribución por Estado")
        acc_state = combined[combined["rejected"] == 0]["addr_state"].value_counts().head(10)
        rej_state = combined[combined["rejected"] == 1]["addr_state"].value_counts().head(10)
        comp = pd.DataFrame({
            "Aceptados": acc_state,
            "Rechazados": rej_state,
        }).fillna(0).astype(int)
        fig = px.bar(comp, barmode="group", height=350,
                     labels={"value": "Cantidad", "addr_state": "Estado", "variable": "Grupo"},
                     color_discrete_map={"Aceptados": "#2ecc71", "Rechazados": "#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Tasa de Rechazo en el Tiempo")
    ts = combined.copy()
    ts["year"] = pd.to_datetime(ts["issue_d"], errors="coerce").dt.year
    if ts["year"].notna().any():
        yearly = ts.groupby("year")["rejected"].mean().reset_index()
        fig = px.line(yearly, x="year", y="rejected", markers=True,
                       labels={"year": "Año", "rejected": "Tasa de Rechazo"},
                       height=350)
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Datos temporales no disponibles para análisis de tasa de rechazo.")
