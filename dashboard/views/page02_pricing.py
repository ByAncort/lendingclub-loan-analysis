import streamlit as st
import pandas as pd
from dashboard.utils.data_loader import load_accepted
from dashboard.utils.charts import (
    plot_heatmap_term_grade, plot_intrate_vs_default_by_subgrade
)

def show():
    st.title("Rentabilidad / Pricing")
    st.markdown("¿La tasa de interés compensa adecuadamente el riesgo asumido?")

    df = load_accepted()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Heatmap: Term × Grade")
        st.markdown("¿Qué combinaciones de plazo y grade tienen peor desempeño?")
        st.plotly_chart(plot_heatmap_term_grade(df), use_container_width=True)

    with c2:
        st.subheader("Tasa de Interés vs Default por Sub-Grade")
        st.markdown("¿El pricing a nivel granular es consistente?")
        st.plotly_chart(plot_intrate_vs_default_by_subgrade(df), use_container_width=True)

    st.markdown("---")

    st.subheader("Tabla Resumen por Grade")
    grade_order = ["A", "B", "C", "D", "E", "F", "G"]
    grp = df.groupby("grade", observed=True).agg(
        total=("bad_loan", "count"),
        default_rate=("bad_loan", "mean"),
        avg_int_rate=("int_rate", "mean"),
    ).reset_index()
    grp["grade"] = pd.Categorical(grp["grade"], categories=grade_order, ordered=True)
    grp = grp.sort_values("grade")
    grp["default_rate"] = grp["default_rate"].map("{:.1%}".format)
    grp["avg_int_rate"] = grp["avg_int_rate"].map("{:.2f}%".format)
    grp["spread_estimado"] = grp["avg_int_rate"]  # proxy simple
    grp.columns = ["Grade", "Total Préstamos", "Tasa Default", "Int. Rate Promedio", "Spread Estimado"]
    st.dataframe(grp, use_container_width=True, hide_index=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Dual-Axis: Default Rate vs Int. Rate por Grade")
        st.markdown("A mayor riesgo → mayor tasa. ¿El spread es suficiente?")
        grade_order = ["A", "B", "C", "D", "E", "F", "G"]
        grp2 = df.groupby("grade", observed=True).agg(
            default_rate=("bad_loan", "mean"),
            avg_int_rate=("int_rate", "mean"),
        ).reset_index()
        grp2["grade"] = pd.Categorical(grp2["grade"], categories=grade_order, ordered=True)
        grp2 = grp2.sort_values("grade")

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=grp2["grade"], y=grp2["default_rate"], name="Default Rate",
                   marker_color="#e74c3c", opacity=0.7,
                   text=grp2["default_rate"].apply(lambda x: f"{x:.1%}"),
                   textposition="outside"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=grp2["grade"], y=grp2["avg_int_rate"], name="Int. Rate Promedio",
                       mode="lines+markers", line=dict(color="#3498db", width=3),
                       marker=dict(size=10)),
            secondary_y=True,
        )
        fig.update_layout(xaxis_title="Grade", height=400)
        fig.update_yaxes(title_text="Tasa de Default", secondary_y=False, tickformat=".0%")
        fig.update_yaxes(title_text="Tasa de Interés Promedio (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Distribución de Int. Rate por Grade")
        import plotly.express as px
        fig = px.box(df, x="grade", y="int_rate", color="grade",
                     category_orders={"grade": list(grade_order)},
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     labels={"grade": "Grade", "int_rate": "Tasa de Interés (%)"},
                     height=400)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
