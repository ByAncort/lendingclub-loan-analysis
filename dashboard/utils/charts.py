import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

COLOR_GOOD = "#2ecc71"
COLOR_BAD = "#e74c3c"
COLOR_PRIMARY = "#3498db"

def kpi_card(val, label, prefix="", suffix="", fmt=",.0f"):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number",
        value=val,
        number={"font": {"size": 48, "color": "#ffffff"}, "prefix": prefix, "suffix": suffix, "valueformat": fmt},
        title={"text": label, "font": {"size": 16, "color": "#e0e0e0"}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#1a1a2e",
        font={"color": "#ffffff"},
    )
    return fig

def plot_grade_default_rate(df):
    grade_order = ["A", "B", "C", "D", "E", "F", "G"]
    grp = df.groupby("grade", observed=True).agg(
        default_rate=("bad_loan", "mean"),
        count=("bad_loan", "count"),
        avg_int_rate=("int_rate", "mean"),
    ).reset_index()
    grp["grade"] = pd.Categorical(grp["grade"], categories=grade_order, ordered=True)
    grp = grp.sort_values("grade")

    fig = make_dual_axis(grp, "grade", "default_rate", "avg_int_rate",
                         "Tasa de Default por Grade", "Grade", "Tasa de Default",
                         "Tasa de Interés Promedio (%)")
    fig.add_hline(y=df["bad_loan"].mean(), line_dash="dash", line_color="gray",
                  annotation_text=f"Default global: {df['bad_loan'].mean():.1%}")
    return fig

def plot_fico_histogram(df):
    fig = go.Figure()
    good = df[df["bad_loan"] == 0]["fico_score"].dropna()
    bad = df[df["bad_loan"] == 1]["fico_score"].dropna()
    fig.add_trace(go.Histogram(x=good, name="Good (Paga)", marker_color=COLOR_GOOD,
                                opacity=0.6, nbinsx=40, histnorm="percent"))
    fig.add_trace(go.Histogram(x=bad, name="Bad (Default)", marker_color=COLOR_BAD,
                                opacity=0.6, nbinsx=40, histnorm="percent"))
    fig.update_layout(barmode="overlay", xaxis_title="FICO Score",
                      yaxis_title="% del total", legend_title="Outcome",
                      height=400)
    return fig

def plot_dti_boxplot(df):
    fig = go.Figure()
    fig.add_trace(go.Box(y=df[df["bad_loan"] == 0]["dti"].dropna(), name="Good (Paga)",
                          marker_color=COLOR_GOOD, boxmean=True))
    fig.add_trace(go.Box(y=df[df["bad_loan"] == 1]["dti"].dropna(), name="Bad (Default)",
                          marker_color=COLOR_BAD, boxmean=True))
    fig.update_layout(yaxis_title="DTI (%)", height=400)
    return fig

def plot_default_by_purpose(df):
    grp = df.groupby("purpose", observed=True).agg(
        default_rate=("bad_loan", "mean"), count=("bad_loan", "count")
    ).reset_index().sort_values("default_rate", ascending=True)
    fig = go.Figure()
    colors = [COLOR_BAD if r >= df["bad_loan"].mean() else COLOR_GOOD for r in grp["default_rate"]]
    fig.add_trace(go.Bar(y=grp["purpose"], x=grp["default_rate"], orientation="h",
                          marker_color=colors, text=grp["default_rate"].apply(lambda x: f"{x:.1%}"),
                          textposition="outside"))
    fig.add_vline(x=df["bad_loan"].mean(), line_dash="dash", line_color="gray",
                  annotation_text=f"Global: {df['bad_loan'].mean():.1%}")
    fig.update_layout(xaxis_title="Tasa de Default", yaxis_title="Propósito",
                      height=400, xaxis_tickformat=".0%")
    return fig

def plot_heatmap_term_grade(df):
    grade_order = ["A", "B", "C", "D", "E", "F", "G"]
    df["term_label"] = df["term"].map({36: "36 meses", 60: "60 meses"})
    pivot = df.pivot_table(index="term_label", columns="grade", values="bad_loan",
                           aggfunc="mean", observed=False)
    pivot = pivot[grade_order]
    fig = px.imshow(pivot, text_auto=".1%", color_continuous_scale="RdYlGn_r",
                    aspect="auto", height=250,
                    labels={"x": "Grade", "y": "Plazo", "color": "Default Rate"})
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig

def plot_hardship_by_grade(df):
    grade_order = ["A", "B", "C", "D", "E", "F", "G"]
    grp = df.groupby("grade", observed=True)["had_hardship"].mean().reset_index()
    grp["grade"] = pd.Categorical(grp["grade"], categories=grade_order, ordered=True)
    grp = grp.sort_values("grade")
    fig = px.bar(grp, x="grade", y="had_hardship", color="had_hardship",
                 color_continuous_scale="Reds", text_auto=".3%",
                 labels={"grade": "Grade", "had_hardship": "Tasa de Hardship"},
                 height=350)
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_tickformat=".1%")
    return fig

def plot_delinquency_cross(df):
    cross = pd.crosstab(df["had_delinquency"], df["bad_loan"], normalize="index")
    cross.index = ["Sin delincuencia previa", "Con delincuencia previa"]
    cross.columns = ["Good (Paga)", "Bad (Default)"]
    fig = go.Figure()
    for col in cross.columns:
        fig.add_trace(go.Bar(name=col, x=cross.index, y=cross[col],
                              marker_color=COLOR_GOOD if "Good" in col else COLOR_BAD))
    fig.update_layout(barmode="group", yaxis_title="Proporción",
                      yaxis_tickformat=".0%", height=350,
                      legend_title="Outcome")
    return fig

def plot_intrate_vs_default_by_subgrade(df):
    grp = df.groupby("sub_grade", observed=True).agg(
        default_rate=("bad_loan", "mean"),
        avg_int_rate=("int_rate", "mean"),
        count=("bad_loan", "count"),
    ).reset_index()
    fig = px.scatter(grp, x="avg_int_rate", y="default_rate", size="count",
                     hover_name="sub_grade", text="sub_grade",
                     labels={"avg_int_rate": "Tasa de Interés Promedio (%)",
                             "default_rate": "Tasa de Default"},
                     height=400)
    fig.update_traces(textposition="top center", marker=dict(color=COLOR_PRIMARY))
    fig.update_layout(yaxis_tickformat=".0%")
    return fig

def plot_rejected_comparison(combined):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("FICO / Risk Score", "DTI (%)"),
        shared_yaxes=False,
    )
    acc = combined[combined["rejected"] == 0]
    rej = combined[combined["rejected"] == 1]
    fig.add_trace(go.Histogram(x=acc["fico_score"].dropna(), name="Aceptados",
                                marker_color=COLOR_GOOD, opacity=0.6, histnorm="percent",
                                nbinsx=30), row=1, col=1)
    fig.add_trace(go.Histogram(x=rej["fico_score"].dropna(), name="Rechazados",
                                marker_color=COLOR_BAD, opacity=0.6, histnorm="percent",
                                nbinsx=30), row=1, col=1)
    fig.add_trace(go.Histogram(x=acc["dti"].dropna(), marker_color=COLOR_GOOD,
                                opacity=0.6, histnorm="percent", nbinsx=30,
                                showlegend=False), row=1, col=2)
    fig.add_trace(go.Histogram(x=rej["dti"].dropna(), marker_color=COLOR_BAD,
                                opacity=0.6, histnorm="percent", nbinsx=30,
                                showlegend=False), row=1, col=2)
    fig.update_layout(height=350, barmode="overlay")
    return fig

def make_dual_axis(df, x_col, y1_col, y2_col, title, xlabel, y1label, y2label):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df[x_col], y=df[y1_col], name=y1label,
                          marker_color=COLOR_BAD, opacity=0.7,
                          text=df[y1_col].apply(lambda x: f"{x:.1%}"),
                          textposition="outside"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y2_col], name=y2label,
                              mode="lines+markers", line=dict(color=COLOR_PRIMARY, width=3),
                              marker=dict(size=10)),
                  secondary_y=True)
    fig.update_layout(title=title, xaxis_title=xlabel, height=400)
    fig.update_yaxes(title_text=y1label, secondary_y=False, tickformat=".0%")
    fig.update_yaxes(title_text=y2label, secondary_y=True)
    return fig

def plot_missing_bar(missing_df, top=20):
    md = missing_df.sort_values("pct_missing", ascending=True).tail(top)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=md.index, x=md["pct_missing"], orientation="h",
                          marker_color="#e67e22",
                          text=md["pct_missing"].apply(lambda x: f"{x:.1f}%"),
                          textposition="outside"))
    fig.add_vline(x=50, line_dash="dash", line_color="red", annotation_text="50%")
    fig.update_layout(xaxis_title="% Missing", yaxis_title="Variable",
                      height=500, margin=dict(l=0, r=0, t=0, b=0))
    return fig

def plot_default_rate_by_home(df):
    order = ["RENT", "MORTGAGE", "OWN", "ANY", "NONE"]
    grp = df.groupby("home_ownership", observed=True)["bad_loan"].mean().reset_index()
    grp.columns = ["home_ownership", "default_rate"]
    grp["home_ownership"] = pd.Categorical(grp["home_ownership"], categories=order, ordered=True)
    grp = grp.dropna().sort_values("home_ownership")
    fig = px.bar(grp, x="home_ownership", y="default_rate",
                 color="default_rate", color_continuous_scale="RdYlGn_r",
                 text_auto=".1%",
                 labels={"home_ownership": "Tipo de Vivienda", "default_rate": "Tasa de Default"},
                 height=350)
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_tickformat=".0%")
    return fig

def plot_default_rate_by_term(df):
    fig = px.bar(df.groupby("term", observed=True)["bad_loan"].mean().reset_index(),
                 x="term", y="bad_loan",
                 color="bad_loan", color_continuous_scale="RdYlGn_r",
                 text_auto=".1%",
                 labels={"term": "Plazo (meses)", "bad_loan": "Tasa de Default"},
                 height=300)
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_tickformat=".0%")
    return fig

def plot_state_map(df):
    grp = df.groupby("addr_state", observed=True).agg(
        default_rate=("bad_loan", "mean"),
        count=("bad_loan", "count"),
    ).reset_index()
    grp.columns = ["state", "default_rate", "count"]
    fig = px.choropleth(grp, locations="state", locationmode="USA-states",
                         color="default_rate", scope="usa",
                         color_continuous_scale="RdYlGn_r",
                         hover_data={"count": True},
                         labels={"default_rate": "Default Rate", "count": "Préstamos"},
                         height=450)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig

def plot_fred_indicators(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    monthly = df.set_index("issue_d").resample("ME").agg(
        default_rate=("bad_loan", "mean"),
        unrate=("unrate", "first"),
        fed_funds=("fed_funds", "first"),
        cpi=("cpi", "first"),
    ).dropna()
    color_map = {"unrate": "#e74c3c", "fed_funds": "#f39c12", "cpi": "#2ecc71"}
    for col, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=monthly.index, y=monthly[col], name=col,
            mode="lines", line=dict(color=color, width=1.5, dash="dot"),
        ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=monthly.index, y=monthly["default_rate"], name="Default Rate",
        mode="lines+markers", line=dict(color=COLOR_BAD, width=2.5),
        marker=dict(size=5),
    ), secondary_y=False)
    fig.update_layout(
        title="Default Rate vs Indicadores Macroeconomicos",
        height=400, hovermode="x unified",
    )
    fig.update_yaxes(title_text="Default Rate", secondary_y=False, tickformat=".1%")
    fig.update_yaxes(title_text="Valor FRED", secondary_y=True)
    return fig

def plot_fred_correlation(df):
    fred_cols = ["unrate", "fed_funds", "cpi"]
    corr_data = []
    for col in fred_cols:
        if col in df.columns:
            c = df[col].corr(df["bad_loan"])
            corr_data.append({"Indicador": col.upper(), "Correlacion con Default": f"{c:+.4f}"})
    return pd.DataFrame(corr_data)

def plot_time_series(df, freq="ME"):
    freq_map = {"M": "ME", "Q": "QE", "Y": "YE"}
    freq = freq_map.get(freq, freq)
    ts = df.set_index("issue_d").resample(freq)["bad_loan"].agg(["mean", "count"]).dropna()
    ts.columns = ["default_rate", "volume"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=ts.index, y=ts["volume"], name="Volumen",
                          marker_color=COLOR_PRIMARY, opacity=0.4), secondary_y=False)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["default_rate"], name="Default Rate",
                              mode="lines+markers",
                              line=dict(color=COLOR_BAD, width=2),
                              marker=dict(size=5)), secondary_y=True)
    fig.update_layout(title="Evolución Temporal", xaxis_title="Fecha", height=400)
    fig.update_yaxes(title_text="Volumen", secondary_y=False)
    fig.update_yaxes(title_text="Tasa de Default", secondary_y=True, tickformat=".0%")
    return fig
