import pandas as pd
import streamlit as st
import plotly.express as px
import os

@st.cache_data
def load_data(sample=True, sample_size=5000):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    
    if sample:
        filename = "accepted_clean.csv"
    else:
        filename = "accepted_clean_sample.csv"
    
    path_acc = os.path.join(PROCESSED_DIR, filename)
    
    if not os.path.exists(path_acc):
        st.error(f"File not found: {path_acc}")
        return pd.DataFrame()
    
    if sample:
        df = pd.read_csv(path_acc, nrows=sample_size, low_memory=False)
    else:
        df = pd.read_csv(path_acc, low_memory=False)
    
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    return df

df = load_data(sample=True, sample_size=5000)

# ── Sidebar ──
st.sidebar.header("Settings")
freq = st.sidebar.selectbox("Time Frequency", ["ME", "QE", "YE"],
                            format_func=lambda x: {"ME": "Monthly", "QE": "Quarterly", "YE": "Annual"}[x])

# ── Title ──
st.title("Credit Risk Dashboard — LendingClub")

if not df.empty:
        st.subheader("Default Rate Trend")
        ts = df.set_index("issue_d").resample(freq)["bad_loan"].agg(["count", "mean"])
        ts.columns = ["total", "default_rate"]
        
        fig1 = px.line(ts, x=ts.index, y="default_rate",
                       markers=True,
                       labels={"default_rate": "Default Rate", "issue_d": ""})
        fig1.update_layout(template="plotly_white", yaxis_tickformat=".2%", height=350, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig1, use_container_width=True)
        
        with st.expander("Summary Table"):
            st.dataframe(ts.style.format({"total": "{:,.0f}", "default_rate": "{:.2%}"}))
