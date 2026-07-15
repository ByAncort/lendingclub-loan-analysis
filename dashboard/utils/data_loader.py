import pandas as pd
import numpy as np
import streamlit as st
import os

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "processed")
INTERIM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "interim")

@st.cache_data(show_spinner="Cargando datos de préstamos...")
def load_accepted():
    path = os.path.join(PROCESSED_DIR, "accepted_clean.csv")
    df = pd.read_csv(path, low_memory=False)
    if "issue_d" in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    if "earliest_cr_line" in df.columns:
        df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
    return df

@st.cache_data(show_spinner="Cargando datos de rechazados...")
def load_rejected():
    path = os.path.join(PROCESSED_DIR, "rejected_clean.csv")
    df = pd.read_csv(path, low_memory=False)
    if "application_date" in df.columns:
        df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce", unit="s")
    return df

@st.cache_data(show_spinner="Cargando datos combinados...")
def load_combined():
    path = os.path.join(PROCESSED_DIR, "combined_summary.csv")
    df = pd.read_csv(path, low_memory=False)
    if "issue_d" in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    return df
