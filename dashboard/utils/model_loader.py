import joblib
import streamlit as st
import os
import json

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")

@st.cache_resource(show_spinner="Cargando modelo predictivo...")
def load_model():
    path = os.path.join(MODELS_DIR, "xgboost_pipeline.pkl")
    if not os.path.exists(path):
        st.warning("Modelo no encontrado. Ejecute `python scripts/train_model.py` primero.")
        return None
    return joblib.load(path)

@st.cache_data(show_spinner="Cargando metadatos del modelo...")
def load_metadata():
    path = os.path.join(MODELS_DIR, "model_metadata.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
