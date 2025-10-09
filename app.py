# app.py
from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="SUTAM", layout="wide")

# (opsiyonel) secrets → env aktarımı; yoksa atlanır
try:
    if "GH_TOKEN" in st.secrets:
        import os
        os.environ.setdefault("GH_TOKEN", st.secrets["GH_TOKEN"])
        os.environ.setdefault("GITHUB_REPO", st.secrets.get("GITHUB_REPO", "cem5113/crime_prediction_data"))
        os.environ.setdefault("GITHUB_WORKFLOW", st.secrets.get("GITHUB_WORKFLOW", "full_pipeline.yml"))
except Exception:
    pass

# Sekmeler
from ui.home import render as render_home

tabs = st.tabs([
    "🏠 Ana Sayfa",
    "🚓 Devriye Planlama",
    "🧭 Suç Tahmini",
    "📊 Suç İstatistikleri",
    "🧾 Raporlar & Öneriler",
])

with tabs[0]:
    render_home()

with tabs[1]:
    st.info("Devriye Planlama buraya gelecek (bir sonraki adımda).")

with tabs[2]:
    st.info("Suç Tahmini bileşenleri burada (harita + Top-K + popup).")

with tabs[3]:
    st.info("Suç İstatistikleri (saat/gün/ay, tür dağılımı) burada.")

with tabs[4]:
    st.info("Raporlar & Operasyonel Öneriler burada.")

