# app.py
from __future__ import annotations
import streamlit as st
from ui.tab_forecast import render as render_forecast
from ui.tab_stats import render as render_stats
from ui.tab_planning import render as render_planning
from services.auth import role_selector_in_sidebar, get_role
from services.logging import audit

st.set_page_config(page_title="SUTAM", layout="wide")
role = role_selector_in_sidebar()
audit(event="app_open", actor=role, payload={"tab": "Home"})

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
    try:
        render_planning()
    except Exception as e:
        st.error("Devriye Planlama yüklenemedi.")
        st.exception(e)

with tabs[2]:
    try:
        render_forecast()
    except Exception as e:
        st.info("Suç Tahmini modülü henüz hazır değil.")
        st.exception(e)

with tabs[3]:
    try:
        render_stats()
    except Exception as e:
        st.error("Suç İstatistikleri modülü yüklenemedi.")
        st.exception(e)

with tabs[4]:
    st.info("🧾 Raporlar & Operasyonel Öneriler burada olacak.")

