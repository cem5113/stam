# sutam/app.py
from __future__ import annotations
import sys
from pathlib import Path

# --- Sağlam import başlığı: proje kökünü sys.path'e ekle ---
_THIS_FILE = Path(__file__).resolve()
_PACKAGE_DIR = _THIS_FILE.parent                 # .../sutam
_PROJECT_ROOT = _PACKAGE_DIR.parent              # .../

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ----------------------------------------------------------------
# Buradan sonrası: her şeyi "mutlak paket" importlarıyla yap
import os
import pandas as pd
import streamlit as st

# ---- Paket-içi importlar (mutlak) ----
from sutam.config.settings import APP_NAME
from sutam.dataio.loaders import load_metadata
from sutam.dataio.bootstrap import get_bootstrap  # ✅ yeni eklendi

# UI sekmeleri
from sutam.ui.home import render as render_home
from sutam.ui.tab_planning import render as render_planning
from sutam.ui.tab_forecast import render as render_forecast
from sutam.ui.tab_stats import render as render_stats
from sutam.ui.tab_reports import render as render_reports

# Servisler
from sutam.services.auth import role_selector_in_sidebar
from sutam.services.logging import audit

# ---- Sayfa ayarı ----
st.set_page_config(page_title=APP_NAME, page_icon="🔎", layout="wide")

# ---- Sidebar: rol & hızlı kontroller ----
with st.sidebar:
    role = role_selector_in_sidebar()
    st.markdown(f"### {APP_NAME}")

    # küçük meta özeti (artifact/release metadata.json varsa)
    try:
        meta_info = load_metadata() or {}
        if meta_info:
            st.caption("Veri/Model meta (özet)")
            st.json(
                {
                    k: meta_info[k]
                    for k in ("model_version", "last_trained_at", "last_data_refresh_at")
                    if k in meta_info
                }
            )
    except Exception:
        pass

    st.markdown("---")
    if st.button("♻️ Cache temizle", use_container_width=True):
        try:
            st.cache_data.clear()
            st.success("Cache temizlendi.")
        except Exception as e:
            st.warning(f"Cache temizlenemedi: {e}")

    if st.button("🔄 Veriyi Yenile", use_container_width=True):
        try:
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Yenileme başarısız: {e}")

    # GitHub token bilgisi
    gh_tok = st.secrets.get("GH_TOKEN", os.environ.get("GH_TOKEN", ""))
    if gh_tok:
        st.info("GH_TOKEN bulundu • Artifact-öncelikli veri çekimi aktif.")
    else:
        st.warning("GH_TOKEN yok • Release/RAW fall-back kullanılacak.")

# ---- Secrets → env (opsiyonel) ----
try:
    if "GH_TOKEN" in st.secrets:
        os.environ.setdefault("GH_TOKEN", st.secrets["GH_TOKEN"])
        os.environ.setdefault("GITHUB_REPO", st.secrets.get("GITHUB_REPO", "cem5113/crime_prediction_data"))
        os.environ.setdefault("GITHUB_WORKFLOW", st.secrets.get("GITHUB_WORKFLOW", "full_pipeline.yml"))

    for k in ("APP_NAME", "APP_ROLE"):
        if k in st.secrets:
            os.environ.setdefault(k, str(st.secrets[k]))
except Exception:
    pass

# ---- Açılışta otomatik veri yükleme (cache sayesinde hızlı) ----
meta, df = get_bootstrap()

app_name = meta.get("app_name") or APP_NAME
st.title(app_name)

if not isinstance(df, pd.DataFrame) or df.empty:
    st.error("Veri yüklenemedi veya boş görünüyor. Veri kaynak/ENV ayarlarını kontrol edin.")
else:
    st.success(f"Veri yüklendi: {len(df):,} satır")

# ---- İlk açılış audit ----
audit(event="app_open", actor=role, payload={"tab": "Home"})

# ---- Sekmeler ----
tabs = st.tabs(
    [
        "🏠 Ana Sayfa",
        "🚓 Devriye Planlama",
        "🧭 Suç Tahmini",
        "📊 Suç İstatistikleri",
        "🧾 Raporlar & Öneriler",
    ]
)

with tabs[0]:
    try:
        render_home()
        audit(event="open_tab", actor=role, payload={"tab": "Home"})
    except Exception as e:
        st.error("Ana Sayfa yüklenemedi.")
        st.exception(e)

with tabs[1]:
    try:
        render_planning()
        audit(event="open_tab", actor=role, payload={"tab": "Planning"})
    except Exception as e:
        st.error("Devriye Planlama yüklenemedi.")
        st.exception(e)

with tabs[2]:
    try:
        render_forecast()
        audit(event="open_tab", actor=role, payload={"tab": "Forecast"})
    except Exception as e:
        st.error("Suç Tahmini modülü yüklenemedi.")
        st.exception(e)

with tabs[3]:
    try:
        render_stats()
        audit(event="open_tab", actor=role, payload={"tab": "Stats"})
    except Exception as e:
        st.error("Suç İstatistikleri yüklenemedi.")
        st.exception(e)

with tabs[4]:
    try:
        render_reports()
        audit(event="open_tab", actor=role, payload={"tab": "Reports"})
    except Exception as e:
        st.error("Raporlar & Öneriler modülü yüklenemedi.")
        st.exception(e)

# ---- Alt bilgi ----
st.markdown("---")
st.caption("© SUTAM • Bu arayüz, doktora çalışmasının kullanıcı vitrini olarak tasarlanmıştır.")
