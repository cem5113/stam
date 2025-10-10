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
import traceback
import pandas as pd
import streamlit as st

# ---- Paket-içi importlar (mutlak) ----
from sutam.config.settings import APP_NAME
from sutam.dataio.loaders import load_metadata
from sutam.dataio.bootstrap import get_bootstrap  # artifact/release/raw/local fallback bekleniyor

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


# ---- Secrets → env (opsiyonel) ----
try:
    if "GH_TOKEN" in st.secrets:
        os.environ.setdefault("GH_TOKEN", st.secrets["GH_TOKEN"])
    # Repo/workflow isimleri secrets'ta yoksa makul varsayılan koy
    if "GITHUB_REPO" in st.secrets:
        os.environ.setdefault("GITHUB_REPO", st.secrets["GITHUB_REPO"])
    else:
        os.environ.setdefault("GITHUB_REPO", "cem5113/crime_prediction_data")

    if "GITHUB_WORKFLOW" in st.secrets:
        os.environ.setdefault("GITHUB_WORKFLOW", st.secrets["GITHUB_WORKFLOW"])
    else:
        os.environ.setdefault("GITHUB_WORKFLOW", "full_pipeline.yml")

    for k in ("APP_NAME", "APP_ROLE"):
        if k in st.secrets:
            os.environ.setdefault(k, str(st.secrets[k]))
except Exception:
    # secrets yoksa sessiz geç
    pass


# ---- Sidebar: rol & hızlı kontroller + TEŞHİS PANELİ ----
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
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("♻️ Cache temizle", use_container_width=True):
            try:
                st.cache_data.clear()
                st.success("Cache temizlendi.")
            except Exception as e:
                st.warning(f"Cache temizlenemedi: {e}")
    with col_c2:
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
        st.warning("GH_TOKEN yok • Release/RAW/yerel fall-back denenecek.")

    # --- DIAG: ENV ve dosya varlığı (gözle gör) ---
    st.caption("GitHub kaynak ayarları (diagnostic)")
    st.code(
        {
            "GITHUB_REPO": os.environ.get("GITHUB_REPO"),
            "GITHUB_WORKFLOW": os.environ.get("GITHUB_WORKFLOW"),
            "GH_TOKEN_set": bool(os.environ.get("GH_TOKEN")),
            "data_dir_exists": os.path.isdir(os.path.join(_PROJECT_ROOT, "data")),
            "has_sf_cells.geojson": os.path.exists(os.path.join(_PROJECT_ROOT, "data", "sf_cells.geojson")),
            "has_events.csv": os.path.exists(os.path.join(_PROJECT_ROOT, "data", "events.csv")),
        },
        language="json",
    )


# ---- Açılışta otomatik veri yükleme (cache sayesinde hızlı) ----
@st.cache_data(show_spinner=True)
def _bootstrap_cached():
    try:
        return get_bootstrap()
    except Exception as e:
        meta_err = {"error": f"{e}\n{traceback.format_exc()}"}
        return meta_err, pd.DataFrame()

meta, df = _bootstrap_cached()

# Başlık: meta.app_name varsa onu kullan
app_name = (meta.get("app_name") if isinstance(meta, dict) else None) or APP_NAME
st.title(app_name)

# Kaynak / Hata bilgisini üstte göster (teşhis için)
with st.container():
    if isinstance(meta, dict) and meta.get("source"):
        st.caption(f"Veri kaynağı: **{meta['source']}**")
    if isinstance(meta, dict) and meta.get("error"):
        st.warning("Bootstrap uyarısı:\n\n" + str(meta["error"]))

# DataFrame durumu
if not isinstance(df, pd.DataFrame) or df.empty:
    st.error("Veri yüklenemedi veya boş görünüyor. Veri kaynak/ENV ayarlarını kontrol edin.")
else:
    st.success(f"Veri yüklendi: {len(df):,} satır")

# ---- İlk açılış audit ----
try:
    audit(event="app_open", actor=role, payload={"tab": "Home"})
except Exception:
    pass

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
