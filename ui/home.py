# ui/home.py
from __future__ import annotations
import streamlit as st
import pandas as pd

from config.settings import (
    APP_NAME, MODEL_VERSION, LAST_TRAINED_AT, LAST_DATA_REFRESH_AT
)
from services.tz import now_sf_str
from dataio.loaders import load_sf_crime_latest, load_metadata


def render():
    st.title(APP_NAME)

    # Üst bilgi şeridi
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", MODEL_VERSION)
    c2.metric("Son eğitim (SF)", LAST_TRAINED_AT or "—")
    c3.metric("Veri güncelleme (SF)", LAST_DATA_REFRESH_AT or "—")
    c4.metric("Şu an (SF)", now_sf_str())

    # Metadata (varsa) + veri kaynağı
    meta = load_metadata() or {}
    with st.expander("ⓘ Model/Veri Meta", expanded=False):
        st.json(meta, expanded=False)

    st.markdown("#### Bugün için risk verisi (örnek yükleme)")
    src_badge = st.empty()
    df_head = st.empty()

    try:
        df, src = load_sf_crime_latest()
        src_txt = {
            "artifact": "✅ Artifact (en güncel)",
            "release":  "⚠️ Release (yedek, güncel olmayabilir)",
        }.get(src, f"📁 {src}")
        src_badge.info(f"Veri kaynağı: **{src_txt}**")

        # Küçük önizleme
        with st.expander("İlk 15 satır (önizleme)", expanded=False):
            st.dataframe(df.head(15), use_container_width=True)
    except Exception as e:
        src_badge.error("Veri yüklenemedi")
        st.exception(e)
        st.stop()

    # Ana ekrandaki küçük kontrol kutuları (harita katmanları ileride bağlanacak)
    st.markdown("#### Hızlı kontroller (hazır/placeholder)")
    colA, colB, colC = st.columns(3)
    colA.toggle("Tahmin katmanı (risk)", value=True, help="Açılışta risk haritası görünür.")
    colB.toggle("Geçici hotspot", value=True, help="Son olaylara dayalı anomali noktaları.")
    colC.toggle("Kalıcı hotspot", value=True, help="Uzun dönem ısı haritası.")

    st.caption("🔌 Katmanlar ileride haritaya bağlanacak. Şimdilik veri akışının canlı çalıştığını doğruladık.")
