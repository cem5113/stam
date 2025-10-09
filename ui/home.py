# ui/home.py
from __future__ import annotations
import streamlit as st

from config.settings import APP_NAME, MODEL_VERSION
from services.tz import now_sf_str
from dataio.loaders import load_sf_crime_latest, load_metadata
from services.metrics import compute_kpis


def _pick_meta(meta: dict) -> tuple[str, str, str]:
    """
    Meta şeması esnek olabilir:
    - model.version / model.last_trained_at / data.last_data_refresh_at
    - veya düz alanlar: model_version / last_trained_at / last_data_refresh_at|data_refresh_at
    """
    mv = (
        meta.get("model", {}).get("version")
        or meta.get("model_version")
        or MODEL_VERSION
    )
    ltr = (
        meta.get("model", {}).get("last_trained_at")
        or meta.get("last_trained_at")
        or "—"
    )
    ldr = (
        meta.get("data", {}).get("last_data_refresh_at")
        or meta.get("last_data_refresh_at")
        or meta.get("data_refresh_at")
        or "—"
    )
    return str(mv), str(ltr), str(ldr)


def render():
    st.title(APP_NAME)

    # Meta
    meta = load_metadata() or {}
    mv, ltr, ldr = _pick_meta(meta)

    # Üst bilgi şeridi
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", mv or "—")
    c2.metric("Son eğitim (SF)", ltr or "—")
    c3.metric("Veri güncelleme (SF)", ldr or "—")
    c4.metric("Şu an (SF)", now_sf_str())

    with st.expander("ⓘ Model/Veri Meta", expanded=False):
        st.json(meta, expanded=False)

    # Veri yükleme + KPI
    try:
        df, src = load_sf_crime_latest()
    except Exception as e:
        st.error("Veri yüklenemedi")
        st.exception(e)
        st.stop()
        return

    kpi = compute_kpis(df)
    s1, s2, s3 = st.columns(3)
    s1.metric("HitRate@Top10", "—" if kpi["hitrate_top10"] is None else f"{kpi['hitrate_top10']}%")
    s2.metric("Brier", "—" if kpi["brier"] is None else f"{kpi['brier']}")
    src_label = {"artifact": "Artifact (güncel)", "release": "Release (yedek)"}.get(src, str(src))
    s3.metric("Veri Kaynağı", src_label)

    # Mini Model Kartı
    with st.expander("ⓘ Model Kartı (mini)", expanded=False):
        st.markdown(
            f"- **Sürüm:** {mv}\n"
            f"- **Son eğitim:** {ltr}\n"
            f"- **Veri güncelleme:** {ldr}\n"
            "- **Notlar:** Düşük olaylı hücrelerde belirsizlik yükselebilir. "
            "Saha görünümünde tahmin seviyesi (Yüksek/Orta/Düşük) sadeleştirilmiştir."
        )

    # Küçük önizleme
    with st.expander("İlk 15 satır (önizleme)", expanded=False):
        st.dataframe(df.head(15), use_container_width=True)

    # Hızlı kontroller (yer tutucu)
    st.markdown("#### Hızlı kontroller")
    colA, colB, colC = st.columns(3)
    colA.toggle("Tahmin katmanı (risk)", value=True, help="Açılışta risk haritası görünür.")
    colB.toggle("Geçici hotspot", value=True, help="Son olaylara dayalı anomali noktaları.")
    colC.toggle("Kalıcı hotspot", value=True, help="Uzun dönem ısı haritası.")
    st.caption("🔌 Katmanlar ileride haritaya bağlanacak. KPI'lar güncel veriden otomatik hesaplanır.")
