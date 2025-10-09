# ui/tab_planning.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from datetime import timedelta
from typing import List, Dict

from dataio.loaders import load_sf_crime_latest
from features.stats_classic import spatial_top_geoid
from patrol.approvals import save_approval, list_approvals
from services.tz import now_sf

def _recent_slice(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    if "date" not in df.columns:
        return df
    dmax = pd.to_datetime(df["date"], errors="coerce").max()
    if pd.isna(dmax):
        return df
    dmin = dmax - pd.Timedelta(days=days)
    m = (pd.to_datetime(df["date"], errors="coerce") >= dmin) & (pd.to_datetime(df["date"], errors="coerce") <= dmax)
    return df[m].copy()

def _propose_routes(df: pd.DataFrame, teams: int, route_len: int, n_alts: int = 4) -> List[Dict]:
    """
    Basit/yer tutucu öneri üretici:
    - Son 7 güne göre en yoğun GEOID'lerden alternatif listeler yapar.
    - Kapsama: seçili GEOID'lerdeki toplam olay / şehir toplamı
    - Çeşitlilik: ardışık alternatifler arası 1 - Jaccard
    """
    dfr = _recent_slice(df, days=7)
    top = spatial_top_geoid(dfr, n=max(teams * route_len * 2, 20))
    geo_pool = top["GEOID"].astype(str).tolist()

    total = float(dfr.get("crime_count", pd.Series([1]*len(dfr))).sum())
    alts = []
    step = max(1, route_len // 2)
    ts_tag = now_sf().strftime("%Y%m%d%H%M%S")

    for i in range(n_alts):
        start = i * step
        cells = geo_pool[start:start + route_len]
        if len(cells) < route_len:
            # havuz yetmezse başa sar
            cells = (cells + geo_pool)[:route_len]

        cov = 0.0
        if "GEOID" in dfr.columns and "crime_count" in dfr.columns and total > 0:
            cov = float(dfr[dfr["GEOID"].astype(str).isin(cells)]["crime_count"].sum()) / total

        alt = {
            "alt_id": f"ALT-{ts_tag}-{i+1}",
            "teams": teams,
            "route_len": route_len,
            "cells": cells,
            "coverage": cov,  # 0..1
        }
        # çeşitlilik (öncekiyle)
        if alts:
            prev = set(alts[-1]["cells"])
            cur  = set(cells)
            jacc = len(prev & cur) / max(1, len(prev | cur))
            alt["diversity"] = 1.0 - jacc
        else:
            alt["diversity"] = 1.0
        alts.append(alt)
    return alts

def render():
    st.subheader("🚓 Devriye Planlama")

    # 1) Veri
    try:
        df, src = load_sf_crime_latest()
    except Exception as e:
        st.error("Veri yüklenemedi.")
        st.exception(e)
        return

    left, mid, right = st.columns([1.05, 1.6, 1.2])

    # 2) Sol panel — parametreler
    with left:
        st.markdown("**Parametreler**")
        k_teams   = st.number_input("Ekip sayısı (K)", min_value=1, max_value=20, value=3, step=1)
        route_len = st.number_input("Rota uzunluğu (hücre sayısı)", min_value=4, max_value=40, value=10, step=1)
        dwell     = st.number_input("Hücre kontrol süresi (dk)", min_value=2, max_value=60, value=8, step=1,
                                    help="Zaman planlaması için yer tutucu (ileride rota süresine katılacak).")
        diversity = st.slider("Çeşitlilik ayarı", min_value=0.0, max_value=1.0, value=0.4, step=0.1,
                              help="Yolların birbirine benzemesini azaltma eğilimi (yer tutucu).")
        gen = st.button("🟢 Devriye Öner", use_container_width=True)

        if gen:
            st.session_state["route_alts"] = _propose_routes(df, int(k_teams), int(route_len), n_alts=4)
            st.success("Öneriler güncellendi.")

    # 3) Orta panel — öneri listesi
    with mid:
        st.markdown("**Önerilen Rotalar**")
        alts = st.session_state.get("route_alts", [])
        if not alts:
            st.info("Öneri üretmek için soldaki **Devriye Öner** düğmesini kullanın.")
        else:
            for alt in alts:
                with st.expander(f"{alt['alt_id']}  •  Kapsama ~ {alt['coverage']*100:0.1f}%  •  Çeşitlilik ~ {alt['diversity']*100:0.0f}%", expanded=False):
                    st.write("Öncelikli GEOID'ler:", ", ".join(alt["cells"]))
                    st.caption("Not: Bu öneriler yer tutucudur; gerçek rota hesabı (yol ağı, süre) faz-2'de eklenecek.")

    # 4) Sağ panel — amir onayı formu + son onaylar
    with right:
        st.markdown("**Amir Onayı**")
        alts = st.session_state.get("route_alts", [])
        alt_ids = [a["alt_id"] for a in alts] if alts else []
        pick = st.selectbox("Onaylanacak alternatif", options=alt_ids, index=0 if alt_ids else None)
        t0 = now_sf()
        t1 = t0 + timedelta(hours=3)

        with st.form("approval_form", clear_on_submit=False):
            dev_code = st.text_input("Devriye/Atama Kodu", value="DV-01")
            teams = st.multiselect("Takımlar", options=["Alpha","Bravo","Charlie","Delta"], default=["Alpha"])
            start = st.text_input("Başlangıç (SF)", value=t0.isoformat(timespec="minutes"))
            end   = st.text_input("Bitiş (SF)", value=t1.isoformat(timespec="minutes"))
            approver = st.text_input("Onaylayan", value="amir.soyad")

            submitted = st.form_submit_button("✅ Onayı Kaydet")
            if submitted:
                if not pick:
                    st.warning("Önce bir alternatif seçin.")
                else:
                    sel = next(a for a in alts if a["alt_id"] == pick)
                    payload = {
                        "alt_id": sel["alt_id"],
                        "assignment": dev_code,
                        "teams": teams,
                        "start": start,
                        "end": end,
                        "approver": approver,
                        # bilgi amaçlı ekler:
                        "cells": sel["cells"],
                        "coverage": f"{sel['coverage']:.4f}",
                        "diversity": f"{sel['diversity']:.4f}",
                    }
                    eid = save_approval(payload)
                    st.success(f"Onay kaydedildi • Kayıt ID: `{eid}`")

        st.markdown("---")
        st.caption("**Son Onaylar**")
        for r in list_approvals(limit=8):
            st.caption(f"• {r.get('ts_sf','-')} | ID:{r.get('event_id','-')} | {r.get('assignment','-')} | {r.get('alt_id','-')}")
