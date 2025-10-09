# ui/tab_planning.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from datetime import timedelta
from typing import List, Dict, Tuple, Optional

from dataio.loaders import load_sf_crime_latest
from features.stats_classic import spatial_top_geoid
from patrol.approvals import save_approval, list_approvals
from services.tz import now_sf, now_sf_str

# ---------------- helpers ----------------
def _latlon_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    for a, b in [("lat","lon"), ("latitude","longitude"), ("y","x"), ("LAT","LON")]:
        if a in df.columns and b in df.columns:
            return a, b
    return None, None

def _recent_slice(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    if "date" not in df.columns:
        return df
    dmax = pd.to_datetime(df["date"], errors="coerce").max()
    if pd.isna(dmax):
        return df
    dmin = dmax - pd.Timedelta(days=days)
    m = (pd.to_datetime(df["date"], errors="coerce") >= dmin) & (pd.to_datetime(df["date"], errors="coerce") <= dmax)
    return df[m].copy()

def _score_by_geoid(df: pd.DataFrame, harm_weighted: bool) -> pd.DataFrame:
    """Skor kolonunu otomatik seç ve GEOID bazında özetle."""
    score_col = (
        "pred_expected" if "pred_expected" in df.columns else
        ("pred_p_occ" if "pred_p_occ" in df.columns else
         ("crime_count" if "crime_count" in df.columns else df.select_dtypes("number").columns[0]))
    )
    if score_col == "pred_p_occ":
        g = df.groupby("GEOID", as_index=False)[score_col].mean()
    else:
        g = df.groupby("GEOID", as_index=False)[score_col].sum()
    if harm_weighted and "harm_multiplier" in df.columns:
        hm = df.groupby("GEOID")["harm_multiplier"].mean().reset_index()
        g = g.merge(hm, on="GEOID", how="left")
        g[score_col] = g[score_col] * g["harm_multiplier"].fillna(1.0)
        g.drop(columns=["harm_multiplier"], inplace=True, errors="ignore")
    g.rename(columns={score_col: "score"}, inplace=True)
    return g.sort_values("score", ascending=False)

def _recent_geoid_penalty(recent_routes: List[List[str]], bias: float) -> Dict[str, float]:
    """Son onaylı rotalardaki GEOID’ler için ceza katsayısı (1 - bias)."""
    recent_set: set[str] = set()
    for r in (recent_routes or []):
        recent_set.update(map(str, r))
    pen = {}
    for gid in recent_set:
        pen[str(gid)] = max(0.0, 1.0 - bias)  # örn. bias=0.4 → 0.6 katsayı
    return pen

def _apply_penalty(geo_scores: pd.DataFrame, penalty: Dict[str, float]) -> pd.DataFrame:
    if not penalty:
        return geo_scores
    g = geo_scores.copy()
    g["score"] = g.apply(lambda r: r["score"] * penalty.get(str(r["GEOID"]), 1.0), axis=1)
    return g.sort_values("score", ascending=False)

def _propose_routes(
    df: pd.DataFrame,
    teams: int,
    route_len: int,
    n_alts: int,
    harm_weighted: bool,
    diversity_bias: float,
    recent_routes: List[List[str]],
) -> List[Dict]:
    """
    Basit/yer tutucu öneri:
    - Son 7 güne göre GEOID skorlarını üret (pred_expected/occ veya count).
    - Çeşitlilik için: son onaylı rotalarda kullanılan GEOID’lere ceza uygula.
    - Alternatifleri döndür (karuselde gezdirilecek).
    """
    dfr = _recent_slice(df, days=7)
    # Eğer skorlar yoksa geleneksel yoğunlukla bir havuz oluştur (yedek)
    if "GEOID" not in dfr.columns:
        return []
    # Skorları topla (otomatik seçim)
    geo_scores = _score_by_geoid(dfr, harm_weighted=harm_weighted)

    # Çok seyrek sahalar için top-N havuzunu genişlet (min havuz = 2×teams×route_len)
    min_pool = max(teams * route_len * 2, 40)
    if len(geo_scores) < min_pool:
        # fallback olarak klasik top listesiyle tamamla
        top_extra = spatial_top_geoid(dfr, n=min_pool)[["GEOID"]].astype(str)
        geo_scores = pd.concat([
            geo_scores[["GEOID","score"]].astype({"GEOID":str}),
            top_extra.assign(score=0.0)
        ], ignore_index=True).drop_duplicates("GEOID")

    # Çeşitlilik cezası
    penalty = _recent_geoid_penalty(recent_routes, diversity_bias)
    ranked = _apply_penalty(geo_scores, penalty)["GEOID"].astype(str).tolist()

    # round-robin alternatifler
    alts: List[Dict] = []
    step = max(1, route_len // 2)
    ts_tag = now_sf().strftime("%Y%m%d%H%M%S")

    # toplam skor (kapsama için payda)
    total_score = float(geo_scores["score"].sum()) or 1.0

    for i in range(n_alts):
        start = i * step
        cells = ranked[start:start + route_len]
        if len(cells) < route_len:
            cells = (cells + ranked)[:route_len]

        cov = float(geo_scores[geo_scores["GEOID"].isin(cells)]["score"].sum()) / total_score

        # önceki alternatifle benzerliği (çeşitlilik göstergesi)
        if alts:
            prev = set(alts[-1]["cells"])
            cur = set(cells)
            jacc = len(prev & cur) / max(1, len(prev | cur))
            div = 1.0 - jacc
        else:
            div = 1.0

        alts.append({
            "alt_id": f"ALT-{ts_tag}-{i+1}",
            "teams": teams,
            "route_len": route_len,
            "cells": cells,
            "coverage": cov,     # 0..1
            "diversity": div,    # 0..1 (yüksek = iyi)
        })
    return alts

# ---------------- UI ----------------
def render():
    st.subheader("🚓 Devriye Planlama")
    st.caption(f"Son kontrol: {now_sf_str()} (SF)")

    # 1) Veri
    try:
        df, _ = load_sf_crime_latest()
    except Exception as e:
        st.error("Veri yüklenemedi.")
        st.exception(e)
        return
    if "GEOID" not in df.columns:
        st.warning("GEOID sütunu yok. Planlama yapılamaz.")
        return

    lat_col, lon_col = _latlon_columns(df)

    left, mid, right = st.columns([1.05, 1.6, 1.2])

    # 2) Sol panel — parametreler
    with left:
        st.markdown("**Parametreler**")
        k_teams   = st.number_input("Ekip sayısı (K)", 1, 20, 3, 1)
        route_len = st.number_input("Rota uzunluğu (hücre)", 4, 40, 10, 1)
        dwell     = st.number_input("Hücre kontrol süresi (dk)", 2, 60, 8, 1,
                                    help="(Şimdilik gösterim; rota süresine Faz-2’de katılacak)")
        harm_weighted = st.checkbox("Harm-weighted kapsama", value=False,
                                    help="Suç etkisine göre ağırlık (varsa harm_multiplier).")
        diversity_bias = st.slider("Çeşitlilik eğilimi", 0.0, 1.0, 0.4, 0.1,
                                   help="Son benzer rotalardaki hücrelere ceza uygular.")

        gen = st.button("🟢 Devriye Öner", use_container_width=True)
        if gen:
            recent_routes = [r.get("route_geoids") or r.get("cells") or [] for r in list_approvals(limit=30)]
            alts = _propose_routes(
                df, int(k_teams), int(route_len), n_alts=5,
                harm_weighted=harm_weighted,
                diversity_bias=float(diversity_bias),
                recent_routes=recent_routes,
            )
            st.session_state["route_alts"] = alts
            st.success("Öneriler güncellendi.")

    # 3) Orta panel — öneri listesi + harita
    with mid:
        st.markdown("**Önerilen Rotalar**")
        alts = st.session_state.get("route_alts", [])
        if not alts:
            st.info("Öneri üretmek için soldaki **Devriye Öner** düğmesini kullanın.")
        else:
            # karusel: seçili alternatif
            alt_ids = [a["alt_id"] for a in alts]
            idx = st.slider("Alternatif seç", 0, len(alts)-1, 0, 1, label_visibility="collapsed")
            sel = alts[idx]
            st.caption(f"Seçili: **{sel['alt_id']}** • Kapsama ≈ {sel['coverage']*100:0.1f}% • Çeşitlilik ≈ {sel['diversity']*100:0.0f}%")

            # harita önizleme (centroid)
            if lat_col and lon_col:
                cent = (
                    df.groupby("GEOID", as_index=False)[[lat_col, lon_col]]
                      .mean(numeric_only=True)
                      .dropna()
                )
                view = cent[cent["GEOID"].astype(str).isin(sel["cells"])].copy()
                view["order"] = range(1, len(view)+1)
                layers = [{
                    "@@type": "ScatterplotLayer",
                    "data": view.to_dict("records"),
                    "get_position": f"[{lon_col}, {lat_col}]",
                    "get_radius": 90,
                    "get_fill_color": "[0, 180, 100]",
                    "pickable": True,
                    "opacity": 0.85,
                }]
                if len(view) >= 2:
                    path = [{"coordinates": [[float(x), float(y)] for x, y in zip(view[lon_col], view[lat_col])]}]
                    layers.append({
                        "@@type": "PathLayer",
                        "data": path,
                        "get_path": "d.coordinates",
                        "get_width": 3,
                        "get_color": [0, 120, 255],
                        "opacity": 0.6,
                    })
                st.pydeck_chart({
                    "initialViewState": {
                        "latitude": float(view[lat_col].mean()) if len(view) else float(df[lat_col].mean()),
                        "longitude": float(view[lon_col].mean()) if len(view) else float(df[lon_col].mean()),
                        "zoom": 12
                    },
                    "layers": layers,
                    "mapProvider": "carto"
                })
            with st.expander("Öncelikli GEOID’ler", expanded=False):
                st.write(", ".join(sel["cells"]))

            # seçimi state'e bırak (onay paneli kullanacak)
            st.session_state["selected_alt"] = sel

    # 4) Sağ panel — amir onayı formu + son onaylar
    with right:
        st.markdown("**Amir Onayı**")
        sel = st.session_state.get("selected_alt")
        if not sel:
            st.info("Sağdaki form için önce bir alternatif seçin.")
        else:
            t0 = now_sf()
            t1 = t0 + timedelta(hours=3)
            with st.form("approval_form", clear_on_submit=False):
                dev_code = st.text_input("Devriye/Atama Kodu", value="DV-01")
                teams = st.multiselect("Takımlar", ["Alpha","Bravo","Charlie","Delta"], default=["Alpha"])
                start = st.text_input("Başlangıç (SF)", value=t0.isoformat(timespec="minutes"))
                end   = st.text_input("Bitiş (SF)", value=t1.isoformat(timespec="minutes"))
                approver = st.text_input("Onaylayan", value="amir.soyad")

                submitted = st.form_submit_button("✅ Onayı Kaydet")
                if submitted:
                    payload = {
                        "alt_id": sel["alt_id"],
                        "assignment": dev_code,
                        "teams": teams,
                        "start": start,
                        "end": end,
                        "approver": approver,
                        # bilgi amaçlı ekler:
                        "route_geoids": sel["cells"],
                        "coverage": f"{sel['coverage']:.4f}",
                        "diversity": f"{sel['diversity']:.4f}",
                    }
                    eid = save_approval(payload)
                    st.success(f"Onay kaydedildi • Kayıt ID: `{eid}`")

        st.markdown("---")
        st.caption("**Son Onaylar**")
        for r in list_approvals(limit=8):
            st.caption(f"• {r.get('ts_sf','-')} | ID:{r.get('event_id','-')} | {r.get('assignment','-')} | {r.get('alt_id','-')}")
