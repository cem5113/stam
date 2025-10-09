# ui/tab_forecast.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from config.settings import RISK_THRESHOLDS
from dataio.loaders import load_sf_crime_latest
from services.tz import now_sf_str

# ---------- helpers ----------
def _pick_category_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["category_grouped", "category", "subcategory_grouped", "subcategory", "crime_type"]:
        if c in df.columns: return c
    return None

def _risk_level(p: float) -> str:
    if np.isnan(p): return "Bilinmiyor"
    return "Yüksek" if p > RISK_THRESHOLDS["mid"] else ("Orta" if p > RISK_THRESHOLDS["low"] else "Düşük")

def _confidence_label(q10: float, q90: float) -> str:
    if np.isnan(q10) or np.isnan(q90): return "—"
    spread = q90 - q10
    if spread <= 0.5: return "Yüksek güven"
    if spread <= 1.5: return "Orta güven"
    return "Düşük güven"

def _latlon_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cand = [("lat","lon"), ("latitude","longitude"), ("y","x"), ("LAT","LON")]
    for y,x in cand:
        if y in df.columns and x in df.columns:
            return y, x
    return None, None

def _expected_col(df: pd.DataFrame) -> str:
    # Tercih sırası: pred_expected → pred_p_occ (olasılık) → crime_count
    if "pred_expected" in df.columns: return "pred_expected"
    if "pred_p_occ"   in df.columns: return "pred_p_occ"
    return "crime_count" if "crime_count" in df.columns else df.columns[0]

def _agg_for_window(df: pd.DataFrame, by_cols: list, val_col: str) -> pd.DataFrame:
    # sum for counts/expected; mean for probability columns
    if val_col == "pred_p_occ":
        agg = df.groupby(by_cols, as_index=False)[val_col].mean()
    else:
        agg = df.groupby(by_cols, as_index=False)[val_col].sum()
    return agg

def _prepare_topk(df: pd.DataFrame, score_col: str, k: int = 10) -> pd.DataFrame:
    out = df.copy()
    if score_col == "pred_p_occ":
        out["risk_level"] = out[score_col].map(_risk_level)
    else:
        # normalize to [0,1] for level — kaba ölçek
        m = out[score_col].max() or 1.0
        out["risk_level"] = (out[score_col] / m).clip(0,1).map(_risk_level)
    # güven etiketi
    if {"pred_q10","pred_q90"}.issubset(out.columns):
        out["güven"] = [_confidence_label(q10, q90) for q10,q90 in zip(out["pred_q10"], out["pred_q90"])]
    else:
        out["güven"] = "—"
    cols = ["GEOID", score_col, "risk_level", "güven"]
    keep = [c for c in cols if c in out.columns]
    return (out[keep].sort_values(score_col, ascending=False).head(k))

# ---------- UI ----------
def render():
    st.subheader("🧭 Suç Tahmini")
    st.caption(f"Son kontrol: {now_sf_str()} (SF)")

    # Veri yükle
    try:
        df_raw, src = load_sf_crime_latest()
    except Exception as e:
        st.error("Veri yüklenemedi.")
        st.exception(e)
        return

    # Filtre paneli
    left, mid, right = st.columns([1.1, 2.2, 1.2])

    with left:
        st.markdown("**Zaman Penceresi**")
        win = st.radio(
            "Seçim",
            ["0–24 (saatlik)", "72 saat (6’şar saat)", "1 hafta (günlük)"],
            index=0,
            help="SF saatine göre pencereler",
        )

        # referans tarih — mevcut verinin en güncel günü
        if "date" in df_raw.columns:
            latest = pd.to_datetime(df_raw["date"], errors="coerce").dropna().max()
        else:
            latest = pd.Timestamp.today()
        d_pick = st.date_input("Referans gün", value=latest.date() if pd.notnull(latest) else pd.Timestamp.today().date())

        # kategori filtresi
        cat_col = _pick_category_col(df_raw)
        cats = sorted(df_raw[cat_col].dropna().unique()) if cat_col else []
        pick_cats = st.multiselect("Suç kategorisi", cats, default=[], help="Boş = tüm kategoriler")

        st.markdown("**Harita Katmanları**")
        layer_risk   = st.checkbox("Tahmin katmanı (risk)", True)
        layer_temp   = st.checkbox("Geçici hotspot", True)
        layer_stable = st.checkbox("Kalıcı hotspot", True)

    # Veri hazırlığı (pencereye göre)
    df = df_raw.copy()
    if cat_col and len(pick_cats) > 0:
        df = df[df[cat_col].isin(pick_cats)]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    score_col = _expected_col(df)

    if win.startswith("0–24"):
        # seçilen günün saatleri
        mask = df["date"].dt.date == pd.to_datetime(d_pick).date() if "date" in df.columns else np.full(len(df), True)
        df_win = df[mask].copy()
        if "event_hour" not in df_win.columns:
            df_win["event_hour"] = 0
        agg = _agg_for_window(df_win, by_cols=["GEOID","event_hour"], val_col=score_col)

    elif win.startswith("72"):
        # son 72 saat → 6’şar saatlik blok
        end = pd.to_datetime(d_pick) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        start = end - pd.Timedelta(hours=72)
        if "date" in df.columns:
            mask = (df["date"] >= start.normalize()) & (df["date"] <= end.normalize())
            df_win = df[mask].copy()
        else:
            df_win = df.copy()
        # block id
        if "event_hour" in df_win.columns:
            df_win["block6h"] = (df_win["event_hour"] // 6).astype(int)
        else:
            df_win["block6h"] = 0
        agg = _agg_for_window(df_win, by_cols=["GEOID","block6h"], val_col=score_col)

    else:
        # 1 hafta → gün bazlı
        end = pd.to_datetime(d_pick)
        start = end - pd.Timedelta(days=6)
        if "date" in df.columns:
            mask = (df["date"].dt.date >= start.date()) & (df["date"].dt.date <= end.date())
            df_win = df[mask].copy()
        else:
            df_win = df.copy()
        if "date" in df_win.columns:
            agg = _agg_for_window(df_win, by_cols=["GEOID","date"], val_col=score_col)
        else:
            agg = _agg_for_window(df_win, by_cols=["GEOID"], val_col=score_col)

    # Top-K tablo
    topk = _prepare_topk(agg if "GEOID" in agg.columns else df, score_col=score_col, k=10)

    with mid:
        st.markdown("**Tahmin Haritası / Önizleme**")
        lat_col, lon_col = _latlon_columns(df_raw)
        if lat_col and lon_col:
            # GEOID bazlı son skor (agg’den max/sum)
            geo_last = agg.groupby("GEOID", as_index=False)[score_col].max()
            # birleştir (ilk görülen lat/lon)
            centroids = df_raw.groupby("GEOID", as_index=False)[[lat_col, lon_col]].first()
            view = centroids.merge(geo_last, on="GEOID", how="left")
            # renk seviyesi (0-1)
            vmax = view[score_col].max() or 1.0
            view["level"] = (view[score_col] / vmax).clip(0,1)
            tooltip = {
                "html": "<b>GEOID:</b> {GEOID} <br/> <b>Skor:</b> {"
                        + score_col + "} <br/> <b>Seviye:</b> {level}",
                "style": {"backgroundColor": "rgba(30,30,30,0.8)", "color": "white"}
            }
            st.pydeck_chart({
                "initialViewState": {"latitude": float(view[lat_col].mean()),
                                     "longitude": float(view[lon_col].mean()),
                                     "zoom": 11},
                "layers": [{
                    "@@type": "ScatterplotLayer",
                    "data": view.to_dict("records"),
                    "get_position": f"[{lon_col}, {lat_col}]",
                    "get_radius": 80,
                    "pickable": True,
                    "opacity": 0.7,
                    "get_fill_color": "[255, (1-level)*200, (1-level)*100]"
                }],
                "tooltip": tooltip,
                "mapProvider": "carto"
            })
            st.caption("Not: Poligon katmanı yerine **nokta/centroid** ile önizleme. Grid geometri gelince PolygonLayer’a geçilecek.")
        else:
            st.info("Harita için lat/lon kolonları bulunamadı. (lat/lon veya latitude/longitude beklenir)")
            st.dataframe(topk, use_container_width=True)

    with right:
        st.markdown("**Top-10 kritik GEOID**")
        st.dataframe(topk, use_container_width=True)
        st.caption("Seviyeler: Yüksek / Orta / Düşük • Güven: q10–q90 yayılımından türetilir (varsa).")

        st.markdown("**Seçili GEOID detay (örnek)**")
        pick = st.selectbox("GEOID seç", topk["GEOID"] if "GEOID" in topk.columns else [])
        if pick:
            sub = df[df["GEOID"] == pick].copy()
            cols = [c for c in ["date","event_hour","crime_count","pred_p_occ","pred_q10","pred_q50","pred_q90","pred_expected"] if c in sub.columns]
            with st.expander(f"{pick} – Son kayıtlar", expanded=False):
                st.dataframe(sub[cols].tail(30), use_container_width=True)

    st.caption("🧪 Bu sekme iskeleti: katman anahtarları ve XAI kısa notları sonraki adımda popup/sağ panelde gösterilecek.")
