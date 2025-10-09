# ui/tab_forecast.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from models.xai import attach_xai

from config.settings import RISK_THRESHOLDS
from dataio.loaders import load_sf_crime_latest
from services.tz import now_sf_str

# Opsiyonel XAI (varsa kullan)
try:
    from models.xai import brief_xai_for_row
except Exception:
    brief_xai_for_row = None

# (Varsa) geçici/kalıcı hotspot fonksiyonlarını kullan; yoksa sessizce atla
try:
    from features.near_repeat import compute_temp_hotspot, compute_stable_hotspot
except Exception:
    compute_temp_hotspot = None
    compute_stable_hotspot = None

# ----------------- helpers -----------------
def _pick_category_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["category_grouped", "category", "subcategory_grouped", "subcategory", "crime_type"]:
        if c in df.columns:
            return c
    return None

def _risk_level_from_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "Bilinmiyor"
    return "Yüksek" if p > RISK_THRESHOLDS["mid"] else ("Orta" if p > RISK_THRESHOLDS["low"] else "Düşük")

def _confidence_label(q10: float, q90: float) -> str:
    if q10 is None or q90 is None or np.isnan(q10) or np.isnan(q90):
        return "—"
    spread = float(q90) - float(q10)
    if spread <= 0.5:  return "Yüksek güven"
    if spread <= 1.5:  return "Orta güven"
    return "Düşük güven"

def _latlon_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cands = [("lat", "lon"), ("latitude", "longitude"), ("y", "x"), ("LAT", "LON")]
    for latc, lonc in cands:
        if latc in df.columns and lonc in df.columns:
            return latc, lonc
    return None, None

def _ensure_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns and "datetime" in out.columns:
        out["date"] = pd.to_datetime(out["datetime"], errors="coerce").dt.date
    if "event_hour" not in out.columns:
        if "datetime" in out.columns:
            out["event_hour"] = pd.to_datetime(out["datetime"], errors="coerce").dt.hour
        else:
            out["event_hour"] = 0
    return out

def _fallback_proxy_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tahmin kolonları yoksa: son 7 güne göre GEOID bazlı normalize (0..1) 'p_proxy'.
    """
    dff = _ensure_time_cols(df)
    if "date" not in dff.columns:
        dff["p_proxy"] = 0.0
        return dff
    dff["date"] = pd.to_datetime(dff["date"], errors="coerce")
    dmax = dff["date"].max()
    if pd.isna(dmax):
        dff["p_proxy"] = 0.0
        return dff
    sub = dff[(dff["date"] >= dmax - pd.Timedelta(days=7)) & (dff["date"] <= dmax)].copy()
    ycol = "crime_count" if "crime_count" in sub.columns else None
    if ycol is None:
        sub["__ones"] = 1
        ycol = "__ones"
    g = sub.groupby("GEOID", as_index=False)[ycol].sum().rename(columns={ycol: "recent_sum"})
    mx = float(g["recent_sum"].max()) if len(g) else 1.0
    g["p_proxy"] = g["recent_sum"] / (mx if mx > 0 else 1.0)
    out = dff.merge(g[["GEOID", "p_proxy"]], on="GEOID", how="left")
    out["p_proxy"] = out["p_proxy"].fillna(0.0)
    return out

def _pick_score_col(df: pd.DataFrame) -> str:
    # Öncelik: pred_p_occ (olasılık) > pred_expected (normalize edilip 0..1) > p_proxy > crime_count
    if "pred_p_occ" in df.columns:      return "pred_p_occ"
    if "pred_expected" in df.columns:   return "pred_expected"
    if "p_proxy" in df.columns:         return "p_proxy"
    if "crime_count" in df.columns:     return "crime_count"
    # fallback: ilk sayısal kolon
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return num_cols[0] if num_cols else df.columns[0]

def _summarize_window_geoid(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    Zaman penceresine indirgenmiş df'yi GEOID bazında tek skora çevirir:
    - Olasılık ('pred_p_occ' / 'p_proxy'): mean
    - Beklenen sayı ('pred_expected') veya sayım: sum
    """
    dfx = df.copy()
    if score_col in ("pred_p_occ", "p_proxy"):
        g = dfx.groupby("GEOID", as_index=False)[score_col].mean()
    else:
        g = dfx.groupby("GEOID", as_index=False)[score_col].sum()
    return g

def _normalize_0_1(s: pd.Series) -> pd.Series:
    mx = float(s.max()) if len(s) else 1.0
    return (s / (mx if mx > 0 else 1.0)).clip(0, 1)

def _prepare_topk(geo_scores: pd.DataFrame, score_col: str, k: int = 10,
                  q10: Optional[pd.Series] = None, q90: Optional[pd.Series] = None) -> pd.DataFrame:
    out = geo_scores.copy()
    # risk seviyesi: olasılık kolonları için doğrudan, diğerlerinde normalize ederek
    if score_col in ("pred_p_occ", "p_proxy"):
        out["risk_level"] = out[score_col].map(_risk_level_from_p)
    else:
        out["risk_norm"] = _normalize_0_1(out[score_col])
        out["risk_level"] = out["risk_norm"].map(_risk_level_from_p)

    # güven etiketi (varsa)
    if q10 is not None and q90 is not None and len(q10) == len(out) == len(q90):
        out["güven"] = [_confidence_label(a, b) for a, b in zip(q10.values, q90.values)]
    else:
        out["güven"] = "—"

    cols = ["GEOID", score_col, "risk_level", "güven"]
    keep = [c for c in cols if c in out.columns]
    return out[keep].sort_values(score_col, ascending=False).head(k)

# ----------------- UI -----------------
def render():
    st.subheader("🧭 Suç Tahmini")
    st.caption(f"Son kontrol: {now_sf_str()} (SF)")

    # --- Veri yükle ---
    try:
        df_raw, src = load_sf_crime_latest()
    except Exception as e:
        st.error("Veri yüklenemedi.")
        st.exception(e)
        return

    # Tahmin kolonları yoksa proxy oluştur
    if not any(c in df_raw.columns for c in ("pred_p_occ", "pred_expected", "pred_q50")):
        df_raw = _fallback_proxy_risk(df_raw)
    df_x = attach_xai(df_raw, topk=3) 
    
    # Sol panel (filtreler)
    left, mid, right = st.columns([1.1, 2.2, 1.2])
    with left:
        st.markdown("**Zaman Penceresi**")
        win = st.radio(
            "Seçim",
            ["0–24 (saatlik)", "72 saat (6’şar saat)", "1 hafta (günlük)"],
            index=0,
            help="SF saatine göre pencereler",
        )

        # referans tarih
        latest = None
        if "date" in df_raw.columns:
            latest = pd.to_datetime(df_raw["date"], errors="coerce").dropna().max()
        d_pick = st.date_input(
            "Referans gün",
            value=(latest.date() if pd.notnull(latest) and latest is not None else pd.Timestamp.today().date()),
        )

        # kategori filtresi
        cat_col = _pick_category_col(df_raw)
        cats = sorted(df_raw[cat_col].dropna().unique()) if cat_col else []
        pick_cats: List[str] = st.multiselect("Suç kategorisi", cats, default=[], help="Boş bırak = tüm kategoriler")

        st.markdown("**Harita Katmanları**")
        layer_risk   = st.checkbox("Tahmin katmanı (risk)", True)
        layer_temp   = st.checkbox("Geçici hotspot", True)
        layer_stable = st.checkbox("Kalıcı hotspot", True)

        k_top = st.number_input("Top-K liste", min_value=5, max_value=50, value=10, step=5)

    # Filtre uygula (kategori)
    df = df_raw.copy()
    if cat_col and pick_cats:
        df = df[df[cat_col].isin(pick_cats)]

    # Tarih-saat alanlarını hazırla
    df = _ensure_time_cols(df)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Skor kolonu
    score_col = _pick_score_col(df)

    # Seçilen pencereye göre veri altkümesi
    if win.startswith("0–24"):
        # seçilen gün (00:00–23:59)
        if "date" in df.columns:
            mask = df["date"].dt.date == pd.to_datetime(d_pick).date()
            df_win = df[mask].copy()
        else:
            df_win = df.copy()

    elif win.startswith("72"):
        # referans günün sonuna kadar 72 saat geriye
        end = pd.to_datetime(d_pick) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        start = end - pd.Timedelta(hours=72)
        if "date" in df.columns:
            mask = (df["date"] >= start.normalize()) & (df["date"] <= end.normalize())
            df_win = df[mask].copy()
        else:
            df_win = df.copy()

    else:
        # 1 hafta (günlük)
        end = pd.to_datetime(d_pick)
        start = end - pd.Timedelta(days=6)
        if "date" in df.columns:
            mask = (df["date"].dt.date >= start.date()) & (df["date"].dt.date <= end.date())
            df_win = df[mask].copy()
        else:
            df_win = df.copy()

    # GEOID bazında tek skor
    geo_scores = _summarize_window_geoid(df_win, score_col=score_col)

    # Top-K tablo (güven etiketini varsa kullan)
    q10 = q90 = None
    if {"pred_q10", "pred_q90"}.issubset(df_win.columns):
        # GEOID bazında q10/q90 için mean alın (hızlı özet)
        q10 = df_win.groupby("GEOID")["pred_q10"].mean()
        q90 = df_win.groupby("GEOID")["pred_q90"].mean()
        # align
        geo_scores = geo_scores.merge(q10.rename("q10"), on="GEOID", how="left")
        geo_scores = geo_scores.merge(q90.rename("q90"), on="GEOID", how="left")
        topk = _prepare_topk(geo_scores, score_col=score_col, k=int(k_top),
                             q10=geo_scores["q10"], q90=geo_scores["q90"])
    else:
        topk = _prepare_topk(geo_scores, score_col=score_col, k=int(k_top))

    # Orta panel — Harita
    with mid:
        st.markdown("**Tahmin Haritası**")
        lat_col, lon_col = _latlon_columns(df_raw)
        if lat_col and lon_col and "GEOID" in df_raw.columns:
            # GEOID → tekil koordinat (ortalama)
            centroids = (
                df_raw.groupby("GEOID", as_index=False)[[lat_col, lon_col]]
                .mean(numeric_only=True)
                .dropna()
            )
            view = centroids.merge(geo_scores, on="GEOID", how="left")
            # 0..1 seviye (olasılık ise direkt kullan, değilse normalize)
            if score_col in ("pred_p_occ", "p_proxy"):
                view["level"] = view[score_col].clip(0, 1)
            else:
                view["level"] = _normalize_0_1(view[score_col])

            # Hotspot katmanları (opsiyonel, mevcutsa)
            layers = []
            tooltip = {
                "html": "<b>GEOID:</b> {GEOID}<br/>"
                        f"<b>Skor:</b> {{{score_col}}}<br/>"
                        "<b>Seviye(0–1):</b> {level}",
                "style": {"backgroundColor": "rgba(30,30,30,0.85)", "color": "white"}
            }

            if layer_risk:
                layers.append({
                    "@@type": "ScatterplotLayer",
                    "data": view.to_dict("records"),
                    "get_position": f"[{lon_col}, {lat_col}]",
                    "get_radius": 90,
                    "pickable": True,
                    "opacity": 0.7,
                    "get_fill_color": "[255, (1-level)*200, (1-level)*100]"
                })

            # Geçici hotspot
            if layer_temp and compute_temp_hotspot is not None:
                try:
                    temp_df = compute_temp_hotspot(df_raw, window_days=2, baseline_days=30)  # GEOID, temp_score
                    if temp_df is not None and "GEOID" in temp_df.columns:
                        view_temp = view.merge(temp_df, on="GEOID", how="left").dropna(subset=["temp_score"])
                        if len(view_temp) > 0:
                            layers.append({
                                "@@type": "ScatterplotLayer",
                                "data": view_temp.to_dict("records"),
                                "get_position": f"[{lon_col}, {lat_col}]",
                                "get_radius": 120,
                                "pickable": True,
                                "opacity": 0.35,
                                "get_fill_color": "[255,140,0]"
                            })
                except Exception:
                    st.caption("Geçici hotspot hesaplanamadı (near_repeat modülü).")

            # Kalıcı hotspot
            if layer_stable and compute_stable_hotspot is not None:
                try:
                    stab_df = compute_stable_hotspot(df_raw, horizon_days=90)  # GEOID, stable_score
                    if stab_df is not None and "GEOID" in stab_df.columns:
                        view_stab = view.merge(stab_df, on="GEOID", how="left").dropna(subset=["stable_score"])
                        if len(view_stab) > 0:
                            layers.append({
                                "@@type": "ScatterplotLayer",
                                "data": view_stab.to_dict("records"),
                                "get_position": f"[{lon_col}, {lat_col}]",
                                "get_radius": 70,
                                "pickable": True,
                                "opacity": 0.35,
                                "get_fill_color": "[0,128,255]"
                            })
                except Exception:
                    st.caption("Kalıcı hotspot hesaplanamadı (near_repeat modülü).")

            if layers:
                st.pydeck_chart({
                    "initialViewState": {
                        "latitude": float(view[lat_col].mean()),
                        "longitude": float(view[lon_col].mean()),
                        "zoom": 11
                    },
                    "layers": layers,
                    "tooltip": tooltip,
                    "mapProvider": "carto"
                })
            else:
                st.info("Harita katmanı yok (seçili katmanlar kapalı olabilir).")
        else:
            st.info("Harita için lat/lon kolonları bulunamadı. (lat/lon veya latitude/longitude beklenir)")

    # Sağ panel — Top-K & GEOID detay
    with right:
        st.markdown(f"**Top-{int(k_top)} kritik GEOID**")
        st.dataframe(topk, use_container_width=True, height=360)
        st.caption("Seviye: Yüksek / Orta / Düşük • Güven: q10–q90 yayılımı (varsa).")

        st.download_button(
            f"⬇️ Top-{int(k_top)} CSV",
            data=topk.to_csv(index=False).encode("utf-8"),
            file_name=f"top{int(k_top)}_geoid.csv",
            mime="text/csv"
        )

        st.markdown("**Seçili GEOID detay**")
        pick = st.selectbox("GEOID seç", topk["GEOID"] if "GEOID" in topk.columns else [])
        if pick is not None and "GEOID" in df.columns:
            sub = df[df["GEOID"].astype(str) == str(pick)].copy()

            # P(olay) / Beklenen
            if score_col in ("pred_p_occ", "p_proxy"):
                p = float(np.nanmean(sub[score_col])) if len(sub) else np.nan
                st.metric("Tahmin seviyesi", _risk_level_from_p(p))
                st.metric("Risk skoru (0–1)", f"{p:.2f}" if not np.isnan(p) else "—")
            else:
                total = float(np.nansum(sub[score_col])) if len(sub) else np.nan
                st.metric("Beklenen/Toplam skor", f"{(0.0 if np.isnan(total) else total):.2f}")

            # Güven (q10/q90 varsa, o GEOID için)
            conf_text = "—"
            if {"pred_q10", "pred_q90"}.issubset(df.columns):
                qsub = sub[["pred_q10", "pred_q90"]].dropna()
                if len(qsub) > 0:
                    conf_text = _confidence_label(float(qsub["pred_q10"].mean()), float(qsub["pred_q90"].mean()))
            st.metric("Güven", conf_text)

            # Gerçekleşen (son 7 / 30 gün)
            if "date" in sub.columns:
                sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
                dmax = sub["date"].max()
                if "crime_count" in sub.columns and pd.notna(dmax):
                    s7  = sub[sub["date"] >= dmax - pd.Timedelta(days=7)]["crime_count"].sum()
                    s30 = sub[sub["date"] >= dmax - pd.Timedelta(days=30)]["crime_count"].sum()
                    st.metric("Son 7 gün", int(s7))
                    st.metric("Son 30 gün", int(s30))

            # İlk 3 etken (XAI)
            if brief_xai_for_row is not None and len(sub) > 0:
                row = sub.sort_values("date", ascending=False).iloc[0].to_dict() if "date" in sub.columns else sub.iloc[0].to_dict()
                facts = brief_xai_for_row(row)
                if facts:
                    st.markdown("**İlk 3 etken (XAI)**")
                    for f in facts:
                        st.markdown(f"- **{f['name']}** — {f['why']}")

            if "xai_reasons" in df_x.columns:
                r = df_x[df_x["GEOID"].astype(str) == str(pick)]
                if len(r) > 0 and isinstance(r.iloc[0]["xai_reasons"], str) and r.iloc[0]["xai_reasons"]:
                    st.markdown("**İlk 3 etken (XAI)**")
                    st.markdown(r.iloc[0]["xai_reasons"])

            # Kısa trend
            if "date" in sub.columns:
                sub_day = sub.assign(day=sub["date"].dt.date).groupby("day", as_index=False)[score_col].sum()
                if len(sub_day) > 0:
                    sub_day = sub_day.sort_values("day").tail(14)
                    st.line_chart(sub_day.set_index("day")[score_col], height=160)
