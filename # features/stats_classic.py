# features/stats_classic.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Tuple

# ---- yardımcılar ----
def _pick_category_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["category_grouped", "category", "subcategory_grouped", "subcategory", "crime_type"]:
        if c in df.columns:
            return c
    return None

def _score_col(df: pd.DataFrame) -> str:
    """
    Skor kolonunu otomatik seç:
    1) pred_expected  2) pred_p_occ (ortalama)  3) crime_count
    """
    if "pred_expected" in df.columns: return "pred_expected"
    if "pred_p_occ"   in df.columns:  return "pred_p_occ"
    if "crime_count"  in df.columns:  return "crime_count"
    # sayı tipindeki ilk kolon fallback
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return num_cols[0] if num_cols else df.columns[0]

# ---- mekânsal özetler ----
def spatial_top_geoid(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """
    GEOID bazında en yüksek risk/yoğunluk listesi.
    - pred_expected/pred_p_occ/crime_count'e göre sıralar
    - pred_p_occ varsa mean, diğerlerinde sum kullanır
    """
    if "GEOID" not in df.columns:
        return pd.DataFrame(columns=["GEOID", "score"])
    score = _score_col(df)
    g = (df.groupby("GEOID", as_index=False)[score]
           .mean() if score == "pred_p_occ"
           else df.groupby("GEOID", as_index=False)[score].sum())
    g = g.rename(columns={score: "score"}).sort_values("score", ascending=False)
    return g.head(n).reset_index(drop=True)

def spatial_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    GEOID başına temel özet: toplam olay, son tarih, skor.
    """
    if "GEOID" not in df.columns:
        return pd.DataFrame()
    score = _score_col(df)
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    agg = {
        score: "sum" if score != "pred_p_occ" else "mean",
    }
    if "crime_count" in out.columns:
        agg["crime_count"] = "sum"
    res = out.groupby("GEOID", as_index=False).agg(agg)
    res = res.rename(columns={score: "score"}).sort_values("score", ascending=False)
    if "date" in out.columns:
        last_seen = out.groupby("GEOID")["date"].max().rename("last_date")
        res = res.merge(last_seen, on="GEOID", how="left")
    return res

# ---- zamansal özetler ----
def hourly_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Saat 0–23 dağılımı (crime_count varsa sum; yoksa satır sayısı)."""
    d = df.copy()
    if "event_hour" not in d.columns:
        return pd.DataFrame({"event_hour": [], "value": []})
    if "crime_count" in d.columns:
        g = d.groupby("event_hour", as_index=False)["crime_count"].sum()
        g = g.rename(columns={"crime_count": "value"})
    else:
        g = d.groupby("event_hour", as_index=False).size().rename(columns={"size":"value"})
    return g.sort_values("event_hour")

def dow_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Haftanın günleri (0=Mon … 6=Sun) dağılımı."""
    d = df.copy()
    if "day_of_week" not in d.columns:
        # date varsa çıkar
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            d["day_of_week"] = d["date"].dt.dayofweek
        else:
            return pd.DataFrame({"day_of_week": [], "value": []})
    if "crime_count" in d.columns:
        g = d.groupby("day_of_week", as_index=False)["crime_count"].sum()
        g = g.rename(columns={"crime_count": "value"})
    else:
        g = d.groupby("day_of_week", as_index=False).size().rename(columns={"size":"value"})
    return g.sort_values("day_of_week")

def heatmap_day_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gün(0–6) × saat(0–23) matrisi — pivot tablo.
    """
    d = df.copy()
    if "date" in d.columns and "day_of_week" not in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["day_of_week"] = d["date"].dt.dayofweek
    if "event_hour" not in d.columns or "day_of_week" not in d.columns:
        return pd.DataFrame()
    val = "crime_count" if "crime_count" in d.columns else None
    if val:
        p = d.pivot_table(index="day_of_week", columns="event_hour", values=val, aggfunc="sum", fill_value=0)
    else:
        p = d.assign(v=1).pivot_table(index="day_of_week", columns="event_hour", values="v", aggfunc="sum", fill_value=0)
    return p.sort_index().sort_index(axis=1)

# ---- tür özetleri ----
def type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Suç türleri (category*) dağılımı — ilk 20."""
    col = _pick_category_col(df)
    if not col:
        return pd.DataFrame(columns=["type","value"])
    if "crime_count" in df.columns:
        g = df.groupby(col, as_index=False)["crime_count"].sum().rename(columns={col:"type","crime_count":"value"})
    else:
        g = df.groupby(col, as_index=False).size().rename(columns={col:"type","size":"value"})
    return g.sort_values("value", ascending=False).head(20).reset_index(drop=True)
