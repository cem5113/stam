# features/stats_classic.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List

# ---------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------
def _pick_category_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["category_grouped", "category", "subcategory_grouped", "subcategory", "crime_type"]:
        if c in df.columns:
            return c
    return None

def _score_col(df: pd.DataFrame) -> Optional[str]:
    """
    Skor kolonunu otomatik seç:
    1) pred_expected  2) pred_p_occ (ortalama)  3) crime_count
    Yoksa None döndür, sayım temelli fallback (_y_col) devreye girer.
    """
    if "pred_expected" in df.columns: return "pred_expected"
    if "pred_p_occ"   in df.columns:  return "pred_p_occ"
    if "crime_count"  in df.columns:  return "crime_count"
    return None

def _y_col(df: pd.DataFrame) -> str:
    """
    Sayım temelli özetler için güvenli hedef kolon.
    Yoksa her satırı 1 sayar.
    """
    for c in ("crime_count", "count", "y", "events"):
        if c in df.columns:
            return c
    df["_ones"] = 1
    return "_ones"

def _ensure_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # date
    if "date" not in out.columns:
        if "datetime" in out.columns:
            out["date"] = pd.to_datetime(out["datetime"], errors="coerce")
        else:
            for cand in ("timestamp", "occurred_at", "reported_at"):
                if cand in out.columns:
                    out["date"] = pd.to_datetime(out[cand], errors="coerce")
                    break
    else:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # event_hour
    if "event_hour" not in out.columns:
        if "datetime" in out.columns:
            out["event_hour"] = pd.to_datetime(out["datetime"], errors="coerce").dt.hour
        elif "date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["date"]):
            out["event_hour"] = out["date"].dt.hour.fillna(0).astype(int)
        else:
            out["event_hour"] = 0

    # day_of_week
    if "day_of_week" not in out.columns:
        if "date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["date"]):
            out["day_of_week"] = out["date"].dt.dayofweek
        else:
            out["day_of_week"] = 0

    # month
    if "month" not in out.columns:
        if "date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["date"]):
            out["month"] = out["date"].dt.month
        else:
            out["month"] = 1
    return out

def _apply_filters(
    df: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    geoids: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    out = df.copy()

    # tarih aralığı
    if date_range and "date" in out.columns:
        d1, d2 = date_range
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        if d1 is not None:
            out = out[out["date"] >= pd.to_datetime(d1)]
        if d2 is not None:
            out = out[out["date"] <= pd.to_datetime(d2)]

    # geoid filtresi
    if geoids and "GEOID" in out.columns:
        out["GEOID"] = out["GEOID"].astype(str)
        geoids = [str(g) for g in geoids]
        out = out[out["GEOID"].isin(geoids)]

    # kategori filtresi (var olan ilk uygun kolon)
    if categories:
        for c in ("category_grouped", "category", "subcategory_grouped", "subcategory", "crime_type"):
            if c in out.columns:
                out = out[out[c].astype(str).isin([str(x) for x in categories])]
                break
    return out

# ---------------------------------------------------------------------
# Mekânsal özetler
# ---------------------------------------------------------------------
def spatial_top_geoid(
    df_raw: pd.DataFrame,
    n: int = 50,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    GEOID bazında en yüksek risk/yoğunluk listesi.
    - Skor kolonları (pred_expected / pred_p_occ / crime_count) önceliklidir.
      * pred_p_occ -> mean
      * diğerleri  -> sum
    - Skor kolonları yoksa, sayım temelli fallback (_y_col) ile sum.
    - Tarih ve kategori filtreleri desteklenir.
    """
    d = _ensure_time_cols(df_raw)
    d = _apply_filters(d, date_range, None, categories)

    if "GEOID" not in d.columns:
        return pd.DataFrame(columns=["GEOID", "score"])

    score = _score_col(d)
    if score is None:
        y = _y_col(d)
        g = d.groupby("GEOID", as_index=False)[y].sum().rename(columns={y: "score"})
    else:
        if score == "pred_p_occ":
            g = d.groupby("GEOID", as_index=False)[score].mean()
        else:
            g = d.groupby("GEOID", as_index=False)[score].sum()
        g = g.rename(columns={score: "score"})

    g = g.sort_values("score", ascending=False).head(n).reset_index(drop=True)
    return g

def spatial_summary(
    df_raw: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    GEOID başına temel özet: toplam skor/sayım, son tarih.
    Skor kolonları yoksa sayım temelli fallback çalışır.
    """
    if "GEOID" not in df_raw.columns:
        return pd.DataFrame()

    out = _ensure_time_cols(df_raw)
    out = _apply_filters(out, date_range, None, categories)

    score = _score_col(out)
    agg: Dict[str, str] = {}

    if score is None:
        y = _y_col(out)
        agg[y] = "sum"
    else:
        agg[score] = "mean" if score == "pred_p_occ" else "sum"
        if "crime_count" in out.columns and score != "crime_count":
            agg["crime_count"] = "sum"

    res = out.groupby("GEOID", as_index=False).agg(agg)

    # skor kolonunu "score" ismine indir
    if score is None:
        res = res.rename(columns={_y_col(out): "score"})
    else:
        res = res.rename(columns={score: "score"})

    res = res.sort_values("score", ascending=False)

    if "date" in out.columns:
        last_seen = out.groupby("GEOID")["date"].max().rename("last_date")
        res = res.merge(last_seen, on="GEOID", how="left")
    return res

# ---------------------------------------------------------------------
# Zamansal özetler
# ---------------------------------------------------------------------
def hourly_distribution(
    df_raw: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    geoids: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Saat 0–23 dağılımı (crime_count varsa sum; yoksa satır sayısı)."""
    d = _ensure_time_cols(df_raw)
    d = _apply_filters(d, date_range, geoids, categories)

    if "event_hour" not in d.columns:
        return pd.DataFrame({"event_hour": [], "value": []})

    y = "crime_count" if "crime_count" in d.columns else _y_col(d)
    g = d.groupby("event_hour", as_index=False)[y].sum().rename(columns={y: "value"})
    return g.sort_values("event_hour")

def dow_distribution(
    df_raw: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    geoids: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Haftanın günleri (0=Mon … 6=Sun) dağılımı."""
    d = _ensure_time_cols(df_raw)
    d = _apply_filters(d, date_range, geoids, categories)

    if "day_of_week" not in d.columns:
        return pd.DataFrame({"day_of_week": [], "value": []})

    y = "crime_count" if "crime_count" in d.columns else _y_col(d)
    g = d.groupby("day_of_week", as_index=False)[y].sum().rename(columns={y: "value"})
    return g.sort_values("day_of_week")

def heatmap_day_hour(
    df_raw: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    geoids: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Gün(0–6) × saat(0–23) matrisi — pivot tablo.
    crime_count varsa sum; yoksa satır sayısı üzerinden.
    """
    d = _ensure_time_cols(df_raw)
    d = _apply_filters(d, date_range, geoids, categories)

    if "event_hour" not in d.columns or "day_of_week" not in d.columns:
        return pd.DataFrame()

    y = "crime_count" if "crime_count" in d.columns else _y_col(d)
    p = (d.groupby(["day_of_week", "event_hour"], as_index=False)[y].sum()
           .pivot(index="day_of_week", columns="event_hour", values=y)
           .fillna(0))
    return p.sort_index().sort_index(axis=1)

# ---------------------------------------------------------------------
# Tür (kategori) özetleri
# ---------------------------------------------------------------------
def type_distribution(
    df_raw: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Suç türleri (category*) dağılımı — ilk 20."""
    d = _ensure_time_cols(df_raw)
    d = _apply_filters(d, date_range, None, categories)

    col = _pick_category_col(d)
    if not col:
        return pd.DataFrame(columns=["type", "value"])

    if "crime_count" in d.columns:
        g = d.groupby(col, as_index=False)["crime_count"].sum().rename(columns={col: "type", "crime_count": "value"})
    else:
        y = _y_col(d)
        g = d.groupby(col, as_index=False)[y].sum().rename(columns={col: "type", y: "value"})
    return g.sort_values("value", ascending=False).head(20).reset_index(drop=True)

# ---------------------------------------------------------------------
# Kapsayıcı özet (hepsi bir arada)
# ---------------------------------------------------------------------
def time_distributions(
    df_raw: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    geoids: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Saat / gün / ay dağılımları ve gün×saat ısı matrisi döner.
    """
    df = _ensure_time_cols(df_raw)
    df = _apply_filters(df, date_range, geoids, categories)
    y = _y_col(df)

    # Saatlik
    by_hour = (df.groupby("event_hour", as_index=False)[y].sum()
                 .rename(columns={y: "value"})
                 .sort_values("event_hour"))

    # Gün (0=Mon)
    dow_map = {0:"Pzt",1:"Sal",2:"Çar",3:"Per",4:"Cum",5:"Cmt",6:"Paz"}
    by_dow = (df.groupby("day_of_week", as_index=False)[y].sum()
                .rename(columns={y: "value"}))
    by_dow["day_name"] = by_dow["day_of_week"].map(dow_map)
    by_dow = by_dow.sort_values("day_of_week")

    # Ay
    by_month = (df.groupby("month", as_index=False)[y].sum()
                  .rename(columns={y: "value"})
                  .sort_values("month"))

    # Gün×saat ısı matrisi
    heat = (df.groupby(["day_of_week","event_hour"], as_index=False)[y].sum()
              .pivot(index="day_of_week", columns="event_hour", values=y))
    heat = heat.reindex(index=sorted(dow_map.keys()), columns=sorted(df["event_hour"].unique()))
    heat.index = [dow_map.get(i, i) for i in heat.index]

    return {"by_hour": by_hour, "by_dow": by_dow, "by_month": by_month, "heat": heat}

def offense_breakdown(
    df_raw: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Tür bazında (ilk uygun kategori kolonunda) toplam sayım.
    """
    df = _ensure_time_cols(df_raw)
    df = _apply_filters(df, date_range, None, None)
    y = _y_col(df)
    for c in ("category_grouped","category","subcategory_grouped","subcategory","crime_type"):
        if c in df.columns:
            bk = (df.groupby(c, as_index=False)[y].sum()
                    .rename(columns={y:"value"})
                    .sort_values("value", ascending=False))
            bk = bk.rename(columns={c: "offense"})
            return bk
    return pd.DataFrame(columns=["offense","value"])
