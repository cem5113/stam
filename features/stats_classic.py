# features/stats_classic.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List

# ---- Güvenli kolon bulucu ----
def _y_col(df: pd.DataFrame) -> str:
    for c in ("crime_count", "count", "y", "events"):
        if c in df.columns: return c
    # yoksa her satırı 1 say
    df["_ones"] = 1
    return "_ones"

def _ensure_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # date
    if "date" not in out.columns:
        if "datetime" in out.columns:
            out["date"] = pd.to_datetime(out["datetime"], errors="coerce").dt.date
        else:
            # eldeki en olası tarih alanlarını dene
            for cand in ("timestamp","occurred_at","reported_at"):
                if cand in out.columns:
                    out["date"] = pd.to_datetime(out[cand], errors="coerce").dt.date
                    break
    # event_hour
    if "event_hour" not in out.columns:
        if "datetime" in out.columns:
            out["event_hour"] = pd.to_datetime(out["datetime"], errors="coerce").dt.hour
        else:
            out["event_hour"] = 0
    # day_of_week
    if "day_of_week" not in out.columns:
        if "date" in out.columns:
            out["day_of_week"] = pd.to_datetime(out["date"], errors="coerce").dt.dayofweek
        else:
            out["day_of_week"] = 0
    # month
    if "month" not in out.columns:
        if "date" in out.columns:
            out["month"] = pd.to_datetime(out["date"], errors="coerce").dt.month
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
        for c in ("category_grouped","category","subcategory_grouped","subcategory"):
            if c in out.columns:
                out = out[out[c].astype(str).isin([str(x) for x in categories])]
                break
    return out

# ---- Ana özet fonksiyonları ----
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
              .pivot(index="day_of_week", columns="event_hour", values=y)
              .reindex(index=sorted(dow_map.keys()), columns=sorted(df["event_hour"].unique())))
    heat.index = [dow_map.get(i, i) for i in heat.index]

    return {"by_hour": by_hour, "by_dow": by_dow, "by_month": by_month, "heat": heat}

def spatial_top_geoid(
    df_raw: pd.DataFrame,
    n: int = 10,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = _ensure_time_cols(df_raw)
    df = _apply_filters(df, date_range, None, categories)
    y = _y_col(df)
    if "GEOID" not in df.columns:
        return pd.DataFrame(columns=["GEOID","value"])
    top = (df.groupby("GEOID", as_index=False)[y].sum()
             .rename(columns={y:"value"})
             .sort_values("value", ascending=False)
             .head(n))
    return top

def offense_breakdown(
    df_raw: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> pd.DataFrame:
    df = _ensure_time_cols(df_raw)
    df = _apply_filters(df, date_range, None, None)
    y = _y_col(df)
    for c in ("category_grouped","category","subcategory_grouped","subcategory"):
        if c in df.columns:
            bk = (df.groupby(c, as_index=False)[y].sum()
                    .rename(columns={y:"value"})
                    .sort_values("value", ascending=False))
            bk = bk.rename(columns={c: "offense"})
            return bk
    return pd.DataFrame(columns=["offense","value"])
