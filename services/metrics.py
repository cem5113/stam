# services/metrics.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def _pick_score_col(df: pd.DataFrame) -> str:
    for c in ("pred_expected", "pred_p_occ", "crime_count"):
        if c in df.columns:
            return c
    return df.columns[0]

def _recent_day(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if "date" not in df.columns:
        return None
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    return None if d.empty else d.max().normalize()

def hitrate_at_k(
    df: pd.DataFrame,
    k: int = 10,
    geoid_col: str = "GEOID",
    label_col: str = "crime_count",
    score_col: Optional[str] = None,
) -> Optional[float]:
    """
    Son gün için: en yüksek skorlu K GEOID içinde gerçekleşen olay oranı.
    """
    if geoid_col not in df.columns or label_col not in df.columns:
        return None
    df = _ensure_date(df)
    day = _recent_day(df)
    if day is not None:
        df = df[pd.to_datetime(df["date"], errors="coerce").dt.normalize() == day]
    if df.empty:
        return None

    s_col = score_col or _pick_score_col(df)

    # GEOID bazında skor ve gerçekleşen olay sayısı
    g = df.groupby(geoid_col, as_index=False).agg(
        score=(s_col, "sum"),
        y=(label_col, "sum"),
    ).sort_values("score", ascending=False)

    top = g.head(max(1, int(k)))
    total_y = float(g["y"].sum())
    if total_y <= 0:
        return None
    return float(top["y"].sum() / total_y)

def brier_score(df: pd.DataFrame) -> Optional[float]:
    """
    Basit Brier: p=pred_p_occ, y=(crime_count>0) ikili hedefi.
    Son 30 gün üzerinden hesaplanır (varsa).
    """
    if "pred_p_occ" not in df.columns:
        return None
    df = _ensure_date(df)
    if "date" in df.columns:
        day = _recent_day(df)
        if day is not None:
            start = day - pd.Timedelta(days=29)
            mask = (pd.to_datetime(df["date"], errors="coerce") >= start) & \
                   (pd.to_datetime(df["date"], errors="coerce") <= day)
            df = df[mask]
    if df.empty:
        return None

    p = pd.to_numeric(df["pred_p_occ"], errors="coerce")
    if "crime_count" in df.columns:
        y = (pd.to_numeric(df["crime_count"], errors="coerce").fillna(0) > 0).astype(int)
    else:
        return None
    m = (~p.isna()) & (~y.isna())
    if m.sum() == 0:
        return None
    return float(np.mean((p[m] - y[m]) ** 2))

def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    UI için güvenli özet.
    """
    hr = hitrate_at_k(df, k=10)
    br = brier_score(df)
    last_day = _recent_day(df)
    return {
        "hitrate_top10": None if hr is None else round(hr * 100, 1),  # yüzde
        "brier": None if br is None else round(br, 4),
        "last_day": None if last_day is None else str(last_day.date()),
    }
