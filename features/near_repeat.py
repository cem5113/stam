# features/near_repeat.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple

def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"])
    return df

def compute_temp_hotspot(
    df: pd.DataFrame,
    window_days: int = 2,
    baseline_days: int = 30,
    value_col: str | None = None,  # "pred_expected" | "pred_p_occ" | "crime_count"
) -> pd.DataFrame:
    """
    Son 'window_days' içindeki toplamı, önceki 'baseline_days' ortalamasına göre karşılaştırır.
    temp_score = (son pencere ort.) / (geçmiş ort. + 1e-6)
    Döndürür: GEOID, temp_score (yüksek = anomali adayı)
    """
    if value_col is None:
        value_col = "pred_expected" if "pred_expected" in df.columns else (
            "pred_p_occ" if "pred_p_occ" in df.columns else "crime_count"
        )
    df = _ensure_date(df)
    if "GEOID" not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=["GEOID","temp_score"])

    end = pd.to_datetime(df["date"].max()).normalize()
    start_w = end - pd.Timedelta(days=window_days-1)
    start_b = end - pd.Timedelta(days=baseline_days)

    daily = (df.groupby(["GEOID", df["date"].dt.date], as_index=False)[value_col]
               .sum().rename(columns={"date":"d"}))
    daily["d"] = pd.to_datetime(daily["d"])

    win = daily[(daily["d"] >= start_w) & (daily["d"] <= end)].groupby("GEOID", as_index=False)[value_col].mean()
    base = daily[(daily["d"] >= start_b) & (daily["d"] <  start_w)].groupby("GEOID", as_index=False)[value_col].mean()

    out = win.merge(base, on="GEOID", how="left", suffixes=("_win","_base"))
    out["temp_score"] = out[f"{value_col}_win"] / (out[f"{value_col}_base"].fillna(0) + 1e-6)
    return out[["GEOID","temp_score"]].sort_values("temp_score", ascending=False)

def compute_stable_hotspot(
    df: pd.DataFrame,
    horizon_days: int = 90,
    value_col: str | None = None,
) -> pd.DataFrame:
    """
    Son 'horizon_days' içindeki ortalama yoğunluk. Yüksek değer = kalıcı hotspot eğilimi.
    Döndürür: GEOID, stable_score (0-1 normalize)
    """
    if value_col is None:
        value_col = "crime_count" if "crime_count" in df.columns else (
            "pred_expected" if "pred_expected" in df.columns else df.columns[0]
        )
    df = _ensure_date(df)
    if "GEOID" not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=["GEOID","stable_score"])

    end = pd.to_datetime(df["date"].max()).normalize()
    start = end - pd.Timedelta(days=horizon_days-1)

    daily = (df[(df["date"] >= start) & (df["date"] <= end)]
               .groupby(["GEOID", df["date"].dt.date], as_index=False)[value_col].sum())
    mean_geo = daily.groupby("GEOID", as_index=False)[value_col].mean().rename(columns={value_col:"stable_raw"})
    vmax = float(mean_geo["stable_raw"].max() or 1.0)
    mean_geo["stable_score"] = (mean_geo["stable_raw"] / vmax).clip(0,1)
    return mean_geo[["GEOID","stable_score"]].sort_values("stable_score", ascending=False)
