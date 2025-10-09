# models/baseline.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

def _pick_y(df: pd.DataFrame) -> str:
    for c in ("pred_expected", "pred_p_occ", "crime_count"):
        if c in df.columns:
            return c
    # sayısal ilk kolon
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return num[0] if num else df.columns[0]

def fit_mean_by_groups(
    df: pd.DataFrame,
    y_col: Optional[str] = None,
    group_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Basit baseline: grup ortalamaları (ör. GEOID × event_hour) + global fallback.
    """
    if y_col is None:
        y_col = _pick_y(df)
    if group_cols is None:
        group_cols = [c for c in ["GEOID", "event_hour"] if c in df.columns]

    dfx = df.copy()
    dfx = dfx.dropna(subset=[y_col])
    if not group_cols:
        mu = float(np.nanmean(dfx[y_col])) if len(dfx) else 0.0
        return {"y_col": y_col, "groups": [], "table": pd.DataFrame(), "global_mean": mu}

    table = (
        dfx.groupby(group_cols, as_index=False)[y_col]
           .mean()
           .rename(columns={y_col: "mean"})
    )
    glob = float(np.nanmean(dfx[y_col])) if len(dfx) else 0.0
    return {"y_col": y_col, "groups": group_cols, "table": table, "global_mean": glob}

def predict_mean_by_groups(model: Dict, df_new: pd.DataFrame) -> pd.Series:
    """
    Eğitimdeki grup ortalamasını yeni veriye uygular; bulunamazsa global_mean.
    """
    y_col = model["y_col"]
    groups: List[str] = model["groups"]
    table: pd.DataFrame = model["table"]
    glob: float = model["global_mean"]

    if not groups or table.empty:
        return pd.Series([glob] * len(df_new), index=df_new.index, name=f"baseline_{y_col}")

    out = df_new.copy()
    out = out.merge(table, on=groups, how="left")
    pred = out["mean"].fillna(glob)
    return pd.to_numeric(pred, errors="coerce").fillna(glob).rename(f"baseline_{y_col}")
