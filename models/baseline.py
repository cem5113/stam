# models/baseline.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, List, Dict

# ---------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------
def _ensure_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    date/datetime kolonlarını güvenle datetime'a çevirir.
    'date' yoksa 'datetime' üzerinden üretmeyi dener.
    """
    out = df.copy()
    if "date" not in out.columns and "datetime" in out.columns:
        out["date"] = pd.to_datetime(out["datetime"], errors="coerce")
    else:
        out["date"] = pd.to_datetime(out.get("date", pd.NaT), errors="coerce")
    return out

def _y_col(df: pd.DataFrame) -> str:
    """
    Sayım temelli hedef kolon.
    Yoksa her satırı 1 sayar.
    """
    for c in ("crime_count", "count", "y", "events"):
        if c in df.columns:
            return c
    df = df.copy()
    df["_ones"] = 1
    return "_ones"

def _pick_y(df: pd.DataFrame) -> str:
    """
    Skor için tercih sırası:
      1) pred_expected
      2) pred_p_occ
      3) crime_count
      4) ilk sayısal kolon (fallback) ya da ilk kolon
    """
    for c in ("pred_expected", "pred_p_occ", "crime_count"):
        if c in df.columns:
            return c
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return nums[0] if nums else df.columns[0]

# ---------------------------------------------------------------------
# 1) Pencere-temelli baseline sınıfı
# ---------------------------------------------------------------------
class BaselineModel:
    """
    Çok hafif bir fallback:
      - Son N gün içinde GEOID bazında toplamı alır
      - Max-normalize ederek [0..1] 'p_proxy' üretir
      - (opsiyonel) Saat etkisi: event_hour varsa GEOID×hour ort. ile çarpar ve yeniden normalize eder
    Gerçek model gelene kadar UI’nin çalışmasını garanti eder.
    """

    def __init__(self, window_days: int = 7, use_hour: bool = True):
        self.window_days = int(window_days)
        self.use_hour = bool(use_hour)
        self.table_geo: Optional[pd.DataFrame] = None
        self.table_geo_hour: Optional[pd.DataFrame] = None

    def fit(self, df_raw: pd.DataFrame) -> "BaselineModel":
        df = _ensure_time(df_raw)
        y = _y_col(df)

        dmax = pd.to_datetime(df["date"], errors="coerce").dropna().max()
        if pd.isna(dmax):
            dsub = df.copy()
        else:
            dmin = dmax - pd.Timedelta(days=self.window_days - 1)
            d = pd.to_datetime(df["date"], errors="coerce")
            dsub = df[(d >= dmin) & (d <= dmax)].copy()

        if "GEOID" not in dsub.columns:
            self.table_geo = pd.DataFrame(columns=["GEOID", "p_proxy"])
            self.table_geo_hour = pd.DataFrame(columns=["GEOID", "event_hour", "hour_weight"])
            return self

        g = (dsub.groupby("GEOID", as_index=False)[y].sum()
                 .rename(columns={y: "recent_sum"}))
        mx = float(g["recent_sum"].max() or 1.0)
        g["p_proxy"] = (g["recent_sum"] / (mx if mx > 0 else 1.0)).clip(0, 1)
        self.table_geo = g[["GEOID", "p_proxy"]].copy()

        # Saat etkisi (opsiyonel)
        if self.use_hour and "event_hour" in dsub.columns:
            gh = (dsub.groupby(["GEOID", "event_hour"], as_index=False)[y].mean()
                    .rename(columns={y: "hour_mean"}))
            # GEOID içinde normalize
            gh["hour_weight"] = gh.groupby("GEOID")["hour_mean"].transform(
                lambda s: (s / (s.max() if s.max() > 0 else 1.0)).clip(0, 1)
            )
            self.table_geo_hour = gh[["GEOID", "event_hour", "hour_weight"]].copy()
        else:
            self.table_geo_hour = None

        return self

    def predict(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Girdi df’ye p_proxy ekler (ve varsa saat düzeltmesini uygular).
        Dönen DataFrame: df + ['p_proxy'] (0..1)
        """
        if self.table_geo is None:
            # fit edilmemişse hızlı bir fit yap
            self.fit(df_raw)

        df = _ensure_time(df_raw)
        out = df.merge(self.table_geo, on="GEOID", how="left") if "GEOID" in df.columns else df.copy()
        if "p_proxy" not in out.columns:
            out["p_proxy"] = 0.0

        if self.table_geo_hour is not None and "event_hour" in out.columns and "GEOID" in out.columns:
            out = out.merge(self.table_geo_hour, on=["GEOID", "event_hour"], how="left")
            out["p_proxy"] = (out["p_proxy"] * out["hour_weight"].fillna(1.0)).clip(0, 1)
            out.drop(columns=["hour_weight"], inplace=True, errors="ignore")

        return out

# ---------------------------------------------------------------------
# 2) Grup-ortalaması baseline (GEOID × event_hour vb.)
# ---------------------------------------------------------------------
def fit_mean_by_groups(
    df: pd.DataFrame,
    y_col: Optional[str] = None,
    group_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Basit baseline: grup ortalamaları (ör. GEOID × event_hour) + global fallback.
    Dönüş: {'y_col','groups','table','global_mean'}
    """
    if y_col is None:
        y_col = _pick_y(df)
    if group_cols is None:
        group_cols = [c for c in ["GEOID", "event_hour"] if c in df.columns]

    dfx = df.copy().dropna(subset=[y_col])
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
    Dönüş: pd.Series (isim: baseline_{y_col})
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
