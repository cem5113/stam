# services/metrics.py
from __future__ import annotations
"""
Basit sayaç/zamanlayıcı (telemetry) + model/DF metrikleri tek dosyada birleşik.
- Prod'da Prometheus/OTel'e taşınabilir (şimdilik in-memory).
"""

# ── Sayaç & zamanlama (hafif telemetry) ───────────────────────────────────────
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Iterable, List

_COUNTERS: Dict[str, int] = {}
_TIMES: Dict[str, float] = {}

def incr(name: str, by: int = 1) -> None:
    """Sayaç artır."""
    _COUNTERS[name] = _COUNTERS.get(name, 0) + int(by)

def get_counter(name: str) -> int:
    """Sayaç değeri (yoksa 0)."""
    return _COUNTERS.get(name, 0)

@contextmanager
def timing(name: str):
    """with timing("load_data"): ...  → süre (saniye) kaydedilir."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _TIMES[name] = time.perf_counter() - t0

def last_timing(name: str) -> float:
    """Son ölçülen süre (saniye)."""
    return _TIMES.get(name, 0.0)

# ── Vektör/metrikler ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

def brier_score_vector(y_true: Iterable[float], p_hat: Iterable[float]) -> float:
    """Brier skoru: mean((p - y)^2) — y∈{0,1}, p∈[0,1]."""
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(p_hat), dtype=float)
    return float(np.mean((p - y) ** 2))

def roc_auc(y_true: Iterable[int], score: Iterable[float]) -> Optional[float]:
    """Sklearn'siz AUC (rank tabanlı). Tüm örnekler tek sınıfsa None."""
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(score), dtype=float)
    if y.size == 0 or y.min() == y.max():
        return None
    order = s.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(s)) + 1.0
    n1 = float((y == 1).sum()); n0 = float((y == 0).sum())
    if n0 == 0 or n1 == 0:
        return None
    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n1 * (n1 + 1) / 2.0) / (n0 * n1)
    return float(auc)

def calibration_table(y_true: Iterable[int], p_hat: Iterable[float], n_bins: int = 10) -> pd.DataFrame:
    """Kalibrasyon tablosu: bin başına ort. tahmin vs. pozitif oranı."""
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(p_hat), dtype=float)
    if y.size == 0:
        return pd.DataFrame(columns=["bin","bin_lo","bin_hi","count","avg_pred","frac_pos"])
    p = np.clip(p, 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins, right=True)
    recs: List[Dict[str, Any]] = []
    for b in range(1, n_bins + 1):
        m = idx == b
        recs.append({
            "bin": b, "bin_lo": bins[b - 1], "bin_hi": bins[b],
            "count": int(m.sum()),
            "avg_pred": float(np.mean(p[m])) if m.any() else np.nan,
            "frac_pos": float(np.mean(y[m])) if m.any() else np.nan
        })
    return pd.DataFrame(recs)

# ── DF tabanlı metrik yardımcıları ───────────────────────────────────────────
def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def _pick_score_col(df: pd.DataFrame) -> str:
    for c in ("pred_expected", "pred_p_occ", "crime_count"):
        if c in df.columns:
            return c
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return nums[0] if nums else df.columns[0]

def _recent_day(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if "date" not in df.columns:
        return None
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    return None if d.empty else d.max().normalize()

# ── DF tabanlı metrikler ─────────────────────────────────────────────────────
def hit_rate_at_k_pairs(
    df_pred: pd.DataFrame,
    df_actual: pd.DataFrame,
    k: int = 10,
    key_col: str = "GEOID",
    actual_y_col: str = "crime_count",
) -> float:
    """
    Ayrık pencerelerde HitRate@K.
      - df_pred: {GEOID, score}/{GEOID, pred_*}
      - df_actual: {GEOID, crime_count} (>=1 → pozitif)
    """
    if key_col not in df_pred.columns or key_col not in df_actual.columns:
        return 0.0
    score_col = _pick_score_col(df_pred)
    topk = (df_pred[[key_col, score_col]].dropna()
           .sort_values(score_col, ascending=False).head(int(k)))
    if actual_y_col not in df_actual.columns:
        df_actual = df_actual.assign(**{actual_y_col: 1})
    pos_geoids = set(df_actual.loc[df_actual[actual_y_col] > 0, key_col].astype(str))
    top_geoids = set(topk[key_col].astype(str))
    if not pos_geoids:
        return 0.0
    hits = len(top_geoids & pos_geoids)
    return float(hits / min(len(pos_geoids), k))

def hitrate_at_k_df(
    df: pd.DataFrame,
    k: int = 10,
    geoid_col: str = "GEOID",
    label_col: str = "crime_count",
    score_col: Optional[str] = None,
) -> Optional[float]:
    """
    Son gün için: en yüksek skorlu K GEOID içinde gerçekleşen olay oranı.
    (date yoksa tüm veri üzerinden)
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
    g = (df.groupby(geoid_col, as_index=False)
           .agg(score=(s_col, "sum"), y=(label_col, "sum"))
           .sort_values("score", ascending=False))

    top = g.head(max(1, int(k)))
    total_y = float(g["y"].sum())
    if total_y <= 0:
        return None
    return float(top["y"].sum() / total_y)

def brier_score_df(df: pd.DataFrame) -> Optional[float]:
    """
    Brier (DF): p=pred_p_occ, y=(crime_count>0). Son 30 gün üzerinden.
    """
    if "pred_p_occ" not in df.columns:
        return None
    df = _ensure_date(df)
    if "date" in df.columns:
        day = _recent_day(df)
        if day is not None:
            start = day - pd.Timedelta(days=29)
            m = (pd.to_datetime(df["date"], errors="coerce") >= start) & \
                (pd.to_datetime(df["date"], errors="coerce") <= day)
            df = df[m]
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
    """UI için kısa özet: HitRate@10 (%) • Brier • son gün."""
    hr = hitrate_at_k_df(df, k=10)
    br = brier_score_df(df)
    last_day = _recent_day(df)
    return {
        "hitrate_top10": None if hr is None else round(hr * 100, 1),
        "brier": None if br is None else round(br, 4),
        "last_day": None if last_day is None else str(last_day.date()),
    }
