# models/uncertainty.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import Optional, Dict, Iterable, Tuple, List

# ------------------------------------------------------------
# Risk / Güven etiketleri (UI ile uyumlu)
# ------------------------------------------------------------
def risk_level_from_prob(p: float, thresholds: Dict[str, float]) -> str:
    """
    p: gerçekleşme olasılığı (0..1)
    thresholds: {"low": 0.33, "mid": 0.66, ...}
    """
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "Bilinmiyor"
    low = float(thresholds.get("low", 0.33))
    mid = float(thresholds.get("mid", 0.66))
    return "Yüksek" if p > mid else ("Orta" if p > low else "Düşük")

def confidence_label(q10: Optional[float], q90: Optional[float]) -> str:
    """
    q10–q90 yayılımından basit güven etiketi.
    """
    if q10 is None or q90 is None:
        return "—"
    try:
        a = float(q10); b = float(q90)
    except Exception:
        return "—"
    if np.isnan(a) or np.isnan(b):
        return "—"
    spread = b - a
    if spread <= 0.5:  return "Yüksek güven"
    if spread <= 1.5:  return "Orta güven"
    return "Düşük güven"

def add_confidence_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    q10/q90 varsa 'confidence' sütunu ekler; yoksa dokunmaz.
    """
    if {"pred_q10", "pred_q90"}.issubset(df.columns):
        out = df.copy()
        out["confidence"] = [
            confidence_label(a, b) for a, b in zip(out["pred_q10"], out["pred_q90"])
        ]
        return out
    return df

# ------------------------------------------------------------
# Poisson belirsizlik (kantiller) ve p_occ
# ------------------------------------------------------------
# Küçük lambda → kesin CDF; büyük lambda → normal yaklaşımı
_Z = {0.1: -1.2815515655446004, 0.5: 0.0, 0.9: 1.2815515655446004}

def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def _poisson_cdf(k: int, lam: float) -> float:
    # Küçük lam’larda güvenli, büyük lam’da yavaş olabilir
    s = 0.0
    for i in range(0, max(0, k) + 1):
        s += _poisson_pmf(i, lam)
    return min(s, 1.0)

def poisson_quantile(lam: float, q: float) -> int:
    """
    Poisson(lam) için yaklaşık/yarı-kesin kantil.
    q tipik olarak {0.1, 0.5, 0.9}.
    """
    if lam <= 0:
        return 0
    if lam >= 50 and q in _Z:
        # normal approx
        val = lam + _Z[q] * math.sqrt(lam)
        return max(0, int(round(val)))
    # artan arama
    k = 0
    k_max = int(max(10, lam + 10 * math.sqrt(max(lam, 1.0))))
    while k < k_max and _poisson_cdf(k, lam) < q:
        k += 1
    return k

def poisson_quantiles(lam: Iterable[float],
                      qs: Tuple[float, float, float] = (0.1, 0.5, 0.9)) -> pd.DataFrame:
    """
    Çoklu λ için q10/q50/q90 kantilleri.
    """
    recs: List[dict] = []
    for L in lam:
        Lf = float(max(0.0, L))
        q10 = poisson_quantile(Lf, qs[0])
        q50 = poisson_quantile(Lf, qs[1])
        q90 = poisson_quantile(Lf, qs[2])
        recs.append({"q10": q10, "q50": q50, "q90": q90})
    return pd.DataFrame(recs)

def add_poisson_uncertainty(df: pd.DataFrame, lam_col: str = "pred_expected") -> pd.DataFrame:
    """
    pred_expected (λ) → {pred_q10,pred_q50,pred_q90} ve (yoksa) pred_p_occ = 1 - e^{-λ}.
    """
    out = df.copy()
    if lam_col not in out.columns:
        return out
    lam = pd.to_numeric(out[lam_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    qs = poisson_quantiles(lam.tolist())
    out["pred_q10"] = qs["q10"].values
    out["pred_q50"] = qs["q50"].values
    out["pred_q90"] = qs["q90"].values
    # olay gerçekleşme olasılığı: 1 - P(X=0) = 1 - e^{-λ}
    if "pred_p_occ" not in out.columns:
        out["pred_p_occ"] = np.nan
    m = out["pred_p_occ"].isna()
    out.loc[m, "pred_p_occ"] = (1.0 - np.exp(-lam[m])).clip(0, 1)
    return out

# ------------------------------------------------------------
# Hepsini tek çağrıda eklemek için kolaylaştırıcı
# ------------------------------------------------------------
def add_uncertainty_and_labels(
    df: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
    lam_col: str = "pred_expected",
) -> pd.DataFrame:
    """
    - λ varsa: q10/q50/q90 + pred_p_occ (yoksa) ekler
    - pred_p_occ varsa: risk seviyesi ekler
    - q10/q90 varsa: confidence etiketi ekler
    """
    thresholds = thresholds or {"low": 0.33, "mid": 0.66}
    out = add_poisson_uncertainty(df, lam_col=lam_col)

    # risk etiketi
    if "pred_p_occ" in out.columns:
        out = out.copy()
        out["risk_level"] = [
            risk_level_from_prob(float(p) if pd.notna(p) else np.nan, thresholds)
            for p in out["pred_p_occ"]
        ]

    # güven etiketi
    out = add_confidence_cols(out)
    return out

__all__ = [
    "risk_level_from_prob",
    "confidence_label",
    "add_confidence_cols",
    "poisson_quantile",
    "poisson_quantiles",
    "add_poisson_uncertainty",
    "add_uncertainty_and_labels",
]
