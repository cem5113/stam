# models/uncertainty.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Dict

# risk etiketi (olasılık) — UI ile aynı eşikler
def risk_level_from_prob(p: float, thresholds: Dict[str, float]) -> str:
    if p is None or np.isnan(p):
        return "Bilinmiyor"
    return "Yüksek" if p > thresholds.get("mid", 0.66) else \
           ("Orta" if p > thresholds.get("low", 0.33) else "Düşük")

# belirsizlik (q10–q90 yayılımından)
def confidence_label(q10: Optional[float], q90: Optional[float]) -> str:
    if q10 is None or q90 is None or np.isnan(q10) or np.isnan(q90):
        return "—"
    spread = float(q90) - float(q10)
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
