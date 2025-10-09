# patrol/scoring.py
from __future__ import annotations
from typing import Iterable, List, Optional
import pandas as pd

def coverage_score(route_geoids: Iterable[str], geo_scores: pd.DataFrame, score_col: str) -> float:
    """
    Kapsama skoru = (rotadaki GEOID'lerin toplam skoru) / (tüm GEOID toplam skoru) * 100
    geo_scores: en az ['GEOID', score_col] içermeli (pencereye göre özetlenmiş)
    """
    if len(geo_scores) == 0:
        return 0.0
    sub = geo_scores[geo_scores["GEOID"].isin(list(route_geoids))]
    num = float(sub[score_col].sum())
    den = float(geo_scores[score_col].sum()) or 1.0
    return 100.0 * num / den

def overlap_ratio(a: Iterable[str], b: Iterable[str]) -> float:
    """İki rota arası örtüşme oranı (Jaccard benzeri): |kesişim| / |birleşim|"""
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / float(len(A | B))

def diversity_score(route_geoids: Iterable[str], recent_routes: Optional[List[List[str]]] = None) -> float:
    """
    Çeşitlilik skoru = 1 - (son N rota ile en yüksek örtüşme)
    Düşük = kötü (benzer), yüksek = iyi (farklı).
    """
    if not recent_routes:
        return 1.0
    max_ov = max(overlap_ratio(route_geoids, r) for r in recent_routes if r)
    return max(0.0, 1.0 - max_ov)
