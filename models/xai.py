# models/xai.py
from __future__ import annotations
from typing import List, Dict
import math

# Saha-dostu kısa etiketler
_LABELS = {
    "nr_7d": "Yakın tekrar (7g)",
    "nr_14d": "Yakın tekrar (14g)",
    "nei_7d_sum": "Komşu yoğunluğu (7g)",
    "911_request_count_hour_range": "911 yoğunluğu (saat)",
    "911_request_count_daily(before_24_hours)": "911 yoğunluğu (24s)",
    "poi_risk_score": "POI risk skoru",
    "poi_risk_score_range": "POI risk skoru (kategori)",
    "bus_stop_count": "Otobüs durağı sayısı",
    "train_stop_count": "Tren durağı sayısı",
    "distance_to_police": "Polise uzaklık",
    "distance_to_government_building": "Kamu binasına uzaklık",
    "precip": "Yağış",
    "wind_speed": "Rüzgâr",
    "event_hour": "Saat etkisi",
}

# Saat dilimi için basit risk ağırlıkları (18–02 daha riskli varsayımı)
_HOUR_WEIGHT = {h: (1.0 if 18 <= h <= 23 or 0 <= h <= 2 else 0.4) for h in range(24)}

def _val(row, k, default=0.0):
    v = row.get(k, default)
    try:
        return float(v) if v is not None and str(v) != "nan" else default
    except Exception:
        return default

def brief_xai_for_row(row: dict) -> List[Dict]:
    """
    Model yoksa bile sahaya okunur bir açıklama üret:
    - Öncelik: yakın tekrar, komşu yoğunluğu, 911 yükü, POI skoru, saat.
    - Her faktör için basit normalize puan ve kısa gerekçe döndür.
    """
    feats: List[Dict] = []

    # 1) Yakın tekrarlar
    nr7  = _val(row, "nr_7d")
    nr14 = _val(row, "nr_14d")
    if nr7 > 0:  feats.append({"name": _LABELS["nr_7d"],  "score": nr7,  "why": "Son 7 günde benzer olay birikti."})
    if nr14 > 0: feats.append({"name": _LABELS["nr_14d"], "score": 0.6*nr14, "why": "Son 14 günde tekrar sinyali var."})

    # 2) Komşu yoğunluğu
    nei = _val(row, "nei_7d_sum")
    if nei > 0: feats.append({"name": _LABELS["nei_7d_sum"], "score": 0.8*nei, "why": "Komşu hücrelerde son 7 günde artış."})

    # 3) 911 yükleri
    r911h = _val(row, "911_request_count_hour_range")
    r911d = _val(row, "911_request_count_daily(before_24_hours)")
    if r911h > 0: feats.append({"name": _LABELS["911_request_count_hour_range"], "score": 1.2*r911h, "why": "Saatlik 911 çağrıları yüksek."})
    if r911d > 0: feats.append({"name": _LABELS["911_request_count_daily(before_24_hours)"], "score": r911d, "why": "Son 24 saatte 911 çağrıları arttı."})

    # 4) POI / çevresel
    poi = max(_val(row, "poi_risk_score"), _val(row, "poi_risk_score_range"))
    if poi > 0: feats.append({"name": _LABELS.get("poi_risk_score_range"), "score": 0.7*poi, "why": "Riskli POI yoğunluğu etkili."})

    # 5) Toplu taşıma & uzaklıklar
    bus = _val(row, "bus_stop_count")
    train = _val(row, "train_stop_count")
    if bus > 0: feats.append({"name": _LABELS["bus_stop_count"], "score": 0.4*bus, "why": "Yaya/hareketlilik artışı (otobüs)."})
    if train > 0: feats.append({"name": _LABELS["train_stop_count"], "score": 0.5*train, "why": "Yaya/hareketlilik artışı (tren)."})

    # 6) Saat etkisi
    hr = int(_val(row, "event_hour", default=-1))
    if 0 <= hr <= 23:
        feats.append({"name": _LABELS["event_hour"], "score": 2.0*_HOUR_WEIGHT.get(hr, 0.4), "why": f"Saat {hr:02d}:00 dilimi görece riskli."})

    # 7) Hava
    precip = _val(row, "precip")
    if precip > 0: feats.append({"name": _LABELS["precip"], "score": 0.3*precip, "why": "Yağış, belirli suç tiplerini etkileyebilir."})

    # Normalize benzeri ölçek: log(1+x) ile yumuşat, büyükten küçüğe sırala
    for f in feats:
        f["score"] = round(math.log1p(max(f["score"], 0.0)), 4)

    feats = sorted(feats, key=lambda x: x["score"], reverse=True)
    return feats[:3]
