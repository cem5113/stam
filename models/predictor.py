# models/predictor.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List

# Paket içi importlar
from .baseline import BaselineModel, fit_mean_by_groups, predict_mean_by_groups
from .uncertainty import add_poisson_uncertainty  # istersen add_uncertainty_and_labels'a da genişletebilirsin

# ---------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------
def _has_predictions(df: pd.DataFrame) -> bool:
    return any(c in df.columns for c in ("pred_p_occ", "pred_expected", "pred_q50"))

# ---------------------------------------------------------------------
# 1) Hafif "Hurdle-like" model
# ---------------------------------------------------------------------
class HurdleLikeModel:
    """
    Hafif/iskelet model:
      - Olasılık (occurrence) ~ grup ortalamaları (proxy, ikili hedef: y>0)
      - Pozitiflerde beklenen sayı ~ aynı grup ortalaması (count hedefi)
    LightGBM/Sklearn yoksa da çalışır; sadece baseline istatistikleri kullanır.
    """

    def __init__(
        self,
        prob_groups: Optional[List[str]] = None,
        count_groups: Optional[List[str]] = None,
        y_col_prob: str = "crime_count",
        y_col_cnt: str = "crime_count",
    ):
        self.prob_groups = prob_groups or ["GEOID", "event_hour"]
        self.count_groups = count_groups or ["GEOID", "event_hour"]
        self.y_col_prob = y_col_prob
        self.y_col_cnt = y_col_cnt
        self._prob_model: Optional[Dict] = None
        self._cnt_model: Optional[Dict] = None

    def fit(self, df: pd.DataFrame) -> "HurdleLikeModel":
        dfx = df.copy()

        # olasılık hedefi: Y>0 ikili
        if self.y_col_prob in dfx.columns:
            y_occ = (pd.to_numeric(dfx[self.y_col_prob], errors="coerce").fillna(0) > 0).astype(int)
            dfx = dfx.assign(_y_occ=y_occ)
        else:
            dfx = dfx.assign(_y_occ=0)

        self._prob_model = fit_mean_by_groups(
            dfx.rename(columns={"_y_occ": "target_prob"}),
            y_col="target_prob",
            group_cols=[c for c in self.prob_groups if c in dfx.columns],
        )

        # pozitiflerde ortalama sayı
        dpos = dfx[dfx["_y_occ"] == 1].copy()
        if len(dpos) == 0:
            dpos = dfx.copy()
        self._cnt_model = fit_mean_by_groups(
            dpos.rename(columns={self.y_col_cnt: "target_cnt"}),
            y_col="target_cnt",
            group_cols=[c for c in self.count_groups if c in dpos.columns],
        )
        return self

    def predict(self, df_new: pd.DataFrame) -> pd.DataFrame:
        if self._prob_model is None or self._cnt_model is None:
            raise RuntimeError("Model fit edilmedi.")
        p = predict_mean_by_groups(self._prob_model, df_new).clip(0, 1)
        mu = predict_mean_by_groups(self._cnt_model, df_new).clip(lower=0)
        out = pd.DataFrame({
            "pred_p_occ": p.values.astype(float),
            "pred_q50": mu.values.astype(float),      # median proxy
            "pred_expected": (p.values * np.maximum(mu.values, 0)).astype(float)
        }, index=df_new.index)
        # kaba belirsizlik bantları (model yoksa placeholder)
        out["pred_q10"] = (out["pred_q50"] * 0.7).astype(float)
        out["pred_q90"] = (out["pred_q50"] * 1.3).astype(float)
        return out

# ---------------------------------------------------------------------
# 2) Genel tahmin güvence katmanı
# ---------------------------------------------------------------------
def ensure_predictions(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Girdi DataFrame’ine aşağıdaki kolonları **mümkün olduğunca** ekler:
      - pred_p_occ (0..1)
      - pred_expected (>=0) varsa belirsizlik: pred_q10, pred_q50, pred_q90
    Yoksa: BaselineModel ile 'p_proxy' üretir ve bunu 'pred_p_occ' olarak kullanır.
    """
    df = df_raw.copy()

    # 1) Zaten beklenen değer varsa → Poisson belirsizliği ekle
    if "pred_expected" in df.columns:
        df = add_poisson_uncertainty(df, lam_col="pred_expected")
        return df

    # 2) Yalnızca olasılık varsa → normalle ve dön
    if "pred_p_occ" in df.columns:
        df["pred_p_occ"] = pd.to_numeric(df["pred_p_occ"], errors="coerce").clip(0, 1)
        return df

    # 3) Hiçbiri yoksa: Baseline fallback (p_proxy)
    if "GEOID" in df.columns:
        base = BaselineModel(window_days=7, use_hour=True).fit(df)
        df = base.predict(df)
        if "p_proxy" in df.columns:
            df["pred_p_occ"] = df["p_proxy"].clip(0, 1)
    else:
        # GEOID yoksa herkese 0 atanır (emin değilsek konservatif)
        df["pred_p_occ"] = 0.0

    return df

class Predictor:
    """
    Gelecekte gerçek model (checkpoint/artifact) bağlanınca bu sınıf genişletilir.
    Şimdilik:
      - ensure_predictions(df) ile tahmin kolonlarını garanti eder.
      - Gerekirse BaselineModel ile p_proxy üretir.
    """
    def __init__(self) -> None:
        self._baseline = BaselineModel(window_days=7, use_hour=True)

    def predict(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        # Eğer girişte tahmin kolonları varsa (veya pred_expected) bunları tamamla.
        if _has_predictions(df_raw):
            return ensure_predictions(df_raw)

        # baseline’ı fit → predict
        self._baseline.fit(df_raw)
        out = self._baseline.predict(df_raw)
        if "p_proxy" in out.columns:
            out["pred_p_occ"] = out["p_proxy"].clip(0, 1)
        return out
