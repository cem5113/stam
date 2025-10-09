# models/predictor.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from .baseline import fit_mean_by_groups, predict_mean_by_groups

class HurdleLikeModel:
    """
    Hafif/iskemle model:
      - Olasılık ~ grup ortalamaları (proxy)
      - Beklenen sayı ~ aynı ortalamanın pozitifler üzerinde uygulanması
    LightGBM/Sklearn yoksa da çalışır; sadece baseline istatistikleri kullanır.
    """

    def __init__(self, prob_groups=None, count_groups=None, y_col_prob="crime_count", y_col_cnt="crime_count"):
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
        # q10/q90 için kaba bant (belirsizlik yoksa placeholder)
        out["pred_q10"] = (out["pred_q50"] * 0.7).astype(float)
        out["pred_q90"] = (out["pred_q50"] * 1.3).astype(float)
        return out
