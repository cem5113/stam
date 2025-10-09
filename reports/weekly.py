# reports/weekly.py
from __future__ import annotations
import pandas as pd
from typing import Optional, List, Dict, Tuple

from reports.builder import (
    daily_tables, planned_patrols_table,
    heuristic_suggestions, export_report_zip
)

def make_weekly(df: pd.DataFrame,
                end: Optional[pd.Timestamp] = None,
                cats: Optional[List[str]] = None
                ) -> Tuple[Dict[str, pd.DataFrame], List[Dict]]:
    """
    Haftalık: end dahil son 7 gün (end-6 .. end).
    """
    end = pd.to_datetime(end or pd.Timestamp.today()).normalize()
    d1 = (end - pd.Timedelta(days=6)).normalize()
    d2 = end
    tables = daily_tables(df, d1, d2, cats=cats)
    planned = planned_patrols_table(limit=1000)
    tips = heuristic_suggestions(tables["subset"], tables["top_geoid"], planned)
    return tables, tips

def export_weekly_zip(df: pd.DataFrame,
                      end: Optional[pd.Timestamp] = None,
                      cats: Optional[List[str]] = None) -> str:
    tables, tips = make_weekly(df, end=end, cats=cats)
    zip_path = export_report_zip("weekly", tables, tips)
    return str(zip_path)
