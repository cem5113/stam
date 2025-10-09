# reports/daily.py
from __future__ import annotations
import pandas as pd
from typing import Optional, List, Dict, Tuple
from datetime import datetime

from reports.builder import (
    daily_tables, planned_patrols_table,
    heuristic_suggestions, export_report_zip
)

def make_daily(df: pd.DataFrame,
               end: Optional[pd.Timestamp] = None,
               cats: Optional[List[str]] = None
               ) -> Tuple[Dict[str, pd.DataFrame], List[Dict]]:
    """
    Günlük: end gününü kapsar (00:00–23:59). d1=d2=end.
    """
    end = pd.to_datetime(end or pd.Timestamp.today()).normalize()
    d1, d2 = end, end
    tables = daily_tables(df, d1, d2, cats=cats)
    planned = planned_patrols_table(limit=500)
    tips = heuristic_suggestions(tables["subset"], tables["top_geoid"], planned)
    return tables, tips

def export_daily_zip(df: pd.DataFrame,
                     end: Optional[pd.Timestamp] = None,
                     cats: Optional[List[str]] = None) -> str:
    tables, tips = make_daily(df, end=end, cats=cats)
    zip_path = export_report_zip("daily", tables, tips)
    return str(zip_path)
