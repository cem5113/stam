# config/settings.py
from __future__ import annotations
import os, json, subprocess
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

APP_NAME = os.getenv("APP_NAME", "SUTAM – Suç Tahmin Modeli")
TZ_SF, TZ_TR = "America/Los_Angeles", "Europe/Istanbul"
DATE_FMT, DATETIME_FMT = "%Y-%m-%d", "%Y-%m-%d %H:%M"

BASE_DIR   = Path(os.getenv("BASE_DIR", ".")).resolve()
DATA_DIR   = Path(os.getenv("DATA_DIR", BASE_DIR/"data")).resolve()
RESULTS_DIR= Path(os.getenv("RESULTS_DIR", BASE_DIR/"results")).resolve()
OUT_DIR    = Path(os.getenv("OUT_DIR", BASE_DIR/"out")).resolve()
LOG_DIR    = Path(os.getenv("LOG_DIR", BASE_DIR/"logs")).resolve()
for p in (OUT_DIR, LOG_DIR): p.mkdir(parents=True, exist_ok=True)

def _git_rev_short(path: Path) -> str | None:
    try: return subprocess.check_output(
        ["git","-C",str(path), "rev-parse","--short","HEAD"], text=True).strip()
    except Exception: return None

def _read_meta() -> dict:
    p = RESULTS_DIR / "metadata.json"
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: pass
    return {}

def _fmt_sf(iso_or_dt) -> str:
    if not iso_or_dt: return "—"
    if isinstance(iso_or_dt, str): 
        iso_or_dt = iso_or_dt.replace("Z","+00:00")
        dt = datetime.fromisoformat(iso_or_dt).astimezone(timezone.utc)
    else: dt = iso_or_dt
    return dt.astimezone(ZoneInfo(TZ_SF)).strftime(DATETIME_FMT)

META = _read_meta()
_git = _git_rev_short(BASE_DIR)
MODEL_VERSION = os.getenv("MODEL_VERSION") or META.get("model_version") or (f"git-{_git}" if _git else "unknown")
LAST_TRAINED_AT      = _fmt_sf(META.get("last_trained_at"))
LAST_DATA_REFRESH_AT = _fmt_sf(META.get("last_data_refresh_at"))

RISK_THRESHOLDS = {"low": 0.33, "mid": 0.66}
RISK_WEIGHTS    = {"very_high":10, "high":8, "mid":6, "low":1}
