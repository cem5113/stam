# services/tz.py
from __future__ import annotations
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    from pytz import timezone as ZoneInfo  # fallback

TZ_SF = "America/Los_Angeles"
TZ_TR = "Europe/Istanbul"
DATETIME_FMT = "%Y-%m-%d %H:%M"

def now_in(tz: str = TZ_SF) -> datetime:
    return datetime.now(ZoneInfo(tz))

def now_sf() -> datetime:
    return now_in(TZ_SF)

def now_tr() -> datetime:
    return now_in(TZ_TR)

def fmt(dt: datetime, fmt_str: str = DATETIME_FMT) -> str:
    return dt.strftime(fmt_str)

def now_sf_str(fmt_str: str = DATETIME_FMT) -> str:
    return fmt(now_sf(), fmt_str)

def to_tz(dt: datetime, tz: str) -> datetime:
    """Naive ise tz atar; aware ise belirtilen tz’ye dönüştürür."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ZoneInfo(tz))
    return dt.astimezone(ZoneInfo(tz))
