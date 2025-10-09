# services/tz.py
from datetime import datetime
from zoneinfo import ZoneInfo
from config.settings import TZ_SF, TZ_TR, DATETIME_FMT

def now_sf_str() -> str:
    return datetime.now(ZoneInfo(TZ_SF)).strftime(DATETIME_FMT)

def to_sf_str(dt: datetime) -> str:
    return dt.astimezone(ZoneInfo(TZ_SF)).strftime(DATETIME_FMT)

def to_tr_str(dt: datetime) -> str:
    return dt.astimezone(ZoneInfo(TZ_TR)).strftime(DATETIME_FMT)
