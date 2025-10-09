# services/logging.py
from __future__ import annotations
from pathlib import Path
import json, uuid
from typing import Any, Dict, Optional

from services.tz import now_sf

LOG_PATH = Path("logs/audit.jsonl")

def _session_id() -> str:
    """Her Streamlit oturumu için kısa bir UID üret/hatırla."""
    try:
        import streamlit as st
        return st.session_state.setdefault("_sid", uuid.uuid4().hex[:12])
    except Exception:
        # UI dışında çağrılırsa
        return uuid.uuid4().hex[:12]

def audit(event: str, actor: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """
    Basit JSONL audit kaydı.
    event: "app_open", "open_tab", "approve_route", ...
    actor: "Amir" / "Kullanıcı" veya kullanıcı adı
    payload: serileştirilebilir ek bilgiler
    """
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts_sf": now_sf().isoformat(timespec="seconds"),
            "session": _session_id(),
            "event": event,
            "actor": actor,
            "payload": payload or {},
        }
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # log yazılamazsa uygulamayı durdurmayalım
        pass

def tail_audit(limit: int = 100) -> list[dict]:
    """Son N audit kaydını oku (UI’de debug için)."""
    if not LOG_PATH.exists():
        return []
    rows: list[dict] = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows[-limit:]
