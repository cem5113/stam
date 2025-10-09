# patrol/approvals.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import hashlib, json
from services.tz import now_sf

LOG_PATH = Path("logs/approvals.jsonl")

def make_approval_hash(payload: dict) -> str:
    # Stabil hash: anahtarları sıralı JSON
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()

def save_approval(payload: dict) -> str:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "hash": make_approval_hash(payload),
        "ts_sf": now_sf().isoformat(timespec="seconds"),
        **payload,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec["hash"]

def list_approvals(limit: int = 20) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    rows: list[dict] = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows[-limit:]
