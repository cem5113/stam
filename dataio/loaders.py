# dataio/loaders.py
from __future__ import annotations
import os, io, zipfile, json, requests
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd

# Ayarlar
from config.settings import DATA_DIR, RESULTS_DIR

DATA_DIR = Path(DATA_DIR); DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(RESULTS_DIR)

GITHUB_REPO     = os.getenv("GITHUB_REPO", "cem5113/crime_prediction_data")   # owner/repo
GITHUB_WORKFLOW = os.getenv("GITHUB_WORKFLOW", "full_pipeline.yml")
GH_TOKEN        = os.getenv("GH_TOKEN", "")

# Release fallback (güncel olmayabilir ama son çare)
CRIME_CSV_URL = os.getenv(
    "CRIME_CSV_URL",
    "https://github.com/cem5113/crime_prediction_data/releases/latest/download/sf_crime.csv",
)

GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

# ----------------- yardımcılar -----------------
def _headers():
    if not GH_TOKEN:
        raise RuntimeError("GH_TOKEN yok (env). Artifact erişimi için gereklidir.")
    return {"Authorization": f"Bearer {GH_TOKEN}", "Accept": "application/vnd.github+json"}

def _artifact_bytes(picks: List[str], artifact_name="sf-crime-pipeline-output") -> Optional[bytes]:
    """Son başarılı run’ın artifact’ından picks içindeki ilk dosyayı döndürür (bytes)."""
    runs_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs?per_page=20"
    runs = requests.get(runs_url, headers=_headers(), timeout=30).json()
    run_ids = [r["id"] for r in runs.get("workflow_runs", []) if r.get("conclusion") == "success"]
    for rid in run_ids:
        arts_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{rid}/artifacts"
        arts = requests.get(arts_url, headers=_headers(), timeout=30).json().get("artifacts", [])
        for a in arts:
            if a.get("name") == artifact_name and not a.get("expired", False):
                z = requests.get(a["archive_download_url"], headers=_headers(), timeout=60).content
                zf = zipfile.ZipFile(io.BytesIO(z))
                names = zf.namelist()
                # tam ve alt klasörlü eşleşme
                for p in picks:
                    for cand in (p, f"crime_prediction_data/{p}"):
                        if cand in names:
                            return zf.read(cand)
                # suffix (sondan) eşleşmesi
                for n in names:
                    if any(n.endswith(p) for p in picks):
                        return zf.read(n)
    return None

def _normalize_geoid(s: pd.Series, L: int = GEOID_LEN) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)", expand=False).str[:L].str.zfill(L)

def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    elif "datetime" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
    return df

# ----------------- public API -----------------
def load_sf_crime_latest() -> Tuple[pd.DataFrame, str]:
    """
    Kaynak sırası:
      1) GitHub Actions artifact: sf_crime_09.csv → sf_crime_08.csv
      2) Release (latest): sf_crime.csv
      3) Yerel cache: data/sf_crime_artifact.csv → sf_crime_09.csv → sf_crime_08.csv → sf_crime.csv
    Dönüş: (df, "artifact" | "release" | "local:<ad>")
    """
    # 1) Artifact
    try:
        blob = _artifact_bytes(["sf_crime_09.csv", "sf_crime_08.csv"])
        if blob:
            (DATA_DIR / "sf_crime_artifact.csv").write_bytes(blob)  # cache
            df = pd.read_csv(io.BytesIO(blob), low_memory=False)
            df = _parse_dates(df)
            if "GEOID" in df.columns: df["GEOID"] = _normalize_geoid(df["GEOID"])
            return df, "artifact"
    except Exception as e:
        print("artifact erişimi başarısız:", e)

    # 2) Release latest
    try:
        r = requests.get(CRIME_CSV_URL, timeout=60); r.raise_for_status()
        (DATA_DIR / "sf_crime_release.csv").write_bytes(r.content)
        df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
        df = _parse_dates(df)
        if "GEOID" in df.columns: df["GEOID"] = _normalize_geoid(df["GEOID"])
        return df, "release"
    except Exception as e:
        print("release fallback başarısız:", e)

    # 3) Yerel
    for name in ["sf_crime_artifact.csv", "sf_crime_09.csv", "sf_crime_08.csv", "sf_crime.csv"]:
        p = DATA_DIR / name
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            df = _parse_dates(df)
            if "GEOID" in df.columns: df["GEOID"] = _normalize_geoid(df["GEOID"])
            return df, f"local:{name}"

    raise FileNotFoundError("sf_crime verisi hiçbir kaynaktan bulunamadı.")

def load_metadata() -> dict:
    """results/metadata.json → yoksa artifact’tan dene → yoksa {}"""
    p = RESULTS_DIR / "metadata.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        blob = _artifact_bytes(["metadata.json"])
        if blob:
            return json.loads(blob.decode("utf-8"))
    except Exception:
        pass
    return {}
