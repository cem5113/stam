# dataio/loaders.py
from __future__ import annotations
import os, io, zipfile, json, requests
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd

# Ayarlar
from config.settings import DATA_DIR as _DATA_DIR, RESULTS_DIR as _RESULTS_DIR

DATA_DIR = Path(_DATA_DIR); DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(_RESULTS_DIR)

# Ortam değişkenleri (Streamlit secrets → os.environ üzerinden gelebilir)
GITHUB_REPO     = os.getenv("GITHUB_REPO", "cem5113/crime_prediction_data")   # owner/repo
GITHUB_WORKFLOW = os.getenv("GITHUB_WORKFLOW", "full_pipeline.yml")
GH_TOKEN        = os.getenv("GH_TOKEN", "")

# Release fallback (güncel olmayabilir ama son çare)
CRIME_CSV_URL = os.getenv(
    "CRIME_CSV_URL",
    "https://github.com/cem5113/crime_prediction_data/releases/latest/download/sf_crime.csv",
)

# GEOID uzunluğu (grid ID padleme)
GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

# ----------------- yardımcılar -----------------
def _headers():
    if not GH_TOKEN:
        # GH_TOKEN yoksa artifact erişimini atla; caller try/except ile yakalıyor
        raise RuntimeError("GH_TOKEN yok (env). Artifact erişimi için gereklidir.")
    return {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

def _artifact_bytes(picks: List[str], artifact_name: str = "sf-crime-pipeline-output") -> Optional[bytes]:
    """
    Son başarılı run’ın artifact’ından 'picks' içindeki ilk dosyayı döndürür (bytes).
    'picks' tam ad ya da sonek eşleşmesi yapabilir.
    """
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

def _ensure_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    date → pandas datetime (saat bilgisi varsa kaybetme), event_hour türet.
    """
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    elif "datetime" in out.columns:
        out["date"] = pd.to_datetime(out["datetime"], errors="coerce")
    else:
        out["date"] = pd.NaT
    if "event_hour" not in out.columns:
        out["event_hour"] = pd.to_datetime(out["date"], errors="coerce").dt.hour.fillna(0).astype(int)
    return out

def _ensure_latlon(df: pd.DataFrame) -> pd.DataFrame:
    # Şimdilik dokunma; UI tarafı lat/lon yoksa uyarı gösteriyor
    return df.copy()

def _parse_and_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_latlon(_ensure_time_cols(df))
    if "GEOID" in df.columns:
        # GEOID’i string ve normalize et
        try:
            df["GEOID"] = _normalize_geoid(df["GEOID"])
        except Exception:
            df["GEOID"] = df["GEOID"].astype(str)
    return df

# ----------------- public API -----------------
def load_sf_crime_latest() -> Tuple[pd.DataFrame, str]:
    """
    Kaynak sırası:
      1) GitHub Actions artifact: sf_crime_09.csv → sf_crime_08.csv
      2) Release (latest): sf_crime.csv
      3) RESULTS_DIR artefact dosyaları: sf_crime_latest.parquet/csv
      4) Yerel cache: data/sf_crime_artifact.csv → sf_crime_09.csv → sf_crime_08.csv → sf_crime.csv
    Dönüş: (df, src_tag) — src_tag ∈ {"artifact","release","results","local:<ad>","empty"}
    """
    # 1) Artifact (GH_TOKEN gerekiyorsa ve varsa)
    try:
        blob = _artifact_bytes(["sf_crime_09.csv", "sf_crime_08.csv"])
        if blob:
            # yerel cache’e yaz
            (DATA_DIR / "sf_crime_artifact.csv").write_bytes(blob)
            df = pd.read_csv(io.BytesIO(blob), low_memory=False)
            return _parse_and_cleanup(df), "artifact"
    except Exception as e:
        print("artifact erişimi başarısız:", e)

    # 2) Release latest
    try:
        r = requests.get(CRIME_CSV_URL, timeout=60); r.raise_for_status()
        (DATA_DIR / "sf_crime_release.csv").write_bytes(r.content)
        df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
        return _parse_and_cleanup(df), "release"
    except Exception as e:
        print("release fallback başarısız:", e)

    # 3) RESULTS_DIR artefact dosyaları
    for cand, tag in [
        (RESULTS_DIR / "sf_crime_latest.parquet", "results"),
        (RESULTS_DIR / "sf_crime_latest.csv",     "results"),
    ]:
        if cand.exists():
            try:
                if cand.suffix.lower().endswith("parquet"):
                    df = pd.read_parquet(cand)
                else:
                    df = pd.read_csv(cand, low_memory=False)
                return _parse_and_cleanup(df), tag
            except Exception as e:
                print("RESULTS okumada hata:", e)
                continue

    # 4) Yerel DATA_DIR
    for name in ["sf_crime_artifact.csv", "sf_crime_09.csv", "sf_crime_08.csv", "sf_crime.csv"]:
        p = DATA_DIR / name
        if p.exists():
            try:
                df = pd.read_csv(p, low_memory=False)
                return _parse_and_cleanup(df), f"local:{name}"
            except Exception:
                continue

    # 5) Hiçbiri yoksa — UI düşmesin
    df = pd.DataFrame({
        "GEOID": [], "date": [], "event_hour": [], "crime_count": [],
        "lat": [], "lon": []
    })
    return df, "empty"

def load_metadata() -> dict:
    """
    results/metadata.json → yoksa artifact’tan dene → yoksa {}
    """
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
