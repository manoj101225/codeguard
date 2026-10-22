"""
PROMISE/AEEEM dataset downloader and loader.

Downloads public CSVs for: camel-1.6, jedit-4.0, ivy-2.0, lucene-2.4, poi-2.5
from the PROMISE Software Engineering Repository (GitHub mirror).

Standard columns across PROMISE ARFF/CSV datasets:
  wmc, dit, noc, cbo, rfc, lcom, ca, ce, npm, lcom3, loc, dam, moa, mfa,
  cam, ic, cbm, amc, max_cc, avg_cc, bug (0/1 label)
"""
import os
import io
import logging
import requests
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public GitHub mirror of the PROMISE repository (raw CSV exports)
# These are the actual, widely-used mirrors in academic ML/SE papers.
# ---------------------------------------------------------------------------
_BASE = (
    "https://raw.githubusercontent.com/klainfo/DefectData/master"
    "/inst/extdata/terapromise/ck"
)

DATASET_URLS = {
    "camel-1.6": f"{_BASE}/camel-1.6.csv",
    "jedit-4.0":  f"{_BASE}/jedit-4.0.csv",
    "ivy-2.0":    f"{_BASE}/ivy-2.0.csv",
    "lucene-2.4": f"{_BASE}/lucene-2.4.csv",
    "poi-2.5":    f"{_BASE}/poi-2.5.csv",
}

# Columns to use as features (CK metrics + process metrics where available)
FEATURE_COLS = [
    "wmc", "dit", "noc", "cbo", "rfc", "lcom",
    "ca", "ce", "npm", "loc", "dam", "moa", "mfa",
    "cam", "ic", "cbm", "amc", "max_cc", "avg_cc",
]

LABEL_COL = "bug"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _local_path(name: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{name}.csv")


def _download(name: str, url: str) -> pd.DataFrame:
    """Download a PROMISE CSV and save locally."""
    log.info("Downloading %s from %s …", name, url)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    path = _local_path(name)
    with open(path, "wb") as f:
        f.write(r.content)
    log.info("Saved → %s", path)
    return pd.read_csv(io.BytesIO(r.content))


def load_dataset(name: str, force_download: bool = False) -> pd.DataFrame:
    """
    Load a PROMISE dataset by name.  Downloads on first call, reads from cache
    on subsequent calls.  Returns a cleaned DataFrame with FEATURE_COLS + LABEL_COL.
    """
    path = _local_path(name)
    if os.path.exists(path) and not force_download:
        df = pd.read_csv(path)
    elif name in DATASET_URLS:
        df = _download(name, DATASET_URLS[name])
    else:
        raise ValueError(f"Unknown dataset '{name}'. Known: {list(DATASET_URLS)}")

    df = _clean(df, name)
    return df


def _clean(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalise column names and handle missing/extra columns."""
    # Lower-case all columns
    df.columns = [c.lower().strip() for c in df.columns]

    # Map common alternative label column names to 'bug'
    for alt in ("defects", "label", "class", "defect", "bugged"):
        if alt in df.columns and LABEL_COL not in df.columns:
            df = df.rename(columns={alt: LABEL_COL})

    # Ensure label is strictly binary int (0 or 1)
    if LABEL_COL in df.columns:
        s = df[LABEL_COL].astype(str).str.lower().str.strip()
        bool_map = {"true": 1, "false": 0, "yes": 1, "no": 0}
        mapped = s.map(bool_map)
        numeric_val = pd.to_numeric(s, errors="coerce")
        df[LABEL_COL] = (
            mapped.fillna(numeric_val)
            .fillna(0)
            .apply(lambda v: 1 if float(v) > 0 else 0)
            .astype(int)
        )

    # Keep only known feature columns that exist in this dataset
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    cols_to_keep = available_features + ([LABEL_COL] if LABEL_COL in df.columns else [])

    # Preserve a 'name' column if present (Java class name)
    if "name" in df.columns:
        cols_to_keep = ["name"] + cols_to_keep
    elif "class" in df.columns and "class" not in available_features:
        cols_to_keep = ["class"] + cols_to_keep

    df = df[cols_to_keep].copy()

    # Fill numeric NaNs with column medians
    num_cols = [c for c in df.columns if c not in ("name", "class")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    log.info(
        "%s: %d rows, %d features, defect rate=%.1f%%",
        name, len(df), len(available_features),
        100 * df[LABEL_COL].mean() if LABEL_COL in df.columns else float("nan")
    )
    return df


def get_features_labels(df: pd.DataFrame):
    """Return (X, y, feature_names) from a cleaned DataFrame."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values.astype(float)
    y = df[LABEL_COL].values.astype(int)
    return X, y, available


def download_all(force: bool = False):
    """Pre-download all configured datasets."""
    results = {}
    for name in DATASET_URLS:
        try:
            df = load_dataset(name, force_download=force)
            results[name] = {"rows": len(df), "ok": True}
        except Exception as e:
            results[name] = {"ok": False, "error": str(e)}
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for n, r in download_all().items():
        print(f"{n}: {r}")
