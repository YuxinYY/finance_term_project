"""Shared helpers for the 2026-11-20 75-feature LightGBM grid (Phase 1, development).

Data: Chloe's 11.20 VIF-selected feature files (per-round fit/transform, train-window medians;
checked 2026-11-20: labels match the s5 panel on every row, no split overlap, no Method G).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA_DIR = Path("/Users/kietnguyen/Downloads/non-aggregated feature data sets (75 features 11.20)")
OUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path("/private/tmp/claude-501/-Users-kietnguyen-Downloads-pd-agent-kit/"
                 "95463a29-60e3-428c-b43c-50deac56a531/scratchpad/grid75_cache")
ROUNDS = ["round 01", "round 02", "round 03"]
SPLITS = ["train", "validation", "test"]
META = ["id", "fs_year", "pred_start_dt", "label"]


def feature_list():
    return json.loads((DATA_DIR / "vif_selected_features.json").read_text())["selected_features"]


def load(rnd, split):
    """Load one file as float32 features + int label (cached as parquet in the scratchpad)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{rnd.replace(' ', '_')}_{split}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_csv(DATA_DIR / rnd / f"features_{split}_vif.csv", low_memory=False)
    feats = feature_list()
    assert list(df.columns[4:]) == feats, "feature order differs from vif_selected_features.json"
    df[feats] = df[feats].astype("float32")
    df["label"] = df["label"].astype("int8")
    df = df[["id", "fs_year", "label", *feats]]
    df.to_parquet(cache)
    return df
