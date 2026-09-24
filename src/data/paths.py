"""Data locations, read from config/pipeline.yaml so that no path is hard-coded in a script."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG = Path("config/pipeline.yaml")
COLLECTION_KEY = {"ID1000": "bids_root", "PIOP1": "piop1_root", "PIOP2": "piop2_root"}


@lru_cache(maxsize=1)
def _paths() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["paths"]


def path(name: str) -> Path:
    """Directory configured under paths.<name>; relative paths resolve from the repository root."""
    return Path(_paths()[name])


def collection_root(collection: str) -> Path:
    """Root directory of an AOMIC collection: ID1000, PIOP1 or PIOP2."""
    return path(COLLECTION_KEY[collection])
