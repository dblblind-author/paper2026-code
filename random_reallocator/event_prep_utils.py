"""
Small utility module used by the multi-run random reallocator.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd


# ----------------------------
# IO helpers
# ----------------------------

def load_pickle(path: Union[str, Path]) -> Any:
    """Load any Python object from a pickle file."""
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """Save any Python object to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ----------------------------
# Event prep helpers
# ----------------------------

def deduplicate_events(
    df: pd.DataFrame,
    subset: Union[str, Sequence[str]] = ("title",),
    keep: str = "first",
) -> pd.DataFrame:
    """Drop duplicates for events (defaults to `title`)."""
    if isinstance(subset, str):
        subset = (subset,)
    if not all(c in df.columns for c in subset):
        missing = [c for c in subset if c not in df.columns]
        raise KeyError(f"Missing columns for deduplication: {missing}")
    return df.drop_duplicates(subset=list(subset), keep=keep).copy()


def collapse_events_with_weight(
    df: pd.DataFrame,
    *,
    subset: Union[str, Sequence[str]] = ("title",),
    weight_col: str = "event_weight",
    keep: str = "first",
) -> pd.DataFrame:
    """
    Collapse duplicates and attach a weight equal to the number of instances.

    The resulting DataFrame has one row per unique subset value(s), plus a
    weight column storing counts.
    """
    if isinstance(subset, str):
        subset = (subset,)
    if not all(c in df.columns for c in subset):
        missing = [c for c in subset if c not in df.columns]
        raise KeyError(f"Missing columns for deduplication: {missing}")

    counts = df.groupby(list(subset)).size().rename(weight_col).reset_index()
    base = df.drop_duplicates(subset=list(subset), keep=keep).copy()
    out = base.merge(counts, on=list(subset), how="left")
    return out


def add_sequential_event_id(
    df: pd.DataFrame,
    id_col: str = "event_id",
    start: int = 1,
) -> pd.DataFrame:
    """Add an integer id column (1..N) if missing."""
    out = df.copy()
    if id_col not in out.columns or out[id_col].isna().any():
        out[id_col] = np.arange(start, start + len(out), dtype=int)
    return out


def prep_events_table(
    events: pd.DataFrame,
    *,
    dedupe_subset: Union[str, Sequence[str]] = ("title",),
    dedupe: bool = True,
    weight_by_instances: bool = False,
    weight_col: str = "event_weight",
    id_col: str = "event_id",
) -> pd.DataFrame:
    """
    Lightweight prep that does NOT require geopandas:
    - optionally drop duplicates (e.g., by title),
    - add sequential event_id.

    Set weight_by_instances=True to collapse duplicates and weight by counts.
    """
    out = events.copy()
    if weight_by_instances and dedupe_subset is not None:
        out = collapse_events_with_weight(out, subset=dedupe_subset, weight_col=weight_col)
    elif dedupe and dedupe_subset is not None:
        out = deduplicate_events(out, subset=dedupe_subset)
    out = add_sequential_event_id(out, id_col=id_col)
    return out
