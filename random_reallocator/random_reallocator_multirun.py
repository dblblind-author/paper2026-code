"""
Run many random event reallocations (outside the notebook) and save combined output.
"""

from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pandas as pd

from event_prep_utils import load_pickle, save_pickle, prep_events_table


# ----------------------------
# 1) Config
# ----------------------------
GRID_PKL = Path("hex_pop.pkl")
EVENTS_PKL = Path("gdf_2024_final_v3_ready_to_realloc.pkl")


# Column names
CELL_ID_COL = "hex_id"

# Event handling
WEIGHT_BY_TITLE_INSTANCES = True  # collapse to unique titles with weights
WEIGHT_COL = "event_weight"
TITLE_COL = "title"
EVENT_ID_COL = "event_id"

# Allocation behavior
GROUP_BY_CATEGORY = True

# Multi-run
N_RUNS = 100

# Output
OUT_DIR = Path("output")
OUT_PATH = OUT_DIR / "reallocated_random_100_runs.pkl"


def _ensure_numpy_pickle_compat() -> None:
    # Compatibility shim for pickles created with newer numpy module paths.
    try:
        import numpy.core as _np_core
        sys.modules.setdefault("numpy._core", _np_core)
        if hasattr(_np_core, "_multiarray_umath"):
            sys.modules.setdefault("numpy._core._multiarray_umath", _np_core._multiarray_umath)
    except Exception:
        pass


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    return prep_events_table(
        events,
        dedupe_subset=TITLE_COL,
        weight_by_instances=WEIGHT_BY_TITLE_INSTANCES,
        weight_col=WEIGHT_COL,
        id_col=EVENT_ID_COL,
    )


def _get_cell_ids(grid: pd.DataFrame, cell_id_col: str) -> np.ndarray:
    if cell_id_col not in grid.columns:
        raise KeyError(f"grid must contain column '{cell_id_col}'")
    cells = grid[cell_id_col].dropna().drop_duplicates()
    try:
        cells = cells.sort_values(kind="mergesort")
    except Exception:
        pass
    return cells.to_numpy()


def _allocate_random(
    events_df: pd.DataFrame,
    *,
    cell_ids: np.ndarray,
    rng: np.random.Generator,
    group_by_category: bool,
    weight_col: str,
    include_weights: bool,
) -> pd.DataFrame:
    if "event_id" not in events_df.columns:
        raise KeyError("events_df must contain column 'event_id'")
    if group_by_category and "category" not in events_df.columns:
        raise KeyError("group_by_category=True but 'category' not in events_df")

    n_cells = len(cell_ids)
    include_category_col = "category" in events_df.columns

    def run_block(block_events: pd.DataFrame, category: Optional[str]) -> pd.DataFrame:
        use_category = category is not None or include_category_col
        if block_events is None or len(block_events) == 0:
            cols = ["event_id", "assigned_cell_id", "chosen_from_hub"]
            if use_category:
                cols = ["event_id", "category"] + cols[1:]
            if include_weights and weight_col in events_df.columns:
                insert_at = 2 if use_category else 1
                cols.insert(insert_at, weight_col)
            return pd.DataFrame(columns=cols)

        block_all = block_events[["event_id"]].copy()
        if use_category and "category" in block_events.columns:
            block_all["category"] = block_events["category"].to_numpy()
        E = len(block_all)

        if n_cells <= 0:
            assigned = np.array([None] * E, dtype=object)
        else:
            assigned_idx = rng.integers(0, n_cells, size=E, endpoint=False)
            assigned = cell_ids[assigned_idx].astype(object)

        out = pd.DataFrame({
            "event_id": block_all["event_id"].to_numpy(),
            "assigned_cell_id": assigned,
            "chosen_from_hub": assigned,
        })
        if use_category:
            if category is not None:
                out.insert(1, "category", np.full(E, category, dtype=object))
            else:
                out.insert(1, "category", block_events["category"].to_numpy())
        if include_weights and weight_col in block_events.columns:
            insert_at = 2 if use_category else 1
            out.insert(insert_at, weight_col, block_events[weight_col].to_numpy())

        cols = ["event_id", "assigned_cell_id", "chosen_from_hub"]
        if use_category:
            cols = ["event_id", "category"] + cols[1:]
        if include_weights and weight_col in out.columns:
            insert_at = 2 if use_category else 1
            cols.insert(insert_at, weight_col)
        return out[cols]

    blocks = []
    if group_by_category:
        for cat, block in events_df.groupby("category", sort=False):
            blocks.append(run_block(block, category=cat))
    else:
        blocks.append(run_block(events_df, category=None))

    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(
        columns=["event_id", "assigned_cell_id", "chosen_from_hub"]
    )


def _build_cell_summary(
    allocations: pd.DataFrame,
    *,
    cell_ids: np.ndarray,
    cell_id_col: str,
    weight_col: str,
    include_weights: bool,
) -> pd.DataFrame:
    n = len(cell_ids)
    initial_demand = np.zeros(n, dtype=np.float64)
    satisfied = np.zeros(n, dtype=np.float64)
    unmet = np.zeros(n, dtype=np.float64)
    events_hosted = np.zeros(n, dtype=np.float64)
    times_hub = np.zeros(n, dtype=np.float64)

    if n > 0 and len(allocations) > 0 and "assigned_cell_id" in allocations.columns:
        id2idx = {cid: i for i, cid in enumerate(cell_ids)}
        if include_weights and weight_col in allocations.columns:
            weights = allocations[weight_col].to_numpy(dtype=float)
        else:
            weights = np.ones(len(allocations), dtype=float)

        assigned_idx = allocations["assigned_cell_id"].map(id2idx)
        mask = assigned_idx.notna()
        idx = assigned_idx[mask].astype(int).to_numpy()
        w = weights[mask.to_numpy()]
        if idx.size:
            counts = np.bincount(idx, weights=w, minlength=n)
            events_hosted = counts.astype(float)
            times_hub = events_hosted.copy()

    return pd.DataFrame({
        cell_id_col: cell_ids,
        "initial_demand": initial_demand,
        "satisfied": satisfied,
        "unmet": unmet,
        "times_chosen_as_hub": times_hub,
        "events_hosted": events_hosted,
    })


def main() -> None:
    _ensure_numpy_pickle_compat()
    grid = load_pickle(GRID_PKL)
    events = load_pickle(EVENTS_PKL)

    events = _prepare_events(events)
    if TITLE_COL not in events.columns:
        raise KeyError(f"Expected '{TITLE_COL}' in events for title->event_id mapping output")
    if EVENT_ID_COL not in events.columns:
        raise KeyError(f"Expected '{EVENT_ID_COL}' in events for title->event_id mapping output")
    title_to_event_id = dict(zip(events[TITLE_COL], events[EVENT_ID_COL]))

    cell_ids = _get_cell_ids(grid, CELL_ID_COL)

    allocations_runs = []
    cell_summary_runs = []

    # Use non-deterministic seeds by default so each run differs.
    seed_base = int(np.random.SeedSequence().entropy % (2**32 - 1))

    for run_id in range(N_RUNS):
        seed = (seed_base + run_id) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        allocations = _allocate_random(
            events,
            cell_ids=cell_ids,
            rng=rng,
            group_by_category=GROUP_BY_CATEGORY,
            weight_col=WEIGHT_COL,
            include_weights=WEIGHT_BY_TITLE_INSTANCES,
        )

        cell_summary = _build_cell_summary(
            allocations,
            cell_ids=cell_ids,
            cell_id_col=CELL_ID_COL,
            weight_col=WEIGHT_COL,
            include_weights=WEIGHT_BY_TITLE_INSTANCES,
        )

        allocations.insert(0, "run_id", run_id)
        cell_summary.insert(0, "run_id", run_id)

        allocations_runs.append(allocations)
        cell_summary_runs.append(cell_summary)

    allocations_all = pd.concat(allocations_runs, ignore_index=True)
    cell_summary_all = pd.concat(cell_summary_runs, ignore_index=True)

    save_pickle(
        {
            "allocations": allocations_all,
            "cell_summary": cell_summary_all,
            "title_to_event_id": title_to_event_id,
        },
        OUT_PATH,
    )

    print(f"Saved combined output to {OUT_PATH}")


if __name__ == "__main__":
    main()
