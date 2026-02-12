"""
Run greedy distance reallocation and deterministic checks.
"""

from pathlib import Path
from typing import Tuple
import sys

import numpy as np
import pandas as pd

from greedy_reallocator import (
    load_pickle,
    save_pickle,
    prep_events_table,
    GreedyDistanceReallocator,
)


# ----------------------------
# 1) Config
# ----------------------------
GRID_PKL = Path("hex_pop.pkl")
# Adjacency neighborhood (touching cells) on H3 level 9 grid.
NEIGH_PKL = Path("adjacent_neigh.pkl")
EVENTS_PKL = Path("gdf_filtered.pkl")

# Column names
CELL_ID_COL = "hex_id"
POP_COL = "population"

# Event handling
# "event": collapse same titles and add weight column
# "instance": keep all instances and reallocate each
EVENTS_MODE = "event"  # "event" | "instance"
WEIGHT_COL = "event_weight"
TITLE_COL = "title"
EVENT_ID_COL = "event_id"

# Allocation behavior
GROUP_BY_CATEGORY = True
CHECK_DETERMINISM = True

# Output
OUT_PATH_WS = Path("output/greedy_realloc_worst_served.pkl")
OUT_PATH_SI = Path("output/greedy_realloc_sum_improvement.pkl")


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
    if EVENTS_MODE == "event":
        return prep_events_table(
            events,
            dedupe_subset=TITLE_COL,
            dedupe=False,
            weight_by_instances=True,
            weight_col=WEIGHT_COL,
            id_col=EVENT_ID_COL,
        )
    if EVENTS_MODE == "instance":
        return prep_events_table(
            events,
            dedupe_subset=TITLE_COL,
            dedupe=False,
            weight_by_instances=False,
            weight_col=WEIGHT_COL,
            id_col=EVENT_ID_COL,
        )
    raise ValueError("EVENTS_MODE must be 'event' or 'instance'")


def _run_allocator(
    gr: GreedyDistanceReallocator,
    events: pd.DataFrame,
    *,
    objective_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return gr.allocate(
        events,
        group_by_category=GROUP_BY_CATEGORY,
        objective_mode=objective_mode,
        progress="none",
        event_weight_col=(WEIGHT_COL if EVENTS_MODE == "event" else None),
    )


def _assert_or_diff(df1: pd.DataFrame, df2: pd.DataFrame, label: str) -> bool:
    if df1.equals(df2):
        return True
    print(f"[determinism] {label} mismatch")
    try:
        diff = df1.compare(df2)
        print(diff.head(10))
    except Exception:
        print(f"left shape={df1.shape}, right shape={df2.shape}")
    return False


def _summary(allocations: pd.DataFrame, cell_ids: np.ndarray, dist: np.ndarray) -> Tuple[int, float, float]:
    if allocations.empty:
        return 0, float("inf"), float("inf")
    id2idx = {cid: i for i, cid in enumerate(cell_ids)}
    assigned = allocations["assigned_cell_id"].map(id2idx)
    assigned = assigned[assigned.notna()].astype(int).to_numpy()
    if assigned.size == 0:
        return 0, float("inf"), float("inf")
    dmin = np.min(dist[assigned], axis=0)
    return int(np.unique(assigned).size), float(np.max(dmin)), float(np.mean(dmin))


def main() -> None:
    _ensure_numpy_pickle_compat()
    grid = load_pickle(GRID_PKL)
    neigh = load_pickle(NEIGH_PKL)
    events = load_pickle(EVENTS_PKL)

    events = _prepare_events(events)
    if TITLE_COL not in events.columns:
        raise KeyError(f"Expected '{TITLE_COL}' in events for title->event_id mapping output")
    if EVENT_ID_COL not in events.columns:
        raise KeyError(f"Expected '{EVENT_ID_COL}' in events for title->event_id mapping output")
    title_to_event_id = dict(zip(events[TITLE_COL], events[EVENT_ID_COL]))

    gr = GreedyDistanceReallocator.from_grid(
        grid,
        neigh,
        pop_col=POP_COL,
        cell_id_col=CELL_ID_COL,
    )

    # ----------------------------
    # Tests / Runs
    # ----------------------------
    if CHECK_DETERMINISM:
        print("\n[determinism] running for objective_mode=worst_served")
        alloc_ws_1, summary_ws_1 = _run_allocator(gr, events, objective_mode="worst_served")
        alloc_ws_2, summary_ws_2 = _run_allocator(gr, events, objective_mode="worst_served")
        _assert_or_diff(alloc_ws_1, alloc_ws_2, "allocations (worst_served)")
        _assert_or_diff(summary_ws_1, summary_ws_2, "cell_summary (worst_served)")

        print("\n[determinism] running for objective_mode=sum_improvement")
        alloc_si_1, summary_si_1 = _run_allocator(gr, events, objective_mode="sum_improvement")
        alloc_si_2, summary_si_2 = _run_allocator(gr, events, objective_mode="sum_improvement")
        _assert_or_diff(alloc_si_1, alloc_si_2, "allocations (sum_improvement)")
        _assert_or_diff(summary_si_1, summary_si_2, "cell_summary (sum_improvement)")
    else:
        alloc_ws_1, summary_ws_1 = _run_allocator(gr, events, objective_mode="worst_served")
        alloc_si_1, summary_si_1 = _run_allocator(gr, events, objective_mode="sum_improvement")

    save_pickle(
        {
            "allocations": alloc_ws_1,
            "cell_summary": summary_ws_1,
            "title_to_event_id": title_to_event_id,
        },
        OUT_PATH_WS,
    )
    print(f"Saved greedy output to {OUT_PATH_WS}")

    save_pickle(
        {
            "allocations": alloc_si_1,
            "cell_summary": summary_si_1,
            "title_to_event_id": title_to_event_id,
        },
        OUT_PATH_SI,
    )
    print(f"Saved greedy output to {OUT_PATH_SI}")

    print("\n[comparison] worst_served vs sum_improvement")
    cell_ids = gr.grid.cell_ids
    dist = gr.grid.dist
    ws_unique, ws_worst, ws_mean = _summary(alloc_ws_1, cell_ids, dist)
    si_unique, si_worst, si_mean = _summary(alloc_si_1, cell_ids, dist)
    print(
        "worst_served: unique_hosts={}, max_d1={:.3f}, mean_d1={:.3f}".format(
            ws_unique, ws_worst, ws_mean
        )
    )
    print(
        "sum_improvement: unique_hosts={}, max_d1={:.3f}, mean_d1={:.3f}".format(
            si_unique, si_worst, si_mean
        )
    )


if __name__ == "__main__":
    main()
