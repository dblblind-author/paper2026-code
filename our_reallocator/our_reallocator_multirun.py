"""
Run many event-unit reallocations (outside the notebook) and save combined output.
"""

from pathlib import Path
import argparse
import sys

import pandas as pd

from event_unit_reallocator import (
    EventUnitReallocator,
    load_pickle,
    save_pickle,
    prep_events_table,
)


# ----------------------------
# 1) Config
# ----------------------------
GRID_PKL = Path("hex_pop.pkl")
NEIGH_PKL = Path("neigh_dict.pkl")
EVENTS_PKL = Path("gdf_filtered.pkl")

# Column names
POP_COL = "population"
CELL_ID_COL = "hex_id"

# Event handling
GROUP_BY = "event"  # "event" | "row"
WEIGHT_MODE = "direct"  # "direct" | "inverse" | "par"
WEIGHT_COL = "event_weight"
TITLE_COL = "title"
EVENT_ID_COL = "event_id"

# Allocation behavior
GROUP_BY_CATEGORY = True
PICK_WITH = "unmet_demand"  # "population" | "unmet_demand"

# Multi-run
N_RUNS = 50
RNG_SEED = 42

# Output
OUT_PATH = Path("output/reallocated_ours_50_runs.pkl")
PROGRESS = "none"  # "tqdm" | "print" | "none"


def _ensure_numpy_pickle_compat() -> None:
    # Compatibility shim for pickles created with newer numpy module paths.
    try:
        import numpy.core as _np_core
        sys.modules.setdefault("numpy._core", _np_core)
        if hasattr(_np_core, "_multiarray_umath"):
            sys.modules.setdefault("numpy._core._multiarray_umath", _np_core._multiarray_umath)
    except Exception:
        pass


def _prepare_events(
    events: pd.DataFrame,
    *,
    group_by: str,
    weight_mode: str,
) -> pd.DataFrame:
    return prep_events_table(
        events,
        group_by=group_by,
        weight_mode=weight_mode,
        dedupe_subset=TITLE_COL,
        weight_col=WEIGHT_COL,
        id_col=EVENT_ID_COL,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple event-unit reallocations.")
    parser.add_argument(
        "--group-by",
        choices=["event", "row"],
        default=None,
        help="Group events by 'event' (collapse duplicate titles) or keep all 'row' entries.",
    )
    parser.add_argument(
        "--weight-mode",
        choices=["direct", "inverse", "par"],
        default=None,
        help="Weighting mode when group_by='title': direct, inverse, or par (no weights).",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Override output pickle path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    group_by = (args.group_by or GROUP_BY).strip().lower()
    weight_mode = (args.weight_mode or WEIGHT_MODE).strip().lower()
    if group_by in {"row", "rows", "record", "records", "raw"} and weight_mode != "par":
        raise ValueError("group_by='row' requires weight_mode='par' (no weights available)")
    out_path = Path(args.out_path) if args.out_path else OUT_PATH

    _ensure_numpy_pickle_compat()
    grid = load_pickle(GRID_PKL)
    neigh = load_pickle(NEIGH_PKL)
    events = load_pickle(EVENTS_PKL)

    events = _prepare_events(events, group_by=group_by, weight_mode=weight_mode)
    if TITLE_COL not in events.columns:
        raise KeyError(f"Expected '{TITLE_COL}' in events for title->event_id mapping output")
    if EVENT_ID_COL not in events.columns:
        raise KeyError(f"Expected '{EVENT_ID_COL}' in events for title->event_id mapping output")
    title_to_event_id = dict(zip(events[TITLE_COL], events[EVENT_ID_COL]))

    er = EventUnitReallocator.from_grid(
        grid,
        neigh,
        pop_col=POP_COL,
        cell_id_col=CELL_ID_COL,
    )

    allocations_runs = []
    cell_summary_runs = []

    for run_id in range(N_RUNS):
        seed = RNG_SEED + run_id
        allocations, cell_summary = er.allocate(
            events,
            group_by_category=GROUP_BY_CATEGORY,
            pick_with=PICK_WITH,
            rng_seed=seed,
            progress=PROGRESS,
            event_weight_col=(
                WEIGHT_COL
                if (group_by in {"event", "events", "instance", "instances", "title", "titles"} and weight_mode in {"direct", "inverse"})
                else None
            ),
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
        out_path,
    )

    print(f"Saved combined output to {out_path}")


if __name__ == "__main__":
    main()
