# greedy_reallocator

Deterministic greedy reallocation with two objective modes.

## Main files
- `build_neighborhood.py`: build `adjacent_neigh.pkl` from `milano_h3_r9.gpkg`.
- `greedy_reallocator_run.py`: run allocator and save two outputs.
- `greedy_reallocator.py`: core implementation and utility functions.

## `build_neighborhood.py` config
- `GRID_PKL = hex_pop.pkl`
- `GPKG_PATH = milano_h3_r9.gpkg`
- `GPKG_LAYER = milano_h3_r9`
- `GPKG_ID_COL = h3_id`
- `CELL_ID_COL = hex_id`
- `OUT_PATH = adjacent_neigh.pkl`

Run:
```bash
cd greedy_reallocator
python build_neighborhood.py
```

## `greedy_reallocator_run.py` config parameters
- `GRID_PKL`, `NEIGH_PKL`, `EVENTS_PKL`
- `EVENTS_MODE`: `event` (collapse titles + weights) or `instance` (all rows)
- `GROUP_BY_CATEGORY` (bool)
- `CHECK_DETERMINISM` (bool)
- outputs:
  - `OUT_PATH_WS = output/greedy_realloc_worst_served.pkl`
  - `OUT_PATH_SI = output/greedy_realloc_sum_improvement.pkl`

Run:
```bash
cd greedy_reallocator
python greedy_reallocator_run.py
```

## Output schema
Each pickle stores:
- `allocations`
- `cell_summary`
- `title_to_event_id`

## Linking with other folders
Produced `.pkl` files are consumed by `cityscore/evaluate_realloc_cityscore.py` via `--files`.
