# random_reallocator

Generate random multi-run allocations.

## Main files
- `random_reallocator_multirun.py`: uniform random cell sampling.
- `random_reallocator_multirun_pop_weighted.py`: population-weighted sampling.
- `event_prep_utils.py`: shared event preparation utilities.

## Inputs (defaults)
- `hex_pop.pkl`
- `gdf_2024_final_v3_ready_to_realloc.pkl`

## Configuration parameters (top-of-file constants)
Common:
- `N_RUNS` (default `100`)
- `GROUP_BY_CATEGORY` (bool)
- `WEIGHT_BY_TITLE_INSTANCES` (bool)
- `WEIGHT_COL`, `TITLE_COL`, `EVENT_ID_COL`
- `OUT_PATH`

Weighted variant only:
- `POP_COL` (default `population`)

## Run
```bash
cd random_reallocator
python random_reallocator_multirun.py
python random_reallocator_multirun_pop_weighted.py
```

## Output files
- `output/reallocated_random_100_runs.pkl`
- `output/reallocated_random_pop_weighted_100_runs.pkl`

Each pickle contains:
- `allocations`
- `cell_summary`
- `title_to_event_id`

## Linking with other folders
Use these output pickles in `cityscore/evaluate_realloc_cityscore.py --files ...`.
