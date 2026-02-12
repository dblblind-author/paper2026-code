# our_reallocator

Custom multi-run event-unit reallocator.

## Main files
- `our_reallocator_multirun.py`: run N reallocations and write one combined pickle.
- `event_unit_reallocator.py`: core algorithm + utilities.

## Inputs (defaults)
- `hex_pop.pkl`
- `neigh_dict.pkl`
- `gdf_filtered.pkl`

## CLI parameters (`our_reallocator_multirun.py`)
- `--group-by {event,row}`
- `--weight-mode {direct,inverse,par}`
- `--out-path PATH`

Behavior:
- `group-by=event`: collapse duplicate titles.
- `group-by=row`: keep each row.
- `weight-mode=direct`: weight = count of duplicated title rows.
- `weight-mode=inverse`: weight = 1 / count.
- `weight-mode=par`: no weights.
- Constraint: `group-by=row` requires `weight-mode=par`.

## Script constants you can change
- `N_RUNS` (default `50`)
- `RNG_SEED` (default `42`)
- `PICK_WITH` (`unmet_demand` or `population`)
- `GROUP_BY_CATEGORY` (bool)

## Output
Default: `output/reallocated_ours_50_runs.pkl`

Stored dict keys:
- `allocations`
- `cell_summary`
- `title_to_event_id`

## Linking with other folders
Output pickle is consumed by `cityscore/evaluate_realloc_cityscore.py --files ...`.
