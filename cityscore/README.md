# cityscore

Evaluate reallocation outputs with CityScore and create summary CSVs + NIL choropleth.

## Main files
- `evaluate_realloc_cityscore.py`: main evaluator for single-run and multi-run allocations.
- `isochrones.py`: helper utilities (`StaticISO`, hex-id normalization, fast scoring wrapper).
- `CityScoreToolkit_plain_v2.py`: toolkit with ISO generation and scoring internals.

## Inputs expected by `evaluate_realloc_cityscore.py`
- `--iso` (default: `iso/walk_15.pkl`): static isochrone object (`StaticISO`).
- `--hex-to-nil` (default: `hex_to_nil.csv`): must include `hex_id`, `geometry`, `population`, `NIL`.
- `--nils` (default: `NIL/NIL_WM.shp`): NIL boundaries.
- `--files`: one or more allocation pickles; each must contain dict key `allocations`.
- `--non-relocatable` (default: `gdf_2024_final_non_relocatable.pkl`): events added to every run.

Allocation schema expected in `allocations`:
- required: `assigned_cell_id`
- optional for multi-run: `run_id`
- recommended: `event_id`, `category`, weight columns.

## CLI parameters (`evaluate_realloc_cityscore.py`)
- `--iso PATH`
- `--hex-to-nil PATH`
- `--hex-crs EPSG` (default `EPSG:6707`)
- `--events-crs EPSG` (default `EPSG:4326`)
- `--nils PATH`
- `--out-dir PATH`
- `--limit-runs INT`
- `--workers INT`
- `--non-relocatable PATH`
- `--no-randomize-within-cell`
- `--random-seed INT`
- `--files PATH [PATH ...]`

## Outputs
Under `--out-dir` (default `outputs/cityscore_eval`):
- `summary.csv`: per-method aggregates.
- `by_run.csv`: per-run metrics (multi-run only).
- `nil_summary_<method>.csv`: NIL-level means.
- `nil_summary_<method>_by_run.csv`: NIL-by-run details (multi-run only).
- `map_<method>_cityscore_mean.png`
- `map_<method>_pop_weighted_cityscore_mean.png`

## Linking with other folders
- Upstream: consumes `.pkl` from `our_reallocator/output/`, `random_reallocator/output/`, `greedy_reallocator/output/`.
- Downstream: produced NIL summary CSVs can feed `cityscore_popw_plot/` and `plot_difference/`.

## Example
```bash
cd cityscore
python evaluate_realloc_cityscore.py \
  --files to_evaluate/reallocated_2024_par_50_runs.pkl \
          to_evaluate/reallocated_random_100_runs.pkl \
          to_evaluate/greedy_realloc_sum_improvement.pkl \
  --out-dir outputs/cityscore_eval_to_evaluate
```
