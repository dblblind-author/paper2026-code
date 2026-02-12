# Workspace Guide

This repository contains multiple scripts for:
- parsing events,
- generating reallocation scenarios,
- evaluating CityScore,
- plotting and comparing NIL-level results.

## Folder map
- `parser_baseline_method/`: parse raw event text/JSON into normalized event JSON.
- `our_reallocator/`: custom reallocation (`direct` / `inverse` / `par`) over multiple runs.
- `random_reallocator/`: random and population-weighted random multi-run reallocations.
- `greedy_reallocator/`: deterministic greedy reallocations.
- `cityscore/`: evaluate reallocated outputs with CityScore, produce NIL summaries/maps.
- `cityscore_baseline/`: evaluate baseline (real) event datasets with CityScore.
- `cityscore_baseline_plot/`: yearly facet plots from baseline CityScore pickles.
- `cityscore_baseline_seasons/`: seasonal and annual baseline statistics.
- `cityscore_popw_plot/`: bivariate NIL plots (population vs pop-weighted CityScore).
- `plot_difference/`: category-level before/after comparisons and visualizations.

## Data flow schema

### Reallocation evaluation flow
1. Generate allocations:
   - `our_reallocator/our_reallocator_multirun.py`
   - `random_reallocator/random_reallocator_multirun.py`
   - `random_reallocator/random_reallocator_multirun_pop_weighted.py`
   - `greedy_reallocator/greedy_reallocator_run.py`
2. Place/copy produced allocation `.pkl` files into `cityscore/to_evaluate/` (or pass absolute paths).
3. Evaluate with `cityscore/evaluate_realloc_cityscore.py`.
4. Use outputs in `cityscore/outputs/...` for:
   - direct reporting,
   - `cityscore_popw_plot/` (expects NIL summary CSV + baseline CityScore pickle),
   - `plot_difference/` (expects reallocated pickle + baseline/non-relocatable inputs).

### Baseline flow
1. Run `cityscore_baseline/evaluate_real_cityscore.py` to score real events.
2. Use generated `cityscore_*.pkl` in:
   - `cityscore_baseline_plot/plot_yearly_cityscore.py`
   - `cityscore_popw_plot/make_baseline_plots.py` (for baseline side of comparison)
   - `cityscore_baseline_seasons/seasonal_cityscore_stats.py` (seasonal statistics).

## File-to-file linkage examples
- Reallocator output (`allocations` table with `assigned_cell_id`) -> `cityscore/evaluate_realloc_cityscore.py --files ...`.
- `cityscore/outputs/.../nil_summary_*.csv` -> `cityscore_popw_plot/DATASETS[*].path` when `kind: "csv"`.
- Baseline cityscore pickle (`cityscore_walk_...pkl`) -> `cityscore_baseline_plot` and `cityscore_popw_plot`.
- `plot_difference/make_nil_event_counts.py` output `outputs/nil_events_avg_across_runs_with_nonreloc.csv` -> `plot_difference/make_nil_visualizations.py --nil-avg`.

## Notes on parameters
- CLI scripts expose parameters with `--help`.
- Non-CLI scripts in this repo use constants at the top of each file (for example `GRID_PKL`, `EVENTS_PKL`, `OUT_PATH`, `N_RUNS`).
- Folder-level `README.md` files document each script's parameters and defaults.
