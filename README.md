# Workspace Guide

This repository contains multiple scripts for:
- scraping raw event pages,
- parsing events,
- filtering and geocoding yearly GeoDataFrames,
- clustering event locations across years,
- generating reallocation scenarios,
- evaluating CityScore,
- plotting and comparing NIL-level results.

## Folder map
- `event_scraper/`: scrape raw Milanotoday event pages into a raw JSON feed.
- `parser_baseline_method/`: parse raw event text/JSON into normalized event JSON.
- `llm_elaboration/`: LLM-only extraction/elaboration from pre-scraped event records.
- `filtering/`: flatten JSON events, geocode locations, and rebuild yearly event GeoDataFrames.
- `clustering/`: DBSCAN elbow analysis, clustering summaries, and stable-cluster visualizations for 2023 vs 2024.
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

### Event parsing/elaboration flow
1. Parse/scrape raw events (external preprocessing step).
2. Run `event_scraper/scrape_events.py` if you need to collect the raw event pages first.
3. Run `llm_elaboration/elaborate_events.py` on raw event records to get structured fields (`data_inizio`, `orari_*`, `price`, `category`).
4. Use `filtering/` scripts to flatten, clean, geocode, and rebuild yearly GeoDataFrames from the structured JSON.

### Clustering flow
1. Prepare yearly GeoDataFrames for clustering as `gdf_2023` and `gdf_2024`.
2. Run `clustering/dbscan_clustering.py` to generate elbow plots and clustering summary CSVs.
3. Run the plotting helpers in `clustering/` to inspect unique-title distributions, DBSCAN clusters, and Hungarian-matched stable clusters between years.

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
3. Preferred event filename convention is non-`_final` (for example `gdf_2024.pkl`).
   `cityscore_baseline_seasons` still supports `_final` files as fallback for compatibility.
4. The clustering scripts follow the same naming convention and now default to `gdf_2023` / `gdf_2024`.


## Notes on parameters
- CLI scripts expose parameters with `--help`.
- Non-CLI scripts in this repo use constants at the top of each file (for example `GRID_PKL`, `EVENTS_PKL`, `OUT_PATH`, `N_RUNS`).
- Folder-level `README.md` files document each script's parameters and defaults.
