# plot_difference

Compare baseline vs reallocated event distributions at NIL/category level.

## Main files
- `make_nil_event_counts.py`: per-run and average NIL event counts.
- `make_nil_visualizations.py`: heatmap, bar charts, optional choropleths, and before/after CSV.

## `make_nil_event_counts.py` parameters
- `--reallocated` (default `reallocated_2024_par_50_runs.pkl`)
- `--hex-to-nil` (default `hex_to_nil.csv`)
- `--non-relocatable` (default `gdf_2024_non_relocatable.pkl`)
- `--baseline` (default `gdf_2024_final_v3.pkl`)
- `--out-by-run` (default `outputs/nil_events_by_run_with_nonreloc.csv`)
- `--out-avg` (default `outputs/nil_events_avg_across_runs_with_nonreloc.csv`)
- `--out-baseline` (default `outputs/nil_events_baseline_gdf_2024_final_v3.csv`)

## `make_nil_visualizations.py` parameters
- `--baseline-gdf`
- `--reallocated-pkl`
- `--non-relocatable`
- `--hex-to-nil`
- `--nil-avg` (links to output of `make_nil_event_counts.py`)
- `--nils-shp` (optional)
- `--out-dir` (default `outputs/viz`)
- `--year` (default `2024`)
- `--categories` (comma-separated override)
- `--top-k` (default `30`)

## Output files (`make_nil_visualizations.py`)
- `nil_category_change_matrix.png`
- `nil_category_top10_posneg_bars.png`
- `nil_category_before_after_diff.csv`
- `nil_category_choropleths.png` (only if `--nils-shp` provided)

## Linking with other folders
- Reallocated pickle can come directly from `our_reallocator/output/`, `random_reallocator/output/`, or `greedy_reallocator/output/`.
- Baseline and non-relocatable GeoDataFrames should match the dataset used in `cityscore/` evaluations.
