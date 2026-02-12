# cityscore_baseline

Compute CityScore on real/baseline event GeoDataFrames.

## Main files
- `evaluate_real_cityscore.py`: batch scoring for multiple event files, ISO files, and alpha values.
- `evaluate_realloc_cityscore.py`: legacy reallocation evaluator variant kept for compatibility.
- `isochrones.py`, `CityScoreToolkit_plain_v2.py`: toolkit utilities.

## `evaluate_real_cityscore.py`
Purpose: produce baseline CityScore pickles and run summary.

Parameters:
- `--events PATH [PATH ...]` (default: `to_evaluate/gdf_2023_final.pkl`, `to_evaluate/gdf_2024_final.pkl`)
- `--iso-dir DIR` (default: `iso`)
- `--out-dir DIR` (default: `outputs`)
- `--alphas CSV` (default: `0.06,0.08,0.1`)
- `--hex-to-nil PATH` (default: `hex_to_nil.csv`)

Outputs:
- `outputs/cityscore_<iso_name>_alpha<alpha>_<year>.pkl`
- `outputs/summary_cityscore_runs_<timestamp>.csv`

## `evaluate_realloc_cityscore.py`
Same evaluation style as `cityscore/evaluate_realloc_cityscore.py` but without non-relocatable merge/random-in-cell options.

Parameters:
- `--iso`, `--hex-to-nil`, `--hex-crs`, `--events-crs`, `--nils`, `--out-dir`, `--limit-runs`, `--workers`, `--files`

## Linking with other folders
- Downstream baseline artifacts are consumed by:
  - `cityscore_baseline_plot/plot_yearly_cityscore.py`
  - `cityscore_popw_plot/make_baseline_plots.py`
  - `cityscore_baseline_seasons/seasonal_cityscore_stats.py` (if reused there).
