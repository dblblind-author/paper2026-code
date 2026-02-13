# cityscore_baseline_seasons

Compute seasonal and annual CityScore statistics.

## Main files
- `seasonal_cityscore_stats.py`: seasonal/annual metrics.
- `isochrones.py`, `CityScoreToolkit_plain_v2.py`: scoring utilities.

## `seasonal_cityscore_stats.py` parameters
- `--events PATH [PATH ...]` (default `to_evaluate/gdf_2023.pkl to_evaluate/gdf_2024.pkl`)
- `--iso PATH [PATH ...]` (default `iso/walk_15min_4.5kmh.pkl iso/bike_15min_15kmh.pkl`)
- `--alpha FLOAT` (default `0.08`)
- `--hex-to-nil PATH` (default `hex_to_nil.csv`)
- `--date-col COL` (default `date`)
- `--out-dir DIR` (default `outputs`)

Compatibility note:
- If `gdf_2023.pkl` / `gdf_2024.pkl` are missing, the script automatically falls back to `gdf_2023_final.pkl` / `gdf_2024_final.pkl` when present.

## Output
- `outputs/summary_cityscore_seasons_<timestamp>.csv`

Columns include annual/seasonal:
- `cityscore_mean`
- `cityscore_pop_weighted_mean`
- `cityscore_gini`
- population share bins (`pop_share_0_20` ... `pop_share_80_100`).

## Linking
- Inputs usually align with baseline events used in `cityscore_baseline/`.
- Output is a compact stats table for reporting or external plotting.
