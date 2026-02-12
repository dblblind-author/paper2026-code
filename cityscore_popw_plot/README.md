# cityscore_popw_plot

Generate NIL-level comparison maps/scatter plots using bivariate bins:
- x axis: pop-weighted CityScore
- y axis: total NIL population

## Main file
- `make_baseline_plots.py`

This script is config-driven (no CLI). Edit constants in file top section.

## Key configuration parameters
- `HEX_TO_NIL_CSV`, `NIL_SHP`: reference geospatial data.
- `DATASETS`: dictionary of named inputs.
  - `kind: "pkl"` expects a CityScore GeoDataFrame pickle.
  - `kind: "csv"` expects NIL summary with `NIL`, `pop_total`, `cityscore_pw_mean`.
- `RUN_KEYS`: which dataset keys to render.
- `TERTILES_REF_KEY`: dataset key used to fix tertile thresholds across all plots.
- `SCATTER_LABEL_QUERY`: optional NIL label filter for extra labeled scatter output.

## Expected link inputs
- Baseline side: pickle from `cityscore_baseline/outputs/`.
- Reallocated side: NIL summary CSV from `cityscore/outputs/.../nil_summary_*.csv`.

## Outputs
For each dataset key in `RUN_KEYS`:
- `<key>_scatter.png`
- `<key>_scatter_<suffix>.png` (if label query enabled)
- `<key>_map.png`

## Example workflow
1. Copy/point baseline pickle into this folder.
2. Copy/point reallocated NIL summary CSV into this folder.
3. Update `DATASETS` paths and run:
```bash
cd cityscore_popw_plot
python make_baseline_plots.py
```
