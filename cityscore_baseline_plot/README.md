# cityscore_baseline_plot

Create yearly facet maps (rows=years, cols=modes) from baseline CityScore pickles.

## Main file
- `plot_yearly_cityscore.py`

## Input naming convention
The script searches `--input_dir` with pattern:
- `cityscore_<mode>_*_<year>.pkl`

Typical files come from `cityscore_baseline/outputs/` after renaming/moving.

## CLI parameters
- `--input_dir DIR` (default `.`)
- `--minutes INT` (default `15`, used only in output filename)
- `--years INT [INT ...]` (default `2023 2024`)
- `--modes STR [STR ...]` (default `walk bike`)
- `--out_png PATH` (default: `<input_dir>/plots_cityscore/cityscore_facets_<minutes>min_2x2.png`)

## Output
- One PNG facet map with shared colorbar.

## Example
```bash
cd cityscore_baseline_plot
python plot_yearly_cityscore.py --input_dir . --years 2023 2024 --modes walk bike
```
