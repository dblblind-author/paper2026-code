# Filtering Pipeline

## Scripts

- `flatten_and_filter.py`
  - expands the source event JSON into date-level rows
  - writes both expanded and cleaned tabular outputs

- `geocode_events.py`
  - geocodes cleaned rows with Google Geocoding
  - uses a local cache file
  - reads the API key from `GOOGLE_MAPS_API_KEY`

- `build_yearly_pickles_from_json.py`
  - runs the end-to-end pipeline from the source JSON
  - geocodes locations with the cache
  - builds yearly GeoDataFrame pickles
  - filters rows against the Milan NIL shapefile
  - optionally compares rebuilt outputs with final pickles

- `pipeline_utils.py`
  - shared helpers used by the scripts above


## Typical Usage

Prepare intermediate cleaned files:

```bash
python filtering/flatten_and_filter.py \
  --input-json "filtering/all_events_llm (3).json"
```

Geocode the cleaned rows:

```bash
python filtering/geocode_events.py \
  --input-csv filtering/events_valid_light.csv
```

Run the full end-to-end yearly build:

```bash
python filtering/build_yearly_pickles_from_json.py \
  --input-json "filtering/all_events_llm (3).json" \
  --final-2023 filtering/gdf_2023_final_v3.pkl \
  --final-2024 filtering/gdf_2024_final_v3.pkl \
  --out-2023 filtering/gdf_2023_rebuilt_from_json_v3.pkl \
  --out-2024 filtering/gdf_2024_rebuilt_from_json_v3.pkl \
  --delta-2023 filtering/delta_table_2023_v3.csv \
  --delta-2024 filtering/delta_table_2024_v3.csv
```

