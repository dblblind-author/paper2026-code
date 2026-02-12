# parser_baseline_method

Rule-based parser + TF-IDF category assignment for raw event records.

## Main file
- `tfidf_dateparser_parse_events.py`

## Inputs
- default `raw_events.json`
- supports JSON input; non-JSON input is treated as text blocks.

## CLI parameters
- `--input PATH` (default `raw_events.json`)
- `--output PATH` (default `events_tfidf_processed.json`)
- `--default-year INT` (default `2023`)
- `--evaluated-url PATH` (optional allowlist file)

## What it extracts
- normalized `title`, `location`, `price`, `description`, `url`
- inferred `category` via TF-IDF similarity over keyword prototypes
- `dates` list with opening/closing hours, including date ranges and weekday mappings

## Run
```bash
cd parser_baseline_method
python tfidf_dateparser_parse_events.py --input raw_events.json --output events_tfidf_processed.json
```

## Linking with other folders
This output is an intermediate semantic JSON. Reallocator/CityScore scripts in this repo expect GeoDataFrame-style `.pkl` inputs, so a separate conversion/geocoding step is needed before feeding downstream modules.
