# llm_elaboration

## What is included
- `elaborate_events.py`: plain Python pipeline that reads pre-scraped events and calls OpenAI to extract structured fields.

## Responsibility split
- LLM (`llm_process`):
  - extracts `data_inizio`, `data_fine`, `orari_inizio`, `orari_fine`, `price`, `category`, and normalized `location` from raw text fields.
- Custom deterministic functions (Python):
  - `_load_input`: validates input structure/required keys.
  - `_build_user_content`: builds the prompt payload from raw event fields.
  - `_clean_hour`: normalizes missing hour values (`None`, `"None"`, empty strings) to `None`.
  - `_expand_dates` (used with `--include-dates`): converts start/end dates + weekday maps into per-day `dates` rows.
  - `elaborate_events`: orchestrates calls, attaches source metadata (`title`, `description`, `url`), and writes final records.

## Input schema expected
Input JSON must be a dict or list of dicts containing:
- `titolo`
- `data_ora`
- `prezzo`
- `location`
- `event_info`
- `url`

## Output schema (per event)
- `title`
- `location`
- `price`
- `category`
- `description`
- `url`
- `data_inizio`
- `data_fine`
- `orari_inizio`
- `orari_fine`
- `dates` (only when `--include-dates` is enabled)

## CLI parameters
- `--input PATH` (default `raw_events.json`)
- `--output PATH` (default `events_llm_elaborated.json`)
- `--model MODEL` (default `gpt-5-nano`)
- `--api-key-env ENV_NAME` (default `OPENAI_API_KEY`)
- `--max-events INT`
- `--delay-seconds FLOAT`
- `--include-dates`

## Run
```bash
cd llm_elaboration
export OPENAI_API_KEY="<your_key>"
python elaborate_events.py --input raw_events.json --output events_llm_elaborated.json --include-dates
```
