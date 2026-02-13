# llm_elaboration

Refactor of `scraper_llm.ipynb` focused only on the LLM elaboration step.

## What is included
- `elaborate_events.py`: plain Python pipeline that reads pre-scraped events and calls OpenAI to extract structured fields.
- `scraper_llm.ipynb`: original notebook kept as legacy reference.

## What is intentionally excluded
- Web scraping logic (`requests`/`BeautifulSoup`/`playwright`) is not part of the refactored script.
- Notebook analytics/plotting cells are not part of the refactored script.

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

## About normalization in the original notebook
Yes, custom normalization/post-processing code is present in the notebook, not only in the LLM prompt. Examples:
- `build_date_schedule_std`
- `transform_entry` / `transform`
- `categorize_time`

In this refactor, only elaboration logic is kept; optional date expansion (`--include-dates`) mirrors the lightweight `transform` behavior.
