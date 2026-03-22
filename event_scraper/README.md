# event_scraper

Standalone raw scraper for Milanotoday event pages.

## Main files
- `scrape_events.py`: discovers event URLs from listing pages and fetches raw event content for downstream elaboration.

## `scrape_events.py`
Purpose:
- collect listing-page URLs with Playwright
- fetch event-page HTML with `requests`
- extract raw fields into a JSON format that can be consumed by `llm_elaboration/elaborate_events.py`

Notes:
- this is the earliest stage in the pipeline
- it produces raw records, not final GeoDataFrames
- it does not contain any hardcoded API key
