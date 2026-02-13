#!/usr/bin/env python3
"""
LLM-only event elaboration pipeline extracted from scraper_llm.ipynb.

This script intentionally keeps only the elaboration step:
- input: pre-scraped raw events JSON
- processing: OpenAI JSON extraction
- output: structured event JSON

No web scraping is performed here.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

INPUT_REQUIRED_KEYS = ("titolo", "data_ora", "prezzo", "location", "event_info", "url")
IT_WEEKDAYS = ["lunedi", "martedi", "mercoledi", "giovedi", "venerdi", "sabato", "domenica"]

SYSTEM_PROMPT = """
Sei un parser testuale rigoroso.
Dato il seguente testo, estrai:

- data_inizio: data di inizio evento, formato YYYY-MM-DD.
- data_fine: data di fine evento, formato YYYY-MM-DD (uguale a data_inizio se viene fornita una sola data).
- orari_inizio: mappa con le chiavi (tutte minuscole, senza accenti): lunedi, martedi, mercoledi, giovedi, venerdi, sabato, domenica.
  Ogni valore e una stringa HH:MM (24h) se l'orario di APERTURA e presente per quel giorno, altrimenti "None".
- orari_fine: mappa con le stesse chiavi, con l'orario di CHIUSURA HH:MM, altrimenti "None".
- price: uno dei soli valori ["gratis", "a pagamento", "non disponibile"].
- category: una (e solo una) tra ["community_lifestyle", "art_exhibitions", "Music_concerts", "sport_fitness", "theatre_shows"].

Regole:
- Non inventare dati: se qualcosa e assente o ambiguo, usa "None" per orari o null per le date.
- Normalizza orari in HH:MM.
- Mantieni location come testo sorgente (rimuovendo eventuali duplicazioni di parole/spazi).

Usa SOLO questi campi sorgente:
- PREZZO <- campo prezzo
- DATA <- campo data_ora
- CATEGORIA <- descrizione/titolo
- LOCATION <- campo location

Rispondi SOLO con un JSON valido in questo schema:
{
  "data_inizio": "YYYY-MM-DD" | null,
  "data_fine": "YYYY-MM-DD" | null,
  "orari_inizio": {
    "lunedi": "HH:MM" | "None",
    "martedi": "HH:MM" | "None",
    "mercoledi": "HH:MM" | "None",
    "giovedi": "HH:MM" | "None",
    "venerdi": "HH:MM" | "None",
    "sabato": "HH:MM" | "None",
    "domenica": "HH:MM" | "None"
  },
  "orari_fine": {
    "lunedi": "HH:MM" | "None",
    "martedi": "HH:MM" | "None",
    "mercoledi": "HH:MM" | "None",
    "giovedi": "HH:MM" | "None",
    "venerdi": "HH:MM" | "None",
    "sabato": "HH:MM" | "None",
    "domenica": "HH:MM" | "None"
  },
  "price": "gratis" | "a pagamento" | "non disponibile",
  "category": "community_lifestyle" | "art_exhibitions" | "Music_concerts" | "sport_fitness" | "theatre_shows",
  "location": "string" | "N/A"
}
""".strip()


def _load_input(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        events = data
    elif isinstance(data, dict) and isinstance(data.get("events"), list):
        events = data["events"]
    elif isinstance(data, dict):
        events = [data]
    else:
        raise ValueError("Input JSON must be a dict or list of dicts")

    for idx, event in enumerate(events):
        missing = [k for k in INPUT_REQUIRED_KEYS if k not in event]
        if missing:
            raise KeyError(f"Event index {idx} missing keys: {missing}")
    return events


def _build_user_content(event: Dict[str, Any]) -> str:
    description = f"{event.get('titolo', '')}{event.get('event_info', '')}"
    return (
        f"DESCRIZIONE: ---{description}---\n"
        f"PREZZO: ---{event.get('prezzo', '')}---\n"
        f"DATA: ---{event.get('data_ora', '')}---\n"
        f"LOCATION: ---{event.get('location', '')}---"
    )


def _clean_hour(value: Any) -> Any:
    if value in (None, "", "None", "none", "N/A", "n/a"):
        return None
    return value


def _expand_dates(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    start = payload.get("data_inizio")
    end = payload.get("data_fine")
    if not start or not end:
        return []

    start_dt = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    open_map = payload.get("orari_inizio", {}) or {}
    close_map = payload.get("orari_fine", {}) or {}

    out: List[Dict[str, Any]] = []
    d = start_dt
    one_day = timedelta(days=1)
    while d <= end_dt:
        wd = IT_WEEKDAYS[d.weekday()]
        out.append(
            {
                "date": d.isoformat(),
                "weekday": wd,
                "opening_hour": _clean_hour(open_map.get(wd)),
                "closing_hour": _clean_hour(close_map.get(wd)),
            }
        )
        d += one_day
    return out


def llm_process(client: OpenAI, model: str, event: Dict[str, Any]) -> Dict[str, Any]:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_content(event)},
        ],
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content or "{}"
    return json.loads(content)


def elaborate_events(
    client: OpenAI,
    model: str,
    events: List[Dict[str, Any]],
    *,
    include_dates: bool,
    max_events: int | None,
    delay_seconds: float,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    selected = events if max_events is None else events[:max_events]

    for idx, event in enumerate(selected, start=1):
        parsed = llm_process(client, model, event)

        row: Dict[str, Any] = {
            "title": event.get("titolo"),
            "location": parsed.get("location", event.get("location")),
            "price": parsed.get("price"),
            "category": parsed.get("category"),
            "description": event.get("event_info"),
            "url": event.get("url"),
            "data_inizio": parsed.get("data_inizio"),
            "data_fine": parsed.get("data_fine"),
            "orari_inizio": parsed.get("orari_inizio", {}),
            "orari_fine": parsed.get("orari_fine", {}),
        }
        if include_dates:
            row["dates"] = _expand_dates(parsed)

        output.append(row)
        print(f"[{idx}/{len(selected)}] elaborated: {row.get('title')}")

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refactored LLM elaboration pipeline (no scraping)."
    )
    parser.add_argument(
        "--input",
        default="raw_events.json",
        help="Input JSON with raw events from scraper (default: raw_events.json)",
    )
    parser.add_argument(
        "--output",
        default="events_llm_elaborated.json",
        help="Output JSON path (default: events_llm_elaborated.json)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-nano",
        help="OpenAI model for extraction (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Env var name containing OpenAI API key (default: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional cap on number of events to process",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay between API calls",
    )
    parser.add_argument(
        "--include-dates",
        action="store_true",
        help="Also build expanded per-day schedule in `dates`",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Missing API key: set env var {args.api_key_env} before running."
        )

    events = _load_input(Path(args.input))
    client = OpenAI(api_key=api_key)

    result = elaborate_events(
        client,
        args.model,
        events,
        include_dates=args.include_dates,
        max_events=args.max_events,
        delay_seconds=args.delay_seconds,
    )

    out_path = Path(args.output)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(result)} records to {out_path}")


if __name__ == "__main__":
    main()
