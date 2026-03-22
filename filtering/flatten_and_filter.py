#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_utils import DEFAULT_INPUT_JSON, clean_expanded_events, flatten_events, load_events


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand event JSON into date-level rows and write cleaned tabular outputs."
    )
    parser.add_argument("--input-json", default=str(DEFAULT_INPUT_JSON))
    parser.add_argument("--expanded-json", default=str(ROOT / "events_expanded.json"))
    parser.add_argument("--expanded-csv", default=str(ROOT / "events_expanded.csv"))
    parser.add_argument("--cleaned-json", default=str(ROOT / "events_valid_light.json"))
    parser.add_argument("--cleaned-csv", default=str(ROOT / "events_valid_light.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_events(Path(args.input_json))

    expanded_df = flatten_events(events)
    cleaned_df = clean_expanded_events(expanded_df)

    Path(args.expanded_json).write_text(
        json.dumps(expanded_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    expanded_df.to_csv(args.expanded_csv, index=False, encoding="utf-8")

    Path(args.cleaned_json).write_text(
        json.dumps(cleaned_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cleaned_df.to_csv(args.cleaned_csv, index=False, encoding="utf-8")

    print(f"[flatten] expanded rows: {len(expanded_df)}")
    print("[flatten] cleaned rows by year:", cleaned_df.groupby("year").size().to_dict())
    print(f"[flatten] wrote: {args.expanded_json}")
    print(f"[flatten] wrote: {args.expanded_csv}")
    print(f"[flatten] wrote: {args.cleaned_json}")
    print(f"[flatten] wrote: {args.cleaned_csv}")


if __name__ == "__main__":
    main()
