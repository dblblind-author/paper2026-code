#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from pipeline_utils import (
    DEFAULT_GEOCODE_CACHE,
    ROOT,
    attach_coordinates_from_locations,
    build_geodataframe,
    build_unique_location_dict,
    geocode_unique_locations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geocode cleaned event rows with Google Geocoding using an environment variable API key."
    )
    parser.add_argument("--input-csv", default=str(ROOT / "events_valid_light.csv"))
    parser.add_argument("--output-json", default=str(ROOT / "events_geocoded.json"))
    parser.add_argument("--output-csv", default=str(ROOT / "events_geocoded.csv"))
    parser.add_argument("--cache", default=str(DEFAULT_GEOCODE_CACHE))
    parser.add_argument("--api-key-env", default="GOOGLE_MAPS_API_KEY")
    parser.add_argument("--city-suffix", default="Milano")
    parser.add_argument("--pause-seconds", type=float, default=0.25)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--refresh-missing-cache",
        action="store_true",
        help="Retry locations already cached with null coordinates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    unique_locations = build_unique_location_dict(df["location"].tolist())
    api_key = os.getenv(args.api_key_env)
    norm_to_coords = geocode_unique_locations(
        unique_locations,
        api_key=api_key,
        cache_path=Path(args.cache),
        city_suffix=args.city_suffix,
        pause_seconds=args.pause_seconds,
        max_retries=args.max_retries,
        refresh_missing_cache=args.refresh_missing_cache,
    )

    geocoded_df = attach_coordinates_from_locations(df, norm_to_coords)
    geocoded_gdf = build_geodataframe(geocoded_df)
    geocoded_df = pd.DataFrame(geocoded_gdf).copy()
    geocoded_df["geometry_wkt"] = geocoded_df["geometry"].map(lambda geom: geom.wkt if geom is not None else None)

    Path(args.output_json).write_text(
        json.dumps(geocoded_df.to_dict(orient="records"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    geocoded_df.to_csv(args.output_csv, index=False, encoding="utf-8")

    success_count = geocoded_df["coordinates"].notna().sum()
    missing_count = geocoded_df["coordinates"].isna().sum()
    print(f"[geocode] input rows: {len(df)}")
    print(f"[geocode] unique locations: {len(unique_locations)}")
    print(f"[geocode] rows with coordinates: {int(success_count)}")
    print(f"[geocode] rows without coordinates: {int(missing_count)}")
    print(f"[geocode] wrote: {args.output_json}")
    print(f"[geocode] wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
