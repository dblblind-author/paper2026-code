#!/usr/bin/env python3
"""
Build yearly filtered event GeoDataFrame pickles directly from the source JSON.

Pipeline:
1. expand date-level rows from the event JSON
2. geocode unique locations with a cached Google Geocoding lookup
3. create geometries from the returned coordinates
4. spatially filter rows against the Milan NIL boundary
5. optionally compare rebuilt yearly pickles against provided final pickles

Notes:
- no coordinates are borrowed from existing yearly pickles
- no API key is stored in this repository; set GOOGLE_MAPS_API_KEY at runtime
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd

from pipeline_utils import (
    DEFAULT_COMPARE_2023,
    DEFAULT_COMPARE_2024,
    DEFAULT_GEOCODE_CACHE,
    DEFAULT_INPUT_JSON,
    DEFAULT_NIL_SHP,
    ROOT,
    attach_coordinates_from_locations,
    build_delta_table,
    build_geodataframe,
    build_unique_location_dict,
    clean_expanded_events,
    filter_within_milan,
    flatten_events,
    geocode_unique_locations,
    load_events,
    load_pickle,
    normalize_frame,
    save_pickle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build yearly filtered event pickles from the original JSON."
    )
    parser.add_argument("--input-json", default=str(DEFAULT_INPUT_JSON))
    parser.add_argument("--nils", default=str(DEFAULT_NIL_SHP))
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
    parser.add_argument("--final-2023", default=str(DEFAULT_COMPARE_2023))
    parser.add_argument("--final-2024", default=str(DEFAULT_COMPARE_2024))
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Build the yearly pickles without comparing them to provided final pickles.",
    )
    parser.add_argument("--out-2023", default=str(ROOT / "gdf_2023_rebuilt_from_json.pkl"))
    parser.add_argument("--out-2024", default=str(ROOT / "gdf_2024_rebuilt_from_json.pkl"))
    parser.add_argument("--delta-2023", default=str(ROOT / "delta_table_2023.csv"))
    parser.add_argument("--delta-2024", default=str(ROOT / "delta_table_2024.csv"))
    return parser.parse_args()


def maybe_load_final(path: Path, skip_compare: bool):
    if skip_compare or not path.exists():
        return None
    return load_pickle(path)


def main() -> None:
    args = parse_args()

    events = load_events(Path(args.input_json))
    nils_gdf = gpd.read_file(args.nils)

    expanded_df = flatten_events(events)
    cleaned_df = clean_expanded_events(expanded_df)

    print(f"[flatten] expanded rows: {len(expanded_df)}")
    print("[flatten] cleaned rows by year:", cleaned_df.groupby("year").size().to_dict())

    unique_locations = build_unique_location_dict(cleaned_df["location"].tolist())
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

    geocoded_df = attach_coordinates_from_locations(cleaned_df, norm_to_coords)
    geocoded_gdf = build_geodataframe(geocoded_df, crs="EPSG:4326")

    geocoded_success = int(geocoded_df["coordinates"].notna().sum())
    geocoded_missing = int(geocoded_df["coordinates"].isna().sum())
    print(f"[geocode] unique locations: {len(unique_locations)}")
    print(f"[geocode] rows with coordinates: {geocoded_success}")
    print(f"[geocode] rows without coordinates: {geocoded_missing}")
    print(f"[geocode] cache: {args.cache}")

    final_paths: Dict[int, Path] = {
        2023: Path(args.final_2023),
        2024: Path(args.final_2024),
    }
    out_paths: Dict[int, Path] = {
        2023: Path(args.out_2023),
        2024: Path(args.out_2024),
    }
    delta_paths: Dict[int, Path] = {
        2023: Path(args.delta_2023),
        2024: Path(args.delta_2024),
    }

    for year in (2023, 2024):
        year_gdf = geocoded_gdf[geocoded_gdf["year"] == year].copy()
        final_gdf = maybe_load_final(final_paths[year], args.skip_compare)
        output_crs: Optional[str] = final_gdf.crs if final_gdf is not None else "EPSG:4326"

        rebuilt_gdf = filter_within_milan(year_gdf, nils_gdf, output_crs=output_crs)
        save_pickle(rebuilt_gdf, out_paths[year])

        year_with_coords = int(year_gdf["coordinates"].notna().sum())
        print(f"[{year}] input cleaned rows: {len(year_gdf)}")
        print(f"[{year}] rows with coordinates before Milan filter: {year_with_coords}")
        print(f"[{year}] rebuilt pickle rows after Milan boundary filter: {len(rebuilt_gdf)}")
        print(f"[{year}] wrote: {out_paths[year]}")

        if final_gdf is None:
            continue

        final_norm = normalize_frame(final_gdf.copy())
        delta_df = build_delta_table(rebuilt_gdf, final_norm, year)
        delta_df.to_csv(delta_paths[year], index=False, encoding="utf-8")

        print(f"[{year}] provided final pickle rows: {len(final_gdf)}")
        print(f"[{year}] delta rows: {len(delta_df)}")
        print(f"[{year}] wrote: {delta_paths[year]}")


if __name__ == "__main__":
    main()
