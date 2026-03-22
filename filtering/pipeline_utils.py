#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_JSON = ROOT / "all_events_llm (3).json"
DEFAULT_NIL_SHP = ROOT / "NIL" / "NIL_WM.shp"
DEFAULT_GEOCODE_CACHE = ROOT / "location_geocode_cache.json"
DEFAULT_COMPARE_2023 = ROOT / "gdf_2023_final_v3.pkl"
DEFAULT_COMPARE_2024 = ROOT / "gdf_2024_final_v3.pkl"

KEY_COLS = [
    "title",
    "location",
    "price",
    "category",
    "date",
    "weekday",
    "opening_hour",
    "closing_hour",
    "url",
]

OUTPUT_COLS = KEY_COLS + ["coordinates", "geometry"]


def ensure_pickle_compat() -> None:
    """Compatibility shim for pickle files written with numpy._core paths."""
    core = getattr(np, "_core", None)
    if core is None:
        core = np.core
        setattr(np, "_core", core)
    sys.modules.setdefault("numpy._core", core)
    for name in ("multiarray", "_multiarray_umath"):
        module = getattr(core, name, None)
        if module is not None:
            sys.modules.setdefault(f"numpy._core.{name}", module)


def load_pickle(path: Path):
    ensure_pickle_compat()
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def normalize_text(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return value


def normalize_hour(value: object) -> object:
    value = normalize_text(value)
    if value in (None, "None", "none", "nan", "NaN"):
        return None
    return value


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("title", "location", "price", "weekday", "url"):
        if col in out.columns:
            out[col] = out[col].map(normalize_text)
    if "category" in out.columns:
        out["category"] = out["category"].map(
            lambda v: normalize_text(str(v).lower())
            if v is not None and not (isinstance(v, float) and pd.isna(v))
            else None
        )
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    if "opening_hour" in out.columns:
        out["opening_hour"] = out["opening_hour"].map(normalize_hour)
    if "closing_hour" in out.columns:
        out["closing_hour"] = out["closing_hour"].map(normalize_hour)
    return out


def load_events(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of events in {path}")
    return data


def flatten_events(events: Iterable[dict], years: Sequence[int] = (2023, 2024)) -> pd.DataFrame:
    year_set = set(years)
    expanded: List[dict] = []

    for event in events:
        for day in event.get("dates", []):
            year = int(str(day["date"])[:4])
            if year not in year_set:
                continue

            expanded.append(
                {
                    "title": event["title"],
                    "location": event["location"],
                    "price": event["price"],
                    "category": str(event["category"]).lower(),
                    "date": day["date"],
                    "weekday": day["weekday"],
                    "opening_hour": day.get("opening_hour"),
                    "closing_hour": day.get("closing_hour"),
                    "description": event.get("description"),
                    "url": event["url"],
                    "year": year,
                }
            )

    return normalize_frame(pd.DataFrame(expanded))


def clean_expanded_events(expanded_df: pd.DataFrame) -> pd.DataFrame:
    light = expanded_df.drop(columns=["description"], errors="ignore").copy()
    light = light.dropna(subset=["date", "opening_hour", "location"]).copy()
    return light


def normalize_location(value: object) -> Optional[str]:
    value = normalize_text(value)
    if value is None or not isinstance(value, str):
        return None
    return " ".join(value.lower().split())


def build_unique_location_dict(locations: Iterable[object]) -> Dict[str, str]:
    unique: Dict[str, str] = {}
    for location in locations:
        normalized = normalize_location(location)
        if normalized is None:
            continue
        if normalized not in unique:
            unique[normalized] = str(location).strip()
    return unique


def load_geocode_cache(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "locations" in data and isinstance(data["locations"], dict):
        return data["locations"]

    # Backward-compatible fallback for a raw mapping.
    if isinstance(data, dict):
        out: Dict[str, dict] = {}
        for key, value in data.items():
            coords: Optional[List[float]]
            if isinstance(value, dict):
                out[key] = value
                continue
            if value is None:
                coords = None
            else:
                coords = list(value)
            out[key] = {"original": None, "query": None, "coordinates": coords}
        return out

    raise ValueError(f"Unsupported geocode cache format in {path}")


def save_geocode_cache(path: Path, entries: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "locations": {key: entries[key] for key in sorted(entries)},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _coords_from_cache_entry(entry: Optional[dict]) -> Optional[Tuple[float, float]]:
    if not entry:
        return None
    coords = entry.get("coordinates")
    if coords is None:
        return None
    if len(coords) != 2:
        return None
    return (float(coords[0]), float(coords[1]))


def geocode_google(query: str, api_key: str, timeout_seconds: int = 30) -> Optional[Tuple[float, float]]:
    params = urllib.parse.urlencode({"address": query, "key": api_key})
    url = f"https://maps.googleapis.com/maps/api/geocode/json?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "paper2026-filtering-pipeline"})

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    status = payload.get("status")
    if status not in {"OK", "ZERO_RESULTS"}:
        return None
    if status == "ZERO_RESULTS":
        return None

    results = payload.get("results") or []
    if not results:
        return None

    location = results[0].get("geometry", {}).get("location", {})
    lat = location.get("lat")
    lng = location.get("lng")
    if lat is None or lng is None:
        return None
    return (float(lat), float(lng))


def geocode_unique_locations(
    unique_norm_to_original: Dict[str, str],
    api_key: Optional[str],
    *,
    cache_path: Path = DEFAULT_GEOCODE_CACHE,
    city_suffix: str = "Milano",
    pause_seconds: float = 0.25,
    max_retries: int = 3,
    refresh_missing_cache: bool = False,
    log: Callable[[str], None] = print,
) -> Dict[str, Optional[Tuple[float, float]]]:
    entries = load_geocode_cache(cache_path)

    missing = []
    for norm_key, original in unique_norm_to_original.items():
        entry = entries.get(norm_key)
        if entry is None:
            missing.append((norm_key, original))
            continue
        if entry.get("coordinates") is None and refresh_missing_cache:
            missing.append((norm_key, original))

    if missing and not api_key:
        raise ValueError(
            "Missing geocoding API key. Set the GOOGLE_MAPS_API_KEY environment variable "
            "or populate the geocode cache before rerunning."
        )

    total = len(unique_norm_to_original)
    for index, (norm_key, original) in enumerate(unique_norm_to_original.items(), start=1):
        entry = entries.get(norm_key)
        if entry is not None and (entry.get("coordinates") is not None or not refresh_missing_cache):
            continue

        query = original
        if city_suffix and city_suffix.lower() not in original.lower():
            query = f"{original}, {city_suffix}"
        log(f"[{index}/{total}] Geocoding: {query}")

        coords: Optional[Tuple[float, float]] = None
        for attempt in range(1, max_retries + 1):
            coords = geocode_google(query, api_key=api_key) if api_key else None
            if coords is not None:
                break
            time.sleep(pause_seconds * attempt)

        entries[norm_key] = {
            "original": original,
            "query": query,
            "coordinates": list(coords) if coords is not None else None,
        }
        save_geocode_cache(cache_path, entries)
        time.sleep(pause_seconds)

    return {
        norm_key: _coords_from_cache_entry(entries.get(norm_key))
        for norm_key in unique_norm_to_original
    }


def attach_coordinates_from_locations(
    cleaned_df: pd.DataFrame,
    norm_to_coords: Dict[str, Optional[Tuple[float, float]]],
) -> pd.DataFrame:
    merged = cleaned_df.copy()
    merged["coordinates"] = merged["location"].map(
        lambda value: norm_to_coords.get(normalize_location(value))
    )
    return merged


def build_geodataframe(
    df: pd.DataFrame,
    *,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    out = df.copy()
    out["geometry"] = out["coordinates"].map(
        lambda value: Point(value[1], value[0]) if value is not None else None
    )
    gdf = gpd.GeoDataFrame(out, geometry="geometry", crs=crs)
    return gdf


def filter_within_milan(
    gdf: gpd.GeoDataFrame,
    nils_gdf: gpd.GeoDataFrame,
    *,
    output_crs: Optional[str] = None,
) -> gpd.GeoDataFrame:
    working = gdf.dropna(subset=["geometry"]).copy()
    if output_crs:
        working = working.to_crs(output_crs)

    nils_local = nils_gdf.to_crs(working.crs) if working.crs is not None else nils_gdf.copy()
    nils_local = nils_local[["geometry"]].copy()

    joined = working.sjoin(nils_local, how="inner", predicate="intersects")
    joined = joined.drop(columns=["index_right"], errors="ignore")
    joined = joined.drop_duplicates(subset=KEY_COLS)
    return joined.loc[:, OUTPUT_COLS].copy()


def build_delta_table(new_gdf: gpd.GeoDataFrame, final_gdf: gpd.GeoDataFrame, year: int) -> pd.DataFrame:
    new_counts = (
        normalize_frame(pd.DataFrame(new_gdf[["url"]].copy()))
        .groupby("url", dropna=False)
        .size()
        .reset_index(name="new_rows")
    )
    final_counts = (
        normalize_frame(pd.DataFrame(final_gdf[["url"]].copy()))
        .groupby("url", dropna=False)
        .size()
        .reset_index(name="final_rows")
    )

    diff = new_counts.merge(final_counts, on="url", how="outer")
    diff["new_rows"] = diff["new_rows"].fillna(0).astype(int)
    diff["final_rows"] = diff["final_rows"].fillna(0).astype(int)
    diff = diff[diff["new_rows"] != diff["final_rows"]].copy()

    def classify(row: pd.Series) -> str:
        if row["new_rows"] == 0:
            return "only_in_final"
        if row["final_rows"] == 0:
            return "only_in_new"
        return "row_count_mismatch"

    diff["year"] = year
    diff["diff_type"] = diff.apply(classify, axis=1)
    return diff[["year", "url", "new_rows", "final_rows", "diff_type"]].sort_values(
        ["diff_type", "url"]
    )
