#!/usr/bin/env python3
"""
Compute CityScore statistics for annual and seasonal subsets.
Seasons: spring (Mar-May), summer (Jun-Aug), autumn (Sep-Nov), winter (Dec-Feb).
Outputs a CSV with CityScore mean, population-weighted mean, gini, and population share bins.
"""

import argparse
import os
import pickle
import sys
import types
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd


# -----------------------------
# Compatibility helpers
# -----------------------------

def _ensure_optional_deps():
    """Stub optional deps so CityScore toolkit can import without full stack."""
    try:
        import osmnx  # noqa: F401
    except Exception:
        ox = types.ModuleType("osmnx")
        ox.settings = types.SimpleNamespace(
            cache_folder=None,
            use_cache=None,
            log_console=None,
        )
        sys.modules["osmnx"] = ox

    try:
        import tobler  # noqa: F401
    except Exception:
        tb = types.ModuleType("tobler")
        tb.util = types.SimpleNamespace(h3fy=None)
        sys.modules["tobler"] = tb


def _load_pickle(path: str):
    """Load pickle, with a numpy._core fallback for older numpy installs."""
    try:
        import numpy as np
        sys.modules.setdefault("numpy._core", np.core)
        try:
            sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        except Exception:
            pass
        try:
            sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)
        except Exception:
            pass
    except Exception:
        pass
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------
# Metrics
# -----------------------------

def pop_weighted_mean(group: pd.DataFrame) -> float:
    x = pd.to_numeric(group["cityscore"], errors="coerce").to_numpy()
    w = pd.to_numeric(group["population"], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not mask.any() or w[mask].sum() == 0:
        return float("nan")
    return float(np.average(x[mask], weights=w[mask]))


def gini_coefficient(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if np.any(x < 0):
        x = x - np.min(x)
    if np.all(x == 0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(gini)


def population_share_bins(cityscore_nil: pd.DataFrame) -> dict:
    df = cityscore_nil.copy()
    df["cityscore"] = pd.to_numeric(df["cityscore"], errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")

    valid = df["cityscore"].notna() & df["population"].notna() & (df["population"] > 0)
    df = df.loc[valid, ["cityscore", "population"]].copy()
    if df.empty:
        return {"0-20": np.nan, "20-40": np.nan, "40-60": np.nan, "60-80": np.nan, "80-100": np.nan}

    bins = [-np.inf, 20, 40, 60, 80, np.inf]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    df["bin"] = pd.cut(df["cityscore"], bins=bins, labels=labels, right=False, include_lowest=True)

    pop_total = df["population"].sum()
    shares = df.groupby("bin", dropna=False)["population"].sum() / pop_total * 100

    out = {label: float(shares.get(label, np.nan)) for label in labels}
    return out


# -----------------------------
# Helpers
# -----------------------------

def _infer_year_label(path: str) -> str:
    base = os.path.basename(path)
    for token in base.split("_"):
        if token.isdigit() and len(token) == 4:
            return token
    # fallback: try regex
    import re
    match = re.search(r"(20\d{2})", base)
    return match.group(1) if match else os.path.splitext(base)[0]


def _resolve_event_paths(paths: list[str]) -> list[str]:
    """
    Resolve event pickle paths while preferring non-`_final` names.
    Falls back to the `_final` variant when needed for compatibility.
    """
    resolved: list[str] = []
    for p in paths:
        if os.path.exists(p):
            resolved.append(p)
            continue

        alt = None
        if p.endswith(".pkl") and "_final.pkl" not in p:
            alt = p[:-4] + "_final.pkl"
        elif p.endswith("_final.pkl"):
            alt = p.replace("_final.pkl", ".pkl")

        if alt and os.path.exists(alt):
            print(f"[compat] Using {alt} (fallback for missing {p})")
            resolved.append(alt)
        else:
            resolved.append(p)
    return resolved


def _load_hex_meta(hex_to_nil_csv: str) -> pd.DataFrame:
    df = pd.read_csv(hex_to_nil_csv)
    hex_meta = df[["hex_id", "population", "NIL"]].copy()
    hex_meta["hex_id"] = hex_meta["hex_id"].astype(str)
    return hex_meta


def _ensure_events_gdf(events_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if not isinstance(events_gdf, gpd.GeoDataFrame):
        if "geometry" in events_gdf.columns:
            events_gdf = gpd.GeoDataFrame(events_gdf, geometry="geometry")
        else:
            raise TypeError("events_gdf is not a GeoDataFrame and has no geometry column.")
    if events_gdf.crs is None:
        events_gdf = events_gdf.set_crs("EPSG:4326", allow_override=True)
    return events_gdf


def _add_season_column(events_gdf: gpd.GeoDataFrame, date_col: str) -> gpd.GeoDataFrame:
    df = events_gdf.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise ValueError(f"All values in {date_col} are NaT after parsing.")
    month = df[date_col].dt.month

    def _season(m: int) -> str:
        if m in (3, 4, 5):
            return "Spring"
        if m in (6, 7, 8):
            return "Summer"
        if m in (9, 10, 11):
            return "Autumn"
        return "Winter"

    df["season"] = month.map(_season)
    return df


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute seasonal and annual CityScore stats for selected ISO files.")
    parser.add_argument(
        "--events",
        nargs="*",
        default=[
            "to_evaluate/gdf_2023.pkl",
            "to_evaluate/gdf_2024.pkl",
        ],
        help="Pickle GeoDataFrame paths to score.",
    )
    parser.add_argument(
        "--iso",
        nargs="*",
        default=[
            "iso/walk_15min_4.5kmh.pkl",
            "iso/bike_15min_15kmh.pkl",
        ],
        help="ISO pickle paths to use.",
    )
    parser.add_argument("--alpha", type=float, default=0.08, help="Alpha value.")
    parser.add_argument(
        "--hex-to-nil",
        default="hex_to_nil.csv",
        help="CSV mapping hex_id to NIL and population.",
    )
    parser.add_argument("--date-col", default="date", help="Date column in events GDF.")
    parser.add_argument("--out-dir", default="outputs", help="Output directory for results.")
    args = parser.parse_args()

    _ensure_optional_deps()
    import isochrones  # noqa: WPS433

    os.makedirs(args.out_dir, exist_ok=True)

    iso_paths = [p for p in args.iso if os.path.exists(p)]
    if not iso_paths:
        raise FileNotFoundError("No ISO files found. Check --iso paths.")
    args.events = _resolve_event_paths(args.events)

    hex_meta = _load_hex_meta(args.hex_to_nil)

    summary_rows = []

    for iso_path in iso_paths:
        iso_name = os.path.splitext(os.path.basename(iso_path))[0]
        static_iso = _load_pickle(iso_path)

        for events_path in args.events:
            year_label = _infer_year_label(events_path)
            events_gdf = _load_pickle(events_path)
            events_gdf = _ensure_events_gdf(events_gdf)
            events_gdf = _add_season_column(events_gdf, args.date_col)

            # Seasons + annual
            season_order = ["Spring", "Summer", "Autumn", "Winter", "Annual"]
            for season in season_order:
                if season == "Annual":
                    subset = events_gdf
                else:
                    subset = events_gdf[events_gdf["season"] == season]

                if subset.empty:
                    cityscore_gdf = gpd.GeoDataFrame(columns=["hex_id", "cityscore"])
                else:
                    cityscore_gdf, _ = isochrones.fast_score_day(
                        subset,
                        static_iso=static_iso,
                        alpha=args.alpha,
                    )

                cityscore_gdf = cityscore_gdf.copy()
                if "hex_id" in cityscore_gdf.columns:
                    cityscore_gdf["hex_id"] = cityscore_gdf["hex_id"].astype(str)

                cityscore_mean = pd.to_numeric(cityscore_gdf.get("cityscore"), errors="coerce").mean()
                cityscore_gini = gini_coefficient(cityscore_gdf.get("cityscore"))

                cityscore_nil = cityscore_gdf.merge(hex_meta, on="hex_id", how="left")
                pop_weighted_cityscore_mean = pop_weighted_mean(cityscore_nil)

                shares = population_share_bins(cityscore_nil)

                summary_rows.append(
                    {
                        "year": year_label,
                        "season": season,
                        "mode": "Walk" if "walk" in iso_name else "Bike",
                        "iso_name": iso_name,
                        "alpha": args.alpha,
                        "cityscore_mean": cityscore_mean,
                        "cityscore_pop_weighted_mean": pop_weighted_cityscore_mean,
                        "cityscore_gini": cityscore_gini,
                        "pop_share_0_20": shares["0-20"],
                        "pop_share_20_40": shares["20-40"],
                        "pop_share_40_60": shares["40-60"],
                        "pop_share_60_80": shares["60-80"],
                        "pop_share_80_100": shares["80-100"],
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    # Sort for readability
    season_cat = pd.Categorical(summary_df["season"], categories=["Spring", "Summer", "Autumn", "Winter", "Annual"], ordered=True)
    summary_df = summary_df.assign(season=season_cat)
    summary_df = summary_df.sort_values(["year", "season", "mode"]).reset_index(drop=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(args.out_dir, f"summary_cityscore_seasons_{ts}.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"Wrote {len(summary_df)} rows to: {summary_path}")


if __name__ == "__main__":
    main()
