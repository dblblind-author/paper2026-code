#!/usr/bin/env python3
"""
Evaluate CityScore for real event GeoDataFrames using precomputed isochrones.

Runs each ISO in a directory across multiple alphas and event pickles.
Saves per-run CityScore GeoDataFrame pickles and a summary CSV.
"""

import argparse
import os
import pickle
import re
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


# -----------------------------
# Helpers
# -----------------------------

def _infer_year_label(path: str) -> str:
    base = os.path.basename(path)
    match = re.search(r"(20\\d{2})", base)
    return match.group(1) if match else os.path.splitext(base)[0]


def _load_hex_meta(hex_to_nil_csv: str) -> pd.DataFrame:
    df = pd.read_csv(hex_to_nil_csv)
    hex_meta = df[["hex_id", "population", "NIL"]].copy()
    hex_meta["hex_id"] = hex_meta["hex_id"].astype(str)
    return hex_meta


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CityScore for real event GDFs across ISO files and alphas."
    )
    parser.add_argument(
        "--events",
        nargs="*",
        default=[
            "to_evaluate/gdf_2023_final.pkl",
            "to_evaluate/gdf_2024_final.pkl",
        ],
        help="Pickle GeoDataFrame paths to score.",
    )
    parser.add_argument("--iso-dir", default="iso", help="Directory with ISO .pkl files.")
    parser.add_argument("--out-dir", default="outputs", help="Output directory for results.")
    parser.add_argument(
        "--alphas",
        default="0.06,0.08,0.1",
        help="Comma-separated list of alpha values.",
    )
    parser.add_argument(
        "--hex-to-nil",
        default="hex_to_nil.csv",
        help="CSV mapping hex_id to NIL and population.",
    )
    args = parser.parse_args()

    _ensure_optional_deps()
    import isochrones  # noqa: WPS433

    os.makedirs(args.out_dir, exist_ok=True)

    iso_paths = [
        os.path.join(args.iso_dir, f)
        for f in os.listdir(args.iso_dir)
        if f.endswith(".pkl")
    ]
    iso_paths = sorted(iso_paths)
    if not iso_paths:
        raise FileNotFoundError(f"No .pkl ISO files found in {args.iso_dir}")

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    if not alphas:
        raise ValueError("No alpha values provided.")

    hex_meta = _load_hex_meta(args.hex_to_nil)

    summary_rows = []

    for iso_path in iso_paths:
        iso_name = os.path.splitext(os.path.basename(iso_path))[0]
        static_iso = _load_pickle(iso_path)

        for alpha in alphas:
            alpha_tag = f"{alpha:.2f}"
            for events_path in args.events:
                year_label = _infer_year_label(events_path)
                events_gdf = _load_pickle(events_path)
                if not isinstance(events_gdf, gpd.GeoDataFrame):
                    if "geometry" in events_gdf.columns:
                        events_gdf = gpd.GeoDataFrame(events_gdf, geometry="geometry")
                    else:
                        raise TypeError(f"{events_path} is not a GeoDataFrame and has no geometry column.")
                if events_gdf.crs is None:
                    events_gdf = events_gdf.set_crs("EPSG:4326", allow_override=True)

                cityscore_gdf, _ = isochrones.fast_score_day(
                    events_gdf,
                    static_iso=static_iso,
                    alpha=alpha,
                )

                out_name = f"cityscore_{iso_name}_alpha{alpha_tag}_{year_label}.pkl"
                out_path = os.path.join(args.out_dir, out_name)
                cityscore_gdf.to_pickle(out_path)

                cityscore_gdf = cityscore_gdf.copy()
                cityscore_gdf["hex_id"] = cityscore_gdf["hex_id"].astype(str)
                cityscore_mean = pd.to_numeric(cityscore_gdf["cityscore"], errors="coerce").mean()
                cityscore_gini = gini_coefficient(cityscore_gdf["cityscore"])
                cityscore_nil = cityscore_gdf.merge(hex_meta, on="hex_id", how="left")
                pop_weighted_cityscore_mean = pop_weighted_mean(cityscore_nil)

                summary_rows.append(
                    {
                        "iso_file": os.path.basename(iso_path),
                        "iso_name": iso_name,
                        "alpha": alpha,
                        "events_file": os.path.basename(events_path),
                        "year": year_label,
                        "n_events": int(len(events_gdf)),
                        "n_hex": int(len(cityscore_gdf)),
                        "cityscore_mean": cityscore_mean,
                        "cityscore_pop_weighted_mean": pop_weighted_cityscore_mean,
                        "cityscore_gini": cityscore_gini,
                        "output_pickle": out_name,
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(args.out_dir, f"summary_cityscore_runs_{ts}.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"Wrote {len(summary_df)} summary rows to: {summary_path}")


if __name__ == "__main__":
    main()
