#!/usr/bin/env python3
"""
Evaluate CityScore and population-weighted CityScore for reallocated event files.

- Single-run allocations -> scalar metrics
- Multi-run allocations -> mean + std across runs
- Saves CSV summaries and NIL-level maps (cityscore mean + pop-weighted mean)
"""

import argparse
import os
import pickle
import sys
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt


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
        # Ensure submodule import works for pickles referencing numpy._core.multiarray
        try:
            sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        except Exception:
            pass
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            import numpy as np
            sys.modules.setdefault("numpy._core", np.core)
            with open(path, "rb") as f:
                return pickle.load(f)
        raise


# -----------------------------
# Parallel helpers
# -----------------------------

_WORKER_STATIC_ISO = None
_WORKER_HEX_GEOM_MAP = None
_WORKER_HEX_META = None
_WORKER_HEX_POINTS_CRS = None
_WORKER_EVENTS_CRS = None


def _init_worker(iso_path: str, hex_to_nil_csv: str, hex_crs: str, events_crs: str):
    global _WORKER_STATIC_ISO, _WORKER_HEX_GEOM_MAP, _WORKER_HEX_META, _WORKER_HEX_POINTS_CRS, _WORKER_EVENTS_CRS
    _ensure_optional_deps()
    _WORKER_STATIC_ISO = _load_pickle(iso_path)
    hex_to_nil_gdf = build_hex_points(hex_to_nil_csv, source_crs=hex_crs)
    _WORKER_HEX_GEOM_MAP, _WORKER_HEX_META = _prepare_hex_maps(hex_to_nil_gdf)
    _WORKER_HEX_POINTS_CRS = hex_crs
    _WORKER_EVENTS_CRS = events_crs


def _score_one_run(payload):
    run_id, alloc_df = payload
    result = score_allocations(
        alloc_df,
        static_iso=_WORKER_STATIC_ISO,
        hex_geom_map=_WORKER_HEX_GEOM_MAP,
        hex_meta=_WORKER_HEX_META,
        hex_points_crs=_WORKER_HEX_POINTS_CRS,
        events_target_crs=_WORKER_EVENTS_CRS,
    )
    return run_id, result["cityscore_mean"], result["pop_weighted_cityscore_mean"], result["nil_summary"]


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


def nil_summary(cityscore_nil: pd.DataFrame) -> pd.DataFrame:
    df = cityscore_nil.copy()
    df["cityscore"] = pd.to_numeric(df["cityscore"], errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce").clip(lower=0)

    grp = df.groupby("NIL", dropna=False)
    cityscore_mean = grp["cityscore"].mean()
    pop_total = grp["population"].sum()
    n_hex = grp.size()

    valid = df["cityscore"].notna() & df["population"].notna() & (df["population"] > 0)
    weighted = df.loc[valid, ["NIL", "cityscore", "population"]].copy()
    weighted["wsum"] = weighted["cityscore"] * weighted["population"]
    wgrp = weighted.groupby("NIL", dropna=False)
    wsum = wgrp["wsum"].sum()
    wpop = wgrp["population"].sum()
    cityscore_pw_mean = (wsum / wpop).reindex(cityscore_mean.index)

    return pd.DataFrame(
        {
            "NIL": cityscore_mean.index,
            "cityscore_mean": cityscore_mean.values,
            "cityscore_pw_mean": cityscore_pw_mean.values,
            "pop_total": pop_total.values,
            "n_hex": n_hex.values,
        }
    )


# -----------------------------
# Scoring pipeline
# -----------------------------

def build_hex_points(hex_to_nil_csv: str, source_crs: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(hex_to_nil_csv)
    geom = df["geometry"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs=source_crs)
    return gdf


def _prepare_hex_maps(hex_to_nil_gdf: gpd.GeoDataFrame):
    hp = hex_to_nil_gdf.copy()
    hp["hex_id_int"] = pd.to_numeric(hp["hex_id"], errors="coerce")
    valid = hp["hex_id_int"].notna()
    hex_geom_map = pd.Series(
        hp.loc[valid, "geometry"].values,
        index=hp.loc[valid, "hex_id_int"].astype("int64").values,
    )
    hex_meta = hp[["hex_id", "population", "NIL"]].copy()
    hex_meta["hex_id"] = hex_meta["hex_id"].astype(str)
    return hex_geom_map, hex_meta


def build_events_gdf(
    alloc_df: pd.DataFrame,
    hex_geom_map: pd.Series,
    source_crs: str,
    target_crs: str,
) -> gpd.GeoDataFrame:
    df = alloc_df.copy()
    df["assigned_cell_id"] = pd.to_numeric(df["assigned_cell_id"], errors="coerce").astype("Int64")
    df["geometry"] = df["assigned_cell_id"].map(hex_geom_map)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=source_crs)
    gdf = gdf.dropna(subset=["geometry"]).copy()
    if target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


def score_allocations(
    alloc_df: pd.DataFrame,
    static_iso,
    hex_geom_map: pd.Series,
    hex_meta: pd.DataFrame,
    hex_points_crs: str,
    events_target_crs: str,
):
    import isochrones

    events_gdf = build_events_gdf(
        alloc_df,
        hex_geom_map=hex_geom_map,
        source_crs=hex_points_crs,
        target_crs=events_target_crs,
    )

    scored, _ = isochrones.fast_score_day(events_gdf, static_iso=static_iso)

    # merge for NIL + population
    scored = scored.copy()
    scored["hex_id"] = scored["hex_id"].astype(str)
    cityscore_nil = scored.merge(hex_meta, on="hex_id", how="left")

    cityscore_mean = pd.to_numeric(scored["cityscore"], errors="coerce").mean()
    pop_weighted_cityscore_mean = pop_weighted_mean(cityscore_nil)

    return {
        "cityscore_mean": cityscore_mean,
        "pop_weighted_cityscore_mean": pop_weighted_cityscore_mean,
        "nil_summary": nil_summary(cityscore_nil),
    }


def plot_nil_map(nils_gdf: gpd.GeoDataFrame, summary: pd.DataFrame, value_col: str, title: str, out_path: str):
    merged = nils_gdf.merge(summary[["NIL", value_col]], on="NIL", how="left")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    merged.plot(
        column=value_col,
        cmap="viridis",
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgrey", "label": "missing"},
    )
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iso", default="iso/walk_15.pkl", help="Path to static ISO pickle")
    parser.add_argument("--hex-to-nil", default="hex_to_nil.csv", help="CSV with hex_id, geometry, population, NIL")
    parser.add_argument("--hex-crs", default="EPSG:6707", help="CRS for hex_to_nil geometry")
    parser.add_argument("--events-crs", default="EPSG:4326", help="CRS required by CityScore (typically EPSG:4326)")
    parser.add_argument("--nils", default="NIL/NIL_WM.shp", help="NIL boundaries shapefile for plotting")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--limit-runs", type=int, default=None, help="Limit number of runs per method (debug)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for multi-run methods")
    parser.add_argument("--files", nargs="*", default=[
        "greedy_realloc_final.pkl",
        "random_realloc_final.pkl",
        "our_realloc_final.pkl",
    ], help="Reallocation pickle files to score")

    args = parser.parse_args()

    t0 = datetime.now()
    print(f"[{t0.strftime('%H:%M:%S')}] Starting evaluation")
    _ensure_optional_deps()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading static ISO: {args.iso}")
    # Load static ISO
    static_iso = _load_pickle(args.iso)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading hex->NIL: {args.hex_to_nil}")
    # Load hex -> NIL centroids/metadata
    hex_to_nil_gdf = build_hex_points(args.hex_to_nil, source_crs=args.hex_crs)
    hex_geom_map, hex_meta = _prepare_hex_maps(hex_to_nil_gdf)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading NIL shapes: {args.nils}")
    # NIL shapes for plotting
    nils_gdf = gpd.read_file(args.nils)

    out_dir = args.out_dir or os.path.join("outputs", "cityscore_eval")
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    per_run_rows = []

    for path in args.files:
        label = os.path.splitext(os.path.basename(path))[0]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading allocations: {path}")
        obj = _load_pickle(path)
        alloc = obj["allocations"]

        is_multirun = "run_id" in alloc.columns and alloc["run_id"].nunique() > 1

        run_results = []
        nil_summaries = []

        if is_multirun:
            runs_iter = list(alloc["run_id"].unique())
            if args.limit_runs is not None:
                runs_iter = runs_iter[: args.limit_runs]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {label}: {len(runs_iter)} runs")
            if args.workers and args.workers > 1:
                try:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {label}: using {args.workers} workers")
                    with ProcessPoolExecutor(
                        max_workers=args.workers,
                        initializer=_init_worker,
                        initargs=(args.iso, args.hex_to_nil, args.hex_crs, args.events_crs),
                    ) as ex:
                        futures = []
                        for run_id in runs_iter:
                            sub = alloc[alloc["run_id"] == run_id]
                            futures.append(ex.submit(_score_one_run, (run_id, sub)))

                        for fut in as_completed(futures):
                            run_id, cityscore_mean, pop_weighted_cityscore_mean, nil_summary_df = fut.result()
                            run_results.append(
                                {
                                    "method": label,
                                    "run_id": run_id,
                                    "cityscore_mean": cityscore_mean,
                                    "pop_weighted_cityscore_mean": pop_weighted_cityscore_mean,
                                }
                            )
                            tmp_nil = nil_summary_df.copy()
                            tmp_nil["run_id"] = run_id
                            nil_summaries.append(tmp_nil)
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {label}: parallel failed ({e}); falling back to sequential")
                    args.workers = 1
            if not args.workers or args.workers <= 1:
                for run_id in runs_iter:
                    sub = alloc[alloc["run_id"] == run_id]
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {label}: run {run_id}")
                    result = score_allocations(
                        sub,
                        static_iso=static_iso,
                        hex_geom_map=hex_geom_map,
                        hex_meta=hex_meta,
                        hex_points_crs=args.hex_crs,
                        events_target_crs=args.events_crs,
                    )
                    run_results.append(
                        {
                            "method": label,
                            "run_id": run_id,
                            "cityscore_mean": result["cityscore_mean"],
                            "pop_weighted_cityscore_mean": result["pop_weighted_cityscore_mean"],
                        }
                    )
                    tmp_nil = result["nil_summary"].copy()
                    tmp_nil["run_id"] = run_id
                    nil_summaries.append(tmp_nil)

            run_df = pd.DataFrame(run_results)
            per_run_rows.append(run_df)

            summary_rows.append(
                {
                    "method": label,
                    "n_runs": int(run_df["run_id"].nunique()),
                    "cityscore_mean": run_df["cityscore_mean"].mean(),
                    "cityscore_sd": run_df["cityscore_mean"].std(ddof=1),
                    "pop_weighted_cityscore_mean": run_df["pop_weighted_cityscore_mean"].mean(),
                    "pop_weighted_cityscore_sd": run_df["pop_weighted_cityscore_mean"].std(ddof=1),
                }
            )

            nil_df = pd.concat(nil_summaries, ignore_index=True)
            nil_agg = (
                nil_df
                .groupby("NIL", dropna=False)
                .agg(
                    cityscore_mean=("cityscore_mean", "mean"),
                    cityscore_mean_sd=("cityscore_mean", "std"),
                    cityscore_pw_mean=("cityscore_pw_mean", "mean"),
                    cityscore_pw_mean_sd=("cityscore_pw_mean", "std"),
                    pop_total=("pop_total", "mean"),
                    n_hex=("n_hex", "mean"),
                )
                .reset_index()
            )

            nil_df.to_csv(os.path.join(out_dir, f"nil_summary_{label}_by_run.csv"), index=False)
            nil_agg.to_csv(os.path.join(out_dir, f"nil_summary_{label}.csv"), index=False)

            plot_nil_map(
                nils_gdf,
                nil_agg,
                value_col="cityscore_mean",
                title=f"{label}: CityScore (mean across runs)",
                out_path=os.path.join(out_dir, f"map_{label}_cityscore_mean.png"),
            )
            plot_nil_map(
                nils_gdf,
                nil_agg,
                value_col="cityscore_pw_mean",
                title=f"{label}: Pop-weighted CityScore (mean across runs)",
                out_path=os.path.join(out_dir, f"map_{label}_pop_weighted_cityscore_mean.png"),
            )

        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {label}: single run")
            result = score_allocations(
                alloc,
                static_iso=static_iso,
                hex_geom_map=hex_geom_map,
                hex_meta=hex_meta,
                hex_points_crs=args.hex_crs,
                events_target_crs=args.events_crs,
            )
            summary_rows.append(
                {
                    "method": label,
                    "n_runs": 1,
                    "cityscore_mean": result["cityscore_mean"],
                    "cityscore_sd": float("nan"),
                    "pop_weighted_cityscore_mean": result["pop_weighted_cityscore_mean"],
                    "pop_weighted_cityscore_sd": float("nan"),
                }
            )

            result["nil_summary"].to_csv(os.path.join(out_dir, f"nil_summary_{label}.csv"), index=False)

            plot_nil_map(
                nils_gdf,
                result["nil_summary"],
                value_col="cityscore_mean",
                title=f"{label}: CityScore",
                out_path=os.path.join(out_dir, f"map_{label}_cityscore_mean.png"),
            )
            plot_nil_map(
                nils_gdf,
                result["nil_summary"],
                value_col="cityscore_pw_mean",
                title=f"{label}: Pop-weighted CityScore",
                out_path=os.path.join(out_dir, f"map_{label}_pop_weighted_cityscore_mean.png"),
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    if per_run_rows:
        by_run_df = pd.concat(per_run_rows, ignore_index=True)
        by_run_df.to_csv(os.path.join(out_dir, "by_run.csv"), index=False)

    # print summary
    print("\nCityScore summary")
    print(summary_df.to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Done. Total time: {datetime.now() - t0}")


if __name__ == "__main__":
    main()
