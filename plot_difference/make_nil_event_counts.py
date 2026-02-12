#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely import wkt


class FixingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return FixingUnpickler(f).load()


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-run NIL event counts and average across runs."
    )
    parser.add_argument(
        "--reallocated",
        default="reallocated_2024_final_v3_par_50_runs.pkl",
        help="Path to reallocated pickle with 'allocations'.",
    )
    parser.add_argument(
        "--hex-to-nil",
        default="hex_to_nil.csv",
        help="Path to hex-to-NIL mapping CSV (must contain hex_id and NIL).",
    )
    parser.add_argument(
        "--non-relocatable",
        default="gdf_2024_final_v3_non_relocatable.pkl",
        help="Path to non-relocatable GeoDataFrame pickle.",
    )
    parser.add_argument(
        "--out-by-run",
        default="outputs/nil_events_by_run_with_nonreloc.csv",
        help="Output CSV path for per-run NIL event counts.",
    )
    parser.add_argument(
        "--out-avg",
        default="outputs/nil_events_avg_across_runs_with_nonreloc.csv",
        help="Output CSV path for average NIL event counts across runs.",
    )
    parser.add_argument(
        "--baseline",
        default="gdf_2024_final_v3.pkl",
        help="Path to baseline GeoDataFrame pickle (counts events per NIL).",
    )
    parser.add_argument(
        "--out-baseline",
        default="outputs/nil_events_baseline_gdf_2024_final_v3.csv",
        help="Output CSV path for baseline NIL event counts.",
    )
    args = parser.parse_args()

    realloc_path = Path(args.reallocated)
    hex_to_nil_path = Path(args.hex_to_nil)
    non_reloc_path = Path(args.non_relocatable)

    obj = load_pickle(realloc_path)
    alloc = obj.get("allocations")
    if alloc is None:
        raise SystemExit("allocations not found in reallocated pickle")

    hex_to_nil = pd.read_csv(hex_to_nil_path)
    if "hex_id" not in hex_to_nil.columns or "NIL" not in hex_to_nil.columns:
        raise SystemExit(
            f"hex_to_nil.csv must contain 'hex_id' and 'NIL'. Columns: {list(hex_to_nil.columns)}"
        )
    if "geometry" not in hex_to_nil.columns:
        raise SystemExit(
            f"hex_to_nil.csv must contain 'geometry' WKT. Columns: {list(hex_to_nil.columns)}"
        )
    hex_to_nil["geometry"] = hex_to_nil["geometry"].apply(wkt.loads)

    gdf_nr = load_pickle(non_reloc_path)
    if "title" not in gdf_nr.columns:
        raise SystemExit(
            f"non-relocatable gdf must contain 'title'. Columns: {list(gdf_nr.columns)}"
        )
    if "geometry" not in gdf_nr.columns:
        raise SystemExit(
            f"non-relocatable gdf must contain 'geometry'. Columns: {list(gdf_nr.columns)}"
        )

    hex_gdf = gpd.GeoDataFrame(hex_to_nil, geometry="geometry", crs=gdf_nr.crs)

    # Allocations -> NIL by assigned_cell_id -> hex_id
    alloc = alloc.copy()
    alloc["assigned_cell_id"] = pd.to_numeric(alloc["assigned_cell_id"], errors="coerce")
    counts_alloc = (
        alloc.merge(
            hex_to_nil[["hex_id", "NIL"]],
            how="left",
            left_on="assigned_cell_id",
            right_on="hex_id",
        )
        .groupby(["run_id", "NIL"], dropna=False)
        .size()
        .reset_index(name="alloc_count")
    )

    # Non-relocatable unique titles -> NIL via nearest hex centroid
    nr_unique = gdf_nr.drop_duplicates(subset=["title"]).copy()
    nr_gdf = gpd.GeoDataFrame(nr_unique, geometry="geometry", crs=gdf_nr.crs)
    nr_join = gpd.sjoin_nearest(
        nr_gdf, hex_gdf[["hex_id", "NIL", "geometry"]], how="left"
    )
    nr_counts = (
        nr_join.groupby("NIL", dropna=False)
        .size()
        .reset_index(name="nr_count")
    )

    unknown_label = "UNKNOWN"
    counts_alloc["NIL"] = counts_alloc["NIL"].fillna(unknown_label)
    nr_counts["NIL"] = nr_counts["NIL"].fillna(unknown_label)

    run_ids = sorted(alloc["run_id"].dropna().unique())
    all_nils = sorted(set(counts_alloc["NIL"]).union(set(nr_counts["NIL"])))

    full_index = pd.MultiIndex.from_product(
        [run_ids, all_nils], names=["run_id", "NIL"]
    ).to_frame(index=False)

    counts = (
        full_index.merge(counts_alloc, on=["run_id", "NIL"], how="left")
        .merge(nr_counts, on="NIL", how="left")
    )
    counts["alloc_count"] = counts["alloc_count"].fillna(0).astype(int)
    counts["nr_count"] = counts["nr_count"].fillna(0).astype(int)
    counts["event_count"] = counts["alloc_count"] + counts["nr_count"]

    per_run = counts[["run_id", "NIL", "event_count"]]
    avg_counts = (
        per_run.groupby("NIL", dropna=False)["event_count"]
        .mean()
        .reset_index(name="avg_event_count")
    )

    out_by_run = Path(args.out_by_run)
    out_avg = Path(args.out_avg)
    out_by_run.parent.mkdir(parents=True, exist_ok=True)
    out_avg.parent.mkdir(parents=True, exist_ok=True)

    per_run.to_csv(out_by_run, index=False)
    avg_counts.to_csv(out_avg, index=False)

    print(f"Wrote {out_by_run}")
    print(f"Wrote {out_avg}")
    print(f"Unique non-reloc titles: {nr_unique['title'].nunique()}")

    # Optional baseline counts from a single GeoDataFrame
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            raise SystemExit(f"Baseline file not found: {baseline_path}")

        gdf_base = load_pickle(baseline_path)
        if "geometry" not in gdf_base.columns:
            raise SystemExit(
                f"baseline gdf must contain 'geometry'. Columns: {list(gdf_base.columns)}"
            )

        if "title" in gdf_base.columns:
            base_unique = gdf_base.drop_duplicates(subset=["title"]).copy()
        else:
            base_unique = gdf_base.copy()

        base_gdf = gpd.GeoDataFrame(base_unique, geometry="geometry", crs=gdf_base.crs)

        hex_gdf_base = hex_gdf
        if (
            base_gdf.crs is not None
            and hex_gdf.crs is not None
            and base_gdf.crs != hex_gdf.crs
        ):
            hex_gdf_base = hex_gdf.to_crs(base_gdf.crs)

        base_join = gpd.sjoin_nearest(
            base_gdf, hex_gdf_base[["hex_id", "NIL", "geometry"]], how="left"
        )
        base_counts = (
            base_join.groupby("NIL", dropna=False)
            .size()
            .reset_index(name="event_count")
        )

        out_baseline = Path(args.out_baseline)
        out_baseline.parent.mkdir(parents=True, exist_ok=True)
        base_counts.to_csv(out_baseline, index=False)
        print(f"Wrote {out_baseline}")


if __name__ == "__main__":
    main()
