#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from shapely import wkt
from typing import Optional, List, Tuple


class FixingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return FixingUnpickler(f).load()


def parse_categories(value: Optional[str], fallback: List[str]) -> List[str]:
    if value:
        return [v.strip() for v in value.split(",") if v.strip()]
    return fallback


def load_hex_to_nil(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    if "hex_id" not in df.columns or "NIL" not in df.columns:
        raise SystemExit(
            f"hex_to_nil must contain 'hex_id' and 'NIL'. Columns: {list(df.columns)}"
        )
    if "geometry" not in df.columns:
        raise SystemExit(
            f"hex_to_nil must contain 'geometry' (WKT). Columns: {list(df.columns)}"
        )
    df["geometry"] = df["geometry"].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry="geometry")


def assign_nil_by_nearest(points_gdf: gpd.GeoDataFrame, hex_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if points_gdf.crs is None and hex_gdf.crs is not None:
        points_gdf = points_gdf.set_crs(hex_gdf.crs, allow_override=True)
    elif points_gdf.crs is not None and hex_gdf.crs is None:
        hex_gdf = hex_gdf.set_crs(points_gdf.crs, allow_override=True)
    elif points_gdf.crs != hex_gdf.crs:
        hex_gdf = hex_gdf.to_crs(points_gdf.crs)

    return gpd.sjoin_nearest(points_gdf, hex_gdf[["hex_id", "NIL", "geometry"]], how="left")


def baseline_counts_by_category_nil(baseline_gdf: gpd.GeoDataFrame, hex_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    gdf = baseline_gdf.copy()
    if "title" in gdf.columns:
        gdf = gdf.drop_duplicates(subset=["title"]).copy()

    if "NIL" not in gdf.columns:
        gdf = assign_nil_by_nearest(gdf, hex_gdf)

    if "category" not in gdf.columns:
        raise SystemExit("Baseline gdf must contain 'category' column.")

    counts = (
        gdf.groupby(["NIL", "category"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    return counts


def reallocated_avg_counts_by_category_nil(
    realloc_pkl: Path,
    non_reloc_pkl: Path,
    hex_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    obj = load_pickle(realloc_pkl)
    alloc = obj.get("allocations")
    if alloc is None:
        raise SystemExit("allocations not found in reallocated pickle")
    if "category" not in alloc.columns:
        raise SystemExit("allocations must contain 'category' column")

    # allocations -> NIL by assigned_cell_id -> hex_id
    alloc = alloc.copy()
    alloc["assigned_cell_id"] = pd.to_numeric(alloc["assigned_cell_id"], errors="coerce")
    hex_map = hex_gdf[["hex_id", "NIL"]].copy()
    hex_map["hex_id"] = pd.to_numeric(hex_map["hex_id"], errors="coerce")

    alloc = alloc.merge(hex_map, how="left", left_on="assigned_cell_id", right_on="hex_id")

    counts_alloc = (
        alloc.groupby(["run_id", "NIL", "category"], dropna=False)
        .size()
        .rename("alloc_count")
        .reset_index()
    )

    # non-relocatable unique titles -> NIL via nearest hex centroid
    gdf_nr = load_pickle(non_reloc_pkl)
    if "title" in gdf_nr.columns:
        gdf_nr = gdf_nr.drop_duplicates(subset=["title"]).copy()
    if "category" not in gdf_nr.columns:
        raise SystemExit("non-relocatable gdf must contain 'category' column")

    nr_join = assign_nil_by_nearest(gdf_nr, hex_gdf)
    counts_nr = (
        nr_join.groupby(["NIL", "category"], dropna=False)
        .size()
        .rename("nr_count")
        .reset_index()
    )

    # Build full grid run_id x NIL x category
    run_ids = sorted(counts_alloc["run_id"].dropna().unique())
    nils = sorted(set(counts_alloc["NIL"]).union(set(counts_nr["NIL"])))
    cats = sorted(set(counts_alloc["category"]).union(set(counts_nr["category"])))

    full_index = pd.MultiIndex.from_product(
        [run_ids, nils, cats], names=["run_id", "NIL", "category"]
    ).to_frame(index=False)

    merged = (
        full_index
        .merge(counts_alloc, on=["run_id", "NIL", "category"], how="left")
        .merge(counts_nr, on=["NIL", "category"], how="left")
    )
    merged["alloc_count"] = merged["alloc_count"].fillna(0).astype(int)
    merged["nr_count"] = merged["nr_count"].fillna(0).astype(int)
    merged["event_count"] = merged["alloc_count"] + merged["nr_count"]

    avg_counts = (
        merged.groupby(["NIL", "category"], dropna=False)["event_count"]
        .mean()
        .rename("n")
        .reset_index()
    )
    return avg_counts


def adjust_to_nil_totals(avg_counts: pd.DataFrame, nil_avg_path: Path) -> pd.DataFrame:
    if not nil_avg_path.exists():
        return avg_counts
    nil_avg = pd.read_csv(nil_avg_path)
    if "NIL" not in nil_avg.columns or "avg_event_count" not in nil_avg.columns:
        raise SystemExit(
            f"nil_avg must contain 'NIL' and 'avg_event_count'. Columns: {list(nil_avg.columns)}"
        )

    totals = (
        avg_counts.groupby("NIL", dropna=False)["n"]
        .sum()
        .rename("sum_n")
        .reset_index()
    )
    merged = avg_counts.merge(totals, on="NIL", how="left").merge(nil_avg, on="NIL", how="left")

    def scale_row(row):
        if pd.isna(row["avg_event_count"]) or row["sum_n"] == 0:
            return row["n"]
        return row["n"] * (row["avg_event_count"] / row["sum_n"])

    merged["n"] = merged.apply(scale_row, axis=1)
    return merged[["NIL", "category", "n"]]


def plot_nil_category_heatmap_posneg_scaled_counts(
    df_counts: pd.DataFrame,
    year: int,
    categories: List[str],
    scenario_order: Tuple[str, str] = ("real", "reallocated"),
    top_k: Optional[int] = None,
    cmap_name: str = "RdBu_r",
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
):
    def short_nil(s: str) -> str:
        if not isinstance(s, str):
            return s
        cut = len(s)
        for sep in [" - ", "-", " ("]:
            j = s.find(sep)
            if j != -1:
                cut = min(cut, j)
        return s[:cut].strip()

    s0, s1 = scenario_order
    d = df_counts[df_counts["year"] == year].copy()
    if d.empty:
        raise ValueError(f"No data for year {year}")

    if "n" not in d.columns:
        raise ValueError("df_counts must include column 'n' with counts.")

    g = d.set_index(["scenario", "NIL", "category"])["n"]

    nils = sorted(d["NIL"].unique())
    idx = pd.MultiIndex.from_product(
        [list(scenario_order), nils, categories],
        names=["scenario", "NIL", "category"],
    )
    g = g.reindex(idx, fill_value=0).reset_index(name="n")

    piv = g.pivot_table(
        index=["NIL", "category"],
        columns="scenario",
        values="n",
        fill_value=0,
    )

    real = (
        piv[s0]
        .unstack("category")
        .reindex(columns=categories)
        .astype(float)
    )
    realloc = (
        piv[s1]
        .unstack("category")
        .reindex(columns=categories)
        .astype(float)
    )

    change = realloc - real

    abs_change = change.abs().copy()
    score = pd.Series(0.0, index=abs_change.index)
    for col in categories:
        col_vals = abs_change[col]
        col_max = col_vals.max()
        if col_max > 0:
            score += col_vals / col_max

    if top_k is not None and top_k < len(score):
        order = score.sort_values(ascending=False).head(top_k).index.tolist()
        change = change.loc[order]
    else:
        order = score.sort_values(ascending=False).index.tolist()
        change = change.loc[order]

    change_norm = change.copy()
    for col in categories:
        v = change[col].values
        pos_max = v[v > 0].max() if np.any(v > 0) else 0.0
        neg_min = v[v < 0].min() if np.any(v < 0) else 0.0

        scaled = np.zeros_like(v, dtype=float)
        if pos_max > 0:
            scaled[v > 0] = v[v > 0] / pos_max
        if neg_min < 0:
            scaled[v < 0] = v[v < 0] / abs(neg_min)
        change_norm[col] = scaled

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        change_norm.loc[order, categories].values,
        aspect="auto",
        cmap=cmap_name,
        norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1),
    )
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short_nil(n) for n in order])
    ax.invert_yaxis()

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Category-normalized change")

    if title is None:
        title = (
            f"Normalized Change in Event Distribution after Reallocation ({year})"
        )
    ax.set_title(title, pad=10)
    plt.tight_layout()
    return fig, ax


def plot_nil_category_choropleths_posneg_scaled(
    nils_gdf: gpd.GeoDataFrame,
    change: pd.Series,
    categories: List[str],
    nil_col: str = "NIL",
    cmap_name: str = "RdBu_r",
    ncols: int = 3,
    figsize_per_plot: Tuple[int, int] = (5, 5),
    title: Optional[str] = None,
):
    if not isinstance(change.index, pd.MultiIndex):
        raise ValueError("`change` must be a MultiIndex Series with (category, NIL).")

    if change.index.names != ["category", "NIL"]:
        if len(change.index.levels) == 2:
            change.index = change.index.set_names(["category", "NIL"])
        else:
            raise ValueError("Expected MultiIndex with exactly two levels: (category, NIL).")

    available_cats = change.index.get_level_values("category").unique()
    missing = [c for c in categories if c not in available_cats]
    if missing:
        raise ValueError(f"Categories not found in `change` index: {missing}")

    n_cat = len(categories)
    nrows = int(np.ceil(n_cat / ncols))
    fig_w = figsize_per_plot[0] * ncols
    fig_h = figsize_per_plot[1] * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    base = nils_gdf.copy()
    norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)

    for i, cat in enumerate(categories):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        cat_series = change.xs(cat, level="category")
        v = cat_series.reindex(base[nil_col]).fillna(0.0).values

        pos_max = v[v > 0].max() if np.any(v > 0) else 0.0
        neg_min = v[v < 0].min() if np.any(v < 0) else 0.0

        scaled = np.zeros_like(v, dtype=float)
        if pos_max > 0:
            scaled[v > 0] = v[v > 0] / pos_max
        if neg_min < 0:
            scaled[v < 0] = v[v < 0] / abs(neg_min)

        col_name = f"{cat}_scaled"
        base[col_name] = scaled

        base.plot(
            column=col_name,
            cmap=cmap_name,
            norm=norm,
            ax=ax,
            legend=True,
            legend_kwds={"label": "Category-normalized change"},
            linewidth=0.2,
            edgecolor="black",
        )
        ax.set_title(cat)
        ax.set_axis_off()

    for j in range(n_cat, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].set_axis_off()

    if title is None:
        title = "NIL choropleths — positives/negatives scaled separately per category"
    fig.suptitle(title, y=0.98)
    plt.tight_layout()
    return fig, axes


def plot_nil_category_posneg_bars_topk(
    change: pd.Series,
    categories: List[str],
    top_k: int = 10,
    ncols: int = 2,
    figsize_per_plot: Tuple[int, int] = (6, 6),
    title: Optional[str] = None,
):
    if not isinstance(change.index, pd.MultiIndex):
        raise ValueError("`change` must be a MultiIndex Series with (category, NIL).")

    if change.index.names != ["category", "NIL"]:
        if len(change.index.levels) == 2:
            change.index = change.index.set_names(["category", "NIL"])
        else:
            raise ValueError("Expected MultiIndex with exactly two levels: (category, NIL).")

    available_cats = change.index.get_level_values("category").unique()
    missing = [c for c in categories if c not in available_cats]
    if missing:
        raise ValueError(f"Categories not found in `change` index: {missing}")

    def short_nil(s: str) -> str:
        if not isinstance(s, str):
            return s
        cut = len(s)
        for sep in [" - ", "-", " ("]:
            j = s.find(sep)
            if j != -1:
                cut = min(cut, j)
        return s[:cut].strip()

    n_cat = len(categories)
    nrows = int(np.ceil(n_cat / ncols))
    fig_w = figsize_per_plot[0] * ncols
    fig_h = figsize_per_plot[1] * nrows

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for i, cat in enumerate(categories):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        cat_series = change.xs(cat, level="category").dropna()
        if cat_series.empty:
            ax.set_title(f"{cat} (no data)")
            ax.set_axis_off()
            continue

        pos = cat_series[cat_series > 0].nlargest(top_k)
        neg = cat_series[cat_series < 0].nsmallest(top_k)

        selected = pd.concat([neg, pos])
        selected = selected[~selected.index.duplicated(keep="first")]
        selected = selected.sort_values()

        y_positions = np.arange(len(selected))
        values = selected.values
        nil_labels = [short_nil(n) for n in selected.index]
        colors = ["blue" if v < 0 else "red" for v in values]

        ax.barh(y_positions, values, color=colors)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(nil_labels)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(cat)
        ax.set_xlabel("Raw change (reallocated − real)")
        ax.invert_yaxis()

    for j in range(n_cat, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].set_axis_off()

    if title is None:
        title = (
            f"Top {top_k} positive and negative NIL changes per category "
            "(raw values, red=positive, blue=negative)"
        )
    fig.suptitle(title, y=0.99)
    plt.tight_layout()
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description="Create NIL visualizations from baseline and reallocated data.")
    parser.add_argument("--baseline-gdf", default="gdf_2024_final_v3.pkl")
    parser.add_argument("--reallocated-pkl", default="reallocated_2024_final_v3_par_50_runs.pkl")
    parser.add_argument("--non-relocatable", default="gdf_2024_final_v3_non_relocatable.pkl")
    parser.add_argument("--hex-to-nil", default="hex_to_nil.csv")
    parser.add_argument("--nil-avg", default="outputs/nil_events_avg_across_runs_with_nonreloc.csv")
    parser.add_argument("--nils-shp", default=None, help="Optional NIL polygons shapefile for choropleths.")
    parser.add_argument("--out-dir", default="outputs/viz")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--categories", default=None, help="Comma-separated list of categories.")
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hex_gdf = load_hex_to_nil(Path(args.hex_to_nil))

    baseline_gdf = load_pickle(Path(args.baseline_gdf))
    if not isinstance(baseline_gdf, gpd.GeoDataFrame):
        baseline_gdf = gpd.GeoDataFrame(baseline_gdf, geometry="geometry")

    hex_gdf = hex_gdf.set_crs(baseline_gdf.crs, allow_override=True)

    baseline_counts = baseline_counts_by_category_nil(baseline_gdf, hex_gdf)

    realloc_counts = reallocated_avg_counts_by_category_nil(
        Path(args.reallocated_pkl),
        Path(args.non_relocatable),
        hex_gdf,
    )

    realloc_counts = adjust_to_nil_totals(realloc_counts, Path(args.nil_avg))

    categories = parse_categories(
        args.categories,
        sorted(set(baseline_counts["category"]).union(set(realloc_counts["category"]))),
    )

    df_counts = pd.concat(
        [
            baseline_counts.assign(year=args.year, scenario="real"),
            realloc_counts.assign(year=args.year, scenario="reallocated"),
        ],
        ignore_index=True,
    )

    # Heatmap matrix (normalized change)
    fig, ax = plot_nil_category_heatmap_posneg_scaled_counts(
        df_counts=df_counts,
        year=args.year,
        categories=categories,
        scenario_order=("real", "reallocated"),
        top_k=args.top_k,
        cmap_name="RdBu_r",
        figsize=(11, 8),
        title="Normalized Change in Event Distribution after Reallocation (2024)",
    )
    heatmap_path = out_dir / "nil_category_change_matrix.png"
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Bar charts: top positives/negatives per category (raw change)
    base_piv = baseline_counts.pivot_table(
        index="NIL", columns="category", values="n", fill_value=0
    ).reindex(columns=categories, fill_value=0)
    realloc_piv = realloc_counts.pivot_table(
        index="NIL", columns="category", values="n", fill_value=0
    ).reindex(columns=categories, fill_value=0)
    change = (realloc_piv - base_piv).T.stack()
    change.index = change.index.set_names(["category", "NIL"])

    fig, axes = plot_nil_category_posneg_bars_topk(
        change=change,
        categories=categories,
        top_k=10,
        ncols=2,
        figsize_per_plot=(6, 6),
        title="Top 10 positive and negative NIL changes per category",
    )
    bars_path = out_dir / "nil_category_top10_posneg_bars.png"
    fig.savefig(bars_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # CSV: per NIL, per category (before/after/diff)
    base_piv = baseline_counts.pivot_table(
        index="NIL", columns="category", values="n", fill_value=0
    ).reindex(columns=categories, fill_value=0)
    realloc_piv = realloc_counts.pivot_table(
        index="NIL", columns="category", values="n", fill_value=0
    ).reindex(columns=categories, fill_value=0)

    diff_piv = realloc_piv - base_piv

    out_df = pd.DataFrame(index=sorted(set(base_piv.index).union(set(realloc_piv.index))))
    for cat in categories:
        out_df[f"before_{cat}"] = base_piv.reindex(out_df.index, fill_value=0)[cat].values
        out_df[f"after_{cat}"] = realloc_piv.reindex(out_df.index, fill_value=0)[cat].values
        out_df[f"diff_{cat}"] = diff_piv.reindex(out_df.index, fill_value=0)[cat].values
    out_df = out_df.reset_index().rename(columns={"index": "NIL"})

    csv_path = out_dir / "nil_category_before_after_diff.csv"
    out_df.to_csv(csv_path, index=False)

    # Choropleths (optional)
    if args.nils_shp:
        nils_gdf = gpd.read_file(args.nils_shp)
        if "NIL" not in nils_gdf.columns:
            raise SystemExit("NIL polygons file must contain 'NIL' column.")

        # change series with MultiIndex (category, NIL)
        base_piv = baseline_counts.pivot_table(
            index="NIL", columns="category", values="n", fill_value=0
        ).reindex(columns=categories, fill_value=0)
        realloc_piv = realloc_counts.pivot_table(
            index="NIL", columns="category", values="n", fill_value=0
        ).reindex(columns=categories, fill_value=0)
        change = (realloc_piv - base_piv).T.stack()
        change.index = change.index.set_names(["category", "NIL"])

        fig, axes = plot_nil_category_choropleths_posneg_scaled(
            nils_gdf=nils_gdf,
            change=change,
            categories=categories,
            nil_col="NIL",
            cmap_name="RdBu_r",
            ncols=3,
            figsize_per_plot=(5, 5),
            title="NIL choropleths — positives/negatives scaled separately per category",
        )
        choropleth_path = out_dir / "nil_category_choropleths.png"
        fig.savefig(choropleth_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {choropleth_path}")
    else:
        print("Skipping choropleths: --nils-shp not provided.")

    print(f"Wrote {heatmap_path}")
    print(f"Wrote {bars_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
