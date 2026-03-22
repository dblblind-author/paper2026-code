#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

import contextily as cx
from cluster_utils import (
    compute_clusters,
    get_cluster_centroids,
    load_gdf,
    match_clusters_hungarian,
    to_3857,
)


def plot_stable_only(
    cat: str,
    clustered_2023: gpd.GeoDataFrame,
    clustered_2024: gpd.GeoDataFrame,
    stable_pairs: list[tuple[int, int]],
    out_path: Path,
    point_size: int = 14,
    centroid_size: int = 120,
    basemap=None,
) -> None:
    c23 = clustered_2023[clustered_2023["category"].str.lower() == cat.lower()]
    c24 = clustered_2024[clustered_2024["category"].str.lower() == cat.lower()]
    cents23 = get_cluster_centroids(c23)
    cents24 = get_cluster_centroids(c24)

    if not stable_pairs:
        return

    midpoints = []
    for cl23, cl24 in stable_pairs:
        p1 = cents23.loc[cents23.cluster == cl23, "geometry"].values[0]
        p2 = cents24.loc[cents24.cluster == cl24, "geometry"].values[0]
        midpoints.append({"geometry": gpd.points_from_xy([(p1.x + p2.x) / 2], [(p1.y + p2.y) / 2])[0]})
    stable_centroids_mid = gpd.GeoDataFrame(midpoints, geometry="geometry", crs=c23.crs)

    xmin, ymin, xmax, ymax = gpd.GeoSeries([c23.unary_union, c24.unary_union], crs=c23.crs).total_bounds
    dx, dy = (xmax - xmin) * 0.08, (ymax - ymin) * 0.08
    extent = (xmin - dx, xmax + dx, ymin - dy, ymax + dy)

    xmin, xmax, ymin, ymax = extent
    width, height = (xmax - xmin), (ymax - ymin)
    if width > height:
        pad = (width - height) / 2
        ymin -= pad
        ymax += pad
    elif height > width:
        pad = (height - width) / 2
        xmin -= pad
        xmax += pad

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if basemap is None:
        raise RuntimeError("Basemap is required but not provided.")
    cx.add_basemap(ax, source=basemap, crs=c23.crs, zorder=1)

    c23.plot(ax=ax, color="#dc2626", alpha=0.7, markersize=point_size, zorder=3)
    c24.plot(ax=ax, color="#2563eb", alpha=0.7, markersize=point_size, zorder=3)

    stable_centroids_mid.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=1.6,
        markersize=centroid_size,
        marker="o",
        zorder=4,
    )

    ax.set_title(f"Stable clusters — {cat.replace('_',' ').title()} | eps={EPS} minPts={MIN_PTS}", fontsize=14)
    ax.set_axis_off()

    if ax.get_legend():
        ax.get_legend().remove()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


EPS = 150
MIN_PTS = 3


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot stable DBSCAN clusters by category")
    parser.add_argument("--gdf-2023", default="gdf_2023")
    parser.add_argument("--gdf-2024", default="gdf_2024")
    parser.add_argument("--out", default="output/dbscan_stable_eps150_minpts3")
    args = parser.parse_args()

    gdf_2023 = load_gdf(Path(args.gdf_2023)).drop_duplicates(subset="title", keep="first")
    gdf_2024 = load_gdf(Path(args.gdf_2024)).drop_duplicates(subset="title", keep="first")

    gdf_2023 = to_3857(gdf_2023)
    gdf_2024 = to_3857(gdf_2024)

    categories = sorted(
        set(gdf_2023["category"].dropna().unique().tolist())
        | set(gdf_2024["category"].dropna().unique().tolist())
    )

    out_dir = Path(args.out)

    for cat in categories:
        c23 = compute_clusters(gdf_2023, cat, EPS, MIN_PTS)
        c24 = compute_clusters(gdf_2024, cat, EPS, MIN_PTS)
        cents23 = get_cluster_centroids(c23)
        cents24 = get_cluster_centroids(c24)
        pairs = match_clusters_hungarian(cents23, cents24, max_dist=EPS)
        out_path = out_dir / f"{cat}_stable.png"
        plot_stable_only(
            cat,
            c23,
            c24,
            pairs,
            out_path,
            basemap=cx.providers.CartoDB.Positron,
        )

    print(f"Saved plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
