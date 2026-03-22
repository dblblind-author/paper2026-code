#!/usr/bin/env python3
from __future__ import annotations

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


def plot_stable_overlay(
    cat: str,
    c23: gpd.GeoDataFrame,
    c24: gpd.GeoDataFrame,
    pairs: list[tuple[int, int]],
    out_path: Path,
    point_size: int = 14,
    centroid_size: int = 120,
) -> None:
    if c23.empty and c24.empty:
        return

    cents23 = get_cluster_centroids(c23)
    cents24 = get_cluster_centroids(c24)

    midpoints = []
    for cl23, cl24 in pairs:
        p1 = cents23.loc[cents23.cluster == cl23, "geometry"].values[0]
        p2 = cents24.loc[cents24.cluster == cl24, "geometry"].values[0]
        midpoints.append({"geometry": gpd.points_from_xy([(p1.x + p2.x) / 2], [(p1.y + p2.y) / 2])[0]})
    stable_centroids_mid = gpd.GeoDataFrame(midpoints, geometry="geometry", crs=c23.crs)

    bounds = gpd.GeoSeries([c23.unary_union, c24.unary_union], crs=c23.crs).total_bounds
    xmin, ymin, xmax, ymax = bounds
    dx, dy = (xmax - xmin) * 0.08, (ymax - ymin) * 0.08
    xmin, xmax = xmin - dx, xmax + dx
    ymin, ymax = ymin - dy, ymax + dy

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

    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=c23.crs, zorder=1)

    if not c23.empty:
        c23.plot(ax=ax, color="#dc2626", alpha=0.7, markersize=point_size, zorder=3)
    if not c24.empty:
        c24.plot(ax=ax, color="#2563eb", alpha=0.7, markersize=point_size, zorder=3)

    if not stable_centroids_mid.empty:
        stable_centroids_mid.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=1.6,
            markersize=centroid_size,
            marker="o",
            zorder=4,
        )

    ax.set_title(f"Stable clusters — {cat.replace('_',' ').title()} | eps={EPS} minPts={MIN_PTS}")
    ax.set_axis_off()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


EPS = 150
MIN_PTS = 3


def main() -> int:
    gdf_2023 = load_gdf(Path("gdf_2023")).drop_duplicates(subset="title", keep="first")
    gdf_2024 = load_gdf(Path("gdf_2024")).drop_duplicates(subset="title", keep="first")

    gdf_2023 = to_3857(gdf_2023)
    gdf_2024 = to_3857(gdf_2024)

    categories = sorted(
        set(gdf_2023["category"].dropna().unique().tolist())
        | set(gdf_2024["category"].dropna().unique().tolist())
    )

    out_dir = Path("output") / "dbscan_stable_overlay_eps150_minpts3"

    for cat in categories:
        c23 = compute_clusters(gdf_2023, cat, EPS, MIN_PTS)
        c24 = compute_clusters(gdf_2024, cat, EPS, MIN_PTS)
        pairs = match_clusters_hungarian(get_cluster_centroids(c23), get_cluster_centroids(c24), max_dist=EPS)
        plot_stable_overlay(cat, c23, c24, pairs, out_dir / f"{cat}_stable_overlay.png")

    print(f"Saved plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
