#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import contextily as cx

from cluster_utils import load_gdf, to_3857

def plot_category_clusters(
    gdf: gpd.GeoDataFrame,
    category: str,
    year: int,
    eps: int,
    min_pts: int,
    out_dir: Path,
) -> None:
    sub = gdf[gdf["category"] == category].copy()
    if sub.empty:
        return

    coords = np.vstack([sub.geometry.x, sub.geometry.y]).T
    labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(coords)
    sub["cluster"] = labels

    xmin, ymin, xmax, ymax = sub.total_bounds
    dx, dy = (xmax - xmin) * 0.08, (ymax - ymin) * 0.08
    xmin, xmax = xmin - dx, xmax + dx
    ymin, ymax = ymin - dy, ymax + dy

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=sub.crs, zorder=1)

    clusters = np.unique(labels)
    cmap = plt.get_cmap("tab20")
    for i, cl in enumerate(clusters):
        pts = sub[sub["cluster"] == cl]
        color = "#bdbdbd" if cl == -1 else cmap(i % 20)
        ax.scatter(pts.geometry.x, pts.geometry.y, s=10, c=[color], alpha=0.7, zorder=3)

    ax.set_title(f"{category.replace('_',' ').title()} | {year} | eps={eps} minPts={min_pts}")
    ax.set_axis_off()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{category}_{year}_clusters.png", dpi=200)
    plt.close(fig)


def main() -> int:
    eps = 150
    min_pts = 3

    gdf_2023 = load_gdf(Path("gdf_2023")).drop_duplicates(subset="title", keep="first")
    gdf_2024 = load_gdf(Path("gdf_2024")).drop_duplicates(subset="title", keep="first")

    gdf_2023 = to_3857(gdf_2023)
    gdf_2024 = to_3857(gdf_2024)

    categories = sorted(
        set(gdf_2023["category"].dropna().unique().tolist())
        | set(gdf_2024["category"].dropna().unique().tolist())
    )

    out_dir = Path("output") / "dbscan_clusters_unique_eps150_minpts3"
    for cat in categories:
        plot_category_clusters(gdf_2023, cat, 2023, eps, min_pts, out_dir)
        plot_category_clusters(gdf_2024, cat, 2024, eps, min_pts, out_dir)

    print(f"Saved plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
