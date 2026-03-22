#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

from cluster_utils import load_gdf


def plot_category(gdf: gpd.GeoDataFrame, year: int, category: str, out_dir: Path) -> None:
    sub = gdf[gdf["category"] == category].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sub.geometry.x, sub.geometry.y, s=6, alpha=0.7)
    ax.set_title(f"{category} | {year} | unique titles")
    ax.set_axis_off()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{category}_{year}.png", dpi=200)
    plt.close(fig)


def main() -> int:
    gdf_2023 = load_gdf(Path("gdf_2023")).drop_duplicates(subset="title", keep="first")
    gdf_2024 = load_gdf(Path("gdf_2024")).drop_duplicates(subset="title", keep="first")

    categories = sorted(
        set(gdf_2023["category"].dropna().unique().tolist())
        | set(gdf_2024["category"].dropna().unique().tolist())
    )

    out_dir = Path("output") / "unique_title_plots"
    for cat in categories:
        plot_category(gdf_2023, 2023, cat, out_dir)
        plot_category(gdf_2024, 2024, cat, out_dir)

    print(f"Saved plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
