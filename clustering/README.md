# clustering

DBSCAN-based spatial clustering analysis for the 2023 and 2024 event GeoDataFrames.

## Main files
- `dbscan_clustering.py`: main analysis pipeline. Computes k-NN elbow curves, DBSCAN summaries, centroid matching across years, and DBCV scores.
- `cluster_utils.py`: shared helpers for loading GeoDataFrames, CRS conversion, DBSCAN clustering, centroid extraction, and Hungarian matching.
- `plot_unique_titles.py`: raw scatter plots by category after deduplicating repeated titles.
- `plot_dbscan_clusters_unique.py`: per-category DBSCAN maps on the unique-title view.
- `plot_dbscan_stable_clusters.py`: per-category stable-cluster plots using Hungarian centroid matching between 2023 and 2024.
- `plot_dbscan_stable_overlay.py`: overlay version of the stable-cluster plots.

## Default inputs
Scripts now default to:
- `gdf_2023`
- `gdf_2024`

These inputs are expected to be pickle-loaded GeoDataFrames available in the working directory when the scripts are run.

## `dbscan_clustering.py`
Purpose:
- run elbow analysis for DBSCAN
- compare clustering structure between 2023 and 2024
- support both event occurrence counts and title-deduplicated runs

Parameters:
- `--gdf-2023 PATH` (default: `gdf_2023`)
- `--gdf-2024 PATH` (default: `gdf_2024`)
- `--out DIR` (default: `output`)
- `--elbow-summary-only`

Outputs:
- `output/knn_elbow_table.csv`
- `output/clustering_summary.csv`
- `output/cluster_summary_elbow.csv` when `--elbow-summary-only` is used
- `output/knn_elbow/*.png`

## Plotting scripts
`plot_unique_titles.py`
- makes one scatter plot per category and year
- output: `output/unique_title_plots/*.png`

`plot_dbscan_clusters_unique.py`
- runs DBSCAN with fixed `eps=150`, `minPts=3` on unique titles
- output: `output/dbscan_clusters_unique_eps150_minpts3/*.png`

`plot_dbscan_stable_clusters.py`
- runs DBSCAN with fixed `eps=150`, `minPts=3`
- matches 2023 and 2024 cluster centroids with the Hungarian algorithm
- marks matched cluster midpoints
- output: `output/dbscan_stable_eps150_minpts3/*.png`

`plot_dbscan_stable_overlay.py`
- same matching idea as above, but as an overlay variant
- output: `output/dbscan_stable_overlay_eps150_minpts3/*.png`
