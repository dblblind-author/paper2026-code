#!/usr/bin/env python3
from __future__ import annotations

import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN


class FixNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_gdf(path: Path) -> gpd.GeoDataFrame:
    with path.open("rb") as f:
        gdf = FixNumpyUnpickler(f).load()
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf


def to_3857(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(3857)


def compute_clusters(
    gdf: gpd.GeoDataFrame,
    category: str,
    eps: int,
    min_samples: int,
) -> gpd.GeoDataFrame:
    sub = gdf[gdf["category"] == category].copy()
    if sub.empty:
        sub["cluster"] = -1
        return sub
    coords = np.vstack([sub.geometry.x, sub.geometry.y]).T
    model = DBSCAN(eps=eps, min_samples=min_samples)
    sub["cluster"] = model.fit_predict(coords)
    return sub


def get_cluster_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    valid = gdf[gdf["cluster"] >= 0].copy()
    if valid.empty:
        return gpd.GeoDataFrame(columns=["cluster", "geometry"], crs=gdf.crs)
    grouped = valid.groupby("cluster")["geometry"].apply(lambda s: s.unary_union.centroid).reset_index()
    return gpd.GeoDataFrame(grouped, geometry="geometry", crs=gdf.crs)


def match_clusters_hungarian(
    cents_23: gpd.GeoDataFrame,
    cents_24: gpd.GeoDataFrame,
    max_dist: float,
) -> list[tuple[int, int]]:
    if cents_23.empty or cents_24.empty:
        return []
    dist_rows = [cents_24.distance(geom).to_numpy() for geom in cents_23.geometry]
    distance_matrix = np.vstack(dist_rows)
    masked = distance_matrix.copy()
    masked[masked > max_dist] = 1e12
    row_ind, col_ind = linear_sum_assignment(masked)

    pairs = []
    for i, j in zip(row_ind, col_ind):
        if distance_matrix[i, j] <= max_dist:
            pairs.append((int(cents_23.iloc[i]["cluster"]), int(cents_24.iloc[j]["cluster"])))
    return pairs
