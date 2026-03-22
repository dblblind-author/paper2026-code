#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from cluster_utils import load_gdf, to_3857

# =========================
# CONFIGURATION
# =========================
EPS_GRID = [150, 200, 250, 300, 350]
MIN_PTS_LIST = [3, 4, 5, 6, 7]

def prep_mode(gdf: gpd.GeoDataFrame, unique_titles: bool) -> gpd.GeoDataFrame:
    if unique_titles:
        gdf = gdf.drop_duplicates(subset="title", keep="first")
    return gdf.copy()


# =========================
# KNN ELBOW (DBSCAN)
# =========================

def knn_distances(coords: np.ndarray, k: int) -> np.ndarray:
    if coords.shape[0] < k:
        return np.array([])
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    return np.sort(distances[:, -1])


def find_elbow(distances: np.ndarray) -> tuple[int | None, float | None]:
    n = distances.shape[0]
    if n < 3:
        return None, None
    x = np.arange(n)
    y = distances
    # Line from first to last point
    x1, y1 = 0, y[0]
    x2, y2 = n - 1, y[-1]
    dx = x2 - x1
    dy = y2 - y1
    denom = np.hypot(dx, dy)
    if denom == 0:
        return None, None
    # Perpendicular distance from each point to the line
    dist = np.abs(dy * x - dx * y + x2 * y1 - y2 * x1) / denom
    idx = int(np.argmax(dist))
    return idx, float(y[idx])


def plot_knn_single(
    distances: np.ndarray,
    elbow_idx: int | None,
    title: str,
    out_path: Path,
    color: str = "#1f77b4",
) -> None:
    if distances.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(distances, color=color, alpha=0.8)
    if elbow_idx is not None:
        ax.axvline(elbow_idx, color=color, alpha=0.5, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Points (sorted by k-NN distance)")
    ax.set_ylabel("k-NN distance (m)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_knn_multi(
    distances_by_cat: dict[str, np.ndarray],
    elbow_by_cat: dict[str, int | None],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("tab10")
    plotted = False
    for i, (cat, distances) in enumerate(distances_by_cat.items()):
        if distances.size == 0:
            continue
        color = cmap(i % 10)
        ax.plot(distances, color=color, alpha=0.55, label=cat)
        elbow_idx = elbow_by_cat.get(cat)
        if elbow_idx is not None:
            ax.axvline(elbow_idx, color=color, alpha=0.35, linestyle="--")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel("Points (sorted by k-NN distance)")
    ax.set_ylabel("k-NN distance (m)")
    ax.legend(loc="best", fontsize=7, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =========================
# CLUSTERING + MATCHING + DBCV
# =========================

class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        self.count -= 1
        return True


def compute_labels(coords: np.ndarray, eps: int, min_samples: int) -> np.ndarray:
    if coords.shape[0] == 0:
        return np.array([])
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)


def centroids_from_labels(coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if coords.shape[0] == 0:
        return np.zeros((0, 2))
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "cluster": labels})
    df = df[df["cluster"] >= 0]
    if df.empty:
        return np.zeros((0, 2))
    grouped = df.groupby("cluster")[["x", "y"]].mean().reset_index(drop=True)
    return grouped.to_numpy()


def match_centroids(c23: np.ndarray, c24: np.ndarray, max_dist: float) -> tuple[int, float | None]:
    if c23.size == 0 or c24.size == 0:
        return 0, None
    # Distance matrix
    diff = c23[:, None, :] - c24[None, :, :]
    D = np.sqrt((diff**2).sum(axis=2))
    big = 1e12
    D_masked = D.copy()
    D_masked[D_masked > max_dist] = big
    row_ind, col_ind = linear_sum_assignment(D_masked)
    distances = []
    for i, j in zip(row_ind, col_ind):
        if D[i, j] <= max_dist:
            distances.append(float(D[i, j]))
    if not distances:
        return 0, None
    return len(distances), float(np.mean(distances))


def build_knn_edges(coords: np.ndarray, min_pts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = coords.shape[0]
    if n < min_pts or n == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    nbrs = NearestNeighbors(n_neighbors=min_pts)
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    core_dist = distances[:, -1]
    rows = np.repeat(np.arange(n), min_pts)
    cols = indices.ravel()
    dists = distances.ravel()
    mrd = np.maximum.reduce([core_dist[rows], core_dist[cols], dists])

    edge_map: dict[tuple[int, int], float] = {}
    for r, c, w in zip(rows, cols, mrd):
        if r == c:
            continue
        a, b = (r, c) if r < c else (c, r)
        prev = edge_map.get((a, b))
        if prev is None or w < prev:
            edge_map[(a, b)] = float(w)

    if not edge_map:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    u = np.fromiter((k[0] for k in edge_map.keys()), dtype=int)
    v = np.fromiter((k[1] for k in edge_map.keys()), dtype=int)
    w = np.fromiter(edge_map.values(), dtype=float)
    return u, v, w


def build_knn_cache(coords: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for min_pts in MIN_PTS_LIST:
        cache[min_pts] = build_knn_edges(coords, min_pts)
    return cache


def compute_core_distances(coords: np.ndarray, min_pts: int) -> np.ndarray:
    if coords.shape[0] < min_pts:
        return np.array([])
    nbrs = NearestNeighbors(n_neighbors=min_pts)
    nbrs.fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    return distances[:, -1]


def mst_max_edge_for_cluster(coords: np.ndarray, core: np.ndarray, idx: np.ndarray) -> float:
    if idx.size <= 1:
        return 0.0
    nodes = idx
    n = nodes.size
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    max_edge = 0.0

    base = nodes[0]
    dists = np.linalg.norm(coords[nodes] - coords[base], axis=1)
    min_edge = np.maximum(np.maximum(core[nodes], core[base]), dists)
    min_edge[0] = 0.0

    for _ in range(1, n):
        i = int(np.argmin(np.where(in_tree, np.inf, min_edge)))
        if np.isinf(min_edge[i]):
            return np.inf
        in_tree[i] = True
        if min_edge[i] > max_edge:
            max_edge = float(min_edge[i])
        new_node = nodes[i]
        dists = np.linalg.norm(coords[nodes] - coords[new_node], axis=1)
        mrd = np.maximum(np.maximum(core[nodes], core[new_node]), dists)
        min_edge = np.minimum(min_edge, mrd)
    return max_edge


def min_mrd_between_sets(
    coords: np.ndarray, core: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray, block: int = 256
) -> float:
    if idx_a.size == 0 or idx_b.size == 0:
        return np.inf
    min_val = np.inf
    for i0 in range(0, idx_a.size, block):
        ia = idx_a[i0 : i0 + block]
        a = coords[ia]
        core_a = core[ia]
        for j0 in range(0, idx_b.size, block):
            ib = idx_b[j0 : j0 + block]
            b = coords[ib]
            core_b = core[ib]
            diff = a[:, None, :] - b[None, :, :]
            dist = np.sqrt((diff**2).sum(axis=2))
            mrd = np.maximum(np.maximum(dist, core_a[:, None]), core_b[None, :])
            block_min = float(mrd.min())
            if block_min < min_val:
                min_val = block_min
    return min_val


def dbcv_score_full(coords: np.ndarray, labels: np.ndarray, min_pts: int) -> float | None:
    if labels.size == 0:
        return None
    mask = labels >= 0
    n_nonnoise = int(mask.sum())
    if n_nonnoise == 0:
        return None

    core = compute_core_distances(coords, min_pts)
    if core.size == 0:
        return None

    clusters = np.unique(labels[mask])
    sizes = {c: int((labels == c).sum()) for c in clusters}
    sep_min = {c: np.inf for c in clusters}

    for c in clusters:
        idx_c = np.where(labels == c)[0]
        idx_other = np.where((labels >= 0) & (labels != c))[0]
        sep_min[c] = min_mrd_between_sets(coords, core, idx_c, idx_other)

    validity_sum = 0.0
    for c in clusters:
        size = sizes[c]
        idx_c = np.where(labels == c)[0]
        s_c = mst_max_edge_for_cluster(coords, core, idx_c)
        d_c = sep_min[c]
        if np.isinf(d_c):
            v_c = 0.0
        elif np.isinf(s_c):
            v_c = -1.0
        else:
            denom = max(d_c, s_c)
            v_c = (d_c - s_c) / denom if denom > 0 else 0.0
        validity_sum += (size / n_nonnoise) * v_c

    return float(validity_sum)


# =========================
# PIPELINE
# =========================

def run_knn_elbows(
    gdf_2024: gpd.GeoDataFrame,
    categories: list[str],
    mode: str,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    coords_all = np.vstack([gdf_2024.geometry.x, gdf_2024.geometry.y]).T

    for min_pts in MIN_PTS_LIST:
        # Case A: all categories together
        distances = knn_distances(coords_all, min_pts)
        elbow_idx, elbow_dist = find_elbow(distances)
        plot_knn_single(
            distances,
            elbow_idx,
            f"{mode} | all categories | minPts={min_pts}",
            out_dir / "knn_elbow" / f"{mode}_all_minPts{min_pts}.png",
        )
        rows.append(
            {
                "mode": mode,
                "case": "all",
                "minPts": min_pts,
                "category": "ALL",
                "n_points": int(coords_all.shape[0]),
                "elbow_index": elbow_idx,
                "elbow_distance_m": elbow_dist,
            }
        )

        # Case B: per-category (plot all lines together)
        distances_by_cat: dict[str, np.ndarray] = {}
        elbows_by_cat: dict[str, int | None] = {}
        for cat in categories:
            sub = gdf_2024[gdf_2024["category"] == cat]
            coords = np.vstack([sub.geometry.x, sub.geometry.y]).T
            d = knn_distances(coords, min_pts)
            idx, dist = find_elbow(d)
            distances_by_cat[cat] = d
            elbows_by_cat[cat] = idx
            rows.append(
                {
                    "mode": mode,
                    "case": "by_category",
                    "minPts": min_pts,
                    "category": cat,
                    "n_points": int(coords.shape[0]),
                    "elbow_index": idx,
                    "elbow_distance_m": dist,
                }
            )

        plot_knn_multi(
            distances_by_cat,
            elbows_by_cat,
            f"{mode} | by category | minPts={min_pts}",
            out_dir / "knn_elbow" / f"{mode}_by_category_minPts{min_pts}.png",
        )

    return pd.DataFrame(rows)


def run_clustering_summary(
    gdf_2023: gpd.GeoDataFrame,
    gdf_2024: gpd.GeoDataFrame,
    categories: list[str],
    mode: str,
) -> pd.DataFrame:
    rows = []
    coords23_all = np.vstack([gdf_2023.geometry.x, gdf_2023.geometry.y]).T
    coords24_all = np.vstack([gdf_2024.geometry.x, gdf_2024.geometry.y]).T

    for min_pts in MIN_PTS_LIST:
        for eps in EPS_GRID:
            # Case A: all categories together
            labels23 = compute_labels(coords23_all, eps, min_pts)
            labels24 = compute_labels(coords24_all, eps, min_pts)
            n23 = int(pd.Series(labels23[labels23 >= 0]).nunique()) if labels23.size else 0
            n24 = int(pd.Series(labels24[labels24 >= 0]).nunique()) if labels24.size else 0
            c23 = centroids_from_labels(coords23_all, labels23)
            c24 = centroids_from_labels(coords24_all, labels24)
            matched, mean_shift = match_centroids(c23, c24, max_dist=eps)
            dbcv23 = dbcv_score_full(coords23_all, labels23, min_pts)
            dbcv24 = dbcv_score_full(coords24_all, labels24, min_pts)
            rows.append(
                {
                    "mode": mode,
                    "case": "all",
                    "minPts": min_pts,
                    "eps": eps,
                    "category": "ALL",
                    "clusters_2023": n23,
                    "clusters_2024": n24,
                    "matched_clusters": matched,
                    "mean_centroid_shift_m": mean_shift,
                    "dbcv_2023": dbcv23,
                    "dbcv_2024": dbcv24,
                }
            )

            # Case B: per-category
            for cat in categories:
                sub23 = gdf_2023[gdf_2023["category"] == cat]
                sub24 = gdf_2024[gdf_2024["category"] == cat]
                coords23 = np.vstack([sub23.geometry.x, sub23.geometry.y]).T
                coords24 = np.vstack([sub24.geometry.x, sub24.geometry.y]).T
                labels23 = compute_labels(coords23, eps, min_pts)
                labels24 = compute_labels(coords24, eps, min_pts)
                n23 = int(pd.Series(labels23[labels23 >= 0]).nunique()) if labels23.size else 0
                n24 = int(pd.Series(labels24[labels24 >= 0]).nunique()) if labels24.size else 0
                c23 = centroids_from_labels(coords23, labels23)
                c24 = centroids_from_labels(coords24, labels24)
                matched, mean_shift = match_centroids(c23, c24, max_dist=eps)
                dbcv23 = dbcv_score_full(coords23, labels23, min_pts)
                dbcv24 = dbcv_score_full(coords24, labels24, min_pts)
                rows.append(
                    {
                        "mode": mode,
                        "case": "by_category",
                        "minPts": min_pts,
                        "eps": eps,
                        "category": cat,
                        "clusters_2023": n23,
                        "clusters_2024": n24,
                        "matched_clusters": matched,
                        "mean_centroid_shift_m": mean_shift,
                        "dbcv_2023": dbcv23,
                        "dbcv_2024": dbcv24,
                    }
                )

    return pd.DataFrame(rows)


def round_eps(value: float | None) -> int | None:
    if value is None or np.isnan(value):
        return None
    return int(round(value / 50.0) * 50)


def run_clustering_summary_from_elbow(
    gdf_2023: gpd.GeoDataFrame,
    gdf_2024: gpd.GeoDataFrame,
    categories: list[str],
    elbow_table: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    rows = []
    coords23_all = np.vstack([gdf_2023.geometry.x, gdf_2023.geometry.y]).T
    coords24_all = np.vstack([gdf_2024.geometry.x, gdf_2024.geometry.y]).T

    df_mode = elbow_table[elbow_table["mode"] == mode].copy()
    for _, row in df_mode.iterrows():
        case = row["case"]
        min_pts = int(row["minPts"])
        cat = row["category"]
        eps = round_eps(row["elbow_distance_m"])
        if eps is None or eps == 0:
            continue

        if case == "all":
            labels23 = compute_labels(coords23_all, eps, min_pts)
            labels24 = compute_labels(coords24_all, eps, min_pts)
            n23 = int(pd.Series(labels23[labels23 >= 0]).nunique()) if labels23.size else 0
            n24 = int(pd.Series(labels24[labels24 >= 0]).nunique()) if labels24.size else 0
            c23 = centroids_from_labels(coords23_all, labels23)
            c24 = centroids_from_labels(coords24_all, labels24)
            matched, mean_shift = match_centroids(c23, c24, max_dist=eps)
            dbcv23 = dbcv_score_full(coords23_all, labels23, min_pts)
            dbcv24 = dbcv_score_full(coords24_all, labels24, min_pts)
            rows.append(
                {
                    "mode": mode,
                    "case": "all",
                    "minPts": min_pts,
                    "eps": eps,
                    "category": "ALL",
                    "clusters_2023": n23,
                    "clusters_2024": n24,
                    "matched_clusters": matched,
                    "mean_centroid_shift_m": mean_shift,
                    "dbcv_2023": dbcv23,
                    "dbcv_2024": dbcv24,
                }
            )
        else:
            if cat not in categories:
                continue
            sub23 = gdf_2023[gdf_2023["category"] == cat]
            sub24 = gdf_2024[gdf_2024["category"] == cat]
            coords23 = np.vstack([sub23.geometry.x, sub23.geometry.y]).T
            coords24 = np.vstack([sub24.geometry.x, sub24.geometry.y]).T
            labels23 = compute_labels(coords23, eps, min_pts)
            labels24 = compute_labels(coords24, eps, min_pts)
            n23 = int(pd.Series(labels23[labels23 >= 0]).nunique()) if labels23.size else 0
            n24 = int(pd.Series(labels24[labels24 >= 0]).nunique()) if labels24.size else 0
            c23 = centroids_from_labels(coords23, labels23)
            c24 = centroids_from_labels(coords24, labels24)
            matched, mean_shift = match_centroids(c23, c24, max_dist=eps)
            dbcv23 = dbcv_score_full(coords23, labels23, min_pts)
            dbcv24 = dbcv_score_full(coords24, labels24, min_pts)
            rows.append(
                {
                    "mode": mode,
                    "case": "by_category",
                    "minPts": min_pts,
                    "eps": eps,
                    "category": cat,
                    "clusters_2023": n23,
                    "clusters_2024": n24,
                    "matched_clusters": matched,
                    "mean_centroid_shift_m": mean_shift,
                    "dbcv_2023": dbcv23,
                    "dbcv_2024": dbcv24,
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="DBSCAN clustering and elbow analysis")
    parser.add_argument("--gdf-2023", default="gdf_2023")
    parser.add_argument("--gdf-2024", default="gdf_2024")
    parser.add_argument("--out", default="output")
    parser.add_argument("--elbow-summary-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf_2023 = load_gdf(Path(args.gdf_2023))
    gdf_2024 = load_gdf(Path(args.gdf_2024))

    categories = sorted(gdf_2024["category"].dropna().unique().tolist())

    elbow_tables = []
    summary_tables = []
    elbow_summary_tables = []

    for mode, unique_titles in [("occurrence", False), ("unique_title", True)]:
        g23 = prep_mode(gdf_2023, unique_titles)
        g24 = prep_mode(gdf_2024, unique_titles)
        g23 = to_3857(g23)
        g24 = to_3857(g24)

        if not args.elbow_summary_only:
            elbow_df = run_knn_elbows(g24, categories, mode, out_dir)
            elbow_tables.append(elbow_df)

            summary_df = run_clustering_summary(g23, g24, categories, mode)
            summary_tables.append(summary_df)
        else:
            elbow_df = pd.read_csv(out_dir / "knn_elbow_table.csv")
            elbow_summary_df = run_clustering_summary_from_elbow(
                g23, g24, categories, elbow_df, mode
            )
            elbow_summary_tables.append(elbow_summary_df)

    if args.elbow_summary_only:
        elbow_summary_all = (
            pd.concat(elbow_summary_tables, ignore_index=True)
            if elbow_summary_tables
            else pd.DataFrame()
        )
        elbow_summary_all.to_csv(out_dir / "cluster_summary_elbow.csv", index=False)
        print("Done.")
        print(f"Wrote: {out_dir / 'cluster_summary_elbow.csv'}")
        return 0

    elbow_all = (
        pd.concat(elbow_tables, ignore_index=True) if elbow_tables else pd.DataFrame()
    )
    summary_all = (
        pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame()
    )

    elbow_all.to_csv(out_dir / "knn_elbow_table.csv", index=False)
    summary_all.to_csv(out_dir / "clustering_summary.csv", index=False)

    print("Done.")
    print(f"Wrote: {out_dir / 'knn_elbow_table.csv'}")
    print(f"Wrote: {out_dir / 'clustering_summary.csv'}")
    print(f"Plots in: {out_dir / 'knn_elbow'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
