"""
Greedy reallocator helpers and implementation.

Key ideas
---------
- Provide small helpers for common data-prep steps (geometry parsing, polygon filter, etc.).
- Keep the output schema: allocations + cell_summary.

Typical usage
-------------
    from greedy_reallocator import load_pickle, prep_events, GreedyDistanceReallocator

    grid = load_pickle("hex_pop.pkl")
    neigh = load_pickle("adjacent_neigh.pkl")
    events = load_pickle("gdf_2024.pkl")   # GeoDataFrame or DataFrame with "geometry"

    events = prep_events(events, polygon_path="A090101_ComuneMilano.shp", id_col="event_id")

    gr = GreedyDistanceReallocator.from_grid(grid, neigh, pop_col="population", cell_id_col="hex_id")
    allocations, summary = gr.allocate(events, group_by_category=True, progress="none")
"""

from __future__ import annotations

import pickle
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely import wkt
    from shapely.geometry.base import BaseGeometry
except Exception as _e:  # pragma: no cover
    gpd = None  # type: ignore
    wkt = None  # type: ignore
    BaseGeometry = object  # type: ignore


# ----------------------------
# IO helpers
# ----------------------------

def load_pickle(path: Union[str, Path]) -> Any:
    """Load any Python object from a pickle file."""
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """Save any Python object to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ----------------------------
# Geo helpers
# ----------------------------

def _require_geopandas() -> None:
    if gpd is None:
        raise ImportError(
            "geopandas/shapely are required for geo helpers. Install with: "
            "`pip install geopandas shapely`."
        )


def to_geom(x: Any):
    """
    Coerce a value into a shapely geometry.
    Handles: shapely geometry, WKT string, bytes containing WKT, None/NaN.
    """
    _require_geopandas()

    if isinstance(x, BaseGeometry):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", errors="ignore")
    x = str(x).strip()
    if x == "" or x.lower() in {"nan", "none", "null"}:
        return None
    return wkt.loads(x)


def ensure_geodataframe(
    df: Union[pd.DataFrame, "gpd.GeoDataFrame"],
    geometry_col: str = "geometry",
    crs: str = "EPSG:4326",
) -> "gpd.GeoDataFrame":
    """
    Ensure `df` is a GeoDataFrame with a valid shapely geometry column.
    If geometry values are WKT/bytes, they will be parsed.

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
    geometry_col : str
    crs : str
    """
    _require_geopandas()

    if isinstance(df, gpd.GeoDataFrame):
        gdf = df.copy()
        if gdf.geometry.name != geometry_col and geometry_col in gdf.columns:
            gdf = gdf.set_geometry(geometry_col)
        if gdf.crs is None:
            gdf = gdf.set_crs(crs)
        return gdf

    if geometry_col not in df.columns:
        raise KeyError(f"Expected a '{geometry_col}' column.")
    out = df.copy()
    out[geometry_col] = out[geometry_col].map(to_geom)
    out = out[out[geometry_col].notna()].copy()
    return gpd.GeoDataFrame(out, geometry=geometry_col, crs=crs)


def filter_points_within_polygon(
    points_gdf: "gpd.GeoDataFrame",
    polygon: Union[str, Path, "gpd.GeoDataFrame"],
    predicate: str = "within",
) -> "gpd.GeoDataFrame":
    """
    Spatially filter points to those inside a polygon (or multipolygon).

    polygon can be:
    - path to a shapefile/geojson/etc readable by geopandas
    - a GeoDataFrame of polygons
    """
    _require_geopandas()

    if not isinstance(points_gdf, gpd.GeoDataFrame):
        raise TypeError("points_gdf must be a GeoDataFrame.")
    if isinstance(polygon, (str, Path)):
        poly_gdf = gpd.read_file(str(polygon))
    else:
        poly_gdf = polygon

    # Ensure same CRS
    if points_gdf.crs != poly_gdf.crs:
        points_gdf = points_gdf.to_crs(poly_gdf.crs)

    out = gpd.sjoin(points_gdf, poly_gdf, predicate=predicate, how="inner")
    # Drop sjoin helper columns if present
    for c in ["index_right", "index_left"]:
        if c in out.columns:
            out = out.drop(columns=[c])
    return out


def deduplicate_events(
    df: pd.DataFrame,
    subset: Union[str, Sequence[str]] = ("title",),
    keep: str = "first",
) -> pd.DataFrame:
    """Drop duplicates for events (defaults to `title`)."""
    if isinstance(subset, str):
        subset = (subset,)
    if not all(c in df.columns for c in subset):
        missing = [c for c in subset if c not in df.columns]
        raise KeyError(f"Missing columns for deduplication: {missing}")
    return df.drop_duplicates(subset=list(subset), keep=keep).copy()


def collapse_events_with_weight(
    df: pd.DataFrame,
    *,
    subset: Union[str, Sequence[str]] = ("title",),
    weight_col: str = "event_weight",
    keep: str = "first",
) -> pd.DataFrame:
    """
    Collapse duplicates and attach a weight equal to the number of instances.

    The resulting DataFrame has one row per unique subset value(s), plus a
    weight column storing counts.
    """
    if isinstance(subset, str):
        subset = (subset,)
    if not all(c in df.columns for c in subset):
        missing = [c for c in subset if c not in df.columns]
        raise KeyError(f"Missing columns for deduplication: {missing}")

    counts = df.groupby(list(subset)).size().rename(weight_col).reset_index()
    base = df.drop_duplicates(subset=list(subset), keep=keep).copy()
    out = base.merge(counts, on=list(subset), how="left")
    return out


def _get_neighborhood(neighborhoods: Dict, cid: Any) -> Optional[Iterable[Any]]:
    if cid in neighborhoods:
        return neighborhoods[cid]
    if isinstance(cid, str):
        try:
            cid_int = int(cid)
            if cid_int in neighborhoods:
                return neighborhoods[cid_int]
        except Exception:
            pass
    else:
        cid_str = str(cid)
        if cid_str in neighborhoods:
            return neighborhoods[cid_str]
    return None


def add_sequential_event_id(
    df: pd.DataFrame,
    id_col: str = "event_id",
    start: int = 1,
) -> pd.DataFrame:
    """Add an integer id column (1..N) if missing."""
    out = df.copy()
    if id_col not in out.columns or out[id_col].isna().any():
        out[id_col] = np.arange(start, start + len(out), dtype=int)
    return out


def prep_events(
    events: Union[pd.DataFrame, "gpd.GeoDataFrame"],
    *,
    polygon_path: Optional[Union[str, Path]] = None,
    geometry_col: str = "geometry",
    crs: str = "EPSG:4326",
    dedupe_subset: Union[str, Sequence[str]] = ("title",),
    dedupe: bool = True,
    weight_by_instances: bool = False,
    weight_col: str = "event_weight",
    id_col: str = "event_id",
) -> "gpd.GeoDataFrame":
    """
    One-stop helper to:
    1) ensure GeoDataFrame with parsed geometry,
    2) optionally filter points inside a polygon,
    3) optionally drop duplicates (e.g., by title) OR collapse with weights,
    4) add sequential event_id.

    Returns a GeoDataFrame.
    """
    _require_geopandas()

    gdf = ensure_geodataframe(events, geometry_col=geometry_col, crs=crs)
    if polygon_path is not None:
        gdf = filter_points_within_polygon(gdf, polygon_path)
    if weight_by_instances and dedupe_subset is not None:
        gdf = collapse_events_with_weight(gdf, subset=dedupe_subset, weight_col=weight_col)
    elif dedupe and dedupe_subset is not None:
        gdf = deduplicate_events(gdf, subset=dedupe_subset)
    gdf = add_sequential_event_id(gdf, id_col=id_col)
    return gdf


def prep_events_table(
    events: pd.DataFrame,
    *,
    dedupe_subset: Union[str, Sequence[str]] = ("title",),
    dedupe: bool = True,
    weight_by_instances: bool = False,
    weight_col: str = "event_weight",
    id_col: str = "event_id",
) -> pd.DataFrame:
    """
    Lightweight prep that does NOT require geopandas:
    - optionally drop duplicates (e.g., by title),
    - add sequential event_id.

    Set weight_by_instances=True to collapse duplicates and weight by counts.
    """
    out = events.copy()
    if weight_by_instances and dedupe_subset is not None:
        out = collapse_events_with_weight(out, subset=dedupe_subset, weight_col=weight_col)
    elif dedupe and dedupe_subset is not None:
        out = deduplicate_events(out, subset=dedupe_subset)
    out = add_sequential_event_id(out, id_col=id_col)
    return out


# ----------------------------
# Greedy distance reallocator (deterministic)
# ----------------------------

@dataclass(frozen=True)
class GreedyDistanceGridIndex:
    """Precomputed grid arrays used by GreedyDistanceReallocator."""
    cell_ids: np.ndarray               # shape (n,)
    pop: np.ndarray                    # shape (n,)
    adj: Sequence[np.ndarray]          # length n, each array of neighbor indices
    dist: np.ndarray                   # shape (n, n), shortest-path distances
    id2idx: Dict[Any, int]
    cell_id_col: str
    pop_col: str
    dist_sentinel: float


def _compute_shortest_path_distances(adj: Sequence[np.ndarray]) -> np.ndarray:
    n = len(adj)
    dist = np.full((n, n), -1, dtype=np.int32)
    for i in range(n):
        dist[i, i] = 0
        q: deque[int] = deque([i])
        while q:
            u = q.popleft()
            du = int(dist[i, u])
            for v in adj[u]:
                if dist[i, v] == -1:
                    dist[i, v] = du + 1
                    q.append(int(v))
    if n == 0:
        return dist.astype(np.float64)
    reachable = dist >= 0
    max_dist = int(dist[reachable].max()) if reachable.any() else 0
    dist[~reachable] = max_dist + 1
    return dist.astype(np.float64)


def _rebuild_best_dists(
    dist_lists: List[List[float]],
    layer: int,
    sentinel: float,
) -> np.ndarray:
    n = len(dist_lists)
    best = np.full((n, layer), sentinel, dtype=np.float64)
    if layer <= 0:
        return best
    for j in range(n):
        lst = dist_lists[j]
        if not lst:
            continue
        if len(lst) <= layer:
            vals = np.sort(np.asarray(lst, dtype=np.float64))
        else:
            arr = np.asarray(lst, dtype=np.float64)
            part = np.partition(arr, layer - 1)[:layer]
            vals = np.sort(part)
        best[j, : len(vals)] = vals
    return best


class GreedyDistanceReallocator:
    """
    Greedy, deterministic reallocator using distance objectives.

    Output schema:
    allocations: [event_id, (category), (event_weight), assigned_cell_id, chosen_from_hub]
    cell_summary: [cell_id, initial_demand, satisfied, unmet, times_chosen_as_hub, events_hosted]
    """

    def __init__(self, grid: GreedyDistanceGridIndex):
        self.grid = grid

    @classmethod
    def from_grid(
        cls,
        grid_gdf: Union[pd.DataFrame, "gpd.GeoDataFrame"],
        neighborhoods: Dict,
        *,
        pop_col: str = "population",
        cell_id_col: str = "cell_id",
    ) -> "GreedyDistanceReallocator":
        for col in [cell_id_col, pop_col]:
            if col not in grid_gdf.columns:
                raise KeyError(f"grid_gdf must contain column '{col}'")

        cells_df = grid_gdf[[cell_id_col, pop_col]].copy()
        cells_df = cells_df.dropna(subset=[cell_id_col]).drop_duplicates(subset=[cell_id_col])
        cells_df = cells_df.sort_values(cell_id_col).reset_index(drop=True)

        cell_ids = cells_df[cell_id_col].to_numpy()
        id2idx = {cid: i for i, cid in enumerate(cell_ids)}
        id2idx_str = {str(cid): i for i, cid in enumerate(cell_ids)}
        n = len(cell_ids)

        pop = cells_df[pop_col].astype(float).clip(lower=0).to_numpy()

        adj = [None] * n
        for i, cid in enumerate(cell_ids):
            neigh_raw = _get_neighborhood(neighborhoods, cid)
            if neigh_raw is None:
                neigh_raw = [cid]
            seen = set()
            neigh_idx = []
            for nid in neigh_raw:
                key = str(nid)
                if key in seen:
                    continue
                if nid in id2idx:
                    seen.add(key)
                    neigh_idx.append(id2idx[nid])
                    continue
                if key in id2idx_str:
                    seen.add(key)
                    neigh_idx.append(id2idx_str[key])
            if not neigh_idx:
                neigh_idx = [i]
            adj[i] = np.asarray(neigh_idx, dtype=np.int32)

        dist = _compute_shortest_path_distances(adj)
        dist_sentinel = float(dist.max() + 1.0) if n > 0 else 0.0

        grid = GreedyDistanceGridIndex(
            cell_ids=cell_ids,
            pop=pop,
            adj=adj,
            dist=dist,
            id2idx=id2idx,
            cell_id_col=cell_id_col,
            pop_col=pop_col,
            dist_sentinel=dist_sentinel,
        )
        return cls(grid=grid)

    def _score_candidates(
        self,
        dL: np.ndarray,
        dL_minus: np.ndarray,
        *,
        objective_mode: str,
        pop_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        D = self.grid.dist
        mask = D < dL
        new_L = np.where(mask, np.maximum(D, dL_minus), dL)
        if objective_mode == "worst_served":
            return new_L.max(axis=1)
        if objective_mode == "sum_improvement":
            improvement = dL - new_L
            if pop_weights is not None:
                improvement = improvement * pop_weights
            return improvement.sum(axis=1)
        raise ValueError("objective_mode must be 'worst_served' or 'sum_improvement'")

    def _pick_candidate(
        self,
        scores: np.ndarray,
        *,
        objective_mode: str,
        prefer_farthest: bool = False,
        d1: Optional[np.ndarray] = None,
    ) -> int:
        if scores.size == 0:
            return -1
        if objective_mode == "worst_served":
            best = float(scores.min())
        else:
            best = float(scores.max())
        tie_mask = np.isclose(scores, best, rtol=1e-12, atol=1e-9)
        candidates = np.where(tie_mask)[0]
        if candidates.size == 1:
            return int(candidates[0])
        if prefer_farthest and d1 is not None:
            dvals = d1[candidates]
            max_d = float(np.max(dvals)) if dvals.size else -np.inf
            dmask = np.isclose(dvals, max_d, rtol=1e-12, atol=1e-9)
            candidates = candidates[dmask]
            if candidates.size == 1:
                return int(candidates[0])
        pop = self.grid.pop[candidates]
        max_pop = float(np.max(pop)) if pop.size else -np.inf
        pop_mask = np.isclose(pop, max_pop, rtol=1e-12, atol=1e-9)
        candidates = candidates[pop_mask]
        if candidates.size == 1:
            return int(candidates[0])
        candidate_ids = self.grid.cell_ids[candidates]
        use_numeric = True
        for cid in candidate_ids:
            if isinstance(cid, (bool, np.bool_)):
                use_numeric = False
                break
            if not isinstance(cid, (int, np.integer, float, np.floating)):
                use_numeric = False
                break
        if use_numeric:
            pos = int(np.argmin(candidate_ids.astype(float)))
        else:
            pos = int(np.argmin(np.asarray(candidate_ids, dtype=str)))
        return int(candidates[pos])

    def allocate(
        self,
        events_df: pd.DataFrame,
        *,
        group_by_category: bool = False,
        objective_mode: str = "worst_served",     # "worst_served" | "sum_improvement"
        progress: str = "tqdm",                   # "tqdm" | "print" | "none"
        log_every: int = 500,
        checkpoint_path: Optional[Union[str, Path]] = None,
        checkpoint_every: int = 2000,
        event_weight_col: Optional[str] = None,
        weight_by_pop: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if "event_id" not in events_df.columns:
            raise KeyError("events_df must contain column 'event_id'")
        if group_by_category and "category" not in events_df.columns:
            raise KeyError("group_by_category=True but 'category' not in events_df")
        if event_weight_col is not None and event_weight_col not in events_df.columns:
            raise KeyError(f"events_df must contain column '{event_weight_col}' when event_weight_col is set")
        if objective_mode not in {"worst_served", "sum_improvement"}:
            raise ValueError("objective_mode must be 'worst_served' or 'sum_improvement'")

        try:
            from tqdm.auto import tqdm  # type: ignore
            has_tqdm = True
        except Exception:
            has_tqdm = False
            if progress == "tqdm":
                progress = "print"

        cell_ids = self.grid.cell_ids
        pop = self.grid.pop
        n = len(cell_ids)
        pop_weights = pop if weight_by_pop else None

        include_category_col = "category" in events_df.columns

        def run_block(
            block_df: pd.DataFrame,
            category_label: Optional[str],
        ) -> pd.DataFrame:
            use_category = category_label is not None or include_category_col
            cols_base = ["event_id", "assigned_cell_id", "chosen_from_hub"]
            if use_category:
                cols_base = ["event_id", "category"] + cols_base[1:]
            if event_weight_col is not None:
                insert_at = 2 if use_category else 1
                cols_base.insert(insert_at, event_weight_col)

            if block_df is None or len(block_df) == 0 or n == 0:
                if block_df is None or len(block_df) == 0:
                    return pd.DataFrame(columns=cols_base)
                out = pd.DataFrame({
                    "event_id": block_df["event_id"].to_numpy(),
                    "assigned_cell_id": np.array([None] * len(block_df), dtype=object),
                    "chosen_from_hub": np.array([None] * len(block_df), dtype=object),
                })
                if use_category:
                    if category_label is not None:
                        out.insert(1, "category", np.full(len(block_df), category_label, dtype=object))
                    elif "category" in block_df.columns:
                        out.insert(1, "category", block_df["category"].to_numpy())
                if event_weight_col is not None:
                    insert_at = 2 if use_category else 1
                    out.insert(insert_at, event_weight_col, block_df[event_weight_col].to_numpy())
                return out[cols_base]

            out_event_id: List[Any] = []
            out_assigned: List[Any] = []
            out_hub: List[Any] = []
            out_weight: Optional[List[float]] = [] if event_weight_col is not None else None
            out_cat: Optional[List[Any]] = [] if use_category else None

            layer = 1
            dist_sentinel = self.grid.dist_sentinel
            dist_lists: List[List[float]] = [[] for _ in range(n)]
            best_dists = np.full((n, layer), dist_sentinel, dtype=np.float64)
            dL = best_dists[:, -1]
            dL_minus = np.full(n, -np.inf, dtype=np.float64)

            iterator = block_df.itertuples(index=False)
            if progress == "tqdm" and has_tqdm:
                iterator = tqdm(
                    iterator,
                    total=len(block_df),
                    desc=f"Greedy{'' if category_label is None else f' [{category_label}]'}",
                    leave=False,
                )
            elif progress == "print":
                start_t = time.time()

            for k, ev in enumerate(iterator, start=1):
                scores = self._score_candidates(
                    dL,
                    dL_minus,
                    objective_mode=objective_mode,
                    pop_weights=pop_weights,
                )
                if objective_mode == "sum_improvement":
                    best_score = float(scores.max()) if scores.size else 0.0
                    if best_score <= 0.0:
                        layer += 1
                        best_dists = _rebuild_best_dists(dist_lists, layer, dist_sentinel)
                        dL = best_dists[:, -1] if layer > 0 else np.full(n, dist_sentinel)
                        dL_minus = best_dists[:, -2] if layer > 1 else np.full(n, -np.inf, dtype=np.float64)
                        scores = self._score_candidates(
                            dL,
                            dL_minus,
                            objective_mode=objective_mode,
                            pop_weights=pop_weights,
                        )
                else:
                    best_score = float(scores.min()) if scores.size else dist_sentinel
                    current_worst = float(dL.max()) if dL.size else dist_sentinel
                    if best_score > current_worst:
                        layer += 1
                        best_dists = _rebuild_best_dists(dist_lists, layer, dist_sentinel)
                        dL = best_dists[:, -1] if layer > 0 else np.full(n, dist_sentinel)
                        dL_minus = best_dists[:, -2] if layer > 1 else np.full(n, -np.inf, dtype=np.float64)
                        scores = self._score_candidates(
                            dL,
                            dL_minus,
                            objective_mode=objective_mode,
                            pop_weights=pop_weights,
                        )

                d1 = best_dists[:, 0] if layer > 0 else np.full(n, dist_sentinel)
                prefer_farthest = (
                    objective_mode == "worst_served"
                    and scores.size > 0
                    and np.isclose(float(scores.min()), float(dL.max()), rtol=1e-12, atol=1e-9)
                )
                chosen_idx = self._pick_candidate(
                    scores,
                    objective_mode=objective_mode,
                    prefer_farthest=prefer_farthest,
                    d1=d1,
                )
                if chosen_idx < 0:
                    out_event_id.append(getattr(ev, "event_id"))
                    out_assigned.append(None)
                    out_hub.append(None)
                    if out_weight is not None:
                        out_weight.append(getattr(ev, event_weight_col))
                    if out_cat is not None:
                        if category_label is not None:
                            out_cat.append(category_label)
                        elif "category" in block_df.columns:
                            out_cat.append(getattr(ev, "category"))
                    continue

                dist_vec = self.grid.dist[chosen_idx]
                for j in range(n):
                    dist_lists[j].append(float(dist_vec[j]))

                if layer == 1:
                    best_dists[:, 0] = np.minimum(best_dists[:, 0], dist_vec)
                else:
                    dL_current = best_dists[:, -1]
                    mask = dist_vec < dL_current
                    if mask.any():
                        rows = np.where(mask)[0]
                        for j in rows:
                            row = best_dists[j]
                            d = float(dist_vec[j])
                            pos = int(np.searchsorted(row, d, side="left"))
                            if pos < layer:
                                row[pos + 1:] = row[pos:-1]
                                row[pos] = d

                dL = best_dists[:, -1] if layer > 0 else np.full(n, dist_sentinel)
                dL_minus = best_dists[:, -2] if layer > 1 else np.full(n, -np.inf, dtype=np.float64)

                out_event_id.append(getattr(ev, "event_id"))
                out_assigned.append(cell_ids[chosen_idx])
                out_hub.append(cell_ids[chosen_idx])
                if out_weight is not None:
                    out_weight.append(getattr(ev, event_weight_col))
                if out_cat is not None:
                    if category_label is not None:
                        out_cat.append(category_label)
                    elif "category" in block_df.columns:
                        out_cat.append(getattr(ev, "category"))

                if progress == "print" and (k % log_every == 0 or k == len(block_df)):
                    dt = time.time() - start_t
                    rate = k / max(dt, 1e-6)
                    print(f"[{category_label or 'all'}] {k}/{len(block_df)} | {rate:.1f} ev/s | layer={layer}")

                if checkpoint_path and (k % checkpoint_every == 0):
                    tmp = pd.DataFrame({
                        "event_id": out_event_id,
                        "assigned_cell_id": out_assigned,
                        "chosen_from_hub": out_hub,
                    })
                    if out_cat is not None:
                        tmp.insert(1, "category", out_cat)
                    if out_weight is not None:
                        insert_at = 2 if out_cat is not None else 1
                        tmp.insert(insert_at, event_weight_col, out_weight)
                    tmp.to_csv(str(checkpoint_path), index=False)

            out_df = pd.DataFrame({
                "event_id": out_event_id,
                "assigned_cell_id": out_assigned,
                "chosen_from_hub": out_hub,
            })
            if out_cat is not None:
                out_df.insert(1, "category", out_cat)
            if out_weight is not None:
                insert_at = 2 if out_cat is not None else 1
                out_df.insert(insert_at, event_weight_col, out_weight)

            cols = ["event_id", "assigned_cell_id", "chosen_from_hub"]
            if use_category:
                cols = ["event_id", "category"] + cols[1:]
            if event_weight_col is not None:
                insert_at = 2 if use_category else 1
                cols.insert(insert_at, event_weight_col)
            out_df = out_df[cols]
            return out_df

        blocks: List[pd.DataFrame] = []
        if group_by_category:
            for cat, block in events_df.groupby("category", sort=False):
                blocks.append(run_block(block, category_label=cat))
        else:
            blocks.append(run_block(events_df, category_label=None))

        allocations = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(
            columns=["event_id", "assigned_cell_id", "chosen_from_hub"]
        )

        if include_category_col and "category" not in allocations.columns:
            allocations = allocations.merge(
                events_df[["event_id", "category"]].drop_duplicates(subset=["event_id"]),
                on="event_id",
                how="left",
            )

        cols = ["event_id", "assigned_cell_id", "chosen_from_hub"]
        if "category" in allocations.columns:
            cols = ["event_id", "category"] + cols[1:]
        if event_weight_col is not None and event_weight_col in allocations.columns:
            insert_at = 2 if "category" in allocations.columns else 1
            cols.insert(insert_at, event_weight_col)
        allocations = allocations[cols]

        E_total = len(allocations)
        total_pop = float(pop.sum())
        beta = (total_pop / E_total) if (E_total > 0 and total_pop > 0) else np.inf

        initial_demand = (pop / beta) if np.isfinite(beta) else np.zeros(n, dtype=np.float64)
        satisfied = np.zeros(n, dtype=np.float64)
        unmet = initial_demand.copy()
        times_hub = np.zeros(n, dtype=np.float64)
        events_hosted = np.zeros(n, dtype=np.float64)

        if n > 0 and len(allocations) > 0 and "assigned_cell_id" in allocations.columns:
            id2idx = self.grid.id2idx
            if event_weight_col is not None and event_weight_col in allocations.columns:
                weights = allocations[event_weight_col].to_numpy(dtype=float)
            else:
                weights = np.ones(len(allocations), dtype=float)

            assigned_idx = allocations["assigned_cell_id"].map(id2idx)
            mask = assigned_idx.notna()
            idx = assigned_idx[mask].astype(int).to_numpy()
            w = weights[mask.to_numpy()]
            if idx.size:
                counts = np.bincount(idx, weights=w, minlength=n)
                events_hosted = counts.astype(float)
                times_hub = events_hosted.copy()

        cell_summary = pd.DataFrame({
            self.grid.cell_id_col: cell_ids,
            "initial_demand": initial_demand,
            "satisfied": satisfied,
            "unmet": unmet,
            "times_chosen_as_hub": times_hub,
            "events_hosted": events_hosted,
        })
        return allocations, cell_summary
