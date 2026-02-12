"""
event_unit_reallocator.py

A small, reusable module that turns the original "Random_reallocator.ipynb" notebook
into a clean API you can call in 1–3 lines.

Key ideas
---------
- Precompute grid + neighborhood arrays once (faster repeated calls).
- Provide small helpers for common data-prep steps (geometry parsing, polygon filter, etc.).
- Keep the original output schema: allocations + cell_summary.

Typical usage
-------------
    from event_unit_reallocator import load_pickle, RandomReallocator, prep_events

    grid = load_pickle("hex_pop.pkl")
    neigh = load_pickle("neigh_dict.pkl")
    events = load_pickle("gdf_2024.pkl")   # GeoDataFrame or DataFrame with "geometry"

    events = prep_events(events, polygon_path="path/to/polygon.shp", id_col="event_id")

    rr = RandomReallocator.from_grid(grid, neigh, pop_col="population", cell_id_col="hex_id", rng_seed=123)
    allocations, summary = rr.allocate(events, group_by_category=True, progress="none")
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
    inverse_weight: bool = False,
    keep: str = "first",
) -> pd.DataFrame:
    """
    Collapse duplicates and attach a weight equal to the number of instances.

    If inverse_weight=True, the weight is 1 / count (more instances -> lower weight).

    The resulting DataFrame has one row per unique subset value(s), plus a
    weight column storing counts.
    """
    if isinstance(subset, str):
        subset = (subset,)
    if not all(c in df.columns for c in subset):
        missing = [c for c in subset if c not in df.columns]
        raise KeyError(f"Missing columns for deduplication: {missing}")

    counts = df.groupby(list(subset)).size().rename(weight_col).reset_index()
    if inverse_weight:
        weights = counts[weight_col].astype(float)
        weights = np.where(weights > 0, 1.0 / weights, 0.0)
        counts[weight_col] = weights
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
    group_by: str = "event",
    weight_mode: str = "par",
    dedupe_subset: Union[str, Sequence[str]] = ("title",),
    weight_col: str = "event_weight",
    id_col: str = "event_id",
) -> "gpd.GeoDataFrame":
    """
    Full prep with optional geo filtering (requires geopandas):
    - parse/ensure geometry,
    - optionally filter points within a polygon,
    - add sequential event_id.

    group_by="event" collapses duplicate titles (one row per title).
    group_by="row" keeps all rows (no grouping).

    weight_mode:
      - "direct": weight by counts (more instances -> higher weight)
      - "inverse": weight by 1 / counts (more instances -> lower weight)
      - "par": no weights (each row counts as 1)
    """
    _require_geopandas()

    gdf = ensure_geodataframe(events, geometry_col=geometry_col, crs=crs)
    if polygon_path is not None:
        gdf = filter_points_within_polygon(gdf, polygon_path)
    gdf = _apply_grouping_and_weights(
        gdf,
        group_by=group_by,
        weight_mode=weight_mode,
        dedupe_subset=dedupe_subset,
        weight_col=weight_col,
    )
    gdf = add_sequential_event_id(gdf, id_col=id_col)
    return gdf


def _apply_grouping_and_weights(
    df: pd.DataFrame,
    *,
    group_by: str,
    weight_mode: str,
    dedupe_subset: Union[str, Sequence[str]] = ("title",),
    weight_col: str = "event_weight",
) -> pd.DataFrame:
    group_by = group_by.strip().lower()
    weight_mode = weight_mode.strip().lower()
    if group_by in {"row", "rows", "record", "records", "raw"}:
        if weight_mode != "par":
            raise ValueError("group_by='row' requires weight_mode='par' (no weights available)")
        return df.copy()
    if group_by in {"event", "events", "instance", "instances", "title", "titles"}:
        if dedupe_subset is None:
            raise ValueError("dedupe_subset must be set when group_by='event'")
        if weight_mode == "par":
            return deduplicate_events(df, subset=dedupe_subset)
        if weight_mode in {"direct", "inverse"}:
            return collapse_events_with_weight(
                df,
                subset=dedupe_subset,
                weight_col=weight_col,
                inverse_weight=(weight_mode == "inverse"),
            )
        raise ValueError("weight_mode must be 'direct', 'inverse', or 'par'")
    raise ValueError("group_by must be 'event' or 'row'")


def prep_events_table(
    events: pd.DataFrame,
    *,
    group_by: str = "event",
    weight_mode: str = "par",
    dedupe_subset: Union[str, Sequence[str]] = ("title",),
    weight_col: str = "event_weight",
    id_col: str = "event_id",
) -> pd.DataFrame:
    """
    Lightweight prep that does NOT require geopandas:
    - optionally collapse duplicates (e.g., by title),
    - add sequential event_id.

    group_by="event" collapses duplicate titles (one row per title).
    group_by="row" keeps all rows (no grouping).

    weight_mode:
      - "direct": weight by counts (more instances -> higher weight)
      - "inverse": weight by 1 / counts (more instances -> lower weight)
      - "par": no weights (each row counts as 1)
    """
    out = events.copy()
    out = _apply_grouping_and_weights(
        out,
        group_by=group_by,
        weight_mode=weight_mode,
        dedupe_subset=dedupe_subset,
        weight_col=weight_col,
    )
    out = add_sequential_event_id(out, id_col=id_col)
    return out


# ----------------------------
# Random reallocator (fast to call repeatedly)
# ----------------------------

@dataclass(frozen=True)
class GridIndex:
    """Precomputed grid arrays used by RandomReallocator."""
    cell_ids: np.ndarray               # shape (n,)
    pop: np.ndarray                    # shape (n,)
    adj: Sequence[np.ndarray]          # length n, each array of neighbor indices
    id2idx: Dict[Any, int]
    cell_id_col: str
    pop_col: str


class RandomReallocator:
    """
    Random event -> cell allocator with cached grid/neighborhood structures.

    Compared to the original notebook function, this is faster to *call repeatedly*
    because grid/neighborhood preprocessing is done once in `from_grid()`.
    """

    def __init__(self, grid: GridIndex, rng_seed: Optional[int] = 42):
        self.grid = grid
        self.rng = np.random.default_rng(rng_seed)

    @classmethod
    def from_grid(
        cls,
        grid_gdf: "gpd.GeoDataFrame",
        neighborhoods: Dict,
        *,
        pop_col: str = "population",
        cell_id_col: str = "cell_id",
        rng_seed: Optional[int] = 42,
    ) -> "RandomReallocator":
        """
        Build a reallocator and precompute neighbor adjacency arrays.

        neighborhoods: dict[cell_id] -> iterable of neighboring cell_ids (including itself is ok).
        """
        if gpd is None:
            raise ImportError("geopandas is required for RandomReallocator (grid_gdf is a GeoDataFrame).")

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

        grid = GridIndex(
            cell_ids=cell_ids,
            pop=pop,
            adj=adj,
            id2idx=id2idx,
            cell_id_col=cell_id_col,
            pop_col=pop_col,
        )
        return cls(grid=grid, rng_seed=rng_seed)

    # ---------- public API ----------

    def allocate(
        self,
        events_df: pd.DataFrame,
        *,
        group_by_category: bool = False,
        progress: str = "tqdm",                 # "tqdm" | "print" | "none"
        log_every: int = 500,
        checkpoint_path: Optional[Union[str, Path]] = None,
        checkpoint_every: int = 2000,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Allocate events randomly to cells and return:
        - allocations: [event_id, (category), assigned_cell_id, chosen_from_hub]
        - cell_summary: [cell_id, initial_demand, satisfied, unmet, times_chosen_as_hub, events_hosted]

        Notes
        -----
        - Allocation is uniform over all cells.
        - `chosen_from_hub` is set to the same cell as `assigned_cell_id` for compatibility.
        """
        if "event_id" not in events_df.columns:
            raise KeyError("events_df must contain column 'event_id'")
        if group_by_category and "category" not in events_df.columns:
            raise KeyError("group_by_category=True but 'category' not in events_df")

        # progress setup
        try:
            from tqdm.auto import tqdm  # type: ignore
            has_tqdm = True
        except Exception:
            has_tqdm = False
            if progress == "tqdm":
                progress = "print"

        def run_block(block_events: pd.DataFrame,
                      category: Optional[str]) -> pd.DataFrame:
            if block_events is None or len(block_events) == 0:
                cols = ["event_id", "assigned_cell_id", "chosen_from_hub"]
                if category is not None:
                    cols = ["event_id", "category"] + cols[1:]
                return pd.DataFrame(columns=cols)

            block_all = block_events[["event_id"] + (["category"] if category is not None else [])].copy()
            E = len(block_all)

            n = len(self.grid.cell_ids)
            if n <= 0:
                assigned = np.array([None] * E, dtype=object)
            else:
                assigned_idx = self.rng.integers(0, n, size=E, endpoint=False)
                assigned = self.grid.cell_ids[assigned_idx].astype(object)

            out = pd.DataFrame({
                "event_id": block_all["event_id"].to_numpy(),
                "assigned_cell_id": assigned,
                "chosen_from_hub": assigned,
            })
            if category is not None:
                out.insert(1, "category", block_all["category"].to_numpy())

            # checkpointing
            if checkpoint_path and E > 0:
                cp = Path(checkpoint_path)
                cp.parent.mkdir(parents=True, exist_ok=True)
                if E <= checkpoint_every:
                    out.to_csv(cp, index=False)
                else:
                    for end in range(checkpoint_every, E + checkpoint_every, checkpoint_every):
                        out.iloc[:min(end, E)].to_csv(cp, index=False)
            return out

        # allocations
        blocks = []
        if group_by_category:
            for cat, block in events_df.groupby("category", sort=False):
                blocks.append(run_block(block_events=block, category=cat))
        else:
            blocks.append(run_block(block_events=events_df, category=None))

        allocations = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(
            columns=["event_id", "assigned_cell_id", "chosen_from_hub"]
        )

        cols = ["event_id", "assigned_cell_id", "chosen_from_hub"]
        if "category" in allocations.columns:
            cols = ["event_id", "category"] + cols[1:]
        allocations = allocations[cols]

        # cell summary (serve-demand simulation, using random assignments)
        cell_summary = self._simulate_cell_summary(
            allocations,
            progress=progress,
            has_tqdm=has_tqdm,
            log_every=log_every,
        )
        return allocations, cell_summary

    # ---------- internals ----------

    def _simulate_cell_summary(
        self,
        allocations: pd.DataFrame,
        *,
        progress: str,
        has_tqdm: bool,
        log_every: int,
    ) -> pd.DataFrame:
        n = len(self.grid.cell_ids)
        pop = self.grid.pop

        E_total = len(allocations)
        total_pop = float(pop.sum()) if n > 0 else 0.0
        beta = (total_pop / E_total) if (E_total > 0 and total_pop > 0) else np.inf

        initial_demand = (pop / beta) if np.isfinite(beta) else np.zeros(n, dtype=np.float64)
        unmet = initial_demand.copy()
        satisfied = np.zeros(n, dtype=np.float64)
        events_hosted = np.zeros(n, dtype=np.int64)
        times_hub = np.zeros(n, dtype=np.int64)

        if E_total > 0 and n > 0:
            # Map chosen hubs back to indices (here hub == assigned cell)
            chosen = allocations["chosen_from_hub"].to_numpy()
            chosen_idx = np.fromiter((self.grid.id2idx.get(c, -1) for c in chosen), count=E_total, dtype=np.int64)

            iterator: Iterable[int]
            start_t = time.time()

            if progress == "tqdm" and has_tqdm:
                from tqdm.auto import tqdm  # type: ignore
                iterator = tqdm(range(E_total), desc="simulate", leave=False)
            else:
                iterator = range(E_total)

            for k in iterator:
                hub_i = int(chosen_idx[k])
                if hub_i < 0:
                    continue

                times_hub[hub_i] += 1

                serve = self.grid.adj[hub_i]  # neighbor indices including self
                u = unmet[serve]
                s = float(u.sum())
                if s > 0.0:
                    delta = u / s
                    new_u = np.maximum(u - delta, 0.0)
                    served = u - new_u
                    unmet[serve] = new_u
                    satisfied[serve] += served

                events_hosted[hub_i] += 1

                if progress == "print" and ((k + 1) % log_every == 0 or (k + 1) == E_total):
                    dt = time.time() - start_t
                    rate = (k + 1) / max(dt, 1e-6)
                    print(f"[simulate] {k+1}/{E_total} | {rate:.1f} ev/s | unmet≈{unmet.sum():.2f}")

        cell_summary = pd.DataFrame({
            self.grid.cell_id_col: self.grid.cell_ids,
            "initial_demand": initial_demand,
            "satisfied": satisfied,
            "unmet": unmet,
            "times_chosen_as_hub": times_hub,
            "events_hosted": events_hosted,
        })
        return cell_summary





# ----------------------------
# Event-unit reallocator (ad-hoc / demand-based)
# ----------------------------

@dataclass(frozen=True)
class EventUnitGridIndex:
    """Precomputed grid arrays used by EventUnitReallocator."""
    cell_ids: np.ndarray               # shape (n,)
    pop: np.ndarray                    # shape (n,)
    adj: Sequence[np.ndarray]          # length n, each array of neighbor indices
    rev_adj: Sequence[np.ndarray]      # length n, for each cell j, hubs i where j ∈ adj[i]
    id2idx: Dict[Any, int]
    cell_id_col: str
    pop_col: str


class EventUnitReallocator:
    """
    Demand-based event-unit reallocator with cached grid/neighborhood structures.

    This implements the "ad-hoc" allocator that:
    - computes an initial unmet-demand per cell from population,
    - iteratively chooses a "hub" cell by reachable unmet demand (argmax),
    - assigns the event to a cell in the hub's neighborhood (weighted),
    - serves one event-unit across the assigned cell's neighborhood.

    Output schema:
    allocations: [event_id, (category), (event_weight), assigned_cell_id, chosen_from_hub]
    cell_summary: [cell_id, initial_demand, satisfied, unmet, times_chosen_as_hub, events_hosted]
    """

    def __init__(self, grid: EventUnitGridIndex):
        self.grid = grid

    @classmethod
    def from_grid(
        cls,
        grid_gdf: Union[pd.DataFrame, "gpd.GeoDataFrame"],
        neighborhoods: Dict,
        *,
        pop_col: str = "population",
        cell_id_col: str = "cell_id",
    ) -> "EventUnitReallocator":
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

        # neighborhoods -> adjacency (indices), ensure valid & deduped
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

        # reverse adjacency: for each cell j, which hubs i include j in their neighborhood?
        rev_adj = [[] for _ in range(n)]
        for i in range(n):
            for j in adj[i]:
                rev_adj[j].append(i)
        rev_adj = [
            (np.asarray(lst, dtype=np.int32) if len(lst) else np.asarray([i], dtype=np.int32))
            for i, lst in enumerate(rev_adj)
        ]

        grid = EventUnitGridIndex(
            cell_ids=cell_ids,
            pop=pop,
            adj=adj,
            rev_adj=rev_adj,
            id2idx=id2idx,
            cell_id_col=cell_id_col,
            pop_col=pop_col,
        )
        return cls(grid=grid)

    def allocate(
        self,
        events_df: pd.DataFrame,
        *,
        group_by_category: bool = False,
        pick_with: str = "population",            # "population" | "unmet_demand"
        rng_seed: Optional[int] = 42,
        progress: str = "tqdm",                   # "tqdm" | "print" | "none"
        log_every: int = 500,
        checkpoint_path: Optional[Union[str, Path]] = None,
        checkpoint_every: int = 2000,
        event_weight_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the ad-hoc reallocation.

        Hub scoring is fixed to "reachable_unmet_demand".
        """
        if "event_id" not in events_df.columns:
            raise KeyError("events_df must contain column 'event_id'")
        if group_by_category and "category" not in events_df.columns:
            raise KeyError("group_by_category=True but 'category' not in events_df")
        if event_weight_col is not None and event_weight_col not in events_df.columns:
            raise KeyError(f"events_df must contain column '{event_weight_col}' when event_weight_col is set")
        has_category_col = "category" in events_df.columns

        # progress setup
        try:
            from tqdm.auto import tqdm  # type: ignore
            has_tqdm = True
        except Exception:
            has_tqdm = False
            if progress == "tqdm":
                progress = "print"

        rng = np.random.default_rng(rng_seed)

        cell_ids = self.grid.cell_ids
        pop = self.grid.pop
        adj = self.grid.adj
        rev_adj = self.grid.rev_adj
        id2idx = self.grid.id2idx
        n = len(cell_ids)

        def run_block(
            block_df: pd.DataFrame,
            category_label: Optional[str],
        ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
            movable_E = len(block_df)
            if event_weight_col is not None:
                weights_series = pd.to_numeric(
                    block_df[event_weight_col],
                    errors="coerce",
                ).fillna(1.0).clip(lower=0)
                total_E = float(weights_series.sum())
            else:
                total_E = float(movable_E)
            total_pop = float(pop.sum())

            cols_base = ["event_id", "assigned_cell_id", "chosen_from_hub"]
            if category_label is not None:
                cols_base = ["event_id", "category"] + cols_base[1:]
            if event_weight_col is not None:
                insert_at = 2 if category_label is not None else 1
                cols_base.insert(insert_at, event_weight_col)

            if movable_E == 0 or total_pop <= 0 or n == 0:
                # Return trivial empty output with correct schema
                if movable_E == 0:
                    return pd.DataFrame(columns=cols_base), {
                        "unmet": np.zeros(n, dtype=np.float64),
                        "satisfied": np.zeros(n, dtype=np.float64),
                        "events_hosted": np.zeros(n, dtype=np.float64),
                        "times_hub": np.zeros(n, dtype=np.float64),
                    }
                # total_pop<=0 or n==0: allocate N rows with None assigned
                out = pd.DataFrame({
                    "event_id": block_df["event_id"].to_numpy(),
                    "assigned_cell_id": np.array([None] * len(block_df), dtype=object),
                    "chosen_from_hub": np.array([None] * len(block_df), dtype=object),
                })
                if category_label is not None:
                    out.insert(1, "category", category_label)
                if event_weight_col is not None:
                    weights = block_df[event_weight_col].to_numpy() if event_weight_col in block_df.columns else np.ones(len(block_df), dtype=float)
                    insert_at = 2 if category_label is not None else 1
                    out.insert(insert_at, event_weight_col, weights)
                return out[cols_base], {
                    "unmet": np.zeros(n, dtype=np.float64),
                    "satisfied": np.zeros(n, dtype=np.float64),
                    "events_hosted": np.zeros(n, dtype=np.float64),
                    "times_hub": np.zeros(n, dtype=np.float64),
                }

            # per-block demand vectors (include fixed + movable)
            beta = total_pop / total_E if total_E > 0 else np.inf
            unmet = (pop / beta).astype(np.float64)
            satisfied = np.zeros(n, dtype=np.float64)
            events_hosted = np.zeros(n, dtype=np.float64)
            times_hub = np.zeros(n, dtype=np.float64)  # movable only

            out_event_id: List[Any] = []
            out_assigned: List[Any] = []
            out_hub: List[Any] = []
            out_weight: Optional[List[float]] = [] if event_weight_col is not None else None
            out_cat: Optional[List[Any]] = [] if category_label is not None else None

            # ---- PREP HUB SCORING FOR EVENTS ----
            if movable_E > 0:
                hub_value = np.fromiter(
                    (float(unmet[adj[i]].sum()) for i in range(n)),
                    count=n,
                    dtype=np.float64,
                )

                iterator = block_df.itertuples(index=False)
                if progress == "tqdm" and has_tqdm:
                    iterator = tqdm(
                        iterator,
                        total=movable_E,
                        desc=f"Allocating{'' if category_label is None else f' [{category_label}]'}",
                        leave=False,
                    )
                elif progress == "print":
                    start_t = time.time()

                for k, ev in enumerate(iterator, start=1):
                    if event_weight_col is None:
                        w = 1.0
                    else:
                        try:
                            w = float(getattr(ev, event_weight_col))
                        except Exception:
                            w = 1.0
                    if w < 0:
                        w = 0.0

                    # stop if nothing left (numerically)
                    if float(unmet.sum()) <= 1e-12:
                        out_event_id.append(getattr(ev, "event_id"))
                        out_assigned.append(None)
                        out_hub.append(None)
                        if out_weight is not None:
                            out_weight.append(w)
                        if out_cat is not None:
                            out_cat.append(category_label)
                        continue

                    # pick hub (argmax)
                    h = int(hub_value.argmax())
                    times_hub[h] += w

                    # pick assigned cell within hub's neighborhood
                    neigh = adj[h]
                    if pick_with == "unmet_demand":
                        weights = unmet[neigh]
                    elif pick_with == "population":
                        weights = pop[neigh]
                    else:
                        raise ValueError("pick_with must be 'population' or 'unmet_demand'")

                    if np.all(weights <= 0):
                        probs = np.full(len(neigh), 1.0 / len(neigh))
                    else:
                        probs = weights / float(weights.sum())
                    chosen_local = int(rng.choice(len(neigh), p=probs))
                    assigned_idx = int(neigh[chosen_local])

                    # serve one unit across assigned neighborhood
                    serve = adj[assigned_idx]
                    u = unmet[serve]
                    s = float(u.sum())
                    served = None
                    if s > 0.0 and w > 0.0:
                        w_eff = w if w <= s else s
                        factor = 1.0 - (w_eff / s)
                        new_u = u * factor
                        served = u - new_u
                        unmet[serve] = new_u
                        satisfied[serve] += served
                        events_hosted[assigned_idx] += w_eff

                        # update hub_value incrementally (reachable_unmet_demand)
                        diff = -served
                        for idx_local, j in enumerate(serve):
                            hubs_to_fix = rev_adj[j]
                            hub_value[hubs_to_fix] += diff[idx_local]

                    # record movable allocation
                    out_event_id.append(getattr(ev, "event_id"))
                    out_assigned.append(cell_ids[assigned_idx])
                    out_hub.append(cell_ids[h])
                    if out_weight is not None:
                        out_weight.append(w)
                    if out_cat is not None:
                        out_cat.append(category_label)

                    if progress == "print" and (k % log_every == 0 or k == movable_E):
                        dt = time.time() - start_t
                        rate = k / max(dt, 1e-6)
                        print(f"[{category_label or 'all'}] {k}/{movable_E} movable | {rate:.1f} ev/s | unmet≈{float(unmet.sum()):.2f}")

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

            # Ensure nice column order
            cols = ["event_id", "assigned_cell_id", "chosen_from_hub"]
            if category_label is not None:
                cols = ["event_id", "category"] + cols[1:]
            if event_weight_col is not None:
                insert_at = 2 if category_label is not None else 1
                cols.insert(insert_at, event_weight_col)
            out_df = out_df[cols]

            return out_df, {
                "unmet": unmet,
                "satisfied": satisfied,
                "events_hosted": events_hosted,
                "times_hub": times_hub,
            }

        # ---------------- run blocks ----------------
        blocks: List[pd.DataFrame] = []
        if group_by_category:
            event_groups = {cat: sub for cat, sub in events_df.groupby("category", sort=False)}
            for cat, block in event_groups.items():
                out_df, _ = run_block(block, category_label=cat)
                blocks.append(out_df)
        else:
            out_df, _ = run_block(events_df, category_label=None)
            blocks.append(out_df)

        allocations = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(
            columns=["event_id", "assigned_cell_id", "chosen_from_hub"]
        )

        # Restore category column if it existed in inputs but wasn't emitted
        if has_category_col and "category" not in allocations.columns:
            allocations = allocations.merge(
                events_df[["event_id", "category"]].drop_duplicates(subset=["event_id"]),
                on="event_id",
                how="left",
            )

        # Ensure column order
        cols = ["event_id", "assigned_cell_id", "chosen_from_hub"]
        if "category" in allocations.columns:
            cols = ["event_id", "category", "assigned_cell_id", "chosen_from_hub"]
        if event_weight_col is not None and event_weight_col in allocations.columns:
            insert_at = 2 if "category" in allocations.columns else 1
            cols.insert(insert_at, event_weight_col)
        allocations = allocations[cols]

        # ---------------- coherent cell summary ----------------
        if event_weight_col is not None and event_weight_col in allocations.columns:
            weights = pd.to_numeric(
                allocations[event_weight_col],
                errors="coerce",
            ).fillna(1.0).clip(lower=0).to_numpy(dtype=float)
            E_total = float(weights.sum())
        else:
            E_total = float(len(allocations))
        total_pop = float(pop.sum())
        beta = (total_pop / E_total) if (E_total > 0 and total_pop > 0) else np.inf

        initial_demand = (pop / beta) if np.isfinite(beta) else np.zeros(n, dtype=np.float64)
        unmet = initial_demand.copy()
        satisfied = np.zeros(n, dtype=np.float64)
        times_hub = np.zeros(n, dtype=np.float64)
        events_hosted = np.zeros(n, dtype=np.float64)

        chosen_from_hub_idx = np.array(
            [id2idx.get(h, -1) if pd.notna(h) else -1 for h in allocations["chosen_from_hub"]],
            dtype=np.int64,
        )
        assigned_idx = np.array(
            [id2idx.get(a, -1) if pd.notna(a) else -1 for a in allocations["assigned_cell_id"]],
            dtype=np.int64,
        )
        if event_weight_col is not None and event_weight_col in allocations.columns:
            weights = pd.to_numeric(
                allocations[event_weight_col],
                errors="coerce",
            ).fillna(1.0).clip(lower=0).to_numpy(dtype=float)
        else:
            weights = np.ones(len(allocations), dtype=float)

        for h_idx, a_idx, w in zip(chosen_from_hub_idx, assigned_idx, weights):
            if a_idx < 0:
                continue
            if 0 <= h_idx < n:
                times_hub[h_idx] += w

            serve = adj[a_idx]
            u = unmet[serve]
            s = float(u.sum())
            if s > 0.0 and w > 0.0:
                w_eff = w if w <= s else s
                factor = 1.0 - (w_eff / s)
                new_u = u * factor
                served = u - new_u
                unmet[serve] = new_u
                satisfied[serve] += served

            events_hosted[a_idx] += w

        cell_summary = pd.DataFrame({
            self.grid.cell_id_col: cell_ids,
            "initial_demand": initial_demand,
            "satisfied": satisfied,
            "unmet": unmet,
            "times_chosen_as_hub": times_hub,
            "events_hosted": events_hosted,
        })
        return allocations, cell_summary


def reallocate_events_eventunits_fast(
    grid_gdf: Union[pd.DataFrame, "gpd.GeoDataFrame"],
    neighborhoods: Dict,
    events_df: pd.DataFrame,
    pop_col: str = "population",
    cell_id_col: str = "cell_id",
    group_by_category: bool = False,
    pick_with: str = "population",
    rng_seed: Optional[int] = 42,
    progress: str = "tqdm",
    log_every: int = 500,
    checkpoint_path: Optional[Union[str, Path]] = None,
    checkpoint_every: int = 2000,
    event_weight_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backward-compatible one-call wrapper around EventUnitReallocator.

    Prefer repeated calls with a cached reallocator:
        er = EventUnitReallocator.from_grid(...)
        allocations, summary = er.allocate(...)
    """
    er = EventUnitReallocator.from_grid(
        grid_gdf,
        neighborhoods,
        pop_col=pop_col,
        cell_id_col=cell_id_col,
    )
    return er.allocate(
        events_df,
        group_by_category=group_by_category,
        pick_with=pick_with,
        rng_seed=rng_seed,
        progress=progress,
        log_every=log_every,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        event_weight_col=event_weight_col,
    )




# ----------------------------
# Optional: one-call pipeline
# ----------------------------

def run_random_reallocation(
    *,
    events_path: Union[str, Path],
    grid_path: Union[str, Path],
    neighborhoods_path: Union[str, Path],
    polygon_path: Optional[Union[str, Path]] = None,
    pop_col: str = "population",
    cell_id_col: str = "cell_id",
    rng_seed: int = 42,
    group_by_category: bool = False,
    progress: str = "none",
) -> Tuple[pd.DataFrame, pd.DataFrame, "gpd.GeoDataFrame"]:
    """
    Convenience wrapper that loads inputs from disk and returns (allocations, cell_summary, events_gdf).

    This is the closest "one-liner" replacement for the original notebook workflow.

    Notes
    -----
    - `events_path`, `grid_path`, `neighborhoods_path` are expected to be pickles.
    - If your events file is not a pickle, load it yourself and use `prep_events()` + `RandomReallocator`.
    """
    events = load_pickle(events_path)
    grid = load_pickle(grid_path)
    neigh = load_pickle(neighborhoods_path)

    events_gdf = prep_events(events, polygon_path=polygon_path, id_col="event_id")
    rr = RandomReallocator.from_grid(grid, neigh, pop_col=pop_col, cell_id_col=cell_id_col, rng_seed=rng_seed)
    allocations, summary = rr.allocate(events_gdf, group_by_category=group_by_category, progress=progress)
    return allocations, summary, events_gdf
