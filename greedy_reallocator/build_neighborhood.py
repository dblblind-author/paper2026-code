"""
Build and save a neighborhood dict from the H3 level 9 GPKG.

The output represents adjacency (touching cells) on the H3 level 9 grid.
"""

from pathlib import Path

import geopandas as gpd

from greedy_reallocator import load_pickle, save_pickle


GRID_PKL = Path("hex_pop.pkl")
GPKG_PATH = Path("milano_h3_r9.gpkg")
GPKG_LAYER = "milano_h3_r9"
GPKG_ID_COL = "h3_id"
CELL_ID_COL = "hex_id"
OUT_PATH = Path("adjacent_neigh.pkl")


def _ensure_numpy_pickle_compat() -> None:
    # Compatibility shim for pickles created with newer numpy module paths.
    try:
        import sys
        import numpy.core as _np_core
        sys.modules.setdefault("numpy._core", _np_core)
        if hasattr(_np_core, "_multiarray_umath"):
            sys.modules.setdefault("numpy._core._multiarray_umath", _np_core._multiarray_umath)
    except Exception:
        pass


def _build_neighborhood_from_gpkg(grid_gdf, gpkg_path: Path) -> dict:
    gpkg = gpd.read_file(gpkg_path, layer=GPKG_LAYER)
    if gpkg.crs != grid_gdf.crs:
        gpkg = gpkg.to_crs(grid_gdf.crs)

    # Map gpkg ids to grid cell ids using centroid containment (fallback to nearest)
    gpkg_centroids = gpkg[[GPKG_ID_COL, "geometry"]].copy()
    gpkg_centroids["geometry"] = gpkg_centroids.geometry.centroid
    grid_cells = grid_gdf[[CELL_ID_COL, "geometry"]].copy()

    join = gpd.sjoin(gpkg_centroids, grid_cells, predicate="within", how="left")
    missing = join[join[CELL_ID_COL].isna()].copy()
    if len(missing) > 0:
        nearest = gpd.sjoin_nearest(
            missing.drop(columns=[CELL_ID_COL]),
            grid_cells,
            how="left",
            distance_col="__dist",
        )
        join.loc[missing.index, CELL_ID_COL] = nearest[CELL_ID_COL].to_numpy()

    id_map = dict(zip(join[GPKG_ID_COL], join[CELL_ID_COL]))

    # Build adjacency using polygon touches
    sindex = gpkg.sindex
    neighbors = {}
    for idx, row in gpkg.iterrows():
        geom = row.geometry
        candidates = list(sindex.query(geom, predicate="touches"))
        candidates.append(idx)  # ensure self
        neigh_ids = gpkg.iloc[list(dict.fromkeys(candidates))][GPKG_ID_COL].tolist()
        mapped = [id_map.get(h) for h in neigh_ids if id_map.get(h) is not None]
        self_id = id_map.get(row[GPKG_ID_COL])
        if self_id is not None and self_id not in mapped:
            mapped.append(self_id)
        if self_id is not None:
            neighbors[self_id] = sorted(set(mapped))

    return neighbors


def main() -> None:
    _ensure_numpy_pickle_compat()
    grid = load_pickle(GRID_PKL)
    neigh = _build_neighborhood_from_gpkg(grid, GPKG_PATH)
    save_pickle(neigh, OUT_PATH)
    print(f"Saved neighborhood to {OUT_PATH}")


if __name__ == "__main__":
    main()
