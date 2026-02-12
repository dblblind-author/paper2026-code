import sys, importlib.util
from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import geopandas as gpd

# HELPERS
# ---------- hex_id cleanup ----------

def normalize_hex_id_series(s: pd.Series) -> pd.Series:
    """
    Coerce any dtype (float64/int64/object) to clean string hex IDs:
    - Numeric-looking -> integer-like strings.
    - Else keep as string. Returns pandas 'string' dtype.
    """
    if s is None:
        return pd.Series([], dtype="string")
    s = s.copy()
    # Attempt numeric
    num = pd.to_numeric(s, errors="coerce")
    out = s.astype("string")
    mask = num.notna()
    if mask.any():
        as_int = num[mask].round(0).astype("Int64")
        out.loc[mask] = as_int.astype("string")
    out = out.str.replace(r"\.0$", "", regex=True).str.strip()
    return out.astype("string")


# ---------- drop columns in cols ----------
def drop_if_exists(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    keep = [c for c in df.columns if c not in cols]
    if len(keep) != len(df.columns):
        return df[keep].copy()
    return df


# ---------- load the toolkit ----------
def _load_cityscore_once(toolkit_path: str = "CityScoreToolkit_plain_v2.py"):
    if "cityscore" in sys.modules:
        return sys.modules["cityscore"]
    spec = importlib.util.spec_from_file_location("cityscore", toolkit_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not find CityScore toolkit at: {toolkit_path}")
    cityscore = importlib.util.module_from_spec(spec)
    sys.modules["cityscore"] = cityscore
    spec.loader.exec_module(cityscore)
    if hasattr(cityscore, "set_osmnx_cache"):
        cityscore.set_osmnx_cache(use_cache=True, log_console=False)
    return cityscore


if "cityscore" in sys.modules:
    del sys.modules["cityscore"]
CITYSCORE = _load_cityscore_once("CityScoreToolkit_plain_v2.py")


# ---------- dataclass declaration ----------
@dataclass
class StaticISO:
    AOI_ll: gpd.GeoDataFrame
    grid_noid: gpd.GeoDataFrame  # geometry ONLY -> for toolkit
    grid_ids: gpd.GeoDataFrame  # ['hex_id','geometry'] -> for attach
    isochrones: gpd.GeoDataFrame


# ---------- get ['hex_id','geometry'] ----------
def _ensure_hex_id(AOI_hex: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create/normalize 'hex_id' and return ONLY ['hex_id','geometry'].
    """
    H = AOI_hex.copy()
    # Pick an existing ID if present; else index
    for cand in ("hex_id", "h3_id", "h3", "cell_id", "id"):
        if cand in H.columns:
            H["hex_id"] = H[cand]
            break
    else:
        H["hex_id"] = H.index
    H["hex_id"] = normalize_hex_id_series(H["hex_id"])
    return H[["hex_id", "geometry"]].copy()


# ---------- attach hex_id with spatial join ----------
def _attach_hex_id_by_sjoin(cs: gpd.GeoDataFrame, grid_ids: gpd.GeoDataFrame) -> pd.Series:
    """
    Attach hex_id using centroid spatial join. Returns a 'string' series.
    This works for both points and polygons (taking their centroid to calculate the location).

    """
    # Drop any hex_id on the left side to avoid GeoPandas' internal merge on same-name columns
    left = drop_if_exists(cs, ["hex_id"]).copy()
    if left.crs != grid_ids.crs:
        left = left.to_crs(grid_ids.crs)
    cent = left.copy()
    cent["geometry"] = cent.geometry.centroid

    right = grid_ids.copy()
    right["hex_id"] = normalize_hex_id_series(right["hex_id"])

    try:
        hit = gpd.sjoin(cent, right, how="left", predicate="within")
        hex_out = normalize_hex_id_series(hit["hex_id"])
    except Exception as e:
        # Last-resort fallback: manual nearest/contains without merging on 'hex_id'
        # Build a series initialized as <NA>
        hex_out = pd.Series([pd.NA] * len(cent), index=cent.index, dtype="string")
        # Vectorized contains is fast enough for city-scale grids
        contains = gpd.sjoin(cent, right, how="left", predicate="within")[["index_right"]]
        # Map indices -> hex_id
        mapper = right["hex_id"]
        matched = contains["index_right"].dropna().astype("Int64")
        hex_out.loc[contains.index] = normalize_hex_id_series(mapper.iloc[matched.fillna(-1).astype(int)].values)
    return hex_out


# ---------- compute CityScore for all events in a dataframe ----------
def fast_score_day(
    events_gdf: gpd.GeoDataFrame,
    static_iso: StaticISO,
    save_path: Optional[str] = None,
    name: Optional[str] = None,
    alpha: float = 0.08,
):
    """
    Compute CityScore for one day's events using precomputed isochrones.
    We pass the grid WITHOUT hex_id to the toolkit, then attach hex_id afterward.
    """

    # Build POIs and ensure they don't carry hex_id
    pois = CITYSCORE.get_pois(static_iso.AOI_ll, official_gdf=drop_if_exists(events_gdf, ["hex_id"]))

    # Use grid WITHOUT hex_id to avoid any internal merge on that column
    CityScore, Diagnostic = CITYSCORE.make_intersection(
        static_iso.grid_noid,
        pois,
        gdf=static_iso.isochrones,
        add_hex_steps=True,
        score_from="steps",
        alpha=alpha,
    )

    cs = CityScore.copy()
    # Attach hex_id from the clean grid_ids
    hex_series = _attach_hex_id_by_sjoin(cs, static_iso.grid_ids)
    cs["hex_id"] = normalize_hex_id_series(hex_series)

    if save_path and name:
        cs.to_file(save_path, layer=f"cityscore_{name}", driver="GPKG")
        Diagnostic.to_file(save_path, layer=f"diagnostic_{name}", driver="GPKG")

    return cs, Diagnostic

    if events_gdf.empty:
        return {
            "n_events": 0,
            "n_points": 0,
            "result": gpd.GeoDataFrame(),
            "diagnostic": gpd.GeoDataFrame(),
        }

    # Ensure CRS is WGS84
    gdf_ll = events_gdf
    if gdf_ll.crs is None:
        gdf_ll = gdf_ll.set_crs("EPSG:4326", allow_override=True)
    elif str(gdf_ll.crs).lower() != "epsg:4326":
        gdf_ll = gdf_ll.to_crs("EPSG:4326")

    # Must be points
    if not (gdf_ll.geom_type.isin(["Point", "MultiPoint"]).all()):
        raise ValueError("events_gdf must contain Point/MultiPoint geometries.")

    CityScore, Diagnostic = fast_score_day(
        gdf_ll,
        static_iso,
        save_path=(gpkg_out if save else None),
        name=(name if save else None),
    )

    return {
        "n_events": int(len(events_gdf)),
        "n_points": int(len(gdf_ll)),
        "result": CityScore,
        "diagnostic": Diagnostic,
    }


# ---------- generate unique name for a group ----------
def _unique_name(group: str, year: int, datestr: str) -> str:
    return f"{group}_{year}_{datestr.replace('-', '')}"


# ---------- compute CityScore for a dict of events.----------
from typing import Dict, Any, Optional
import geopandas as gpd
import pandas as pd


def run_year_fast(
        index_year: Dict[str, Any],
        static_iso: StaticISO,
        save_each: bool = False,
        gpkg_out: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Orchestrate CityScore computation for a yearly dictionary. Like the one you obtained from `build_day_event_index`.
    Expects `index_year` to have:
      {
        "year": 2024,
        "weekend_holidays": { "YYYY-MM-DD": [event_dict, ...], ... },
        "weekdays":         { "YYYY-MM-DD": [event_dict, ...], ... }
      }

    Parameters
    ----------
    index_year : dict
        Year index with 'year', 'weekend_holidays', 'weekdays'.
    static_iso : StaticISO
        Precomputed AOI/grid/isochrones container (from `precompute_iso`).
    save_each : bool, default False
        If True, saves each day's CityScore and Diagnostic to `gpkg_out`.
    gpkg_out : str or None
        GeoPackage output path if `save_each=True`.

    Returns
    -------
    dict
        {
          "year": int,
          "weekend_holidays": [ {date,name,n_events,n_points,result,diagnostic}, ... ],
          "weekdays":         [ { ... }, ... ],
          "failures":         [ {date, group, error}, ... ]
        }
    """
    year = index_year.get("year")
    if year is None:
        raise ValueError("index_year must contain a 'year' key (e.g., 2024).")

    if save_each and not gpkg_out:
        raise ValueError("gpkg_out must be provided when save_each=True.")

    print(f"\n=== Using precomputed static ISO (shared) for {year} ===")

    out = {"year": year, "weekend_holidays": [], "weekdays": [], "failures": []}

    for group in ("weekend_holidays", "weekdays"):
        day_map = index_year.get(group, {}) or {}
        day_keys = sorted(day_map.keys())
        total = len(day_keys)
        print(f"[{year}] {group}: {total} day(s)")

        for i, d in enumerate(day_keys, start=1):
            events = day_map[d]
            # relies on your existing helper to turn list[dict] -> GeoDataFrame
            gdf: gpd.GeoDataFrame = _events_to_gdf(events)

            print(f"  - [{i:>3}/{total}] {d}: {len(events)} rows → {len(gdf)} pts", end="")

            if gdf.empty:
                print("  [skip: no geocoded points]")
                continue

            name = _unique_name(group, year, d)  # your existing naming helper

            try:
                cs, diag = fast_score_day(
                    gdf,
                    static_iso,
                    save_path=(gpkg_out if save_each else None),
                    name=(name if save_each else None),
                )
                print("  [ok]")
                out[group].append({
                    "date": d,
                    "name": name,
                    "n_events": int(len(events)),
                    "n_points": int(len(gdf)),
                    "result": cs,
                    "diagnostic": diag,
                })
            except Exception as e:
                print(f"  [FAILED] {e}")
                out["failures"].append({"date": d, "group": group, "error": repr(e)})

    return out
