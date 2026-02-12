import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
from typing import Optional
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# =========================
# FILES
# =========================
HEX_TO_NIL_CSV = "hex_to_nil.csv"
NIL_SHP = "NIL_WM.shp"

# Define datasets you want to plot.
# key: output prefix; path: pickle path; title: used in plot titles
DATASETS = {
    "baseline_2024": {
        "path": "cityscore_walk_15min_4.5kmh_alpha0.08_gdf_2024_final.pkl",
        "title": "Baseline - 2024",
        "kind": "pkl",
    },
    "realloc_2024": {
        "path": "nil_summary_reallocated_2024_final_v3_par_50_runs.csv",
        "title": "Reallocated - 2024",
        "kind": "csv",
    },
}

# Which datasets to run (subset of DATASETS keys)
RUN_KEYS = ["baseline_2024", "realloc_2024"]


# =========================
# CONFIG
# =========================
PALETTE_3x3 = [
    "#2c3f2a", "#4f7449", "#78a770",
    "#6a4b27", "#8f7147", "#b99d73",
    "#a55c24", "#cc8345", "#efb36e",
]

# Scatter layout (match notebook)
SCATTER_FIGSIZE = (9.6, 7.2)
cube_mode = "manual"   # "loc" | "anchor" | "manual"
cube_size = ("18%", "18%")
cube_loc = "upper left"
cube_anchor = (0.20, 0.20)
cube_bounds = [0.06, 0.78, 0.18, 0.18]

# Optional labeled scatter (case-insensitive match on NIL name)
SCATTER_LABEL_QUERY = "Q.RE GALLARATESE"  # set to None to disable
SCATTER_LABEL_SUFFIX = "gallaratese"

DOT_SIZE = 72
EDGE_LW = 0.35
GRID_LW = 1.1
GRID_ALPHA = 0.6

LABEL_FONT_SIZE = 5.0
LABEL_X_OFFSET = 0.02
LABEL_Y_OFFSET = 0.02

MISSING_COLOR = "#eeeeee"
BORDER_COLOR = "black"

xcol = "cityscore_pw_mean"
ycol = "pop_total"

# If set, use this dataset's tertiles as the reference for all runs.
# Set to None to compute tertiles per dataset.
TERTILES_REF_KEY = "baseline_2024"  # e.g. "baseline_2024" or "realloc_2024"


# =========================
# HELPERS
# =========================
def read_hex_to_nil(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["hex_id"] = df["hex_id"].astype(str)
    return df


def load_year_gdf(path: str, hex_to_nil: pd.DataFrame) -> gpd.GeoDataFrame:
    with open(path, "rb") as f:
        gdf = pickle.load(f)

    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf.crs or "EPSG:3857")
    gdf["hex_id"] = gdf["hex_id"].astype(str)

    gdf = gdf.merge(
        hex_to_nil[["hex_id", "NIL", "population"]],
        on="hex_id",
        how="left",
    )
    return gdf


def load_nil_boundaries(path: str) -> gpd.GeoDataFrame:
    nils = gpd.read_file(path)
    nils = gpd.GeoDataFrame(nils, geometry="geometry", crs=nils.crs or "EPSG:4326")
    return nils


def compute_nil_stats(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    data = gdf.dropna(subset=["NIL"]).copy()
    pop = data["population"].fillna(0)
    data["_w_cityscore"] = data["cityscore"] * pop

    grouped = data.groupby("NIL", as_index=False).agg(
        pop_total=("population", "sum"),
        w_cityscore=("_w_cityscore", "sum"),
        cityscore_mean=("cityscore", "mean"),
    )

    grouped["cityscore_pw_mean"] = np.where(
        grouped["pop_total"] > 0,
        grouped["w_cityscore"] / grouped["pop_total"],
        grouped["cityscore_mean"],
    )

    return grouped[["NIL", "pop_total", "cityscore_pw_mean"]]


def load_nil_summary_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["NIL", "pop_total", "cityscore_pw_mean"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    out = df[needed].copy()
    out["NIL"] = out["NIL"].astype(str)
    return out


def compute_tertiles(df: pd.DataFrame, cols: list[str]) -> dict:
    thresholds = {}
    for col in cols:
        s = df[col].dropna()
        q1, q2 = s.quantile([1 / 3, 2 / 3], interpolation="linear").values
        thresholds[col] = (q1, q2)
    return thresholds


def bin_series(s: pd.Series, q1: float, q2: float) -> pd.Series:
    return pd.cut(s, bins=[-np.inf, q1, q2, np.inf], labels=[0, 1, 2]).astype("Int64")


def add_bivariate_bins(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    qx1, qx2 = thresholds[xcol]
    qy1, qy2 = thresholds[ycol]

    out = df.copy()
    out["_sbin"] = bin_series(out[xcol], qx1, qx2)
    out["_pbin"] = bin_series(out[ycol], qy1, qy2)
    out["_bivar"] = out["_pbin"] * 3 + out["_sbin"]
    color_map = dict(enumerate(PALETTE_3x3))
    out["_color"] = out["_bivar"].map(color_map)
    return out


def add_bivariate_cube_inset(ax,
                             palette9,
                             width="26%", height="26%",
                             loc="upper right",
                             bbox_to_anchor=None,
                             bbox_transform=None,
                             manual_bounds=None,
                             label_x="CityScore",
                             label_y="Population"):
    if manual_bounds is not None:
        inax = ax.inset_axes(manual_bounds)
    else:
        if (isinstance(width, str) and "%" in width) or (isinstance(height, str) and "%" in height):
            if isinstance(bbox_to_anchor, (tuple, list)) and len(bbox_to_anchor) == 2:
                bbox_to_anchor = (*bbox_to_anchor, 1, 1)
                if bbox_transform is None:
                    bbox_transform = ax.transAxes
        inax = inset_axes(
            ax,
            width=width,
            height=height,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            bbox_transform=bbox_transform,
            borderpad=0.8,
        )
    inax.set_aspect("equal")

    for pbin in range(3):
        for sbin in range(3):
            idx = pbin * 3 + sbin
            r = Rectangle((sbin, pbin), 1, 1,
                          facecolor=palette9[idx], edgecolor="k", linewidth=0.6)
            inax.add_patch(r)

    inax.set_xlim(0, 3)
    inax.set_ylim(0, 3)
    inax.set_xticks([0.5, 1.5, 2.5])
    inax.set_yticks([0.5, 1.5, 2.5])
    inax.set_xticklabels(["Low", "Mid", "High"], fontsize=8)
    inax.set_yticklabels(["Low", "Mid", "High"], fontsize=8)
    for v in [1, 2]:
        inax.axvline(v, color="k", lw=0.6)
        inax.axhline(v, color="k", lw=0.6)
    inax.set_xlabel(label_x, fontsize=8, labelpad=2)
    inax.set_ylabel(label_y, fontsize=8, labelpad=2)
    for s in inax.spines.values():
        s.set_linewidth(0.8)
    return inax


def label_high_cityscore(ax, df: pd.DataFrame, thresholds: dict):
    qx1, qx2 = thresholds[xcol]
    qy1, qy2 = thresholds[ycol]

    x_span = (qx2 - qx1) if qx2 > qx1 else df[xcol].max() - df[xcol].min()
    y_span = (qy2 - qy1) if qy2 > qy1 else df[ycol].max() - df[ycol].min()

    label_df = df[df["_sbin"] == 2].dropna(subset=[xcol, ycol, "NIL"])
    for _, r in label_df.iterrows():
        x, y = r[xcol], r[ycol]
        lx = x + LABEL_X_OFFSET * x_span
        ly = y + LABEL_Y_OFFSET * y_span
        ax.plot([x, lx], [y, ly], color="0.3", lw=0.6, alpha=0.8, zorder=3)
        ax.text(
            lx,
            ly,
            r["NIL"],
            fontsize=LABEL_FONT_SIZE,
            color="black",
            ha="left",
            va="bottom",
            zorder=4,
            path_effects=[pe.withStroke(linewidth=2, foreground="white", alpha=0.8)],
        )


def label_selected_nil(ax, df: pd.DataFrame, thresholds: dict, nil_query: str):
    """
    Label only NILs whose name matches the query (case-insensitive).
    Matches if name starts with or contains the query.
    """
    if not nil_query:
        return

    q = nil_query.lower().strip()
    if not q:
        return

    qx1, qx2 = thresholds[xcol]
    qy1, qy2 = thresholds[ycol]

    x_span = (qx2 - qx1) if qx2 > qx1 else df[xcol].max() - df[xcol].min()
    y_span = (qy2 - qy1) if qy2 > qy1 else df[ycol].max() - df[ycol].min()

    mask = df["NIL"].str.lower().str.startswith(q) | df["NIL"].str.lower().str.contains(q)
    label_df = df[mask].dropna(subset=[xcol, ycol, "NIL"])

    for _, r in label_df.iterrows():
        x, y = r[xcol], r[ycol]
        lx = x + LABEL_X_OFFSET * x_span
        ly = y + LABEL_Y_OFFSET * y_span
        ax.plot([x, lx], [y, ly], color="0.3", lw=0.6, alpha=0.8, zorder=3)
        ax.text(
            lx,
            ly,
            r["NIL"],
            fontsize=LABEL_FONT_SIZE,
            color="black",
            ha="left",
            va="bottom",
            zorder=4,
            path_effects=[pe.withStroke(linewidth=2, foreground="white", alpha=0.8)],
        )


def plot_scatter(ax, stats_binned, thresholds, title, label_query: Optional[str] = None):
    qx1, qx2 = thresholds[xcol]
    qy1, qy2 = thresholds[ycol]

    # ---- Scatter
    plot_df = stats_binned.dropna(subset=[xcol, ycol, "_color"])
    ax.scatter(
        plot_df[xcol],
        plot_df[ycol],
        c=plot_df["_color"],
        s=DOT_SIZE,
        edgecolor="k",
        linewidth=EDGE_LW,
        alpha=0.95,
        zorder=2,
    )
    ax.set_xlabel("CityScore (PopW mean)")
    ax.set_ylabel("Population")
    ax.set_title(f"Population vs PopW CityScore across Milan NILs ({title})")

    ax.axvline(qx1, color="0.25", lw=GRID_LW, ls="--", alpha=GRID_ALPHA, zorder=1)
    ax.axvline(qx2, color="0.25", lw=GRID_LW, ls="--", alpha=GRID_ALPHA, zorder=1)
    ax.axhline(qy1, color="0.25", lw=GRID_LW, ls="--", alpha=GRID_ALPHA, zorder=1)
    ax.axhline(qy2, color="0.25", lw=GRID_LW, ls="--", alpha=GRID_ALPHA, zorder=1)

    if cube_mode == "manual":
        add_bivariate_cube_inset(ax, PALETTE_3x3, manual_bounds=cube_bounds)
    elif cube_mode == "anchor":
        add_bivariate_cube_inset(
            ax,
            PALETTE_3x3,
            width=cube_size[0],
            height=cube_size[1],
            loc="upper left",
            bbox_to_anchor=cube_anchor,
            bbox_transform=ax.transAxes,
        )
    else:
        add_bivariate_cube_inset(
            ax,
            PALETTE_3x3,
            width=cube_size[0],
            height=cube_size[1],
            loc=cube_loc,
        )

    if label_query:
        label_selected_nil(ax, plot_df, thresholds, label_query)


def plot_map(ax, gdf_binned, title):
    gdf_plot = gdf_binned.copy()
    gdf_plot.plot(
        ax=ax,
        color=gdf_plot["_color"].fillna(MISSING_COLOR),
        edgecolor=BORDER_COLOR,
        linewidth=0.25,
        alpha=1.0,
    )
    ax.set_title(title)
    ax.set_axis_off()


def main():
    nils = load_nil_boundaries(NIL_SHP)

    if not RUN_KEYS:
        raise ValueError("RUN_KEYS is empty. Add dataset keys to run.")

    needs_hex = any(DATASETS[k].get("kind", "pkl") == "pkl" for k in RUN_KEYS)
    hex_to_nil = read_hex_to_nil(HEX_TO_NIL_CSV) if needs_hex else None

    stats_by_key = {}
    for key in RUN_KEYS:
        if key not in DATASETS:
            raise ValueError(f"RUN_KEYS item '{key}' not found in DATASETS.")
        path = DATASETS[key]["path"]
        kind = DATASETS[key].get("kind", "pkl")
        if kind == "pkl":
            gdf = load_year_gdf(path, hex_to_nil)
            stats_by_key[key] = compute_nil_stats(gdf)
        elif kind == "csv":
            stats_by_key[key] = load_nil_summary_csv(path)
        else:
            raise ValueError(f"Unknown dataset kind '{kind}' for key '{key}'.")

    if TERTILES_REF_KEY is not None:
        if TERTILES_REF_KEY not in stats_by_key:
            raise ValueError(f"TERTILES_REF_KEY={TERTILES_REF_KEY} not found in RUN_KEYS.")
        thresholds_ref = compute_tertiles(stats_by_key[TERTILES_REF_KEY], [xcol, ycol])
    else:
        thresholds_ref = None

    for key, stats in stats_by_key.items():
        title = DATASETS[key]["title"]
        thresholds = thresholds_ref or compute_tertiles(stats, [xcol, ycol])
        stats_binned = add_bivariate_bins(stats, thresholds)

        gdf_binned = nils.merge(
            stats_binned[["NIL", "_sbin", "_pbin", "_bivar", "_color"]],
            on="NIL",
            how="left",
        )

        # Scatter PNG (no labels by default)
        fig_scatter, ax_scatter = plt.subplots(figsize=SCATTER_FIGSIZE)
        plot_scatter(ax_scatter, stats_binned, thresholds, title, label_query=None)
        fig_scatter.tight_layout()
        fig_scatter.savefig(f"{key}_scatter.png", dpi=300, bbox_inches="tight")
        plt.close(fig_scatter)

        # Scatter PNG with selected NIL label (optional)
        if SCATTER_LABEL_QUERY:
            fig_scatter_l, ax_scatter_l = plt.subplots(figsize=SCATTER_FIGSIZE)
            plot_scatter(ax_scatter_l, stats_binned, thresholds, title, label_query=SCATTER_LABEL_QUERY)
            fig_scatter_l.tight_layout()
            fig_scatter_l.savefig(
                f"{key}_scatter_{SCATTER_LABEL_SUFFIX}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_scatter_l)

        # Map PNG
        fig_map, ax_map = plt.subplots(figsize=(5.8, 5.8))
        plot_map(ax_map, gdf_binned, title)
        fig_map.tight_layout()
        fig_map.savefig(f"{key}_map.png", dpi=300, bbox_inches="tight")
        plt.close(fig_map)


if __name__ == "__main__":
    main()
