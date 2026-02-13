import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def _load_gdf(path: Path):
    with path.open("rb") as f:
        obj = pickle.load(f)
    return obj.get("result") if isinstance(obj, dict) and "result" in obj else obj


def _find_pickle(input_dir: Path, year: int, mode: str) -> Optional[Path]:
    pattern = f"cityscore_{mode}_*_{year}.pkl"
    matches = sorted(input_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise RuntimeError(f"Multiple matches for {year} {mode}: {names}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CityScore facets (years x methods) from local pickles."
    )
    parser.add_argument(
        "--input_dir",
        default=".",
        help="Directory containing the cityscore_<mode>_*_<year>.pkl files.",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=15,
        help="Minutes label for output naming (does not filter files).",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2023, 2024],
        help="Years to plot (row order).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["walk", "bike"],
        help="Modes to plot (column order).",
    )
    parser.add_argument(
        "--out_png",
        default=None,
        help="Output PNG path. Defaults to plots_cityscore/cityscore_facets_<minutes>min_2x2.png",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)

    # Unified typography for all figure text
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
    })

    availability = {}
    xmins, xmaxs, ymins, ymaxs = [], [], [], []

    for year in args.years:
        for mode in args.modes:
            path = _find_pickle(input_dir, year, mode)
            if path is None:
                availability[(year, mode)] = None
                print(f"⚠️ Missing file for {year} {mode} in {input_dir}")
                continue
            gdf = _load_gdf(path)
            if gdf is None or "cityscore" not in gdf.columns or gdf.empty:
                availability[(year, mode)] = None
                print(f"⚠️ Invalid/empty GDF or missing 'cityscore' in: {path}")
                continue
            availability[(year, mode)] = gdf
            xmin, ymin, xmax, ymax = gdf.total_bounds
            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)

    if not xmins:
        raise RuntimeError("No valid data found among the expected pickles.")

    XMIN, XMAX = float(np.min(xmins)), float(np.max(xmaxs))
    YMIN, YMAX = float(np.min(ymins)), float(np.max(ymaxs))

    fig, axes = plt.subplots(
        nrows=len(args.years),
        ncols=len(args.modes),
        figsize=(9, 7.5),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    for i, year in enumerate(args.years):
        for j, mode in enumerate(args.modes):
            ax = axes[i, j]
            gdf = availability.get((year, mode))
            if gdf is None:
                ax.axis("off")
                continue
            gdf.plot(
                column="cityscore",
                cmap="viridis",
                vmin=0,
                vmax=100,
                linewidth=0,
                ax=ax,
                legend=False,
            )
            ax.set_xlim(XMIN, XMAX)
            ax.set_ylim(YMIN, YMAX)
            ax.set_aspect("equal")
            ax.axis("off")

    # Column headers (top row)
    for j, mode in enumerate(args.modes):
        axes[0, j].set_title(
            mode.capitalize(),
            fontsize=11,
            fontweight="bold",
            pad=6,
        )

    # Row labels (years on the left, vertical)
    for i, year in enumerate(args.years):
        axes[i, 0].text(
            XMIN - (XMAX - XMIN) * 0.03,
            (YMIN + YMAX) / 2,
            str(year),
            va="center",
            ha="right",
            fontsize=11,
            fontweight="bold",
            rotation=90,
        )

    # Shared colorbar (bottom)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=100))
    sm._A = []

    plt.subplots_adjust(
        bottom=0.12, top=0.94, left=0.09, right=0.98, hspace=0.05, wspace=0.05
    )
    cax = fig.add_axes([0.15, 0.055, 0.7, 0.025])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("CityScore", fontsize=11, fontweight="bold", labelpad=8)
    cbar.ax.tick_params(labelsize=9, width=0.8, length=3, pad=2)

    out_png = args.out_png
    if out_png is None:
        out_dir = input_dir / "plots_cityscore"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"cityscore_facets_{args.minutes}min_2x2.png"
    else:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"✅ Saved PNG: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
