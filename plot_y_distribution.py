#!/usr/bin/env python
"""Distribution in y of the vertically integrated wind speed and hydrometeors.

The cells of the torus are grouped into --nbins bins of equal width in y (the bins tile
the periodic domain exactly). For every cell column two vertical integrals are computed:

  wind speed   : sum_k |V|_k dz_k                         (m^2 s^-1)
                 with --density-weighted: sum_k rho_k |V|_k dz_k   (kg m^-1 s^-1)
                 |V| is the horizontal wind speed sqrt(u^2 + v^2); --include-w adds w
                 (interpolated to full levels) to it.
  hydrometeors : sum_k rho_k (qc + qi + qr + qs + qg)_k dz_k       (kg m^-2)
                 i.e. the total condensate water path; missing tracers count as zero.

Per bin, the cell-area weighted mean of the column integrals is plotted as a bar, and
the whiskers span the minimum to maximum column in the bin. The integrals can be
restricted to the levels below --zmax.

Output (in --plot-dir, default <output_dir>/plots):
  y_distribution_NNNN.png     bar charts per selected output time (fixed axis limits)
  y_distribution_hovmoller.png  bin mean as a function of time and y for all times
  y_distribution.gif          with --gif

Usage (inside the uenv, with the icon4py venv activated):
    python plot_y_distribution.py output_sample
    python plot_y_distribution.py output_sample --nbins 6 --stride 10 --gif
    python plot_y_distribution.py output_sample --density-weighted --zmax 12
"""

import argparse
import pathlib

import matplotlib


matplotlib.use("Agg")
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import xarray as xr


HYDROMETEORS = ("qc", "qi", "qr", "qs", "qg")


# --------------------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------------------
def open_output(output_dir: pathlib.Path, basename: str) -> tuple[xr.Dataset, xr.Dataset]:
    """Open the field-group output (netCDF or zarr) and the UGRID grid file next to it."""
    candidates = sorted(output_dir.glob(f"{basename}_*.nc")) + sorted(
        output_dir.glob(f"{basename}_*.zarr")
    )
    if not candidates:
        raise FileNotFoundError(f"no '{basename}_*.nc' or '{basename}_*.zarr' in {output_dir}")
    if len(candidates) > 1:
        print(f"note: several files found, using {candidates[0].name}")
    path = candidates[0]
    if path.suffix == ".zarr":
        data = xr.open_dataset(path, engine="zarr", decode_times=False)
    else:
        data = xr.open_dataset(path, decode_times=False)
    grids = sorted(output_dir.glob("*_ugrid.nc"))
    if not grids:
        raise FileNotFoundError(f"no '*_ugrid.nc' grid file in {output_dir}")
    return data, xr.open_dataset(grids[0])


# --------------------------------------------------------------------------------------
# column integrals
# --------------------------------------------------------------------------------------
def layer_thickness(data: xr.Dataset, zmax_m: float | None) -> np.ndarray:
    """(level, 1) thickness (m) of the full levels; zero above zmax (by full-level height)."""
    h = data["height"].values  # half levels, top to bottom
    dz = h[:-1] - h[1:]
    if zmax_m is not None:
        dz = np.where(0.5 * (h[:-1] + h[1:]) <= zmax_m, dz, 0.0)
    return dz[:, None]


def wind_speed_integral(
    data: xr.Dataset, it: int, dz: np.ndarray, *, density_weighted: bool, include_w: bool
) -> np.ndarray:
    u = data["eastward_wind"].isel(time=it).values
    v = data["northward_wind"].isel(time=it).values
    speed2 = u**2 + v**2
    if include_w:
        w_half = data["upward_air_velocity"].isel(time=it).values
        speed2 = speed2 + (0.5 * (w_half[:-1] + w_half[1:])) ** 2
    weight = dz * data["air_density"].isel(time=it).values if density_weighted else dz
    return (np.sqrt(speed2) * weight).sum(axis=0)


def hydrometeor_path(data: xr.Dataset, it: int, dz: np.ndarray, present: list[str]) -> np.ndarray:
    if not present:
        return np.zeros(data.sizes["cell"])
    q = sum(data[n].isel(time=it).values for n in present)
    return (data["air_density"].isel(time=it).values * q * dz).sum(axis=0)


# --------------------------------------------------------------------------------------
# binning in y
# --------------------------------------------------------------------------------------
def y_bins(y: np.ndarray, ly: float, nbins: int) -> tuple[np.ndarray, np.ndarray]:
    """Bin edges (m) tiling the periodic y-period and the bin index of every cell.

    The edges are centred on the cell centres, so the gap between the last and the first
    row of cells (across the periodic boundary) is split evenly between the two ends.
    """
    y_start = y.min() - 0.5 * (ly - (y.max() - y.min()))
    edges = y_start + np.linspace(0.0, ly, nbins + 1)
    index = np.clip(((y - y_start) // (ly / nbins)).astype(int), 0, nbins - 1)
    return edges, index


def bin_statistics(
    column: np.ndarray, index: np.ndarray, area: np.ndarray, nbins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-weighted mean, minimum and maximum of a per-cell field in every bin."""
    area_sum = np.bincount(index, weights=area, minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(index, weights=area * column, minlength=nbins) / area_sum
    lo = np.full(nbins, np.nan)
    hi = np.full(nbins, np.nan)
    np.fmin.at(lo, index, column)
    np.fmax.at(hi, index, column)
    return mean, lo, hi


# --------------------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------------------
def draw_bars(ax, edges_km, stats, *, color, ylim, ylabel):
    mean, lo, hi = stats
    centres = 0.5 * (edges_km[:-1] + edges_km[1:])
    widths = 0.9 * np.diff(edges_km)
    ax.bar(centres, mean, width=widths, color=color, alpha=0.75, edgecolor="k", linewidth=0.5)
    ax.errorbar(
        centres, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor="k", capsize=3, lw=0.8
    )
    ax.set_xlim(edges_km[0], edges_km[-1])
    ax.set_xticks(edges_km)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", lw=0.4, alpha=0.5)


def write_gif(frames: list[pathlib.Path], gif: pathlib.Path, fps: int) -> None:
    images = [plt.imread(f) for f in frames]
    fig = plt.figure(figsize=(images[0].shape[1] / 100, images[0].shape[0] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(images[0])
    anim = mpl_animation.FuncAnimation(
        fig, lambda i: (im.set_data(images[i]), [im])[1], frames=len(images), blit=True
    )
    anim.save(gif, writer=mpl_animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"animation: {gif}")


def main() -> None:  # noqa: PLR0915 [too-many-statements] linear plotting script
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "output_dir", type=pathlib.Path, help="driver output directory (e.g. output_sample)"
    )
    p.add_argument(
        "--group", default="winds", help="basename of the field group with the 3D fields"
    )
    p.add_argument(
        "--plot-dir", type=pathlib.Path, default=None, help="default: <output_dir>/plots"
    )
    p.add_argument("--nbins", type=int, default=8, help="number of bins in y")
    p.add_argument("--stride", type=int, default=1, help="bar charts for every n-th output time")
    p.add_argument(
        "--times", type=int, nargs="*", default=None, help="explicit output time indices"
    )
    p.add_argument("--zmax", type=float, default=None, help="integrate only below this height (km)")
    p.add_argument(
        "--density-weighted",
        action="store_true",
        help="integrate rho |V| dz instead of |V| dz for the wind speed",
    )
    p.add_argument("--include-w", action="store_true", help="include w in the wind speed")
    p.add_argument(
        "--gif", action="store_true", help="also write an animated GIF of the bar charts"
    )
    p.add_argument("--fps", type=int, default=4)
    args = p.parse_args()

    data, grid = open_output(args.output_dir, args.group)
    y = grid["cell_circumcenter_cartesian_y"].values
    ly = float(grid.attrs["domain_height"])
    area = grid["cell_area"].values
    t = data["time"].values
    t_min = (t - t[0]) / 60.0
    nt = data.sizes["time"]

    present = [n for n in HYDROMETEORS if n in data]
    missing = [n for n in HYDROMETEORS if n not in data]
    if missing:
        print(f"note: {', '.join(missing)} not in the '{args.group}' group, counted as zero")

    edges, index = y_bins(y, ly, args.nbins)
    edges_km = np.round(edges * 1e-3, 6) + 0.0  # no "-0.0" tick label
    counts = np.bincount(index, minlength=args.nbins)
    print(
        f"{np.unique(np.round(y, 3)).size} rows of cell centres in y, cells per bin: {counts.tolist()}"
    )
    if np.any(counts == 0):
        print("warning: some bins contain no cells (shown empty); reduce --nbins")

    dz = layer_thickness(data, None if args.zmax is None else args.zmax * 1e3)
    if args.density_weighted:
        wind_label = r"$\int \rho\,|V|\,dz$  (kg m$^{-1}$ s$^{-1}$)"
    else:
        wind_label = r"$\int |V|\,dz$  (m$^2$ s$^{-1}$)"
    wind_name = "|V| incl. w" if args.include_w else "horizontal |V|"
    hydro_label = (
        r"$\int \rho\,(" + "+".join(f"q_{n[1]}" for n in HYDROMETEORS) + r")\,dz$  (kg m$^{-2}$)"
    )

    # binned statistics for all output times (also used for the fixed axis limits)
    wind = np.empty((3, nt, args.nbins))
    hydro = np.empty((3, nt, args.nbins))
    for it in range(nt):
        col_wind = wind_speed_integral(
            data, it, dz, density_weighted=args.density_weighted, include_w=args.include_w
        )
        col_hydro = hydrometeor_path(data, it, dz, present)
        wind[:, it] = bin_statistics(col_wind, index, area, args.nbins)
        hydro[:, it] = bin_statistics(col_hydro, index, area, args.nbins)

    plot_dir = args.plot_dir or args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    zinfo = f", below {args.zmax:g} km" if args.zmax is not None else ""

    # bar charts
    its = args.times if args.times else list(range(0, nt, args.stride))
    wind_ylim = (0.0, 1.05 * max(float(np.nanmax(wind[2, its])), 1e-12))
    hydro_ylim = (0.0, 1.05 * max(float(np.nanmax(hydro[2, its])), 1e-12))
    frames = []
    for n, it in enumerate(its):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
        draw_bars(ax1, edges_km, wind[:, it], color="tab:blue", ylim=wind_ylim, ylabel=wind_label)
        ax1.set_title(f"vertically integrated {wind_name}")
        draw_bars(
            ax2, edges_km, hydro[:, it], color="tab:green", ylim=hydro_ylim, ylabel=hydro_label
        )
        ax2.set_title("vertically integrated hydrometeors (condensate water path)")
        ax2.set_xlabel("y (km)   (bin edges)")
        fig.suptitle(
            f"t = {t_min[it]:.0f} min    bars: area-weighted mean per y bin, "
            f"whiskers: min-max column{zinfo}",
            fontsize=10,
        )
        out = plot_dir / f"y_distribution_{it:04d}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        frames.append(out)
        print(f"[{n + 1}/{len(its)}] {out}")

    # time-y diagram of the bin means over all output times
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
    t_edges = (
        np.concatenate([[t_min[0]], 0.5 * (t_min[:-1] + t_min[1:]), [t_min[-1]]])
        if nt > 1
        else np.array([t_min[0] - 0.5, t_min[0] + 0.5])
    )
    for ax, field, cmap, label, title in (
        (ax1, wind[0], "viridis", wind_label, f"integrated {wind_name}"),
        (ax2, hydro[0], "Blues", hydro_label, "integrated hydrometeors"),
    ):
        pc = ax.pcolormesh(edges_km, t_edges, field, cmap=cmap, shading="flat")
        fig.colorbar(pc, ax=ax, pad=0.01).set_label(label)
        ax.set_xticks(edges_km)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
        ax.set_xlabel("y (km)   (bin edges)")
        ax.set_title(title)
    ax1.set_ylabel("time (min)")
    fig.suptitle(f"area-weighted mean per y bin ({args.nbins} bins{zinfo})", fontsize=11)
    out = plot_dir / "y_distribution_hovmoller.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(out)

    if args.gif and frames:
        write_gif(frames, plot_dir / "y_distribution.gif", args.fps)


if __name__ == "__main__":
    main()
