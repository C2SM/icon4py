#!/usr/bin/env python
"""Vertically integrated moisture, vertical wind and energy of the AES bubble experiment.

Writes two figures to <output_dir>/plots (or --plot-dir):

  aes_bubble_timeseries.png  (horizontal, cell-area weighted means; time series)
    a) vertically integrated moisture, as changes since the first output (kg m-2):
       water vapour path IWV = int rho qv dz, cloud path int rho (qc + qi) dz,
       precipitating water path int rho (qr + qs + qg) dz, precipitation accumulated
       at the surface, and their sum (constant if water is conserved and there is no
       surface evaporation);
    b) domain maximum and minimum of the vertical wind w (m s-1);
    c) energy changes since the first output (J m-2): internal, potential, kinetic and
       total energy, and the total plus the energy carried out through the surface by
       precipitation;
    d) kinetic energy (J m-2).

  aes_bubble_hovmoeller.png  (x-time diagrams, averaged over y)
    IWV - IWV(t0), condensate path int rho (qc + qi + qr + qs + qg) dz, and the column
    extreme of w (w at the level where |w| is largest in the column).

Energies use the same definitions as plot_warm_bubble_budgets.py (icon4py's
compute_energy.py). The surface precipitation is read from the 'precipitation' group
if present; it is exact only if that group is written every time step.

Usage (inside the uenv, with the icon4py venv):
    .venv/bin/python plot_aes_bubble.py output_aes_bubble/output_sample
    .venv/bin/python plot_aes_bubble.py output_dry_aes_bubble/output_sample --csv
"""

import argparse
import contextlib
import io
import pathlib

import matplotlib


matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from plot_warm_bubble import cell_coordinates, open_output
from plot_warm_bubble_budgets import (
    column_budgets,
    full_level_heights,
    layer_thickness,
    surface_outflow,
)


# categorical slots in fixed order (validated reference palette), plus ink for totals
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
SEQUENTIAL = mcolors.LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f2322"]
)

CLOUD = ("qc", "qi")
PRECIP = ("qr", "qs", "qg")


def water_path(data: xr.Dataset, names: tuple[str, ...], rho_dz: np.ndarray) -> np.ndarray:
    """int rho q dz (kg m-2) per (time, cell) for the sum of the tracers present."""
    present = [n for n in names if n in data]
    if not present:
        return np.zeros((rho_dz.shape[0], rho_dz.shape[2]))
    q = sum(data[n].values for n in present)
    return np.sum(q * rho_dz, axis=1)


def column_extreme(w: np.ndarray) -> np.ndarray:
    """w at the level of largest |w| in each column; w is (time, level, cell)."""
    k = np.argmax(np.abs(w), axis=1)
    return np.take_along_axis(w, k[:, None, :], axis=1)[:, 0, :]


def x_bins(grid: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Bin index of each cell along x (two triangle columns per bin) and the bin edges (m)."""
    x, _, lx, _ = cell_coordinates(grid)
    dx = float(np.median(grid["edge_length"].values))
    nbins = max(1, round(lx / dx))
    x0 = float(x.min()) - 0.25 * dx
    edges = x0 + np.arange(nbins + 1) * (lx / nbins)
    return np.clip(((x - x0) // (lx / nbins)).astype(int), 0, nbins - 1), edges


def bin_mean(field: np.ndarray, idx: np.ndarray, weights: np.ndarray, nbins: int) -> np.ndarray:
    """Area-weighted mean of a (time, cell) field over the cells of each x bin."""
    num = np.zeros((field.shape[0], nbins))
    den = np.bincount(idx, weights=weights, minlength=nbins)
    for b in range(nbins):
        sel = idx == b
        num[:, b] = field[:, sel] @ weights[sel]
    return num / np.where(den > 0, den, np.nan)


def style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.axhline(0.0, color=MUTED, lw=0.6, zorder=0)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(fontsize=8, frameon=False)


def main() -> None:  # noqa: PLR0912, PLR0915 [too-many-branches/statements] linear plotting script
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
        "--precip-group", default="precipitation", help="basename of the precipitation group"
    )
    p.add_argument(
        "--plot-dir", type=pathlib.Path, default=None, help="default: <output_dir>/plots"
    )
    p.add_argument(
        "--end-time", type=float, default=None, help="only plot outputs up to this time (min)"
    )
    p.add_argument("--every", type=int, default=1, metavar="N", help="use every N-th output time")
    p.add_argument("--csv", action="store_true", help="also write the time series as CSV")
    args = p.parse_args()
    if args.every < 1:
        p.error("--every must be >= 1")

    data, grid = open_output(args.output_dir, args.group)
    t_all = data["time"].values
    keep = np.arange(t_all.size)
    if args.end_time is not None:
        keep = keep[(t_all - t_all[0]) / 60.0 <= args.end_time]
    data = data.isel(time=keep[:: args.every]).load()
    t_out = data["time"].values
    t_min = (t_out - t_out[0]) / 60.0
    print(f"using {t_out.size} of {t_all.size} output times")

    area = grid["cell_area"].values
    weights = area / area.sum()
    dz = layer_thickness(data)
    z = full_level_heights(data)

    # vertically integrated moisture, per (time, cell)
    rho_dz = data["air_density"].values * dz[None, :, None]
    iwv = water_path(data, ("qv",), rho_dz)
    cloud = water_path(data, CLOUD, rho_dz)
    precip = water_path(data, PRECIP, rho_dz)

    # vertical wind on half levels, (time, half_level, cell)
    w = data["upward_air_velocity"].values

    # energies (area means, J m-2); column_budgets prints a line per call, silence it
    energy = {
        k: np.empty(t_out.size) for k in ("internal_energy", "potential_energy", "kinetic_energy")
    }
    with contextlib.redirect_stdout(io.StringIO()):
        for it in range(t_out.size):
            budget = column_budgets(data, it, dz, z, weights)
            for k, series in energy.items():
                series[it] = budget[k]
    ie, pe, ke = energy["internal_energy"], energy["potential_energy"], energy["kinetic_energy"]
    te = ie + pe + ke

    try:
        precip_ds, _ = open_output(args.output_dir, args.precip_group)
        sfc_water, sfc_energy = surface_outflow(precip_ds, t_out, weights)
        sfc_water = np.concatenate(([0.0], np.cumsum(sfc_water)))
        sfc_energy = np.concatenate(([0.0], np.cumsum(sfc_energy)))
        if not np.all(np.isfinite(sfc_energy)):
            sfc_energy = None
    except FileNotFoundError:
        print(f"no '{args.precip_group}' output found: plotting without surface precipitation")
        sfc_water = sfc_energy = None

    mean = lambda f: f @ weights  # noqa: E731  (time, cell) -> (time,)
    d_iwv = mean(iwv) - mean(iwv)[0]
    cwp, pwp = mean(cloud), mean(precip)
    total_water = (
        d_iwv + (cwp - cwp[0]) + (pwp - pwp[0]) + (sfc_water if sfc_water is not None else 0.0)
    )

    plot_dir = args.plot_dir or args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.prop_cycle": plt.cycler(color=[BLUE])})

    # ---- time series ----
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True, constrained_layout=True)
    ax = axs[0, 0]
    ax.plot(t_min, d_iwv, color=BLUE, lw=1.5, label=r"$\Delta$IWV (vapour)")
    ax.plot(t_min, cwp - cwp[0], color=ORANGE, lw=1.5, label=r"$\Delta$ cloud path (qc+qi)")
    ax.plot(t_min, pwp - pwp[0], color=AQUA, lw=1.5, label=r"$\Delta$ precip. path (qr+qs+qg)")
    if sfc_water is not None:
        ax.plot(t_min, sfc_water, color=YELLOW, lw=1.5, label="accumulated surface precip.")
    ax.plot(t_min, total_water, color=INK, lw=1.2, ls="--", label="sum")
    style(
        ax,
        f"a) vertically integrated moisture   (IWV$_0$ = {mean(iwv)[0]:.2f} kg m$^{{-2}}$)",
        r"change since $t_0$ (kg m$^{-2}$)",
    )

    ax = axs[0, 1]
    ax.plot(t_min, w.max(axis=(1, 2)), color=ORANGE, lw=1.5, label="max w")
    ax.plot(t_min, w.min(axis=(1, 2)), color=BLUE, lw=1.5, label="min w")
    style(ax, "b) vertical wind, domain extremes", r"w (m s$^{-1}$)")

    ax = axs[1, 0]
    ax.plot(t_min, ie - ie[0], color=BLUE, lw=1.5, label=r"$\Delta E_\mathrm{I}$ internal")
    ax.plot(t_min, pe - pe[0], color=ORANGE, lw=1.5, label=r"$\Delta E_\mathrm{P}$ potential")
    ax.plot(t_min, ke - ke[0], color=AQUA, lw=1.5, label=r"$\Delta E_\mathrm{K}$ kinetic")
    ax.plot(t_min, te - te[0], color=INK, lw=1.5, label=r"$\Delta E$ total (atmosphere)")
    if sfc_energy is not None:
        ax.plot(
            t_min,
            te - te[0] + sfc_energy,
            color=INK,
            lw=1.2,
            ls="--",
            label=r"$\Delta E$ + surface precip. energy outflow",
        )
    style(ax, "c) energy", r"change since $t_0$ (J m$^{-2}$)")

    ax = axs[1, 1]
    ax.plot(t_min, ke, color=AQUA, lw=1.5, label=r"$E_\mathrm{K}$")
    style(ax, "d) kinetic energy", r"J m$^{-2}$")
    ax.get_legend().remove()  # single series, the title names it

    for ax in axs[1]:
        ax.set_xlabel("time (min)", color=MUTED)
    out = plot_dir / "aes_bubble_timeseries.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"figure: {out}")

    # ---- Hovmoeller (x-time), averaged over y ----
    idx, edges = x_bins(grid)
    nb = edges.size - 1
    x_km = edges / 1e3
    t_edges = np.concatenate(([t_min[0]], 0.5 * (t_min[1:] + t_min[:-1]), [t_min[-1]]))
    panels = [
        (
            bin_mean(iwv - iwv[:1], idx, weights, nb),
            r"IWV $-$ IWV($t_0$) (kg m$^{-2}$)",
            DIVERGING.reversed(),
            True,
        ),
        (
            bin_mean(cloud + precip, idx, weights, nb),
            r"condensate path (kg m$^{-2}$)",
            SEQUENTIAL,
            False,
        ),
        (
            bin_mean(column_extreme(w), idx, weights, nb),
            r"column extreme w (m s$^{-1}$)",
            DIVERGING,
            True,
        ),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(13, 5), sharey=True, constrained_layout=True)
    for ax, (field, label, cmap, symmetric) in zip(axs, panels):
        if symmetric:
            vmax = float(np.nanmax(np.abs(field))) or 1.0
            norm = mcolors.Normalize(-vmax, vmax)
        else:
            norm = mcolors.Normalize(0.0, float(np.nanmax(field)) or 1.0)
        mesh = ax.pcolormesh(x_km, t_edges, field, cmap=cmap, norm=norm, shading="flat")
        fig.colorbar(mesh, ax=ax, orientation="horizontal", label=label, shrink=0.9, pad=0.02)
        ax.set_xlabel("x (km)", color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(MUTED)
    axs[0].set_ylabel("time (min)", color=MUTED)
    fig.suptitle("y-mean x-time diagrams", x=0.01, ha="left", fontsize=10, color=INK)
    out = plot_dir / "aes_bubble_hovmoeller.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"figure: {out}")

    print(f"total energy relative change over the run: {te[-1] / te[0] - 1:+.3e}")
    print(f"total water change incl. surface precipitation: {total_water[-1]:+.3e} kg m-2")

    if args.csv:
        columns = {
            "time_min": t_min,
            "iwv": mean(iwv),
            "cloud_path": cwp,
            "precip_path": pwp,
            "w_max": w.max(axis=(1, 2)),
            "w_min": w.min(axis=(1, 2)),
            "internal_energy": ie,
            "potential_energy": pe,
            "kinetic_energy": ke,
            "total_energy": te,
        }
        if sfc_water is not None:
            columns["accumulated_surface_precip"] = sfc_water
        if sfc_energy is not None:
            columns["accumulated_surface_energy_outflow"] = sfc_energy
        csv = plot_dir / "aes_bubble_timeseries.csv"
        np.savetxt(
            csv,
            np.column_stack(list(columns.values())),
            delimiter=",",
            header=",".join(columns),
            comments="",
        )
        print(f"csv: {csv}")


if __name__ == "__main__":
    main()
