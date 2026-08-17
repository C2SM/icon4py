#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Plot the tracer blob experiment: qv panels on the torus and error/mass time series.

Reads the standalone driver's ``icon4py_output_*.nc`` plus the ``*_ugrid.nc`` companion
from an output directory and writes two PNG figures next to them. The translated
reference is the analytic disc moved by u0 * t (torus-periodic in x); the blob options
must match the TracerBlobConfig used in the run (defaults match its defaults).
"""

from __future__ import annotations

import pathlib

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import netCDF4 as nc
import numpy as np
import typer


matplotlib.use("Agg")

cli = typer.Typer()


def _load_qv(output_dir: pathlib.Path) -> tuple:
    # qv as (time, cell, level) plus elapsed seconds per frame
    files = sorted(output_dir.rglob("icon4py_output_*.nc"))
    if not files:
        raise FileNotFoundError(f"no icon4py_output_*.nc under {output_dir}")
    frames, times = [], []
    for path in files:
        with nc.Dataset(path) as ds:
            var = ds.variables["qv"]
            axes = [var.dimensions.index(name) for name in ("time", "cell", "level")]
            frames.append(np.transpose(np.asarray(var[:]), axes))
            time_var = ds.variables["time"]
            times.append(nc.num2date(time_var[:], time_var.units))
    qv = np.concatenate(frames, axis=0)
    dates = np.concatenate(times, axis=0)
    elapsed = np.array([(d - dates[0]).total_seconds() for d in dates])
    return qv, elapsed


def _load_mesh(output_dir: pathlib.Path) -> tuple:
    # vertex coordinates, cell-vertex connectivity, cell centers, domain extents
    try:
        path = next(output_dir.rglob("*_ugrid.nc"))
    except StopIteration as err:
        raise FileNotFoundError(f"no *_ugrid.nc under {output_dir}") from err
    with nc.Dataset(path) as ds:
        vx = np.asarray(ds.variables["cartesian_x_vertices"][:])
        vy = np.asarray(ds.variables["cartesian_y_vertices"][:])
        triangles = np.asarray(ds.variables["vertex_of_cell"][:])  # zero-based, (cell, 3)
        cx = np.asarray(ds.variables["cell_circumcenter_cartesian_x"][:])
        cy = np.asarray(ds.variables["cell_circumcenter_cartesian_y"][:])
        domain_length = float(ds.getncattr("domain_length"))
        domain_height = float(ds.getncattr("domain_height"))
    return vx, vy, triangles, cx, cy, domain_length, domain_height


def _translated_disc(
    cx, cy, *, u0, t, blob_x, blob_y, blob_radius, blob_amplitude, domain_length, domain_height
):
    # analytic initial disc translated by u0 * t, torus-periodic
    center_x = (blob_x + u0 * t) % domain_length
    dx = (cx - center_x + 0.5 * domain_length) % domain_length - 0.5 * domain_length
    dy = (cy - blob_y + 0.5 * domain_height) % domain_height - 0.5 * domain_height
    return np.where(dx**2 + dy**2 <= blob_radius**2, blob_amplitude, 0.0)


@cli.command(help=__doc__)
def plot_tracer_blob(
    output_dir: pathlib.Path,
    *,
    u0: float = 20.0,
    level: int = 0,
    blob_x: float | None = None,
    blob_y: float | None = None,
    blob_radius: float | None = None,
    blob_amplitude: float = 1e-3,
) -> None:
    qv, elapsed = _load_qv(output_dir)
    vx, vy, triangles, cx, cy, domain_length, domain_height = _load_mesh(output_dir)

    # defaults mirror TracerBlobConfig: domain center, quarter of the smaller extent
    blob_x = 0.5 * domain_length if blob_x is None else blob_x
    blob_y = 0.5 * domain_height if blob_y is None else blob_y
    blob_radius = 0.25 * min(domain_length, domain_height) if blob_radius is None else blob_radius

    disc = lambda t: _translated_disc(  # noqa: E731 [lambda-assignment]
        cx,
        cy,
        u0=u0,
        t=t,
        blob_x=blob_x,
        blob_y=blob_y,
        blob_radius=blob_radius,
        blob_amplitude=blob_amplitude,
        domain_length=domain_length,
        domain_height=domain_height,
    )

    # hide triangles that wrap around the periodic seams
    tri = mtri.Triangulation(vx, vy, triangles)
    wraps = (np.ptp(vx[triangles], axis=1) > 0.5 * domain_length) | (
        np.ptp(vy[triangles], axis=1) > 0.5 * domain_height
    )
    tri.set_mask(wraps)

    qv_first = qv[0, :, level]
    qv_last = qv[-1, :, level]
    translated = disc(elapsed[-1])
    difference = qv_last - translated

    fig, axs = plt.subplots(4, 1, figsize=(10, 7), constrained_layout=True, sharex=True)
    vmax = max(qv_first.max(), qv_last.max(), blob_amplitude)
    panels = [
        (qv_first, f"initial qv (t = {elapsed[0]:.0f} s)", "Blues", 0.0, vmax),
        (qv_last, f"final qv (t = {elapsed[-1]:.0f} s)", "Blues", 0.0, vmax),
        (translated, "translated initial (x + u0 t mod Lx)", "Blues", 0.0, vmax),
        (difference, "final - translated", "RdBu_r", -np.abs(difference).max(), None),
    ]
    for ax, (values, title, cmap, vmin, vmax_panel) in zip(axs, panels, strict=True):
        image = ax.tripcolor(
            tri, values, cmap=cmap, vmin=vmin, vmax=-vmin if vmax_panel is None else vmax_panel
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_ylabel("y [m]")
        fig.colorbar(image, ax=ax, label="qv [kg/kg]")
    axs[-1].set_xlabel("x [m]")
    panels_path = output_dir / "tracer_blob_panels.png"
    fig.savefig(panels_path, dpi=150)

    # time series: L2 error vs the translated disc and relative tracer mass drift
    # (cell areas are uniform on the torus, so plain sums are area-weighted)
    references = np.stack([disc(t) for t in elapsed])
    l2_error = np.linalg.norm(qv[:, :, level] - references, axis=1) / np.linalg.norm(
        references, axis=1
    )
    mass = qv.sum(axis=(1, 2))
    mass_drift = mass / mass[0] - 1.0

    fig, (ax_error, ax_mass) = plt.subplots(
        2, 1, figsize=(8, 6), constrained_layout=True, sharex=True
    )
    ax_error.plot(elapsed, l2_error, linewidth=2)
    ax_error.set_ylabel("relative L2 error")
    ax_error.set_title("qv error vs translated initial disc")
    ax_mass.plot(elapsed, mass_drift, linewidth=2)
    ax_mass.set_ylabel("relative mass drift")
    ax_mass.set_title("total tracer mass")
    ax_mass.set_xlabel("t [s]")
    for ax in (ax_error, ax_mass):
        ax.grid(alpha=0.3)
    timeseries_path = output_dir / "tracer_blob_timeseries.png"
    fig.savefig(timeseries_path, dpi=150)

    print(f"wrote {panels_path} and {timeseries_path}")


if __name__ == "__main__":
    cli()
