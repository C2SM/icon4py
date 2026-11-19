#!/usr/bin/env python
"""Plot the cloud evolution of the warm bubble experiment as vertical cross sections.

For every selected output time, one figure with two panels is written:
  left : x-z section at y = y0, cloud condensate (qc + qi) shaded, wind vectors (u, w)
  right: y-z section at x = x0, cloud condensate (qc + qi) shaded, wind vectors (v, w)
Precipitating hydrometeors (qr + qs + qg) are drawn as contour lines.

A second figure per output time (thermo_sections_NNNN.png) shows, on the same two
sections, the temperature and the air density as anomalies from the horizontal-mean
profile of the first output time, so both the bubble and any net warming or cooling of
a level since the start are visible. Isotherms (every 10 K) and the wind vectors are
overlaid. Temperature is diagnosed like icon4py's
compute_virtual_temperature_and_temperature: T = theta_v * exner / (1 + (Rv/Rd - 1) qv - q_cond).
Use --no-thermo to skip it.

A third figure per output time (moisture_sections_NNNN.png) shows, on the same two
sections, the water vapour qv as an anomaly from the horizontal-mean profile of the first
output time (grey lines: full qv) and the relative humidity (black line: saturation,
100 %). The relative humidity is the ratio of the vapour partial pressure to the
saturation vapour pressure, RH = rho qv Rv T / e_sat(T), with icon4py's Tetens formula
over liquid water; --rh-ice uses ice below the melting point instead, as in the
relative-humidity profile of the warm bubble initial condition. Use --no-moist to skip it.

The ICON torus grid is triangular and doubly periodic: cell-centre values are linearly
interpolated onto the section lines (Delaunay triangulation of the periodically tiled
cell centres; the interpolation weights are computed once and reused for every field
and time).

Usage (inside the uenv, with the icon4py venv activated):
    python plot_warm_bubble.py output_sample
    python plot_warm_bubble.py output_sample --stride 5 --gif
    python plot_warm_bubble.py output_sample --follow --zmax 14
    python plot_warm_bubble.py output_sample --x0 10 --y0 -5   # section through (10 km, -5 km)
"""

import argparse
import pathlib

import matplotlib


matplotlib.use("Agg")
import matplotlib.animation as mpl_animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402

from icon4py.model.common import constants
from icon4py.model.common.physics.thermodynamics import compute_pressure


CLOUD_VARS = ("qc", "qi")
PRECIP_VARS = ("qr", "qs", "qg")
ISOTHERM_SPACING = 10.0  # K
# %, bins centred on multiples of 5 % so a uniform RH (e.g. the initial 70 %) does not
# flicker between two colours; supersaturation (> 102.5 %) is shown as 'max'
RH_LEVELS = np.arange(2.5, 105.0, 5.0)
RH_TICKS = np.arange(10.0, 101.0, 10.0)


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
        print(f"note: several files found, using {candidates[0].name} (use --group-file to pick)")
    data = _open(candidates[0])
    grids = sorted(output_dir.glob("*_ugrid.nc"))
    if not grids:
        raise FileNotFoundError(f"no '*_ugrid.nc' grid file in {output_dir}")
    return data, xr.open_dataset(grids[0])


def _open(path: pathlib.Path) -> xr.Dataset:
    if path.suffix == ".zarr":
        return xr.open_dataset(path, engine="zarr", decode_times=False)
    return xr.open_dataset(path, decode_times=False)


def cell_coordinates(grid: xr.Dataset) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Planar cell-centre coordinates (m) and the periods of the torus in x and y."""
    x = grid["cell_circumcenter_cartesian_x"].values
    y = grid["cell_circumcenter_cartesian_y"].values
    lx = float(grid.attrs["domain_length"])
    ly = float(grid.attrs["domain_height"])
    return x, y, lx, ly


def full_level_heights(data: xr.Dataset) -> np.ndarray:
    """Heights (m) of the full levels, ordered like the 'level' dimension (top to bottom)."""
    h = data["height"].values
    return 0.5 * (h[:-1] + h[1:])


# --------------------------------------------------------------------------------------
# interpolation onto a section line
# --------------------------------------------------------------------------------------
def section_weights(
    x: np.ndarray, y: np.ndarray, lx: float, ly: float, px: np.ndarray, py: np.ndarray
) -> sp.csr_matrix:
    """Sparse (npoints x ncells) matrix of linear interpolation weights on the torus."""
    ncell = x.size
    shifts = [(i * lx, j * ly) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    tx = np.concatenate([x + dx for dx, _ in shifts])
    ty = np.concatenate([y + dy for _, dy in shifts])
    owner = np.tile(np.arange(ncell), len(shifts))

    tri = Delaunay(np.column_stack([tx, ty]))
    # wrap the target points into the central copy of the domain
    x_min, y_min = x.min(), y.min()
    qx = (px - x_min) % lx + x_min
    qy = (py - y_min) % ly + y_min
    q = np.column_stack([qx, qy])
    simplex = tri.find_simplex(q)
    if np.any(simplex < 0):
        raise RuntimeError("section point outside the triangulation")
    trans = tri.transform[simplex]
    bary2 = np.einsum("nij,nj->ni", trans[:, :2, :], q - trans[:, 2, :])
    bary = np.column_stack([bary2, 1.0 - bary2.sum(axis=1)])
    cols = owner[tri.simplices[simplex]]
    rows = np.repeat(np.arange(q.shape[0]), 3)
    return sp.csr_matrix((bary.ravel(), (rows, cols.ravel())), shape=(q.shape[0], ncell))


def on_section(w: sp.csr_matrix, field_lev_cell: np.ndarray) -> np.ndarray:
    """(level, cell) field -> (level, npoints) section."""
    return (w @ field_lev_cell.T).T


# --------------------------------------------------------------------------------------
# fields
# --------------------------------------------------------------------------------------
def sum_vars(data: xr.Dataset, names: tuple[str, ...], it: int) -> np.ndarray:
    present = [n for n in names if n in data]
    if not present:
        return np.zeros((data.sizes["level"], data.sizes["cell"]))
    return sum(data[n].isel(time=it).values for n in present)


def w_on_full_levels(data: xr.Dataset, it: int) -> np.ndarray:
    w_half = data["upward_air_velocity"].isel(time=it).values
    return 0.5 * (w_half[:-1] + w_half[1:])


def temperature(data: xr.Dataset, it: int) -> np.ndarray:
    """Temperature (K) from theta_v, exner and the tracers, like icon4py's dycore diagnostics."""
    t_v = (
        data["virtual_potential_temperature"].isel(time=it).values
        * data["exner_function"].isel(time=it).values
    )
    qv = sum_vars(data, ("qv",), it)
    q_cond = sum_vars(data, CLOUD_VARS + PRECIP_VARS, it)
    return t_v / (1.0 + constants.RV_O_RD_MINUS_1 * qv - q_cond)


def relative_humidity(data: xr.Dataset, it: int, over_ice: bool = False) -> np.ndarray:
    """Relative humidity (%) as vapour partial pressure over saturation vapour pressure.

    e = rho_v Rv T with rho_v = rho qv, i.e. the inverse of qv_from_relative_humidity used
    by the warm bubble initial condition. With over_ice, the saturation vapour pressure is
    taken over ice below the melting point (with the same temperature clamp as there).
    """
    temp = temperature(data, it)
    vapour_pressure = (
        data["air_density"].isel(time=it).values
        * sum_vars(data, ("qv",), it)
        * constants.RV
        * temp
    )
    saturation_pressure = compute_pressure.sat_pres_water(temp)
    if over_ice:
        saturation_pressure = np.where(
            temp <= constants.MELTING_TEMPERATURE,
            compute_pressure.sat_pres_ice(
                np.maximum(temp, constants.MINIMUM_TEMPERATURE_ICE_SATURATION)
            ),
            saturation_pressure,
        )
    return 100.0 * vapour_pressure / saturation_pressure


def horizontal_mean(field_lev_cell: np.ndarray, area_weights: np.ndarray) -> np.ndarray:
    """Cell-area weighted horizontal mean profile of a (level, cell) field."""
    return field_lev_cell @ area_weights


def symmetric_levels(vmax: float, n: int = 21) -> np.ndarray:
    vmax = vmax if vmax > 0.0 else 1.0
    return matplotlib.ticker.MaxNLocator(nbins=n - 1, symmetric=True).tick_values(-vmax, vmax)


def locate_cloud_centre(data: xr.Dataset, x: np.ndarray, y: np.ndarray, its) -> tuple[float, float]:
    """Cell with the largest vertically summed cloud + precip condensate over the given times."""
    best, best_cell = -np.inf, None
    for it in its:
        col = sum_vars(data, CLOUD_VARS + PRECIP_VARS, it).sum(axis=0)
        ic = int(np.argmax(col))
        if col[ic] > best:
            best, best_cell = col[ic], ic
    if best <= 0.0:  # no cloud at all: use the domain centre
        return float(0.5 * (x.min() + x.max())), float(0.5 * (y.min() + y.max()))
    return float(x[best_cell]), float(y[best_cell])


# --------------------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------------------
def draw_panel(ax, s, z_km, cloud, precip, uh, w, cloud_levels, precip_levels, quiv, args):
    """One vertical section: shading = cloud, lines = precip, arrows = (uh, w)."""
    s_km = s * 1e-3
    cf = ax.contourf(s_km, z_km, cloud * 1e3, levels=cloud_levels, cmap="Blues", extend="max")
    if precip.max() * 1e3 >= precip_levels[0]:
        ax.contour(
            s_km, z_km, precip * 1e3, levels=precip_levels, colors="tab:red", linewidths=0.8
        )
    ks = slice(None, None, args.zskip)
    ss = slice(None, None, max(1, s.size // args.narrows))
    q = ax.quiver(
        s_km[ss], z_km[ks], uh[ks, ss], w[ks, ss],
        scale=quiv, scale_units="width", width=0.0025, color="k", alpha=0.8,
    )
    ax.set_ylim(0.0, args.zmax)
    ax.set_ylabel("height (km)")
    return cf, q


def draw_anomaly_panel(
    ax, s, z_km, anomaly, *, levels, cmap, uh, w, quiv, args, full=None, full_levels=None,
    full_fmt="%.0f K", full_color="0.35", full_width=0.5, extend="both",
):
    """One vertical section: shading = anomaly, lines = full field, arrows = (uh, w)."""
    s_km = s * 1e-3
    cf = ax.contourf(s_km, z_km, anomaly, levels=levels, cmap=cmap, extend=extend)
    if full is not None and full_levels is not None and len(full_levels) > 0:
        cs = ax.contour(
            s_km, z_km, full, levels=full_levels, colors=full_color, linewidths=full_width
        )
        ax.clabel(cs, fontsize=6, fmt=full_fmt, inline_spacing=2)
    ks = slice(None, None, args.zskip)
    ss = slice(None, None, max(1, s.size // args.narrows))
    q = ax.quiver(
        s_km[ss], z_km[ks], uh[ks, ss], w[ks, ss],
        scale=quiv, scale_units="width", width=0.0025, color="k", alpha=0.6,
    )
    ax.set_ylim(0.0, args.zmax)
    return cf, q


def plot_moisture(
    data, it, *, plot_dir, wxz, wyz, sx, sy, x0, y0, z_km, u, v, w,
    qv_ref, qv_levels, qv_lines, quiv, ref, t_min, args,
) -> pathlib.Path:
    """Figure with the qv anomaly (top) and the relative humidity (bottom) on both sections."""
    qv = data["qv"].isel(time=it).values
    qv_anom = (qv - qv_ref) * 1e3  # g/kg
    rh = relative_humidity(data, it, over_ice=args.rh_ice)
    rh_label = "ice below 0 °C" if args.rh_ice else "liquid water"

    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharey=True, constrained_layout=True)
    rows = (
        dict(field=qv_anom, levels=qv_levels, ticks=qv_levels[::2], cmap="BrBG",
             extend="both", full=qv * 1e3, full_levels=qv_lines, full_fmt="%g",
             full_color="0.35", full_width=0.5),
        dict(field=rh, levels=RH_LEVELS, ticks=RH_TICKS, cmap="YlGnBu", extend="both",
             full=rh, full_levels=[100.0], full_fmt="%.0f %%", full_color="k",
             full_width=1.2),
    )
    for row, r in enumerate(rows):
        for col, (wsec, s, uh) in enumerate(((wxz, sx + x0, u), (wyz, sy + y0, v))):
            cf, q = draw_anomaly_panel(
                axs[row, col], s, z_km, on_section(wsec, r["field"]), levels=r["levels"],
                cmap=r["cmap"], uh=on_section(wsec, uh), w=on_section(wsec, w), quiv=quiv,
                args=args, full=on_section(wsec, r["full"]), full_levels=r["full_levels"],
                full_fmt=r["full_fmt"], full_color=r["full_color"], full_width=r["full_width"],
                extend=r["extend"],
            )
        cb = fig.colorbar(cf, ax=axs[row, :], pad=0.01, shrink=0.9, ticks=r["ticks"])
        cb.set_label(
            "water vapour anomaly  qv - <qv>(t$_0$)  (g/kg)"
            if row == 0
            else f"relative humidity over {rh_label}  (%)"
        )
    for ax in axs[:, 0]:
        ax.set_ylabel("height (km)")
    axs[1, 0].set_xlabel("x (km)")
    axs[1, 1].set_xlabel("y (km)")
    axs[0, 0].set_title(f"x-z section at y = {y0 * 1e-3:.1f} km   (arrows: u, w)")
    axs[0, 1].set_title(f"y-z section at x = {x0 * 1e-3:.1f} km   (arrows: v, w)")
    axs[0, 0].quiverkey(q, 0.05, 0.975, ref, f"{ref:.0f} m/s", labelpos="E", coordinates="figure")
    fig.suptitle(
        f"warm bubble   t = {t_min[it]:.0f} min    (qv anomaly from the horizontal mean at "
        f"t = {t_min[0]:.0f} min, grey lines: qv in g/kg; black line: RH = 100 %)",
        fontsize=11,
    )
    out = plot_dir / f"moisture_sections_{it:04d}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


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


def main() -> None:  # noqa: PLR0912, PLR0915 [too-many-branches/statements] linear plotting script
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", type=pathlib.Path, help="driver output directory (e.g. output_sample)")
    p.add_argument("--group", default="winds", help="basename of the field group with the 3D fields")
    p.add_argument("--plot-dir", type=pathlib.Path, default=None, help="default: <output_dir>/plots")
    p.add_argument("--stride", type=int, default=1, help="plot every n-th output time")
    p.add_argument("--times", type=int, nargs="*", default=None, help="explicit output time indices")
    p.add_argument("--x0", type=float, default=None, help="x (km) of the y-z section (grid coords)")
    p.add_argument("--y0", type=float, default=None, help="y (km) of the x-z section (grid coords)")
    p.add_argument("--follow", action="store_true", help="re-centre the sections on the cloud every frame")
    p.add_argument("--zmax", type=float, default=16.0, help="top of the plots (km)")
    p.add_argument("--npoints", type=int, default=400, help="interpolation points along each section")
    p.add_argument("--narrows", type=int, default=30, help="approx. number of arrow columns per panel")
    p.add_argument("--zskip", type=int, default=2, help="plot arrows on every n-th level")
    p.add_argument("--vector-scale", type=float, default=None, help="m/s per panel width for the arrows")
    p.add_argument("--gif", action="store_true", help="also write an animated GIF of all frames")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--no-thermo", action="store_true", help="skip the temperature/density figures")
    p.add_argument("--no-moist", action="store_true", help="skip the qv/relative humidity figures")
    p.add_argument(
        "--rh-ice", action="store_true",
        help="relative humidity over ice below the melting point (default: over liquid water)",
    )
    args = p.parse_args()

    data, grid = open_output(args.output_dir, args.group)
    x, y, lx, ly = cell_coordinates(grid)
    z_km = full_level_heights(data) * 1e-3
    t = data["time"].values
    t_min = (t - t[0]) / 60.0

    its = args.times if args.times else list(range(0, data.sizes["time"], args.stride))
    plot_dir = args.plot_dir or args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # section geometry: along x (at y0) and along y (at x0), centred on the cloud
    sx = np.linspace(-0.5 * lx, 0.5 * lx, args.npoints, endpoint=False)
    sy = np.linspace(-0.5 * ly, 0.5 * ly, args.npoints, endpoint=False)
    if args.x0 is not None and args.y0 is not None:
        x0, y0 = args.x0 * 1e3, args.y0 * 1e3
    else:
        x0, y0 = locate_cloud_centre(data, x, y, its)
        x0 = args.x0 * 1e3 if args.x0 is not None else x0
        y0 = args.y0 * 1e3 if args.y0 is not None else y0
    print(f"sections through x0 = {x0 * 1e-3:.1f} km, y0 = {y0 * 1e-3:.1f} km"
          + (" (following the cloud)" if args.follow else ""))

    def weights(x0, y0):
        wxz = section_weights(x, y, lx, ly, sx + x0, np.full_like(sx, y0))
        wyz = section_weights(x, y, lx, ly, np.full_like(sy, x0), sy + y0)
        return wxz, wyz

    wxz, wyz = weights(x0, y0)

    # fixed colour scales / arrow scale over all frames, so the animation is comparable
    cloud_max = max(float(sum_vars(data, CLOUD_VARS, it).max()) for it in its) * 1e3
    nice = matplotlib.ticker.MaxNLocator(nbins=12).tick_values(0.0, max(cloud_max, 0.02))
    # lowest level 0.01 g/kg: everything below is shown as clear sky
    cloud_levels = np.concatenate([[0.01], nice[nice > 0.01]])
    precip_levels = np.array([0.1, 0.5, 1.0, 2.0, 4.0, 8.0])
    if args.vector_scale is None:
        umax = max(
            float(np.abs(data[v].isel(time=it).values).max())
            for it in its
            for v in ("eastward_wind", "northward_wind", "upward_air_velocity")
        )
        quiv = max(umax, 1.0) * 6.0
    else:
        quiv = args.vector_scale

    area = grid["cell_area"].values
    area_weights = area / area.sum()
    below_top = z_km <= args.zmax

    if not args.no_moist and "qv" not in data:
        print(f"note: no 'qv' in the '{args.group}' group, skipping the moisture figures")
        args.no_moist = True
    if not args.no_moist:
        # qv anomaly from the horizontal-mean profile at the first output time, colour scale
        # fixed over all selected frames
        qv_ref = horizontal_mean(data["qv"].isel(time=0).values, area_weights)[:, None]
        qv_anom_max = qv_max_all = 0.0
        for it in its:
            qv = data["qv"].isel(time=it).values[below_top]
            qv_anom_max = max(qv_anom_max, float(np.abs(qv - qv_ref[below_top]).max()))
            qv_max_all = max(qv_max_all, float(qv.max()))
        qv_levels = symmetric_levels(qv_anom_max * 1e3)
        qv_lines = matplotlib.ticker.MaxNLocator(nbins=8).tick_values(0.0, qv_max_all * 1e3)
        qv_lines = qv_lines[(qv_lines > 0.0) & (qv_lines < qv_max_all * 1e3)]

    if not args.no_thermo:
        # anomalies from the horizontal-mean profile at the first output time, with colour
        # scales fixed over all selected frames
        t_ref = horizontal_mean(temperature(data, 0), area_weights)[:, None]
        rho_ref = horizontal_mean(data["air_density"].isel(time=0).values, area_weights)[:, None]
        t_anom_max = rho_anom_max = 0.0
        t_min_all, t_max_all = np.inf, -np.inf
        for it in its:
            temp = temperature(data, it)[below_top]
            rho = data["air_density"].isel(time=it).values[below_top]
            t_anom_max = max(t_anom_max, float(np.abs(temp - t_ref[below_top]).max()))
            rho_anom_max = max(rho_anom_max, float(np.abs(rho - rho_ref[below_top]).max()))
            t_min_all, t_max_all = min(t_min_all, float(temp.min())), max(t_max_all, float(temp.max()))
        t_levels = symmetric_levels(t_anom_max)
        rho_levels = symmetric_levels(rho_anom_max * 1e3)
        isotherms = np.arange(
            np.floor(t_min_all / ISOTHERM_SPACING) * ISOTHERM_SPACING, t_max_all, ISOTHERM_SPACING
        )

    frames, thermo_frames, moist_frames = [], [], []
    for n, it in enumerate(its):
        if args.follow and n > 0:
            x0, y0 = locate_cloud_centre(data, x, y, [it])
            wxz, wyz = weights(x0, y0)

        cloud = sum_vars(data, CLOUD_VARS, it)
        precip = sum_vars(data, PRECIP_VARS, it)
        u = data["eastward_wind"].isel(time=it).values
        v = data["northward_wind"].isel(time=it).values
        w = w_on_full_levels(data, it)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True, constrained_layout=True)
        cf, q = draw_panel(
            ax1, sx + x0, z_km, on_section(wxz, cloud), on_section(wxz, precip),
            on_section(wxz, u), on_section(wxz, w), cloud_levels, precip_levels, quiv, args,
        )
        ax1.set_xlabel("x (km)")
        ax1.set_title(f"x-z section at y = {y0 * 1e-3:.1f} km   (arrows: u, w)")
        draw_panel(
            ax2, sy + y0, z_km, on_section(wyz, cloud), on_section(wyz, precip),
            on_section(wyz, v), on_section(wyz, w), cloud_levels, precip_levels, quiv, args,
        )
        ax2.set_ylabel("")
        ax2.set_xlabel("y (km)")
        ax2.set_title(f"y-z section at x = {x0 * 1e-3:.1f} km   (arrows: v, w)")
        ref = 10.0 if quiv > 30 else 5.0
        ax1.quiverkey(q, 0.05, 0.965, ref, f"{ref:.0f} m/s", labelpos="E", coordinates="figure")
        cb = fig.colorbar(cf, ax=[ax1, ax2], pad=0.01, shrink=0.9, ticks=cloud_levels[1::2])
        cb.set_label("cloud water + ice  qc+qi  (g/kg)")
        fig.suptitle(
            f"warm bubble   t = {t_min[it]:.0f} min    "
            f"(red lines: rain+snow+graupel  qr+qs+qg = {', '.join(f'{c:g}' for c in precip_levels)} g/kg)",
            fontsize=11,
        )
        out = plot_dir / f"cross_sections_{it:04d}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        frames.append(out)
        print(f"[{n + 1}/{len(its)}] {out}")

        if not args.no_moist:
            moist_frames.append(
                plot_moisture(
                    data, it, plot_dir=plot_dir, wxz=wxz, wyz=wyz, sx=sx, sy=sy, x0=x0, y0=y0,
                    z_km=z_km, u=u, v=v, w=w, qv_ref=qv_ref, qv_levels=qv_levels,
                    qv_lines=qv_lines, quiv=quiv, ref=ref, t_min=t_min, args=args,
                )
            )
            print(f"[{n + 1}/{len(its)}] {moist_frames[-1]}")

        if args.no_thermo:
            continue
        temp = temperature(data, it)
        rho = data["air_density"].isel(time=it).values
        t_anom = temp - t_ref
        rho_anom = (rho - rho_ref) * 1e3  # g m-3
        fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharey=True, constrained_layout=True)
        for row, (anom, levels, cmap, full) in enumerate(
            ((t_anom, t_levels, "RdBu_r", temp), (rho_anom, rho_levels, "PuOr_r", None))
        ):
            for col, (wsec, s, uh) in enumerate(((wxz, sx + x0, u), (wyz, sy + y0, v))):
                cf, q = draw_anomaly_panel(
                    axs[row, col], s, z_km, on_section(wsec, anom), levels=levels, cmap=cmap,
                    uh=on_section(wsec, uh), w=on_section(wsec, w), quiv=quiv, args=args,
                    full=None if full is None else on_section(wsec, full), full_levels=isotherms,
                )
            cb = fig.colorbar(cf, ax=axs[row, :], pad=0.01, shrink=0.9, ticks=levels[::2])
            cb.set_label(
                "temperature anomaly  T - <T>(t$_0$)  (K)"
                if row == 0
                else r"density anomaly  $\rho$ - <$\rho$>(t$_0$)  (g m$^{-3}$)"
            )
        for ax in axs[:, 0]:
            ax.set_ylabel("height (km)")
        axs[1, 0].set_xlabel("x (km)")
        axs[1, 1].set_xlabel("y (km)")
        axs[0, 0].set_title(f"x-z section at y = {y0 * 1e-3:.1f} km   (arrows: u, w)")
        axs[0, 1].set_title(f"y-z section at x = {x0 * 1e-3:.1f} km   (arrows: v, w)")
        axs[0, 0].quiverkey(q, 0.05, 0.975, ref, f"{ref:.0f} m/s", labelpos="E", coordinates="figure")
        fig.suptitle(
            f"warm bubble   t = {t_min[it]:.0f} min    (anomalies from the horizontal mean at "
            f"t = {t_min[0]:.0f} min; grey lines: isotherms every {ISOTHERM_SPACING:.0f} K)",
            fontsize=11,
        )
        out = plot_dir / f"thermo_sections_{it:04d}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        thermo_frames.append(out)
        print(f"[{n + 1}/{len(its)}] {out}")

    if args.gif and frames:
        write_gif(frames, plot_dir / "cross_sections.gif", args.fps)
    if args.gif and thermo_frames:
        write_gif(thermo_frames, plot_dir / "thermo_sections.gif", args.fps)
    if args.gif and moist_frames:
        write_gif(moist_frames, plot_dir / "moisture_sections.gif", args.fps)


if __name__ == "__main__":
    main()
