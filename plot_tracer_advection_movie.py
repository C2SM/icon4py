#!/usr/bin/env python
"""2D movie of an idealized horizontal tracer advection run on a torus grid.

For the `linear_horizontal_adv` initial condition the tracer is carried by a uniform wind
(u, v) = (Lx, Ly) / s (uniform_xy), (Lx, 0) / s (uniform_x) or (0, Ly) / s (uniform_y),
so the exact solution at time t is the initial profile shifted by (u t, v t) on the
periodic domain. The exact solution is rebuilt here with the same three-point quadrature
the model uses for the initial condition and for the reference of the convergence test
(model/common/.../initial_condition/analytical/linear_horizontal_tracer_advection.py),
so at t = 0 it reproduces the output to round-off (checked and printed).

Every frame has six panels:

  top     numerical tracer | exact solution | error (numerical - exact)
          The tracer colour scale is fixed over all frames: viridis on [0, 1] (the range
          of the initial profile), continued by a yellow -> dark red ramp up to the largest
          overshoot and a light -> dark blue ramp down to the largest undershoot (if any),
          so the size of the over- and undershoots can be read off the colour bar. The
          exact circle (circle_2d) is outlined on the numerical panel. The error scale is
          symmetric and fixed over all frames.
  bottom  min and max of the tracer | relative L1 and Linf errors (as in
          test_tracer_advection_convergence: sum|q - q_ex| / sum|q_ex| and
          max|q - q_ex| / max|q_ex| over the cells of the level) | relative change of the
          tracer mass sum_c A_c q_c (cell areas from the grid file)
          A marker follows the current time.

The run parameters (tracer profile, velocity field, initial centre, decay radius,
advection scheme, limiter, time step) are read from the config .yaml that the driver
copies next to the output; --profile / --velocity / --center / --decay-radius override
them, --no-exact turns the exact solution and the error panels off (e.g. for a velocity
field without an analytical solution).

Output (in --plot-dir, default <output_dir>/plots_movie):
  frames/frame_NNNN.png, <var>_advection.gif and, if ffmpeg is on PATH, <var>_advection.mp4

Usage (inside the uenv, with the icon4py venv activated):
    python plot_tracer_advection_movie.py output_horizontal_advection
    python plot_tracer_advection_movie.py output_horizontal_advection/output_sample --stride 2 --fps 15
    python plot_tracer_advection_movie.py output_x --var qv --level 2 --no-exact
"""

import argparse
import dataclasses
import multiprocessing
import os
import pathlib
import shutil
import subprocess

import matplotlib


matplotlib.use("Agg")
import matplotlib.colors as mpl_colors
import matplotlib.patches as mpl_patches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

from plot_water_path_map import cell_polygons, draw_map
from plot_y_distribution import open_output, write_gif


PROFILES = ("gaussian_2d", "gaussian_2d_offcenter", "gaussian_1d_x", "gaussian_1d_y", "circle_2d")
VELOCITIES = ("uniform_xy", "uniform_x", "uniform_y")


# --------------------------------------------------------------------------------------
# run configuration
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class RunConfig:
    profile: str = "gaussian_2d"
    velocity: str = "uniform_xy"
    center: tuple[float, float] = (0.5, 0.5)
    decay_radius: float = 0.25
    scheme: str = "?"
    limiter: str = "?"
    dtime_s: float | None = None
    experiment: str = ""


def duration_seconds(value) -> float | None:
    """'400 microseconds' / '10 seconds' / a number of seconds -> seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    number, *unit = str(value).split()
    scale = {
        "microseconds": 1e-6,
        "milliseconds": 1e-3,
        "seconds": 1.0,
        "minutes": 60.0,
        "hours": 3600.0,
    }.get(unit[0] if unit else "seconds", 1.0)
    return float(number) * scale


def read_config(sample_dir: pathlib.Path) -> RunConfig:
    cfg = RunConfig()
    for path in sorted(sample_dir.glob("*.yaml")):
        with path.open() as f:
            doc = yaml.safe_load(f) or {}
        ic = doc.get("initial_condition") or {}
        if ic.get("type") != "linear_horizontal_adv":
            continue
        cfg.profile = str(ic.get("tracer_profile", cfg.profile)).lower()
        cfg.velocity = str(ic.get("velocity_field", cfg.velocity)).lower()
        cfg.center = tuple(float(c) for c in ic.get("initial_center", cfg.center))
        cfg.decay_radius = float(ic.get("decay_radius", cfg.decay_radius))
        adv = doc.get("tracer_advection") or {}
        cfg.scheme = str(adv.get("horizontal_advection_type", cfg.scheme))
        cfg.limiter = str(adv.get("horizontal_advection_limiter", cfg.limiter))
        driver = doc.get("driver") or {}
        cfg.dtime_s = duration_seconds(driver.get("dtime"))
        cfg.experiment = str(driver.get("experiment_name", ""))
        print(f"config: {path.name}")
        return cfg
    print(f"note: no linear_horizontal_adv config .yaml in {sample_dir}, using defaults/options")
    return cfg


# --------------------------------------------------------------------------------------
# exact solution (mirrors linear_horizontal_tracer_advection.py)
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class Geometry:
    lx: float
    ly: float
    origin: tuple[float, float]  # (vertex_x.min(), vertex_y.min()), as in the model
    nodes_x: np.ndarray  # (3, ncell) quadrature nodes: midpoints of the triangle edges
    nodes_y: np.ndarray
    area: np.ndarray  # (ncell,)


def build_geometry(grid: xr.Dataset) -> Geometry:
    cx = grid["cell_circumcenter_cartesian_x"].values
    cy = grid["cell_circumcenter_cartesian_y"].values
    lx = float(grid.attrs["domain_length"])
    ly = float(grid.attrs["domain_height"])
    v2c = grid["vertex_of_cell"].values.astype(np.int64)
    if v2c.shape[0] != cx.size:
        v2c = v2c.T
    nvertex = grid.sizes["vertex"]
    v2c = v2c - int(
        grid["vertex_of_cell"].attrs.get("start_index", 1 if v2c.max() == nvertex else 0)
    )
    vx_all = grid["cartesian_x_vertices"].values
    vy_all = grid["cartesian_y_vertices"].values
    vx, vy = vx_all[v2c], vy_all[v2c]
    dx, dy = vx - cx[:, None], vy - cy[:, None]
    vx = np.where(np.abs(dx) > 0.5 * lx, vx - np.sign(dx) * lx, vx)
    vy = np.where(np.abs(dy) > 0.5 * ly, vy - np.sign(dy) * ly, vy)
    alpha = np.array([[0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
    return Geometry(
        lx=lx,
        ly=ly,
        origin=(float(vx_all.min()), float(vy_all.min())),
        nodes_x=alpha @ vx.T,
        nodes_y=alpha @ vy.T,
        area=grid["cell_area"].values,
    )


def wind(velocity: str, lx: float, ly: float) -> tuple[float, float]:
    return {"uniform_xy": (lx, ly), "uniform_x": (lx, 0.0), "uniform_y": (0.0, ly)}[velocity]


def tracer_centre(cfg: RunConfig, geo: Geometry, t: float) -> tuple[float, float]:
    u, v = wind(cfg.velocity, geo.lx, geo.ly)
    return (
        geo.origin[0] + (cfg.center[0] * geo.lx + u * t) % geo.lx,
        geo.origin[1] + (cfg.center[1] * geo.ly + v * t) % geo.ly,
    )


def exact_tracer(cfg: RunConfig, geo: Geometry, t: float) -> np.ndarray:
    xc, yc = tracer_centre(cfg, geo, t)
    dx = geo.nodes_x - xc
    dy = geo.nodes_y - yc
    dx -= geo.lx * np.round(dx / geo.lx)  # minimum image
    dy -= geo.ly * np.round(dy / geo.ly)
    lmin = min(geo.lx, geo.ly)
    if cfg.profile == "circle_2d":
        radius = (geo.lx + geo.ly) / 8.0
        point = np.where(dx**2 + dy**2 <= radius**2, 1.0, 0.0)
    elif cfg.profile in ("gaussian_2d", "gaussian_2d_offcenter"):
        if cfg.profile == "gaussian_2d_offcenter":
            dy = dy - geo.ly / 4
        k = -np.log(1e-3) / (cfg.decay_radius * lmin) ** 2
        point = np.exp(-k * (dx**2 + dy**2))
    elif cfg.profile == "gaussian_1d_x":
        point = np.exp(np.log(1e-3) / (cfg.decay_radius * geo.lx) ** 2 * dx**2)
    elif cfg.profile == "gaussian_1d_y":
        point = np.exp(np.log(1e-3) / (cfg.decay_radius * geo.ly) ** 2 * dy**2)
    else:
        raise ValueError(f"unknown tracer profile {cfg.profile}")
    return point.mean(axis=0)  # weights 1/3 each


# --------------------------------------------------------------------------------------
# frame rendering (one figure per worker, arrays swapped per frame)
# --------------------------------------------------------------------------------------
_S: dict = {}  # shared, read-only state set before the pool forks


def tracer_colours(vmin: float, vmax: float) -> tuple[mpl_colors.Colormap, mpl_colors.Normalize]:
    """viridis on [0, 1] (the range of the initial profile), continued by a yellow -> dark
    red ramp on (1, vmax] for overshoots and a light -> dark blue ramp on [vmin, 0) for
    undershoots, so their size can be read off the colour bar."""
    vmin, vmax = min(vmin, 0.0), max(vmax, 1.0)
    span = vmax - vmin
    stops = []
    if vmin < 0.0:
        stops += [(0.0, "#08306b"), (-vmin / span * 0.999, "#c6dbef")]
    viridis = plt.get_cmap("viridis")
    stops += [((x - vmin) / span, viridis(x)) for x in np.linspace(0.0, 1.0, 64)]
    if vmax > 1.0:
        x1 = (1.0 - vmin) / span
        stops += [
            (x1 + (1.0 - x1) * f, c)
            for f, c in ((0.02, "#fec44f"), (0.35, "#fd8d3c"), (0.7, "#e31a1c"), (1.0, "#67000d"))
        ]
    cmap = mpl_colors.LinearSegmentedColormap.from_list("tracer", stops, N=1024)
    return cmap, mpl_colors.Normalize(vmin, vmax)


def tracer_colorbar(fig, mappable, ax, label: str) -> None:
    cb = fig.colorbar(mappable, ax=ax, shrink=0.9, label=label)
    lo, hi = mappable.norm.vmin, mappable.norm.vmax
    ticks = [t for t in (lo, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, hi) if lo <= t <= hi]
    ticks = sorted({round(t, 3) for t in ticks})
    cb.set_ticks(ticks, labels=[f"{t:g}" for t in ticks])
    for edge in (0.0, 1.0):
        if lo < edge < hi:
            cb.ax.axhline(edge, color="k", lw=1.0)


def _setup_figure():  # noqa: PLR0915 [too-many-statements] linear figure layout
    s = _S
    with_exact = s["exact"] is not None
    ncol = 3 if with_exact else 2
    fig = plt.figure(figsize=(5.4 * ncol, 8.4), dpi=s["dpi"], layout="constrained")
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.0, 0.45])
    axes = {}
    tracer_cmap, tnorm = tracer_colours(s["vmin"], s["vmax"])
    polys, owner, xwin, ywin = s["polys"], s["owner"], s["xwin"], s["ywin"]

    ax = fig.add_subplot(gs[0, 0])
    pc_num = draw_map(
        ax, polys, s["q"][0][owner], norm=tnorm, cmap=tracer_cmap, xwin=xwin, ywin=ywin
    )
    tracer_colorbar(fig, pc_num, ax, s["var"])
    axes["num"] = ax
    circles = []
    if with_exact and s["cfg"].profile == "circle_2d":
        lx, ly = s["geo"].lx * 1e-3, s["geo"].ly * 1e-3
        for ox in (-lx, 0.0, lx):
            for oy in (-ly, 0.0, ly):
                c = mpl_patches.Circle(
                    (0, 0), s["radius_km"], fill=False, ec="white", lw=1.0, ls="--", alpha=0.9
                )
                ax.add_patch(c)
                circles.append((c, ox, oy))
    pcs = {"num": pc_num}
    if with_exact:
        ax = fig.add_subplot(gs[0, 1])
        pcs["ex"] = draw_map(
            ax, polys, s["exact"][0][owner], norm=tnorm, cmap=tracer_cmap, xwin=xwin, ywin=ywin
        )
        tracer_colorbar(fig, pcs["ex"], ax, s["var"])
        ax.set_title("exact solution")
        axes["ex"] = ax
        ax = fig.add_subplot(gs[0, 2])
        enorm = mpl_colors.Normalize(-s["emax"], s["emax"])
        pcs["err"] = draw_map(
            ax,
            polys,
            (s["q"][0] - s["exact"][0])[owner],
            norm=enorm,
            cmap="RdBu_r",
            xwin=xwin,
            ywin=ywin,
        )
        fig.colorbar(pcs["err"], ax=ax, shrink=0.9, label=f"{s['var']} - exact")
        axes["err"] = ax
    for ax in (axes.get("num"), axes.get("ex"), axes.get("err")):
        if ax is not None:
            ax.set_xlabel("x (km)")
            ax.set_ylabel("y (km)")

    # time series
    t = s["t"]
    markers = []
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, s["qmax"], color="#d62728", label="max")
    ax.plot(t, s["qmin"], color="#1f77b4", label="min")
    ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
    ax.axhline(0.0, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("time (s)")
    ax.set_title(f"min / max of {s['var']}", fontsize=10)
    ax.legend(loc="center right", fontsize=8)
    markers += [
        ax.plot([], [], "o", color="#d62728", ms=5)[0],
        ax.plot([], [], "o", color="#1f77b4", ms=5)[0],
    ]
    series = [s["qmax"], s["qmin"]]
    vlines = [ax.axvline(t[0], color="k", lw=0.8, alpha=0.5)]
    if with_exact:
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(t, s["l1"], color="tab:blue", label=r"$L_1$")
        ax.plot(t, s["linf"], color="tab:red", label=r"$L_\infty$")
        ax.set_xlabel("time (s)")
        ax.set_title("relative error vs exact", fontsize=10)
        ax.legend(loc="center right", fontsize=8)
        markers += [
            ax.plot([], [], "o", color="tab:blue", ms=5)[0],
            ax.plot([], [], "o", color="tab:red", ms=5)[0],
        ]
        series += [s["l1"], s["linf"]]
        vlines.append(ax.axvline(t[0], color="k", lw=0.8, alpha=0.5))
    ax = fig.add_subplot(gs[1, ncol - 1])
    ax.plot(t, s["dmass"], color="k")
    ax.set_xlabel("time (s)")
    ax.set_title(r"relative mass change $\sum A q / \sum A q_0 - 1$", fontsize=10)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    markers.append(ax.plot([], [], "o", color="k", ms=5)[0])
    series.append(s["dmass"])
    vlines.append(ax.axvline(t[0], color="k", lw=0.8, alpha=0.5))

    title = fig.suptitle("", fontsize=13)
    _S["fig"] = dict(
        fig=fig,
        pcs=pcs,
        axes=axes,
        circles=circles,
        markers=markers,
        series=series,
        vlines=vlines,
        title=title,
    )


def _render(i: int) -> pathlib.Path:
    s = _S
    if "fig" not in s:
        _setup_figure()
    f = s["fig"]
    owner = s["owner"]
    q = s["q"][i]
    t = s["t"][i]
    f["pcs"]["num"].set_array(q[owner])
    f["axes"]["num"].set_title(f"numerical   min {q.min():.3g}  max {q.max():.4g}", fontsize=10)
    if s["exact"] is not None:
        f["pcs"]["ex"].set_array(s["exact"][i][owner])
        f["pcs"]["err"].set_array((q - s["exact"][i])[owner])
        f["axes"]["err"].set_title(
            f"error   $L_1$ {s['l1'][i]:.3g}   $L_\\infty$ {s['linf'][i]:.3g}", fontsize=10
        )
        if f["circles"]:
            xc, yc = tracer_centre(s["cfg"], s["geo"], t)
            for c, ox, oy in f["circles"]:
                c.center = (xc * 1e-3 + ox, yc * 1e-3 + oy)
    for m, y in zip(f["markers"], f["series"]):
        m.set_data([t], [y[i]])
    for vl in f["vlines"]:
        vl.set_xdata([t, t])
    step = f"   step {round(t / s['dtime']):d}" if s["dtime"] else ""
    f["title"].set_text(f"{s['header']}\nt = {t:.3f} s{step}")
    out = s["frame_dir"] / f"frame_{i:04d}.png"
    f["fig"].savefig(out, dpi=s["dpi"])
    return out


# --------------------------------------------------------------------------------------
def main() -> None:  # noqa: PLR0912, PLR0915 [too-many-branches, too-many-statements] linear plotting script
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="run directory or its output folder (with <basename>_*.nc and *_ugrid.nc)",
    )
    p.add_argument("--basename", default="atmospheric_state")
    p.add_argument("--var", default="qv")
    p.add_argument("--level", type=int, default=0, help="model level to show (default 0, the top)")
    p.add_argument("--stride", type=int, default=1, help="use every n-th output time")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument("--jobs", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 1)))
    p.add_argument("--plot-dir", type=pathlib.Path)
    p.add_argument("--no-exact", action="store_true", help="no exact solution / error panels")
    p.add_argument("--profile", choices=PROFILES, help="override the tracer profile of the config")
    p.add_argument(
        "--velocity", choices=VELOCITIES, help="override the velocity field of the config"
    )
    p.add_argument(
        "--center", type=float, nargs=2, help="override the initial centre (domain fractions)"
    )
    p.add_argument("--decay-radius", type=float, help="override the Gaussian decay radius")
    args = p.parse_args()

    out_dir = args.output_dir
    if not list(out_dir.glob(f"{args.basename}_*")) and (out_dir / "output_sample").is_dir():
        out_dir = out_dir / "output_sample"
    data, grid = open_output(out_dir, args.basename)
    cfg = read_config(out_dir)
    if args.profile:
        cfg.profile = args.profile
    if args.velocity:
        cfg.velocity = args.velocity
    if args.center:
        cfg.center = tuple(args.center)
    if args.decay_radius is not None:
        cfg.decay_radius = args.decay_radius

    times = np.arange(0, data.sizes["time"], args.stride)
    t_all = data["time"].values.astype(float)
    q_all = data[args.var].isel(level=args.level).values  # (time, cell)
    geo = build_geometry(grid)

    with_exact = not args.no_exact
    if with_exact and cfg.velocity not in VELOCITIES:
        print(f"note: no exact solution for velocity field {cfg.velocity}, disabling it")
        with_exact = False
    exact_all = None
    if with_exact:
        exact_all = np.stack([exact_tracer(cfg, geo, t) for t in t_all])
        d0 = np.abs(exact_all[0] - q_all[0]).max()
        print(f"check: max |exact - output| at t = {t_all[0]:g} s: {d0:.3e}")
        if d0 > 1e-6:
            print(
                "WARNING: the exact solution does not match the initial output; check the "
                "profile/velocity/centre options"
            )

    # diagnostics over all output times (series are drawn in full, frames may be strided)
    mass = (q_all * geo.area).sum(axis=1)
    stats = dict(
        qmin=q_all.min(axis=1),
        qmax=q_all.max(axis=1),
        dmass=mass / mass[0] - 1.0,
    )
    if with_exact:
        err = q_all - exact_all
        stats["l1"] = np.abs(err).sum(axis=1) / np.abs(exact_all).sum(axis=1)
        stats["linf"] = np.abs(err).max(axis=1) / np.abs(exact_all).max(axis=1)
        emax = float(np.abs(err).max())
    print(
        f"{args.var}: min {stats['qmin'].min():.4g}  max {stats['qmax'].max():.4g}  "
        f"|mass change| max {np.abs(stats['dmass']).max():.3e}"
    )
    if with_exact:
        print(
            f"relative error at t = {t_all[-1]:g} s: L1 {stats['l1'][-1]:.4g}  Linf {stats['linf'][-1]:.4g}"
        )

    polys, owner, xwin, ywin = cell_polygons(grid)
    plot_dir = args.plot_dir or out_dir / "plots_movie"
    frame_dir = plot_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("frame_*.png"):
        old.unlink()

    mean_edge = grid.attrs.get("mean_edge_length")
    res = f"{float(mean_edge):g} m torus" if mean_edge is not None else "torus"
    header = (
        f"{args.var} (level {args.level}), {cfg.profile} in {cfg.velocity} wind, {res}: "
        f"{cfg.scheme}, limiter {cfg.limiter}"
    )

    _S.update(
        q=q_all[times],
        exact=None if exact_all is None else exact_all[times],
        t=t_all[times],
        polys=polys,
        owner=owner,
        xwin=xwin,
        ywin=ywin,
        var=args.var,
        dpi=args.dpi,
        geo=geo,
        cfg=cfg,
        radius_km=(geo.lx + geo.ly) / 8.0 * 1e-3,
        emax=emax if with_exact else 1.0,
        vmin=float(stats["qmin"].min()),
        vmax=float(stats["qmax"].max()),
        dtime=cfg.dtime_s,
        header=header,
        frame_dir=frame_dir,
        qmin=stats["qmin"][times],
        qmax=stats["qmax"][times],
        dmass=stats["dmass"][times],
        l1=stats.get("l1", np.zeros(0))[times] if with_exact else None,
        linf=stats.get("linf", np.zeros(0))[times] if with_exact else None,
    )
    jobs = max(1, min(args.jobs, len(times)))
    print(f"rendering {len(times)} frames with {jobs} process(es) into {frame_dir}")
    if jobs == 1:
        frames = [_render(i) for i in range(len(times))]
    else:
        ctx = multiprocessing.get_context("fork")
        chunk = -(-len(times) // jobs)
        with ctx.Pool(jobs) as pool:
            frames = pool.map(_render, range(len(times)), chunksize=chunk)

    stem = f"{args.var}_advection"
    write_gif(frames, plot_dir / f"{stem}.gif", args.fps)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        mp4 = plot_dir / f"{stem}.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(args.fps),
                "-i",
                str(frame_dir / "frame_%04d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                str(mp4),
            ],
            check=True,
        )
        print(f"movie: {mp4}")
    else:
        print("note: ffmpeg not on PATH, no .mp4 written")


if __name__ == "__main__":
    main()
