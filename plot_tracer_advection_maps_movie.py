#!/usr/bin/env python
"""2D movie of a horizontal tracer advection run: maps only, one row per frame.

The map row of plot_tracer_advection_movie.py without the time series below it:

  numerical tracer | exact solution | error (numerical - exact)

The colour scales, the exact solution (rebuilt with the model's three-point quadrature and
checked against the t = 0 output), the circle outline and the reading of the run config
are those of plot_tracer_advection_movie.py, see there. With --no-exact only the numerical
tracer is drawn.

Output (in --plot-dir, default <output_dir>/plots_movie_maps):
  frames/frame_NNNN.png, <var>_advection_maps.gif and, if ffmpeg is on PATH,
  <var>_advection_maps.mp4

Usage (inside the uenv, with the icon4py venv activated):
    python plot_tracer_advection_maps_movie.py output_horizontal_advection
    python plot_tracer_advection_maps_movie.py output_horizontal_advection --stride 2 --fps 15
"""

import argparse
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

from plot_tracer_advection_movie import (
    PROFILES,
    VELOCITIES,
    build_geometry,
    exact_tracer,
    read_config,
    tracer_centre,
    tracer_colorbar,
    tracer_colours,
)
from plot_water_path_map import cell_polygons, draw_map
from plot_y_distribution import open_output, write_gif


_S: dict = {}  # shared, read-only state set before the pool forks


def _setup_figure() -> None:
    s = _S
    with_exact = s["exact"] is not None
    ncol = 3 if with_exact else 1
    fig = plt.figure(figsize=(5.4 * ncol, 5.4), dpi=s["dpi"], layout="constrained")
    axs = fig.subplots(1, ncol, squeeze=False)[0]
    cmap, tnorm = tracer_colours(s["vmin"], s["vmax"])
    polys, owner, xwin, ywin = s["polys"], s["owner"], s["xwin"], s["ywin"]

    pcs = {
        "num": draw_map(
            axs[0], polys, s["q"][0][owner], norm=tnorm, cmap=cmap, xwin=xwin, ywin=ywin
        )
    }
    tracer_colorbar(fig, pcs["num"], axs[0], s["var"])
    circles = []
    if with_exact:
        pcs["ex"] = draw_map(
            axs[1], polys, s["exact"][0][owner], norm=tnorm, cmap=cmap, xwin=xwin, ywin=ywin
        )
        tracer_colorbar(fig, pcs["ex"], axs[1], s["var"])
        axs[1].set_title("exact solution", fontsize=10)
        enorm = mpl_colors.Normalize(-s["emax"], s["emax"])
        pcs["err"] = draw_map(
            axs[2],
            polys,
            (s["q"][0] - s["exact"][0])[owner],
            norm=enorm,
            cmap="RdBu_r",
            xwin=xwin,
            ywin=ywin,
        )
        fig.colorbar(pcs["err"], ax=axs[2], shrink=0.9, label=f"{s['var']} - exact")
        if s["cfg"].profile == "circle_2d":
            lx, ly = s["geo"].lx * 1e-3, s["geo"].ly * 1e-3
            radius_km = (s["geo"].lx + s["geo"].ly) / 8.0 * 1e-3
            for ox in (-lx, 0.0, lx):
                for oy in (-ly, 0.0, ly):
                    c = mpl_patches.Circle(
                        (0, 0), radius_km, fill=False, ec="white", lw=1.0, ls="--", alpha=0.9
                    )
                    axs[0].add_patch(c)
                    circles.append((c, ox, oy))
    for ax in axs:
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
    title = fig.suptitle("", fontsize=12)
    s["fig"] = dict(fig=fig, axs=axs, pcs=pcs, circles=circles, title=title)


def _render(i: int) -> pathlib.Path:
    s = _S
    if "fig" not in s:
        _setup_figure()
    f = s["fig"]
    owner = s["owner"]
    q, t = s["q"][i], s["t"][i]
    f["pcs"]["num"].set_array(q[owner])
    f["axs"][0].set_title(f"numerical   min {q.min():.3g}  max {q.max():.4g}", fontsize=10)
    if s["exact"] is not None:
        ex = s["exact"][i]
        err = q - ex
        f["pcs"]["ex"].set_array(ex[owner])
        f["pcs"]["err"].set_array(err[owner])
        l1 = np.abs(err).sum() / np.abs(ex).sum()
        linf = np.abs(err).max() / np.abs(ex).max()
        f["axs"][2].set_title(f"error   $L_1$ {l1:.3g}   $L_\\infty$ {linf:.3g}", fontsize=10)
        if f["circles"]:
            xc, yc = tracer_centre(s["cfg"], s["geo"], t)
            for c, ox, oy in f["circles"]:
                c.center = (xc * 1e-3 + ox, yc * 1e-3 + oy)
    step = f"   step {round(t / s['dtime']):d}" if s["dtime"] else ""
    f["title"].set_text(f"{s['header']}\nt = {t:.3f} s{step}")
    out = s["frame_dir"] / f"frame_{i:04d}.png"
    f["fig"].savefig(out, dpi=s["dpi"])
    return out


def main() -> None:  # noqa: PLR0915 [too-many-statements] linear plotting script
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
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--jobs", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 1)))
    p.add_argument("--plot-dir", type=pathlib.Path)
    p.add_argument("--no-exact", action="store_true", help="only the numerical tracer")
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

    exact_all = None
    emax = 1.0
    if not args.no_exact:
        exact_all = np.stack([exact_tracer(cfg, geo, t) for t in t_all])
        d0 = np.abs(exact_all[0] - q_all[0]).max()
        print(f"check: max |exact - output| at t = {t_all[0]:g} s: {d0:.3e}")
        if d0 > 1e-6:
            print(
                "WARNING: the exact solution does not match the initial output; check the "
                "profile/velocity/centre options"
            )
        emax = float(np.abs(q_all - exact_all).max())  # fixed over all times

    polys, owner, xwin, ywin = cell_polygons(grid)
    plot_dir = args.plot_dir or out_dir / "plots_movie_maps"
    frame_dir = plot_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("frame_*.png"):
        old.unlink()

    mean_edge = grid.attrs.get("mean_edge_length")
    res = f"{float(mean_edge):g} m torus" if mean_edge is not None else "torus"
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
        emax=emax,
        vmin=float(q_all.min()),
        vmax=float(q_all.max()),
        dtime=cfg.dtime_s,
        header=(
            f"{args.var} (level {args.level}), {cfg.profile} in {cfg.velocity} wind, {res}: "
            f"{cfg.scheme}, limiter {cfg.limiter}"
        ),
        frame_dir=frame_dir,
    )
    jobs = max(1, min(args.jobs, len(times)))
    print(f"rendering {len(times)} frames with {jobs} process(es) into {frame_dir}")
    if jobs == 1:
        frames = [_render(i) for i in range(len(times))]
    else:
        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            frames = pool.map(_render, range(len(times)), chunksize=-(-len(times) // jobs))

    stem = f"{args.var}_advection_maps"
    write_gif(frames, plot_dir / f"{stem}.gif", args.fps)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        mp4 = plot_dir / f"{stem}.mp4"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(args.fps),
             "-i", str(frame_dir / "frame_%04d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             "-pix_fmt", "yuv420p", "-c:v", "libx264", str(mp4)],
            check=True,
        )  # fmt: skip
        print(f"movie: {mp4}")
    else:
        print("note: ffmpeg not on PATH, no .mp4 written")


if __name__ == "__main__":
    main()
