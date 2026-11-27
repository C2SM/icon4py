#!/usr/bin/env python
"""Initial and final vertical tracer profiles of an idealized vertical advection run.

For the linear_vertical_adv test case the tracer is carried by a uniform vertical wind
w = +H / s (uniform_positive) or -H / s (uniform_negative), H = model top height, so the
exact solution at time t is the initial profile shifted by w t. It is rebuilt here like the
initial condition and the reference of the convergence test
(model/common/.../analytical/linear_vertical_tracer_advection.py): layer averages by
Simpson's rule from the full- and half-level heights, so at t = 0 it reproduces the output
to round-off (checked and printed).

One panel: q(z) at the first output time and at the final time (--final-time, default the
last output) as solid lines, each with its exact solution dashed. The relative L1 and Linf
errors of the final profile (as in test_tracer_advection_convergence) are given in the
legend.

The exact solution has no boundaries, while the model has a closed lid and ground (no flux
through them): once the edge of the profile (box: half width H/8; Gaussian: decay radius,
where it drops to 1e-3) reaches the model top (or the ground for w < 0), the tracer piles
up in the boundary layer and no longer follows the exact solution. That time is printed,
and a final profile after it is marked in the legend; values beyond the x range (the
pile-up) are clipped and annotated.

All columns of the run are identical in this test case; --cell picks the column (a warning
is printed if they differ). The run parameters are read from the config .yaml that the
driver copies next to the output; --profile / --velocity / --center / --decay-radius
override them.

Output (in --plot-dir, default <output_dir>/plots): <var>_vertical_profiles.png

Usage (inside the uenv, with the icon4py venv activated):
    python plot_vertical_advection_profiles.py output_vertical_advection
    python plot_vertical_advection_profiles.py output_vertical_advection --final-time 0.57
"""

import argparse
import dataclasses
import math
import pathlib

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from plot_tracer_advection_movie import duration_seconds
from plot_y_distribution import open_output


PROFILES = ("gaussian", "box")
VELOCITIES = ("uniform_positive", "uniform_negative")


# --------------------------------------------------------------------------------------
# run configuration
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class RunConfig:
    profile: str = "gaussian"
    velocity: str = "uniform_positive"
    center: float = 0.5
    decay_radius: float = 0.25
    scheme: str = "?"
    limiter: str = "?"
    dtime_s: float | None = None


def read_config(sample_dir: pathlib.Path) -> RunConfig:
    cfg = RunConfig()
    for path in sorted(sample_dir.glob("*.yaml")):
        with path.open() as f:
            doc = yaml.safe_load(f) or {}
        ic = doc.get("initial_condition") or {}
        if ic.get("type") != "linear_vertical_adv":
            continue
        cfg.profile = str(ic.get("tracer_profile", cfg.profile)).lower()
        cfg.velocity = str(ic.get("velocity_field", cfg.velocity)).lower()
        cfg.center = float(ic.get("initial_center", cfg.center))
        cfg.decay_radius = float(ic.get("decay_radius", cfg.decay_radius))
        adv = doc.get("tracer_advection") or {}
        cfg.scheme = str(adv.get("vertical_advection_type", cfg.scheme))
        cfg.limiter = str(adv.get("vertical_advection_limiter", cfg.limiter))
        cfg.dtime_s = duration_seconds((doc.get("driver") or {}).get("dtime"))
        print(f"config: {path.name}")
        return cfg
    print(f"note: no linear_vertical_adv config .yaml in {sample_dir}, using defaults/options")
    return cfg


# --------------------------------------------------------------------------------------
# exact solution (mirrors linear_vertical_tracer_advection.py)
# --------------------------------------------------------------------------------------
def wind(cfg: RunConfig, top: float) -> float:
    return top if cfg.velocity == "uniform_positive" else -top


def half_width(cfg: RunConfig, top: float) -> float:
    """Distance from the centre to the edge of the profile (Gaussian: where it is 1e-3)."""
    return top / 8.0 if cfg.profile == "box" else cfg.decay_radius * top


def point_profile(cfg: RunConfig, dz: np.ndarray, top: float) -> np.ndarray:
    if cfg.profile == "box":
        return np.where(dz**2 <= (top / 8.0) ** 2, 1.0, 0.0)
    k = -1.0 / (cfg.decay_radius * top) ** 2 * math.log(1e-3)
    return np.exp(-k * dz**2)


def exact_tracer(cfg: RunConfig, z_half: np.ndarray, top: float, t: float) -> np.ndarray:
    """Layer averages (Simpson's rule) of the shifted profile, ordered like the output."""
    centre = cfg.center * top + wind(cfg, top) * t
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    q_full = point_profile(cfg, z_full - centre, top)
    q_half = point_profile(cfg, z_half - centre, top)
    return (q_half[:-1] + 4.0 * q_full + q_half[1:]) / 6.0


def boundary_time(cfg: RunConfig, top: float, bottom: float) -> float:
    """Time at which the edge of the exact profile reaches the lid (w > 0) or ground (w < 0)."""
    w = wind(cfg, top)
    z0, r = cfg.center * top, half_width(cfg, top)
    return (top - (z0 + r)) / w if w > 0 else (z0 - r - bottom) / -w


# --------------------------------------------------------------------------------------
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
    p.add_argument("--cell", type=int, default=0, help="column to plot")
    p.add_argument(
        "--final-time", type=float, help="time (s) of the final profile (default: last output)"
    )
    p.add_argument("--xmax", type=float, help="upper limit of the q axis (default: from the data)")
    p.add_argument("--plot-dir", type=pathlib.Path)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--profile", choices=PROFILES, help="override the tracer profile of the config")
    p.add_argument("--velocity", choices=VELOCITIES, help="override the velocity of the config")
    p.add_argument("--center", type=float, help="override the initial centre (fraction of the top)")
    p.add_argument("--decay-radius", type=float, help="override the Gaussian decay radius")
    args = p.parse_args()

    out_dir = args.output_dir
    if not list(out_dir.glob(f"{args.basename}_*")) and (out_dir / "output_sample").is_dir():
        out_dir = out_dir / "output_sample"
    data, _ = open_output(out_dir, args.basename)
    cfg = read_config(out_dir)
    for name in ("profile", "velocity", "center", "decay_radius"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))

    t_all = data["time"].values.astype(float)
    z_half = data["height"].values.astype(float)  # half levels, top -> bottom
    if z_half[0] < z_half[-1]:
        raise SystemExit("error: expected the half-level heights ordered from the top down")
    top, bottom = float(z_half[0]), float(z_half[-1])
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    dz = z_half[:-1] - z_half[1:]
    q_cols = data[args.var].values  # (time, level, cell)
    spread = float(np.abs(q_cols - q_cols[:, :, :1]).max())
    if spread > 0.0:
        print(f"WARNING: the columns differ (max {spread:.3e}); plotting cell {args.cell}")
    q = q_cols[:, :, args.cell]  # (time, level)

    i0 = 0
    i1 = t_all.size - 1
    if args.final_time is not None:
        i1 = int(np.abs(t_all - args.final_time).argmin())
        if abs(t_all[i1] - args.final_time) > 1e-9 * max(1.0, abs(args.final_time)):
            print(f"note: t = {args.final_time:g} s is not an output time, using {t_all[i1]:g} s")

    exact = {i: exact_tracer(cfg, z_half, top, t_all[i]) for i in (i0, i1)}
    d0 = float(np.abs(exact[i0] - q[i0]).max())
    print(f"check: max |exact - output| at t = {t_all[i0]:g} s: {d0:.3e}")
    if d0 > 1e-6:
        print(
            "WARNING: the exact solution does not match the initial output; check the "
            "profile/velocity/centre options"
        )
    w = wind(cfg, top)
    t_bnd = boundary_time(cfg, top, bottom)
    where = "model top" if w > 0 else "ground"
    print(f"w = {w:g} m/s: the tracer edge reaches the {where} at t = {t_bnd:.4g} s")
    after = t_all[i1] > t_bnd + 1e-12
    if after:
        print(
            f"note: the final time {t_all[i1]:g} s is after that; the final profile is shaped by "
            f"the {where}, not comparable with the exact solution (try --final-time)"
        )

    err = q[i1] - exact[i1]
    norms = ""
    if not after:
        l1 = np.abs(err).sum() / np.abs(exact[i1]).sum()
        linf = np.abs(err).max() / np.abs(exact[i1]).max()
        norms = f"\n$L_1$ {l1:.3g}, $L_\\infty$ {linf:.3g}"
        print(f"relative error at t = {t_all[i1]:g} s: L1 {l1:.4g}  Linf {linf:.4g}")
    mass = (q * dz).sum(axis=1)
    print(
        f"{args.var} at t = {t_all[i1]:g} s: min {q[i1].min():.4g}  max {q[i1].max():.4g}  "
        f"mass change {mass[i1] / mass[i0] - 1.0:.3e}"
    )

    before = t_all <= t_bnd + 1e-12
    q_ref = max(float(q[i0].max()), float(q[i1].max()) if not after else 0.0)
    q_ref = max(q_ref, float(q[before].max()) if before.any() and after else 0.0)
    xmax = args.xmax or 1.08 * max(1.0, q_ref)
    xmin = min(0.0, float(q[[i0, i1]].min())) - 0.04 * xmax

    # ----------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 8.0), layout="constrained")
    zkm = z_full * 1e-3
    for i, colour, name in ((i0, "#1f3a93", "initial"), (i1, "#d6249f", "final")):
        note = f", after the {where}" if (i == i1 and after) else ""
        label = f"{name}, t = {t_all[i]:.3g} s{note}" + (norms if i == i1 else "")
        ax.plot(np.clip(q[i], xmin, xmax), zkm, color=colour, lw=1.8, label=label)
        ax.plot(exact[i], zkm, color=colour, lw=1.1, ls="--", alpha=0.85)
        if q[i].max() > xmax:
            k = int(q[i].argmax())
            ax.annotate(
                f"{q[i].max():.3g} \u2192",
                (xmax, zkm[k]),
                xytext=(-4, -4),
                textcoords="offset points",
                ha="right",
                va="top",
                color=colour,
                fontsize=9,
            )
    ax.plot([], [], color="0.3", lw=1.1, ls="--", label="exact solution")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(bottom * 1e-3, top * 1e-3)
    ax.axvline(1.0, color="0.5", lw=0.6, ls=":")
    ax.axvline(0.0, color="0.5", lw=0.6, ls=":")
    ax.set_xlabel(args.var)
    ax.set_ylabel("height (km)")
    ax.legend(fontsize=9, loc="lower right")
    step = f", dt = {cfg.dtime_s * 1e6:g} \u03bcs" if cfg.dtime_s else ""
    ax.set_title(
        f"{args.var}: {cfg.profile} in {cfg.velocity} wind (w = {w:g} m/s)\n"
        f"{q.shape[1]} levels (\u0394z = {dz.mean():g} m){step}\n"
        f"{cfg.scheme}, limiter {cfg.limiter}",
        fontsize=10,
    )

    plot_dir = args.plot_dir or out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"{args.var}_vertical_profiles.png"
    fig.savefig(out, dpi=args.dpi)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
