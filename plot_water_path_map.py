#!/usr/bin/env python
"""Horizontal (x-y) maps of the water paths and the vertically integrated vertical wind speed.

For every selected output time, one figure with four panels is written:

  liquid water path : LWP = sum_k rho_k (qc + qr)_k dz_k             (kg m^-2)
  ice water path    : IWP = sum_k rho_k (qi + qs + qg)_k dz_k        (kg m^-2)
  condensed water   : LWP + IWP                                      (kg m^-2)
  vertical wind     : sum_k |w|_k dz_k                               (m^2 s^-1)
                      with --density-weighted: sum_k rho_k |w|_k dz_k (kg m^-1 s^-1)

--cloud-only restricts the water paths to the non-precipitating hydrometeors (LWP: qc,
IWP: qi). w is interpolated from the half levels to the full levels before taking its
magnitude, so up- and downdrafts both count. Missing tracers count as zero, and --zmax
restricts all integrals to the levels below that height.

Every triangle of the ICON torus grid is drawn with its cell value; triangles crossing
the periodic boundary are drawn on both sides. Each panel has a fixed colour scale over
all selected times (--log: logarithmic scale for the water paths, from --log-min). The
panel titles give the domain (area-weighted) mean.

Output (in --plot-dir, default <output_dir>/plots): water_path_map_NNNN.png and, with
--gif, water_path_map.gif.

A summary figure, cwp_y_profile.png, shows the condensed water path averaged over x
(cell-area weighted) as a function of y: profiles at --profile-lines of the selected
times (left) and a time-y diagram over all selected times (right), e.g. to see a storm
split into two maxima moving apart in y. By default every row of cell centres is its own
y bin; --y-bins N uses N bins of equal width instead.

The maxima of each x-mean profile are located on a periodic cubic spline through the bin
means; a maximum counts if its prominence is at least --min-prominence times the profile
maximum. They are marked in the figure, whose third panel shows their number and, for
two maxima, their separation over time. --split-after T tests that there are exactly two
maxima at every selected time >= T min (e.g. after a storm split), prints the maxima, and
exits with status 1 if the test fails.

Usage (inside the uenv, with the icon4py venv activated):
    python plot_water_path_map.py output_sample
    python plot_water_path_map.py output_sample --stride 5 --log --gif
    python plot_water_path_map.py output_sample --cloud-only --zmax 12
"""

import argparse
import dataclasses
import pathlib

import matplotlib


matplotlib.use("Agg")
import matplotlib.collections as mpl_collections
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import interpolate, signal

from plot_y_distribution import bin_statistics, layer_thickness, open_output, write_gif, y_bins


LIQUID = ("qc", "qr")
ICE = ("qi", "qs", "qg")


# --------------------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------------------
def periodic_window(centres: np.ndarray, period: float) -> tuple[float, float]:
    """A window of one period centred on the cell centres."""
    start = centres.min() - 0.5 * (period - (centres.max() - centres.min()))
    return start, start + period


def cell_polygons(grid: xr.Dataset) -> tuple[np.ndarray, np.ndarray, tuple, tuple]:
    """Triangles (npoly, 3, 2) in km inside one periodic window and the cell of each.

    The vertices are unwrapped to the periodic image nearest to the cell centre; a
    triangle sticking out of the window is repeated, shifted by one period, on the other
    side so the window is completely covered.
    """
    cx = grid["cell_circumcenter_cartesian_x"].values
    cy = grid["cell_circumcenter_cartesian_y"].values
    lx = float(grid.attrs["domain_length"])
    ly = float(grid.attrs["domain_height"])
    v2c = grid["vertex_of_cell"].values.astype(np.int64)
    if v2c.shape[0] != cx.size:  # stored as (nv, cell)
        v2c = v2c.T
    nvertex = grid.sizes["vertex"]
    start_index = grid["vertex_of_cell"].attrs.get("start_index", 1 if v2c.max() == nvertex else 0)
    v2c = v2c - int(start_index)

    vx = grid["cartesian_x_vertices"].values[v2c]
    vy = grid["cartesian_y_vertices"].values[v2c]
    dx = vx - cx[:, None]
    dy = vy - cy[:, None]
    vx = cx[:, None] + dx - lx * np.round(dx / lx)
    vy = cy[:, None] + dy - ly * np.round(dy / ly)

    xwin = periodic_window(cx, lx)
    ywin = periodic_window(cy, ly)
    polys, owner = [], []
    cells = np.arange(cx.size)
    for sx in (-lx, 0.0, lx):
        for sy in (-ly, 0.0, ly):
            px, py = vx + sx, vy + sy
            inside = (
                (px.max(axis=1) > xwin[0])
                & (px.min(axis=1) < xwin[1])
                & (py.max(axis=1) > ywin[0])
                & (py.min(axis=1) < ywin[1])
            )
            polys.append(np.stack([px[inside], py[inside]], axis=-1))
            owner.append(cells[inside])
    xwin_km = (xwin[0] * 1e-3, xwin[1] * 1e-3)
    ywin_km = (ywin[0] * 1e-3, ywin[1] * 1e-3)
    return np.concatenate(polys) * 1e-3, np.concatenate(owner), xwin_km, ywin_km


# --------------------------------------------------------------------------------------
# fields
# --------------------------------------------------------------------------------------
def water_path(data: xr.Dataset, it: int, dz: np.ndarray, names: list[str]) -> np.ndarray:
    if not names:
        return np.zeros(data.sizes["cell"])
    q = sum(data[n].isel(time=it).values for n in names)
    return (data["air_density"].isel(time=it).values * q * dz).sum(axis=0)


def vertical_wind_speed_integral(
    data: xr.Dataset, it: int, dz: np.ndarray, *, density_weighted: bool
) -> np.ndarray:
    """Column integral of |w| (w interpolated from the half to the full levels)."""
    w_half = data["upward_air_velocity"].isel(time=it).values
    speed = np.abs(0.5 * (w_half[:-1] + w_half[1:]))
    weight = dz * data["air_density"].isel(time=it).values if density_weighted else dz
    return (speed * weight).sum(axis=0)


# --------------------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------------------
def draw_map(ax, polys, values, *, norm, cmap, xwin, ywin):
    pc = mpl_collections.PolyCollection(
        polys, array=values, cmap=cmap, norm=norm, edgecolors="face", linewidths=0.1
    )
    pc.set_rasterized(True)
    ax.add_collection(pc)
    ax.set_xlim(*xwin)
    ax.set_ylim(*ywin)
    ax.set_aspect("equal")
    return pc


def y_rows(
    y: np.ndarray, ly: float, nbins: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin centres, bin edges (m) and the bin of every cell for averaging over x.

    Without nbins every distinct y of the cell centres (one row of up- or down-pointing
    triangles) is a bin; the outer edges split the gap across the periodic boundary.
    """
    if nbins is not None:
        edges, index = y_bins(y, ly, nbins)
        return 0.5 * (edges[:-1] + edges[1:]), edges, index
    centres, index = np.unique(np.round(y, 3), return_inverse=True)
    gap = 0.5 * (centres[0] + ly - centres[-1])
    edges = np.concatenate(
        [[centres[0] - gap], 0.5 * (centres[:-1] + centres[1:]), [centres[-1] + gap]]
    )
    return centres, edges, index


@dataclasses.dataclass(frozen=True)
class ProfilePeaks:
    """Significant maxima of a periodic profile and the spline they were located on."""

    y: np.ndarray  # positions (m)
    height: np.ndarray
    prominence: np.ndarray
    #: two peaks only: the minimum separating them divided by the lower peak (nan otherwise)
    dip_ratio: float
    y_fine: np.ndarray  # spline sample (m)
    value_fine: np.ndarray

    @property
    def count(self) -> int:
        return int(self.y.size)


def find_profile_peaks(
    centres: np.ndarray,
    profile: np.ndarray,
    ly: float,
    *,
    min_prominence: float,
    nfine: int = 2048,
) -> ProfilePeaks:
    """Maxima of a periodic profile, located on a periodic cubic spline through it.

    A maximum is significant if its prominence (height above the higher of the two
    minima separating it from any higher maximum) is at least min_prominence times the
    profile maximum. The spline is piecewise cubic: a single global polynomial through
    all points would oscillate between them (Runge) and produce spurious maxima.
    """
    valid = np.isfinite(profile)
    c, v = centres[valid], profile[valid]
    y_fine = c[0] + np.arange(nfine) * (ly / nfine)
    top = float(v.max()) if v.size else 0.0
    if v.size < 4 or top <= 0.0:
        empty = np.empty(0)
        return ProfilePeaks(empty, empty, empty, np.nan, y_fine, np.zeros(nfine))
    spline = interpolate.CubicSpline(
        np.append(c, c[0] + ly), np.append(v, v[0]), bc_type="periodic"
    )
    value_fine = spline(y_fine)
    # tile three periods so maxima and prominences near the periodic boundary are right
    idx, props = signal.find_peaks(np.tile(value_fine, 3), prominence=min_prominence * top)
    keep = (idx >= nfine) & (idx < 2 * nfine)
    idx, prominence = idx[keep] - nfine, props["prominences"][keep]

    dip_ratio = np.nan
    if idx.size == 2:
        lo, hi = np.sort(idx)
        # the two arcs between the peaks on the periodic domain; the shallower valley
        # (higher minimum) is the one that separates them
        inner = value_fine[lo : hi + 1].min()
        outer = np.concatenate([value_fine[hi:], value_fine[: lo + 1]]).min()
        dip_ratio = float(max(inner, outer) / value_fine[idx].min())
    return ProfilePeaks(y_fine[idx], value_fine[idx], prominence, dip_ratio, y_fine, value_fine)


def periodic_distance(a: float, b: float, period: float) -> float:
    d = abs(a - b) % period
    return min(d, period - d)


def plot_cwp_y_profile(
    profiles: np.ndarray,
    peaks: list[ProfilePeaks],
    t_sel: np.ndarray,
    centres_km: np.ndarray,
    edges_km: np.ndarray,
    *,
    ly_km: float,
    nlines: int,
    title: str,
    out: pathlib.Path,
) -> None:
    """x-mean condensed water path vs y: profiles at a few times, a time-y diagram with the
    detected maxima, and the number and separation of the maxima over time."""
    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=(19, 5.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1.15, 0.8]},
    )
    lines = np.unique(np.linspace(0, len(t_sel) - 1, min(nlines, len(t_sel))).round().astype(int))
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.95, len(lines)))
    for k, color in zip(lines, colors, strict=True):
        pk = peaks[k]
        # spline sample, wrapped into the plotted window
        yf = (pk.y_fine * 1e-3 - edges_km[0]) % ly_km + edges_km[0]
        order = np.argsort(yf)
        ax1.plot(yf[order], pk.value_fine[order], lw=1.2, color=color, label=f"{t_sel[k]:.0f} min")
        ax1.plot(centres_km, profiles[k], "o", ms=2.5, color=color)
        if pk.count:
            ax1.plot(
                (pk.y * 1e-3 - edges_km[0]) % ly_km + edges_km[0],
                pk.height,
                "v",
                ms=7,
                mec="k",
                mew=0.6,
                color=color,
            )
    ax1.set_xlim(edges_km[0], edges_km[-1])
    ax1.set_ylim(bottom=0.0)
    ax1.set_xlabel("y (km)")
    ax1.set_ylabel(r"x-mean condensed water path  (kg m$^{-2}$)")
    ax1.grid(lw=0.4, alpha=0.5)
    ax1.legend(fontsize=8, ncols=2)
    ax1.set_title(
        "profiles (dots: bin means, lines: periodic cubic spline, ▼: maxima)", fontsize=10
    )

    if len(t_sel) > 1:
        t_edges = np.concatenate([[t_sel[0]], 0.5 * (t_sel[:-1] + t_sel[1:]), [t_sel[-1]]])
    else:
        t_edges = np.array([t_sel[0] - 0.5, t_sel[0] + 0.5])
    pc = ax2.pcolormesh(edges_km, t_edges, profiles, cmap="YlGnBu", shading="flat")
    fig.colorbar(pc, ax=ax2, pad=0.01).set_label(r"x-mean condensed water path  (kg m$^{-2}$)")
    for t, pk in zip(t_sel, peaks, strict=True):
        if pk.count:
            ax2.plot(
                (pk.y * 1e-3 - edges_km[0]) % ly_km + edges_km[0],
                np.full(pk.count, t),
                ".",
                ms=3,
                color="tab:red",
            )
    ax2.set_xlim(edges_km[0], edges_km[-1])
    ax2.set_xlabel("y (km)")
    ax2.set_ylabel("time (min)")
    ax2.set_title("time-y diagram (red: detected maxima)", fontsize=10)

    counts = np.array([pk.count for pk in peaks])
    separation = np.array(
        [
            periodic_distance(pk.y[0], pk.y[1], ly_km * 1e3) * 1e-3 if pk.count == 2 else np.nan
            for pk in peaks
        ]
    )
    ax3.step(t_sel, counts, where="mid", color="k", lw=1.2)
    ax3.set_ylim(-0.2, max(3, counts.max()) + 0.2)
    ax3.set_yticks(range(max(3, counts.max()) + 1))
    ax3.set_xlabel("time (min)")
    ax3.set_ylabel("number of maxima")
    ax3.grid(lw=0.4, alpha=0.5)
    ax3b = ax3.twinx()
    ax3b.plot(t_sel, separation, "-", color="tab:red", lw=1.2)
    ax3b.set_ylabel("separation of the two maxima (km)", color="tab:red")
    ax3b.tick_params(axis="y", colors="tab:red")
    ax3b.set_ylim(bottom=0.0)
    ax3.set_title("number of maxima (black), separation (red)", fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(out)


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
    p.add_argument("--stride", type=int, default=1, help="plot every n-th output time")
    p.add_argument(
        "--times", type=int, nargs="*", default=None, help="explicit output time indices"
    )
    p.add_argument("--zmax", type=float, default=None, help="integrate only below this height (km)")
    p.add_argument(
        "--cloud-only", action="store_true", help="LWP from qc only and IWP from qi only"
    )
    p.add_argument(
        "--density-weighted",
        action="store_true",
        help="integrate rho |w| dz instead of |w| dz for the vertical wind speed",
    )
    p.add_argument(
        "--log", action="store_true", help="logarithmic colour scale for the water paths"
    )
    p.add_argument(
        "--log-min", type=float, default=1e-3, help="lower end of the log scale (kg m^-2)"
    )
    p.add_argument(
        "--y-bins",
        type=int,
        default=None,
        help="bins of equal width in y for the x-mean profile (default: one per row of cells)",
    )
    p.add_argument(
        "--profile-lines", type=int, default=8, help="number of times drawn as x-mean profiles"
    )
    p.add_argument(
        "--min-prominence",
        type=float,
        default=0.2,
        help="a maximum of the x-mean profile counts if its prominence is at least this "
        "fraction of the profile maximum",
    )
    p.add_argument(
        "--split-after",
        type=float,
        default=None,
        help="test that the x-mean profile has exactly two maxima at every selected time at "
        "or after this time (min); exit status 1 if not",
    )
    p.add_argument("--gif", action="store_true", help="also write an animated GIF of all frames")
    p.add_argument("--fps", type=int, default=4)
    args = p.parse_args()

    data, grid = open_output(args.output_dir, args.group)
    polys, owner, xwin, ywin = cell_polygons(grid)
    area = grid["cell_area"].values
    area_weights = area / area.sum()
    t = data["time"].values
    t_min = (t - t[0]) / 60.0
    its = args.times if args.times else list(range(0, data.sizes["time"], args.stride))

    liquid = [n for n in (("qc",) if args.cloud_only else LIQUID) if n in data]
    ice = [n for n in (("qi",) if args.cloud_only else ICE) if n in data]
    missing = [n for n in (("qc", "qi") if args.cloud_only else LIQUID + ICE) if n not in data]
    if missing:
        print(f"note: {', '.join(missing)} not in the '{args.group}' group, counted as zero")

    dz = layer_thickness(data, None if args.zmax is None else args.zmax * 1e3)

    def fields(it: int) -> list[np.ndarray]:
        lwp = water_path(data, it, dz, liquid)
        iwp = water_path(data, it, dz, ice)
        wind = vertical_wind_speed_integral(data, it, dz, density_weighted=args.density_weighted)
        return [lwp, iwp, lwp + iwp, wind]

    def plus(names: list[str]) -> str:
        return "+".join(names) if names else "none"

    wind_unit = "kg m$^{-1}$ s$^{-1}$" if args.density_weighted else "m$^2$ s$^{-1}$"
    panels = [
        dict(
            title=f"liquid water path ({plus(liquid)})",
            unit="kg m$^{-2}$",
            cmap="Blues",
            water=True,
        ),
        dict(title=f"ice water path ({plus(ice)})", unit="kg m$^{-2}$", cmap="PuRd", water=True),
        dict(
            title="condensed water path (liquid + ice)",
            unit="kg m$^{-2}$",
            cmap="YlGnBu",
            water=True,
        ),
        dict(
            title="vertically integrated vertical wind speed"
            + (r" ($\int\rho|w|dz$)" if args.density_weighted else r" ($\int|w|dz$)"),
            unit=wind_unit,
            cmap="viridis",
            water=False,
        ),
    ]

    # fixed colour scales over all selected times
    vmax = np.zeros(len(panels))
    for it in its:
        vmax = np.maximum(vmax, [float(f.max()) for f in fields(it)])
    for panel, field_max in zip(panels, vmax, strict=True):
        top = field_max if field_max > 0.0 else 1.0
        if panel["water"] and args.log:
            panel["norm"] = mpl_colors.LogNorm(
                vmin=args.log_min, vmax=max(top, 10.0 * args.log_min), clip=True
            )
        else:
            panel["norm"] = mpl_colors.Normalize(vmin=0.0, vmax=top)

    # elongated domains: one column of panels; otherwise 2 x 2
    aspect = (ywin[1] - ywin[0]) / (xwin[1] - xwin[0])
    nrows, ncols = (4, 1) if aspect < 0.5 else (2, 2)
    width = 12.0 if ncols == 1 else 13.0
    height = nrows * (width / ncols) * aspect + 1.0 + 0.4 * nrows

    plot_dir = args.plot_dir or args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    zinfo = f"  (integrated below {args.zmax:g} km)" if args.zmax is not None else ""

    # x-mean condensed water path per y bin
    y_centres, y_edges, y_index = y_rows(
        grid["cell_circumcenter_cartesian_y"].values,
        float(grid.attrs["domain_height"]),
        args.y_bins,
    )
    if np.any(np.bincount(y_index, minlength=y_centres.size) == 0):
        print("warning: some y bins contain no cells (shown empty); reduce --y-bins")
    cwp_profiles = np.empty((len(its), y_centres.size))

    frames = []
    for n, it in enumerate(its):
        frame_fields = fields(it)
        cwp_profiles[n] = bin_statistics(frame_fields[2], y_index, area, y_centres.size)[0]
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(width, height), sharex=True, sharey=True, constrained_layout=True
        )
        for ax, panel, field in zip(axs.ravel(), panels, frame_fields, strict=True):
            pc = draw_map(
                ax,
                polys,
                field[owner],
                norm=panel["norm"],
                cmap=panel["cmap"],
                xwin=xwin,
                ywin=ywin,
            )
            cb = fig.colorbar(
                pc, ax=ax, pad=0.01, extend="max" if panel["water"] and args.log else "neither"
            )
            cb.set_label(panel["unit"])
            ax.set_title(
                f"{panel['title']}    domain mean {float(field @ area_weights):.3g}", fontsize=10
            )
        for ax in axs[-1] if ncols > 1 else axs[-1:]:
            ax.set_xlabel("x (km)")
        for ax in axs[:, 0] if ncols > 1 else axs:
            ax.set_ylabel("y (km)")
        fig.suptitle(f"t = {t_min[it]:.0f} min{zinfo}", fontsize=11)
        out = plot_dir / f"water_path_map_{it:04d}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        frames.append(out)
        print(f"[{n + 1}/{len(its)}] {out}")

    ly = float(grid.attrs["domain_height"])
    t_sel = t_min[its]
    # nfine = max(nfine, c.size * 3)  # at least 3 points per bin
    peaks = [
        find_profile_peaks(y_centres, profile, ly, min_prominence=args.min_prominence, nfine=y_centres.size * 3)
        for profile in cwp_profiles
    ]
    plot_cwp_y_profile(
        cwp_profiles,
        peaks,
        t_sel,
        y_centres * 1e-3,
        y_edges * 1e-3,
        ly_km=ly * 1e-3,
        nlines=args.profile_lines,
        title=(
            f"condensed water path ({plus(liquid + ice)}) averaged over x{zinfo}    "
            f"(maxima: prominence >= {args.min_prominence:g} of the profile maximum)"
        ),
        out=plot_dir / "cwp_y_profile.png",
    )

    if args.gif and frames:
        write_gif(frames, plot_dir / "water_path_map.gif", args.fps)

    if args.split_after is not None and not two_peak_test(peaks, t_sel, ly, args.split_after):
        raise SystemExit(1)


def two_peak_test(
    peaks: list[ProfilePeaks], t_sel: np.ndarray, ly: float, split_after: float
) -> bool:
    """Print the maxima of the x-mean profile at every time >= split_after and whether
    there are exactly two of them at all those times."""
    tested = [(t, pk) for t, pk in zip(t_sel, peaks, strict=True) if t >= split_after - 1e-9]
    if not tested:
        print(f"two-peak test: no selected output time at or after {split_after:g} min")
        return False
    print(f"two-peak test of the x-mean condensed water path for t >= {split_after:g} min:")
    for t, pk in tested:
        where = ", ".join(f"{y * 1e-3:.1f}" for y in np.sort(pk.y))
        detail = (
            f"; separation {periodic_distance(pk.y[0], pk.y[1], ly) * 1e-3:.1f} km, "
            f"dip ratio {pk.dip_ratio:.2f}"
            if pk.count == 2
            else ""
        )
        print(f"  t = {t:5.0f} min: {pk.count} maxima at y = [{where}] km{detail}")
    failed = [t for t, pk in tested if pk.count != 2]
    # the time since when the profile has two maxima without interruption
    since = None
    for t, pk in zip(t_sel[::-1], peaks[::-1], strict=True):
        if pk.count != 2:
            break
        since = t
    if since is not None:
        print(f"  two maxima continuously since t = {since:.0f} min")
    if failed:
        print(f"two-peak test FAILED at t = {', '.join(f'{t:.0f}' for t in failed)} min")
        return False
    print(f"two-peak test PASSED at all {len(tested)} times")
    return True


if __name__ == "__main__":
    main()
