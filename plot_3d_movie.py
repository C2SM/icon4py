#!/usr/bin/env python
"""3D movie of the liquid and frozen hydrometeors and the wind of an ICON4Py torus run.

  liquid  qc + qr          cloud water + rain
  ice     qi + qs + qg     cloud ice + snow + graupel
  wind    (u, v, w) arrows on a coarse sample of the columns and levels, coloured by w

Two outputs (both by default):

  --movie  a slowly rotating matplotlib 3D view per output time: PNG frames, an animated
           GIF and, if ffmpeg is on PATH, an MP4. Every cell whose specific content exceeds
           --threshold is a translucent dot at its centre whose opacity and size grow with
           log q (fixed scale over all times); the floor shows the column condensed water
           path LWP + IWP = sum_k rho (qc + qr + qi + qs + qg) dz as a grey shadow.
  --html   a self-contained interactive WebGL viewer (three.js): drag to rotate, scroll to
           zoom, play/pause and scrub through time, toggle each layer, change the
           threshold, the vertical exaggeration, the point size and the arrow length, and
           show the wind relative to its horizontal mean at each height (storm-relative
           circulation). Copy the .html file to a laptop and open it in a browser (it loads
           three.js from a CDN). With --html-split, the data is written next to it as
           <name>_data.json instead (for hosting the page on a web server).

Coordinates: x, y are the planar cell circumcentres of the torus grid in km, placed in one
periodic window; z is the full-level height (mean of the two bounding half levels, flat
topography). eastward_wind and northward_wind are drawn along x and y, and w is
interpolated from the half to the full levels. Only levels below --zmax are shown (default:
1.5 km above the highest cloud of all selected times).

The view is meant for coarse runs (a few 10^4 cells): every voxel above the threshold is a
point, and all selected times of the fields are read into memory.

Output (in --plot-dir, default <output_dir>/plots_3d):
  frames/frame_NNNN.png, cloud3d.gif (and cloud3d.mp4)   --movie
  cloud3d.html (and cloud3d_data.json)                   --html

Usage (inside the uenv, with the icon4py venv activated):
    python plot_3d_movie.py output_sequential/output_sample
    python plot_3d_movie.py output_sequential --html --stride 2
    python plot_3d_movie.py output_sequential --movie --times 30 60 90 --azim -50 --rotate 0
    python plot_3d_movie.py output_sequential --movie --vec-dx 15 --vec-dz 2 --relative-wind
"""

import argparse
import base64
import dataclasses
import json
import multiprocessing
import pathlib
import re
import shutil
import subprocess

import matplotlib


matplotlib.use("Agg")
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.mplot3d import art3d
from PIL import Image
from scipy import spatial

from plot_water_path_map import cell_polygons, periodic_window
from plot_y_distribution import layer_thickness, open_output


LIQUID = ("qc", "qr")
ICE = ("qi", "qs", "qg")
MAX_CELLS = 100_000  # point clouds of larger grids are unreadable and do not fit in memory
CWP_MIN = 1e-2  # kg m^-2, lower end of the floor shading

# colours shared by the movie and the viewer
LIQUID_CMAP = mpl_colors.LinearSegmentedColormap.from_list("liquid", ["#9ecae1", "#2171b5", "#08306b"])
ICE_CMAP = mpl_colors.LinearSegmentedColormap.from_list("ice", ["#fbb4d9", "#d6249f", "#6a0f5a"])
W_CMAP = mpl_colors.LinearSegmentedColormap.from_list("w", ["#01665e", "#8c8c8c", "#e66101"])
FLOOR_CMAP = mpl_colors.LinearSegmentedColormap.from_list("floor", ["#f2f2f2", "#9a9a9a", "#2b2b2b"])


# --------------------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------------------
def resolve_output_dir(path: pathlib.Path, group: str) -> pathlib.Path:
    """The directory holding <group>_NNNN.nc / .zarr: the path itself or one level down."""
    if not path.is_dir():
        raise SystemExit(f"error: {path} is not a directory")
    name = re.compile(rf"^{re.escape(group)}_\d+\.(nc|zarr)$")
    for directory in [path, *sorted(p for p in path.iterdir() if p.is_dir() and p.suffix != ".zarr")]:
        if any(name.match(p.name) for p in directory.iterdir()):
            return directory
    raise SystemExit(f"error: no {group}_NNNN.nc or .zarr in {path} or its subdirectories")


def read_sum(data: xr.Dataset, names: tuple[str, ...], its: list[int], levels: np.ndarray) -> np.ndarray:
    """Sum of the present variables, (time, level, cell) float32; zeros if none is present."""
    present = [n for n in names if n in data]
    shape = (len(its), levels.size, data.sizes["cell"])
    total = np.zeros(shape, dtype=np.float32)
    for n in present:
        total += data[n].isel(time=its, level=levels).values.astype(np.float32)
    return total


@dataclasses.dataclass
class Fields:
    """Everything the movie and the viewer draw, for the selected times and the shown levels."""

    label: str
    t_min: np.ndarray  # (nt,) minutes since the first output
    x_km: np.ndarray  # (ncell,) cell centres in the periodic window
    y_km: np.ndarray
    xwin: tuple[float, float]  # km
    ywin: tuple[float, float]
    z_km: np.ndarray  # (nlev,) full-level heights of the shown levels, top to bottom
    zmax_km: float
    polys_km: np.ndarray  # (npoly, 3, 2) floor triangles, see plot_water_path_map.cell_polygons
    poly_owner: np.ndarray  # (npoly,) cell of each triangle
    liquid: np.ndarray  # (nt, nlev, ncell) g/kg
    ice: np.ndarray
    cwp: np.ndarray  # (nt, ncell) condensed water path of the whole column, kg m^-2
    vec_cells: np.ndarray  # (m,) wind sample: cell and shown-level index of each arrow
    vec_levels: np.ndarray
    uvw: np.ndarray  # (nt, m, 3) m/s
    w_range: np.ndarray  # (nt, 2) min and max of w over all cells and shown levels, m/s
    has_w: bool
    spacing_km: float  # typical distance between neighbouring cell centres
    liquid_names: list[str]
    ice_names: list[str]


def wind_sample(
    x_km: np.ndarray,
    y_km: np.ndarray,
    xwin: tuple[float, float],
    ywin: tuple[float, float],
    z_km: np.ndarray,
    dx_km: float,
    dz_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cells nearest to a regular lattice of spacing dx_km and the levels nearest to every dz_km."""
    lx, ly = xwin[1] - xwin[0], ywin[1] - ywin[0]
    nx, ny = max(1, round(lx / dx_km)), max(1, round(ly / dx_km))
    gx = xwin[0] + (np.arange(nx) + 0.5) * lx / nx
    gy = ywin[0] + (np.arange(ny) + 0.5) * ly / ny
    gx, gy = (a.ravel() for a in np.meshgrid(gx, gy))
    tree = spatial.cKDTree(np.column_stack([(x_km - xwin[0]) % lx, (y_km - ywin[0]) % ly]), boxsize=[lx, ly])
    _, cells = tree.query(np.column_stack([(gx - xwin[0]) % lx, (gy - ywin[0]) % ly]))
    cells = np.unique(cells)
    targets = np.arange(0.5 * dz_km, z_km.max(), dz_km)
    if targets.size == 0:  # view lower than half the arrow spacing: one layer in the middle
        targets = np.array([0.5 * z_km.max()])
    levels = np.unique(np.array([int(np.argmin(np.abs(z_km - t))) for t in targets], dtype=np.int64))
    cc, ll = np.meshgrid(cells, levels)
    return cc.ravel(), ll.ravel()


def load(args: argparse.Namespace) -> Fields:  # noqa: PLR0915 [too-many-statements] linear reading
    directory = resolve_output_dir(args.output_dir.resolve(), args.group)
    data, grid = open_output(directory, args.group)
    ncell = data.sizes["cell"]
    if ncell > MAX_CELLS:
        raise SystemExit(
            f"error: {ncell} cells; this 3D point view is meant for coarse runs (<= {MAX_CELLS} cells)"
        )
    nt = data.sizes["time"]
    if args.times and any(not -nt <= i < nt for i in args.times):
        raise SystemExit(f"error: --times must be output indices in 0..{nt - 1}")
    its = [i % nt for i in args.times] if args.times else list(range(0, nt, args.stride))
    t = data["time"].values.astype(float)
    t_min = (t[its] - t[0]) / 60.0

    # horizontal: cell centres (m) wrapped into one periodic window, in km
    lx = float(grid.attrs["domain_length"])
    ly = float(grid.attrs["domain_height"])
    cx = grid["cell_circumcenter_cartesian_x"].values
    cy = grid["cell_circumcenter_cartesian_y"].values
    xwin = periodic_window(cx, lx)
    ywin = periodic_window(cy, ly)
    x_km = ((cx - xwin[0]) % lx + xwin[0]) * 1e-3
    y_km = ((cy - ywin[0]) % ly + ywin[0]) * 1e-3
    xwin_km = (xwin[0] * 1e-3, xwin[1] * 1e-3)
    ywin_km = (ywin[0] * 1e-3, ywin[1] * 1e-3)
    polys, owner, _, _ = cell_polygons(grid)
    # equilateral triangles of area A: edge a = sqrt(4 A / sqrt(3)), centre distance a / sqrt(3)
    edge = np.sqrt(4.0 * float(np.median(grid["cell_area"].values)) / np.sqrt(3.0))
    spacing_km = edge / np.sqrt(3.0) * 1e-3

    # vertical: full levels (top to bottom)
    h = data["height"].values
    z_all = 0.5 * (h[:-1] + h[1:]) * 1e-3
    all_levels = np.arange(z_all.size)
    liquid_names = [n for n in LIQUID if n in data]
    ice_names = [n for n in ICE if n in data]
    if not liquid_names and not ice_names:
        raise SystemExit(f"error: none of {LIQUID + ICE} in the {args.group} output")
    print(f"reading {len(its)} output times of {ncell} cells x {z_all.size} levels from {directory}")
    liquid = read_sum(data, LIQUID, its, all_levels) * 1e3  # g/kg
    ice = read_sum(data, ICE, its, all_levels) * 1e3
    dz = layer_thickness(data, None)  # (level, 1) m
    rho = data["air_density"].isel(time=its).values
    cwp = (rho * (liquid + ice) * 1e-3 * dz[None]).sum(axis=1)  # kg m^-2

    # levels to show
    if args.zmax is not None:
        zmax = args.zmax
    else:
        cloudy = ((liquid > args.threshold) | (ice > args.threshold)).any(axis=(0, 2))
        zmax = float(z_all[cloudy].max()) + 1.5 if cloudy.any() else float(h[0]) * 1e-3
        zmax = min(float(np.ceil(zmax)), float(h[0]) * 1e-3)
    shown = np.nonzero(z_all <= zmax)[0]
    if shown.size == 0:
        raise SystemExit(f"error: --zmax {zmax} km is below the lowest model level ({z_all.min():.3f} km)")
    z_km = z_all[shown]
    liquid = liquid[:, shown]
    ice = ice[:, shown]

    # wind sample
    vec_cells, vec_levels = wind_sample(x_km, y_km, xwin_km, ywin_km, z_km, args.vec_dx, args.vec_dz)
    uvw = np.zeros((len(its), vec_cells.size, 3), dtype=np.float32)
    w_range = np.zeros((len(its), 2))
    sample_levels = shown[vec_levels]
    for k, name in enumerate(("eastward_wind", "northward_wind")):
        if name in data:
            field = data[name].isel(time=its).values
            uvw[:, :, k] = field[:, sample_levels, vec_cells]
    has_w = "upward_air_velocity" in data
    if not has_w:
        print("note: no upward_air_velocity in the output, the arrows are horizontal")
    if has_w:
        w_half = data["upward_air_velocity"].isel(time=its).values
        w_full = 0.5 * (w_half[:, :-1] + w_half[:, 1:])
        uvw[:, :, 2] = w_full[:, sample_levels, vec_cells]
        w_range = np.stack([w_full[:, shown].min(axis=(1, 2)), w_full[:, shown].max(axis=(1, 2))], axis=1)
    if args.relative_wind:
        uvw = relative_wind(uvw, vec_levels)

    label = args.title or (f"{directory.parent.name}/{directory.name}" if directory.parent.name else directory.name)
    return Fields(
        label=label,
        t_min=t_min,
        x_km=x_km,
        y_km=y_km,
        xwin=xwin_km,
        ywin=ywin_km,
        z_km=z_km,
        zmax_km=zmax,
        polys_km=polys,
        poly_owner=owner,
        liquid=liquid,
        ice=ice,
        cwp=cwp,
        vec_cells=vec_cells,
        vec_levels=vec_levels,
        uvw=uvw,
        w_range=w_range,
        has_w=has_w,
        spacing_km=spacing_km,
        liquid_names=liquid_names,
        ice_names=ice_names,
    )


def relative_wind(uvw: np.ndarray, vec_levels: np.ndarray) -> np.ndarray:
    """u and v minus their mean over the sampled columns at each level (w unchanged)."""
    out = uvw.copy()
    for lev in np.unique(vec_levels):
        sel = vec_levels == lev
        out[:, sel, :2] -= uvw[:, sel, :2].mean(axis=1, keepdims=True)
    return out


# --------------------------------------------------------------------------------------
# scales shared by all frames
# --------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Scales:
    threshold: float  # g/kg
    liquid_max: float
    ice_max: float
    speed_ref: float  # m/s, drawn with an arrow of length --vec-length * --vec-dx
    w_max: float  # m/s, end of the colour scale of w
    cwp_max: float  # kg m^-2


def nice_ceil(value: float) -> float:
    """Smallest 1, 2, 2.5 or 5 times a power of ten that is >= value."""
    if value <= 0.0:
        return 1.0
    p = 10.0 ** np.floor(np.log10(value))
    return float(next(m * p for m in (1.0, 2.0, 2.5, 5.0, 10.0) if m * p >= value * (1 - 1e-9)))


def scales(f: Fields, threshold: float) -> Scales:
    speed = np.linalg.norm(f.uvw, axis=-1)
    return Scales(
        threshold=threshold,
        liquid_max=max(float(f.liquid.max()), 10 * threshold),
        ice_max=max(float(f.ice.max()), 10 * threshold),
        speed_ref=nice_ceil(float(np.percentile(speed, 99))) if speed.size else 1.0,
        w_max=nice_ceil(float(np.abs(f.uvw[..., 2]).max())) if f.uvw.size else 1.0,
        cwp_max=max(nice_ceil(float(f.cwp.max())), 10 * CWP_MIN),
    )


def log_fraction(q: np.ndarray, qmin: float, qmax: float) -> np.ndarray:
    """log(q / qmin) / log(qmax / qmin), clipped to [0, 1]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.log(np.maximum(q, 1e-30) / qmin) / np.log(qmax / qmin)
    return np.clip(np.nan_to_num(frac, nan=0.0), 0.0, 1.0)


# --------------------------------------------------------------------------------------
# movie (matplotlib)
# --------------------------------------------------------------------------------------
_MOVIE: dict = {}  # set before the worker processes are forked


def voxels(f: Fields, field: np.ndarray, it: int, qmin: float, qmax: float):
    """Positions (km) and log fraction of the shown voxels above qmin at output it."""
    lev, cell = np.nonzero(field[it] > qmin)
    frac = log_fraction(field[it, lev, cell], qmin, qmax)
    return f.x_km[cell], f.y_km[cell], f.z_km[lev], frac


def draw_wind(ax, f: Fields, s: Scales, args: argparse.Namespace, it: int) -> None:
    """Wind arrows with the tail at the sample point, coloured by w."""
    if args.no_wind or not f.uvw.size:
        return
    uvw = f.uvw[it].astype(float)
    keep = np.linalg.norm(uvw, axis=1) > max(args.vec_min_speed, 1e-6)
    if not keep.any():
        return
    length = args.vec_length * args.vec_dx / s.speed_ref  # km per m/s
    wc = W_CMAP(mpl_colors.Normalize(-s.w_max, s.w_max, clip=True)(uvw[keep, 2]))
    wc[:, 3] = 0.85
    ax.quiver(
        f.x_km[f.vec_cells[keep]],
        f.y_km[f.vec_cells[keep]],
        f.z_km[f.vec_levels[keep]],
        uvw[keep, 0] * length,
        uvw[keep, 1] * length,
        uvw[keep, 2] * length,
        length=1.0,
        pivot="tail",
        arrow_length_ratio=0.3,
        colors=np.concatenate([wc, wc, wc]),  # shafts, left heads, right heads
        linewidths=0.8,
    )


def render_frame(k: int) -> pathlib.Path:  # noqa: PLR0915 [too-many-statements] one figure
    f: Fields = _MOVIE["fields"]
    s: Scales = _MOVIE["scales"]
    args = _MOVIE["args"]
    it = _MOVIE["its"][k]
    out = _MOVIE["frame_dir"] / f"frame_{k:04d}.png"

    fig = plt.figure(figsize=(args.width / args.dpi, args.height / args.dpi), dpi=args.dpi)
    ax = fig.add_axes([0.0, -0.05, 0.8, 1.0], projection="3d", computed_zorder=False)
    lx, ly = f.xwin[1] - f.xwin[0], f.ywin[1] - f.ywin[0]
    ax.set_xlim(*f.xwin)
    ax.set_ylim(*f.ywin)
    ax.set_zlim(0.0, f.zmax_km)
    ax.set_box_aspect((lx, ly, f.zmax_km * args.zexag), zoom=args.zoom)
    ax.view_init(elev=args.elev, azim=args.azim + args.rotate * k)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.96, 0.96, 0.97, 1.0))
        axis._axinfo["grid"].update(color=(0.85, 0.85, 0.87, 1.0), linewidth=0.5)

    # floor: column condensed water path
    if not args.no_floor:
        cwp = f.cwp[it][f.poly_owner]
        floor_norm = mpl_colors.LogNorm(CWP_MIN, s.cwp_max, clip=True)
        colors = FLOOR_CMAP(floor_norm(np.maximum(cwp, CWP_MIN)))
        colors[cwp < CWP_MIN, 3] = 0.0
        verts = np.concatenate([f.polys_km, np.zeros((*f.polys_km.shape[:2], 1))], axis=-1)
        keep = cwp >= CWP_MIN
        if keep.any():
            floor = art3d.Poly3DCollection(verts[keep], facecolors=colors[keep], edgecolors="none")
            ax.add_collection3d(floor)

    draw_wind(ax, f, s, args, it)

    # hydrometeors: one collection, so matplotlib sorts all dots by depth
    point_size = (args.point_size * f.spacing_km * _MOVIE["px_per_km"] * 72.0 / args.dpi) ** 2  # pt^2
    xs, ys, zs, rgba, sizes = [], [], [], [], []
    for field, cmap, qmax, show in (
        (f.liquid, LIQUID_CMAP, s.liquid_max, not args.no_liquid),
        (f.ice, ICE_CMAP, s.ice_max, not args.no_ice),
    ):
        if not show:
            continue
        x, y, z, frac = voxels(f, field, it, s.threshold, qmax)
        c = cmap(frac)
        c[:, 3] = args.alpha * (0.25 + 0.75 * frac)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        rgba.append(c)
        sizes.append(point_size * (0.35 + 0.65 * frac))
    if xs and sum(a.size for a in xs):
        ax.scatter(
            np.concatenate(xs),
            np.concatenate(ys),
            np.concatenate(zs),
            c=np.concatenate(rgba),
            s=np.concatenate(sizes),
            marker="o",
            linewidths=0,
            depthshade=False,
        )

    # labels and colour scales
    fig.text(0.02, 0.965, f"t = {f.t_min[k]:5.1f} min", fontsize=15, family="monospace", va="top")
    fig.text(
        0.02,
        0.925,
        f"{f.label}   liquid {'+'.join(f.liquid_names)} (blue)   ice {'+'.join(f.ice_names)} (magenta)   "
        f"q > {s.threshold:g} g/kg   max liquid {f.liquid[it].max():.2f}, ice {f.ice[it].max():.2f} g/kg",
        fontsize=8.5,
        va="top",
        color="0.25",
    )
    bars = [
        (LIQUID_CMAP, mpl_colors.LogNorm(s.threshold, s.liquid_max), f"liquid {'+'.join(f.liquid_names)} (g/kg)", not args.no_liquid and f.liquid_names),
        (ICE_CMAP, mpl_colors.LogNorm(s.threshold, s.ice_max), f"ice {'+'.join(f.ice_names)} (g/kg)", not args.no_ice and f.ice_names),
        (W_CMAP, mpl_colors.Normalize(-s.w_max, s.w_max), (f"arrows: colour w (m/s), {s.speed_ref:g} m/s = {args.vec_length * args.vec_dx:g} km" if f.has_w else f"arrows (no w): {s.speed_ref:g} m/s = {args.vec_length * args.vec_dx:g} km"), not args.no_wind),
        (FLOOR_CMAP, mpl_colors.LogNorm(CWP_MIN, s.cwp_max), "floor: LWP+IWP (kg/m²)", not args.no_floor),
    ]
    y0 = 0.80
    for cmap, norm, title, show in bars:
        if not show:
            continue
        cax = fig.add_axes([0.835, y0, 0.14, 0.022])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=7)
        cax.set_title(title, fontsize=8, loc="left")
        y0 -= 0.12
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    return out


def write_movie(f: Fields, s: Scales, args: argparse.Namespace, plot_dir: pathlib.Path) -> None:
    frame_dir = plot_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("frame_*.png"):
        old.unlink()
    # about how many pixels one km of the horizontal domain spans (sets the dot size)
    px_per_km = 0.55 * args.width * args.zoom / (f.xwin[1] - f.xwin[0])
    _MOVIE.update(fields=f, scales=s, args=args, its=list(range(f.t_min.size)), frame_dir=frame_dir, px_per_km=px_per_km)
    n = f.t_min.size
    if args.jobs > 1 and n > 1:
        with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
            frames = []
            for i, path in enumerate(pool.imap(render_frame, range(n))):
                frames.append(path)
                print(f"\rframe {i + 1}/{n}", end="", flush=True)
    else:
        frames = []
        for i in range(n):
            frames.append(render_frame(i))
            print(f"\rframe {i + 1}/{n}", end="", flush=True)
    print(f"\nframes: {frame_dir}/frame_NNNN.png")

    gif = plot_dir / "cloud3d.gif"
    images = []
    for path in frames:
        image = Image.open(path).convert("RGB")
        if 0 < args.gif_width < image.width:
            image = image.resize((args.gif_width, round(image.height * args.gif_width / image.width)), Image.Resampling.LANCZOS)
        images.append(image.quantize(colors=255, method=Image.Quantize.MEDIANCUT))
    images[0].save(gif, save_all=True, append_images=images[1:], duration=round(1000 / args.fps), loop=0)
    print(f"animation: {gif}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        mp4 = plot_dir / "cloud3d.mp4"
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(frame_dir / "frame_%04d.png"),
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)]
        subprocess.run(cmd, check=True)
        print(f"video: {mp4}")
    else:
        print("note: no ffmpeg on PATH, so no MP4 (convert the frames elsewhere, e.g.\n"
              f"  ffmpeg -framerate {args.fps} -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p cloud3d.mp4)")


# --------------------------------------------------------------------------------------
# interactive viewer (three.js)
# --------------------------------------------------------------------------------------
def b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def viewer_records(f: Fields, s: Scales, args: argparse.Namespace) -> list[dict]:
    """Header and one record per output time; arrays are base64 little-endian binaries.

    Voxel index = shown level * ncell + cell; voxel value byte = 255 * log fraction between
    the export threshold (threshold / 10) and the maximum; floor byte 0 = below CWP_MIN,
    1..255 = log fraction up to the maximum; wind int8 = component / wind_unit.
    """
    qmin = s.threshold / 10.0
    ncell = f.x_km.size
    wind_unit = max(float(np.abs(f.uvw).max()) / 127.0, 1e-3) if f.uvw.size else 1.0
    header = {
        "kind": "header",
        "title": f.label,
        "nCell": int(ncell),
        "x": b64(f.x_km, "<f4"),
        "y": b64(f.y_km, "<f4"),
        "z": b64(f.z_km, "<f4"),
        "xwin": list(f.xwin),
        "ywin": list(f.ywin),
        "zmax": f.zmax_km,
        "spacing": f.spacing_km,
        "times": [round(float(t), 3) for t in f.t_min],
        "liquid": {"names": f.liquid_names, "min": qmin, "max": s.liquid_max, "threshold": s.threshold},
        "ice": {"names": f.ice_names, "min": qmin, "max": s.ice_max, "threshold": s.threshold},
        "wind": {
            "cell": b64(f.vec_cells, "<u4"),
            "level": b64(f.vec_levels, "<u4"),
            "unit": wind_unit,
            "speedRef": s.speed_ref,
            "wMax": s.w_max,
            "dx": args.vec_dx,
            "dz": args.vec_dz,
            "lengthFactor": args.vec_length,
            "relative": bool(args.relative_wind),
            "hasW": f.has_w,
        },
        "floor": {
            "tri": b64(f.polys_km.reshape(-1), "<f4"),
            "owner": b64(f.poly_owner, "<u4"),
            "min": CWP_MIN,
            "max": s.cwp_max,
        },
        "zexag": args.zexag,
    }
    records = [header]
    for it in range(f.t_min.size):
        rec = {
            "t": round(float(f.t_min[it]), 3),
            "s": [round(float(v), 4) for v in (f.liquid[it].max(), f.ice[it].max(), *f.w_range[it])],
        }
        for key, field, qmax in (("L", f.liquid, s.liquid_max), ("I", f.ice, s.ice_max)):
            lev, cell = np.nonzero(field[it] > qmin)
            val = np.rint(255 * log_fraction(field[it, lev, cell], qmin, qmax))
            rec[key] = {"i": b64(lev * ncell + cell, "<u4"), "v": b64(val, "u1")}
        rec["W"] = b64(np.clip(np.rint(f.uvw[it] / wind_unit), -127, 127).reshape(-1), "i1")
        cwp = f.cwp[it]
        floor = np.where(cwp >= CWP_MIN, 1 + np.rint(254 * log_fraction(cwp, CWP_MIN, s.cwp_max)), 0)
        rec["F"] = b64(floor, "u1")
        records.append(rec)
    return records


def write_viewer(f: Fields, s: Scales, args: argparse.Namespace, plot_dir: pathlib.Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    records = viewer_records(f, s, args)
    # one record per line; "<" escaped so the JSON can sit inside a <script> element
    text = "[\n" + ",\n".join(json.dumps(r, separators=(",", ":")) for r in records) + "\n]\n"
    text = text.replace("<", "\\u003c")
    html = plot_dir / f"{args.html_name}.html"
    title = args.title or "Cloud 3D Viewer"
    page = VIEWER_TEMPLATE
    if args.html_split:
        data = plot_dir / f"{args.html_name}_data.json"
        data.write_text(text)
        page = page.replace("__DATA_SRC__", data.name).replace("__DATA_INLINE__", "")
        print(f"viewer data: {data} ({data.stat().st_size / 1e6:.1f} MB)")
    else:
        page = page.replace("__DATA_SRC__", "").replace("__DATA_INLINE__", text)
    # the title only in the part before the data, so neither substitution can rewrite the other
    head, sep, rest = page.partition('<script id="cloud3d-data"')
    page = head.replace("__TITLE__", title.replace("&", "&amp;").replace("<", "&lt;")) + sep + rest
    if not args.html_fragment:
        page = (
            '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "</head>\n<body>\n" + page + "\n</body>\n</html>\n"
        )
    html.write_text(page)
    print(f"viewer: {html} ({html.stat().st_size / 1e6:.1f} MB)")


VIEWER_TEMPLATE = (pathlib.Path(__file__).with_name("plot_3d_movie_viewer.html")).read_text()


# --------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", type=pathlib.Path, help="driver output directory, or the directory above it")
    p.add_argument("--group", default="atmospheric_state", help="basename of the field group with the 3D fields")
    p.add_argument("--plot-dir", type=pathlib.Path, default=None, help="default: <output_dir>/plots_3d")
    p.add_argument("--movie", action="store_true", help="write the matplotlib movie")
    p.add_argument("--html", action="store_true", help="write the interactive viewer")
    p.add_argument("--stride", type=int, default=1, help="use every n-th output time")
    p.add_argument("--times", type=int, nargs="*", default=None, help="explicit output time indices")
    p.add_argument("--title", default=None, help="title of the frames and the viewer page")
    p.add_argument("--threshold", type=float, default=0.01, help="smallest liquid / ice content drawn (g/kg)")
    p.add_argument("--zmax", type=float, default=None, help="top of the view (km); default: above the highest cloud")
    p.add_argument("--zexag", type=float, default=2.5, help="vertical exaggeration")
    p.add_argument("--vec-dx", type=float, default=10.0, help="horizontal spacing of the wind arrows (km)")
    p.add_argument("--vec-dz", type=float, default=2.5, help="vertical spacing of the wind arrows (km)")
    p.add_argument("--vec-length", type=float, default=0.9, help="arrow length of the reference speed, in units of --vec-dx")
    p.add_argument("--vec-min-speed", type=float, default=0.0, help="hide arrows slower than this (m/s)")
    p.add_argument("--relative-wind", action="store_true", help="subtract the horizontal mean u, v at each level")
    mg = p.add_argument_group("movie")
    mg.add_argument("--elev", type=float, default=22.0, help="camera elevation (deg)")
    mg.add_argument("--azim", type=float, default=-60.0, help="camera azimuth of the first frame (deg)")
    mg.add_argument("--rotate", type=float, default=1.0, help="azimuth increment per frame (deg)")
    mg.add_argument("--zoom", type=float, default=1.1, help="zoom of the 3D box")
    mg.add_argument("--point-size", type=float, default=1.4, help="dot diameter relative to the cell spacing")
    mg.add_argument("--alpha", type=float, default=0.1, help="opacity of the densest dots")
    mg.add_argument("--no-liquid", action="store_true")
    mg.add_argument("--no-ice", action="store_true")
    mg.add_argument("--no-wind", action="store_true")
    mg.add_argument("--no-floor", action="store_true")
    mg.add_argument("--width", type=int, default=1600, help="frame width (px)")
    mg.add_argument("--height", type=int, default=1000, help="frame height (px)")
    mg.add_argument("--dpi", type=int, default=120)
    mg.add_argument("--fps", type=int, default=8)
    mg.add_argument("--gif-width", type=int, default=960, help="width of the GIF (px); 0: frame width")
    mg.add_argument("--jobs", type=int, default=4, help="frames rendered in parallel")
    hg = p.add_argument_group("viewer")
    hg.add_argument("--html-name", default="cloud3d", help="basename of the viewer file")
    hg.add_argument("--html-split", action="store_true", help="write the data to <name>_data.json")
    hg.add_argument("--html-fragment", action="store_true", help="omit the <html>/<head>/<body> wrapper")
    args = p.parse_args()
    if args.stride < 1:
        p.error("--stride must be >= 1")
    if args.threshold <= 0:
        p.error("--threshold must be > 0")
    if not args.movie and not args.html:
        args.movie = args.html = True

    f = load(args)
    s = scales(f, args.threshold)
    plot_dir = args.plot_dir or resolve_output_dir(args.output_dir, args.group) / "plots_3d"
    print(
        f"view: {f.xwin[1] - f.xwin[0]:.0f} x {f.ywin[1] - f.ywin[0]:.0f} x {f.zmax_km:.1f} km "
        f"({f.z_km.size} levels), {f.vec_cells.size} wind arrows, max liquid {s.liquid_max:.3g} g/kg, "
        f"max ice {s.ice_max:.3g} g/kg, |w| <= {s.w_max:g} m/s"
    )
    if args.html:
        write_viewer(f, s, args, plot_dir)
    if args.movie:
        write_movie(f, s, args, plot_dir)


if __name__ == "__main__":
    main()
