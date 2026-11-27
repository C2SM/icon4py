#!/usr/bin/env python
"""3D movie of the five hydrometeor species and the wind of an ICON4Py torus run.

Like plot_3d_movie.py, but every species is its own layer with its own colour instead of
being summed into liquid and ice:

  qc  cloud water   cyan
  qi  cloud ice     green
  qr  rain          blue
  qs  snow          magenta
  qg  graupel       amber
  wind  (u, v, w) arrows on a coarse sample of the columns and levels, shaded by w in grey
        so they do not compete with the five species colours (movie: light grey = down,
        black = up; viewer: dim slate = down, white = up)

Two outputs (both by default):

  --movie  a slowly rotating matplotlib 3D view per output time: PNG frames, an animated
           GIF and, if ffmpeg is on PATH, an MP4. Every cell whose specific content of a
           species exceeds --threshold is a translucent dot of that species' colour at its
           centre whose opacity and size grow with log q (fixed scale per species over all
           times); the floor shows the column condensed water path
           sum_k rho (qc + qi + qr + qs + qg) dz as a grey shadow.
  --html   a self-contained interactive WebGL viewer (three.js): drag to rotate, scroll to
           zoom, play/pause and scrub through time, toggle each species, the wind and the
           floor, change the threshold, the vertical exaggeration, the point size and the
           arrow length, and show the wind relative to its horizontal mean at each height.
           Copy the .html file to a laptop and open it in a browser (it loads three.js from
           a CDN). With --html-split, the data is written next to it as <name>_data.json.

Coordinates, level selection and the wind sample are the same as in plot_3d_movie.py.
Species that are not in the output are skipped (e.g. qg with a scheme without graupel).

Output (in --plot-dir, default <output_dir>/plots_3d_species):
  frames/frame_NNNN.png, species3d.gif (and species3d.mp4)   --movie
  species3d.html (and species3d_data.json)                   --html

Usage (inside the uenv, with the icon4py venv activated):
    python plot_3d_species_movie.py output_sequential/output_sample
    python plot_3d_species_movie.py output_sequential --html --stride 2
    python plot_3d_species_movie.py output_sequential --movie --times 30 60 90 --azim -50 --rotate 0
    python plot_3d_species_movie.py output_sequential --movie --species qr qs qg --no-wind
"""

import argparse
import dataclasses
import json
import multiprocessing
import pathlib
import shutil
import subprocess

import matplotlib


matplotlib.use("Agg")
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import art3d
from PIL import Image

from plot_3d_movie import (
    CWP_MIN,
    FLOOR_CMAP,
    MAX_CELLS,
    b64,
    log_fraction,
    nice_ceil,
    relative_wind,
    resolve_output_dir,
    wind_sample,
)
from plot_water_path_map import cell_polygons, periodic_window
from plot_y_distribution import layer_thickness, open_output


@dataclasses.dataclass(frozen=True)
class Species:
    name: str  # output variable
    long_name: str
    colour_name: str
    movie_colours: tuple[str, str, str]  # light tint, base, dark shade on the light movie background
    viewer_colour: str  # bright base colour on the dark viewer scene


SPECIES = (
    Species("qc", "cloud water", "cyan", ("#9edcf2", "#1a9fd0", "#05506e"), "#3fd0ff"),
    Species("qi", "cloud ice", "green", ("#b9e8b3", "#2f9e44", "#0f4d1f"), "#4fe08a"),
    Species("qr", "rain", "blue", ("#9fb1f5", "#2a4fd6", "#0d1f66"), "#3d6bff"),
    Species("qs", "snow", "magenta", ("#fbb4d9", "#d6249f", "#6a0f5a"), "#ff5fcf"),
    Species("qg", "graupel", "amber", ("#fcd9a0", "#e69500", "#7a4a00"), "#ffb52e"),
)
SPECIES_BY_NAME = {sp.name: sp for sp in SPECIES}
CMAPS = {sp.name: mpl_colors.LinearSegmentedColormap.from_list(sp.name, sp.movie_colours) for sp in SPECIES}
# w: light grey (downward) to black (upward) on the light movie background
W_CMAP = mpl_colors.LinearSegmentedColormap.from_list("w_grey", ["#c4c4c4", "#7a7a7a", "#000000"])


# --------------------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------------------
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
    q: dict[str, np.ndarray]  # species name -> (nt, nlev, ncell) g/kg, in SPECIES order
    cwp: np.ndarray  # (nt, ncell) condensed water path of the whole column (all species read), kg m^-2
    vec_cells: np.ndarray  # (m,) wind sample: cell and shown-level index of each arrow
    vec_levels: np.ndarray
    uvw: np.ndarray  # (nt, m, 3) m/s
    w_range: np.ndarray  # (nt, 2) min and max of w over all cells and shown levels, m/s
    has_w: bool
    spacing_km: float  # typical distance between neighbouring cell centres


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

    # species: the requested ones that are in the output, in SPECIES order
    requested = [sp.name for sp in SPECIES if sp.name in args.species]
    missing = [n for n in requested if n not in data]
    if missing:
        print(f"note: {' '.join(missing)} not in the {args.group} output, skipped")
    names = [n for n in requested if n in data]
    if not names:
        raise SystemExit(f"error: none of {' '.join(requested)} in the {args.group} output")
    print(f"reading {len(its)} output times of {ncell} cells x {z_all.size} levels from {directory}")
    q = {n: data[n].isel(time=its).values.astype(np.float32) * np.float32(1e3) for n in names}  # g/kg
    dz = layer_thickness(data, None)  # (level, 1) m
    rho = data["air_density"].isel(time=its).values
    cwp = (rho * sum(q.values()) * 1e-3 * dz[None]).sum(axis=1)  # kg m^-2

    # levels to show
    if args.zmax is not None:
        zmax = args.zmax
    else:
        cloudy = np.zeros(z_all.size, dtype=bool)
        for field in q.values():
            cloudy |= (field > args.threshold).any(axis=(0, 2))
        zmax = float(z_all[cloudy].max()) + 1.5 if cloudy.any() else float(h[0]) * 1e-3
        zmax = min(float(np.ceil(zmax)), float(h[0]) * 1e-3)
    shown = np.nonzero(z_all <= zmax)[0]
    if shown.size == 0:
        raise SystemExit(f"error: --zmax {zmax} km is below the lowest model level ({z_all.min():.3f} km)")
    z_km = z_all[shown]
    q = {n: field[:, shown] for n, field in q.items()}

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
        q=q,
        cwp=cwp,
        vec_cells=vec_cells,
        vec_levels=vec_levels,
        uvw=uvw,
        w_range=w_range,
        has_w=has_w,
        spacing_km=spacing_km,
    )


# --------------------------------------------------------------------------------------
# scales shared by all frames
# --------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Scales:
    threshold: float  # g/kg
    q_max: dict[str, float]  # species name -> end of its colour scale, g/kg
    speed_ref: float  # m/s, drawn with an arrow of length --vec-length * --vec-dx
    w_max: float  # m/s, end of the colour scale of w
    cwp_max: float  # kg m^-2


def scales(f: Fields, threshold: float) -> Scales:
    speed = np.linalg.norm(f.uvw, axis=-1)
    return Scales(
        threshold=threshold,
        q_max={n: max(float(field.max()), 10 * threshold) for n, field in f.q.items()},
        speed_ref=nice_ceil(float(np.percentile(speed, 99))) if speed.size else 1.0,
        w_max=nice_ceil(float(np.abs(f.uvw[..., 2]).max())) if f.uvw.size else 1.0,
        cwp_max=max(nice_ceil(float(f.cwp.max())), 10 * CWP_MIN),
    )


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
    """Wind arrows with the tail at the sample point, shaded by w."""
    if args.no_wind or not f.uvw.size:
        return
    uvw = f.uvw[it].astype(float)
    keep = np.linalg.norm(uvw, axis=1) > max(args.vec_min_speed, 1e-6)
    if not keep.any():
        return
    length = args.vec_length * args.vec_dx / s.speed_ref  # km per m/s
    wc = W_CMAP(mpl_colors.Normalize(-s.w_max, s.w_max, clip=True)(uvw[keep, 2]))
    wc[:, 3] = 0.8
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
        verts = np.concatenate([f.polys_km, np.zeros((*f.polys_km.shape[:2], 1))], axis=-1)
        keep = cwp >= CWP_MIN
        if keep.any():
            floor = art3d.Poly3DCollection(verts[keep], facecolors=colors[keep], edgecolors="none")
            ax.add_collection3d(floor)

    draw_wind(ax, f, s, args, it)

    # hydrometeors: one collection, so matplotlib sorts all dots of all species by depth
    point_size = (args.point_size * f.spacing_km * _MOVIE["px_per_km"] * 72.0 / args.dpi) ** 2  # pt^2
    xs, ys, zs, rgba, sizes = [], [], [], [], []
    for name, field in f.q.items():
        x, y, z, frac = voxels(f, field, it, s.threshold, s.q_max[name])
        c = CMAPS[name](frac)
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
    maxima = "  ".join(f"{n} {field[it].max():.2f}" for n, field in f.q.items())
    fig.text(
        0.02,
        0.925,
        f"{f.label}   q > {s.threshold:g} g/kg   max (g/kg): {maxima}",
        fontsize=8.5,
        va="top",
        color="0.25",
    )
    bars = [
        (CMAPS[n], mpl_colors.LogNorm(s.threshold, s.q_max[n]), f"{n} {SPECIES_BY_NAME[n].long_name} (g/kg)", True)
        for n in f.q
    ]
    wind_title = (
        f"arrows: shade w (m/s), {s.speed_ref:g} m/s = {args.vec_length * args.vec_dx:g} km"
        if f.has_w
        else f"arrows (no w): {s.speed_ref:g} m/s = {args.vec_length * args.vec_dx:g} km"
    )
    bars.append((W_CMAP, mpl_colors.Normalize(-s.w_max, s.w_max), wind_title, not args.no_wind))
    bars.append((FLOOR_CMAP, mpl_colors.LogNorm(CWP_MIN, s.cwp_max), f"floor: path of {'+'.join(f.q)} (kg/m²)", not args.no_floor))
    y0 = 0.82
    for cmap, norm, title, show in bars:
        if not show:
            continue
        cax = fig.add_axes([0.835, y0, 0.14, 0.022])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=7)
        cax.set_title(title, fontsize=8, loc="left")
        y0 -= 0.105
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
    frames = []
    if args.jobs > 1 and n > 1:
        with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
            for i, path in enumerate(pool.imap(render_frame, range(n))):
                frames.append(path)
                print(f"\rframe {i + 1}/{n}", end="", flush=True)
    else:
        for i in range(n):
            frames.append(render_frame(i))
            print(f"\rframe {i + 1}/{n}", end="", flush=True)
    print(f"\nframes: {frame_dir}/frame_NNNN.png")

    gif = plot_dir / f"{args.name}.gif"
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
        mp4 = plot_dir / f"{args.name}.mp4"
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(frame_dir / "frame_%04d.png"),
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)]
        subprocess.run(cmd, check=True)
        print(f"video: {mp4}")
    else:
        print("note: no ffmpeg on PATH, so no MP4 (convert the frames elsewhere, e.g.\n"
              f"  ffmpeg -framerate {args.fps} -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p {args.name}.mp4)")


# --------------------------------------------------------------------------------------
# interactive viewer (three.js)
# --------------------------------------------------------------------------------------
def viewer_records(f: Fields, s: Scales, args: argparse.Namespace) -> list[dict]:
    """Header and one record per output time; arrays are base64 little-endian binaries.

    Voxel index = shown level * ncell + cell; voxel value byte = 255 * log fraction between
    the export threshold (threshold / 10) and the species maximum; floor byte 0 = below
    CWP_MIN, 1..255 = log fraction up to the maximum; wind int8 = component / wind_unit.
    Record "s" = [max of each species in header order..., w min, w max].
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
        "threshold": s.threshold,
        "species": [
            {
                "name": n,
                "longName": SPECIES_BY_NAME[n].long_name,
                "colourName": SPECIES_BY_NAME[n].colour_name,
                "colour": SPECIES_BY_NAME[n].viewer_colour,
                "min": qmin,
                "max": s.q_max[n],
            }
            for n in f.q
        ],
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
            "s": [round(float(v), 4) for v in (*(field[it].max() for field in f.q.values()), *f.w_range[it])],
            "Q": {},
        }
        for n, field in f.q.items():
            lev, cell = np.nonzero(field[it] > qmin)
            val = np.rint(255 * log_fraction(field[it, lev, cell], qmin, s.q_max[n]))
            rec["Q"][n] = {"i": b64(lev * ncell + cell, "<u4"), "v": b64(val, "u1")}
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
    html = plot_dir / f"{args.name}.html"
    title = args.title or "Hydrometeor 3D Viewer"
    page = VIEWER_TEMPLATE
    if args.html_split:
        data = plot_dir / f"{args.name}_data.json"
        data.write_text(text)
        page = page.replace("__DATA_SRC__", data.name).replace("__DATA_INLINE__", "")
        print(f"viewer data: {data} ({data.stat().st_size / 1e6:.1f} MB)")
    else:
        page = page.replace("__DATA_SRC__", "").replace("__DATA_INLINE__", text)
    # the title only in the part before the data, so neither substitution can rewrite the other
    head, sep, rest = page.partition('<script id="species3d-data"')
    page = head.replace("__TITLE__", title.replace("&", "&amp;").replace("<", "&lt;")) + sep + rest
    if not args.html_fragment:
        page = (
            '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "</head>\n<body>\n" + page + "\n</body>\n</html>\n"
        )
    html.write_text(page)
    print(f"viewer: {html} ({html.stat().st_size / 1e6:.1f} MB)")


VIEWER_TEMPLATE = (pathlib.Path(__file__).with_name("plot_3d_species_movie_viewer.html")).read_text()


# --------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", type=pathlib.Path, help="driver output directory, or the directory above it")
    p.add_argument("--group", default="atmospheric_state", help="basename of the field group with the 3D fields")
    p.add_argument("--plot-dir", type=pathlib.Path, default=None, help="default: <output_dir>/plots_3d_species")
    p.add_argument("--name", default="species3d", help="basename of the GIF, MP4 and viewer files")
    p.add_argument("--movie", action="store_true", help="write the matplotlib movie")
    p.add_argument("--html", action="store_true", help="write the interactive viewer")
    p.add_argument("--species", nargs="+", choices=list(SPECIES_BY_NAME), default=list(SPECIES_BY_NAME),
                   help="species to draw (default: all five)")
    p.add_argument("--stride", type=int, default=1, help="use every n-th output time")
    p.add_argument("--times", type=int, nargs="*", default=None, help="explicit output time indices")
    p.add_argument("--title", default=None, help="title of the frames and the viewer page")
    p.add_argument("--threshold", type=float, default=0.01, help="smallest specific content drawn (g/kg)")
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
    mg.add_argument("--no-wind", action="store_true")
    mg.add_argument("--no-floor", action="store_true")
    mg.add_argument("--width", type=int, default=1600, help="frame width (px)")
    mg.add_argument("--height", type=int, default=1000, help="frame height (px)")
    mg.add_argument("--dpi", type=int, default=120)
    mg.add_argument("--fps", type=int, default=8)
    mg.add_argument("--gif-width", type=int, default=960, help="width of the GIF (px); 0: frame width")
    mg.add_argument("--jobs", type=int, default=4, help="frames rendered in parallel")
    hg = p.add_argument_group("viewer")
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
    plot_dir = args.plot_dir or resolve_output_dir(args.output_dir, args.group) / "plots_3d_species"
    maxima = ", ".join(f"{n} {v:.3g}" for n, v in s.q_max.items())
    print(
        f"view: {f.xwin[1] - f.xwin[0]:.0f} x {f.ywin[1] - f.ywin[0]:.0f} x {f.zmax_km:.1f} km "
        f"({f.z_km.size} levels), {f.vec_cells.size} wind arrows, max (g/kg) {maxima}, |w| <= {s.w_max:g} m/s"
    )
    if args.html:
        write_viewer(f, s, args, plot_dir)
    if args.movie:
        write_movie(f, s, args, plot_dir)


if __name__ == "__main__":
    main()
