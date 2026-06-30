#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the ICON4Py logo as scalable vector graphics (SVG).

The logo is a true 3D icosahedron (the icosahedral base grid is ICON4Py's
defining geometry) projected to 2D through a parametrizable perspective
camera. Markers identify the topological entities of the mesh:

- colored circles at the 20 cell centres (triangular face centroids),
- colored squares at the 30 edge midpoints,
- nothing on the 12 vertices.

Every marker is defined as a flat shape lying in the tangent plane at its
position in 3D and only then projected, so it is foreshortened with the
correct perspective on faces that are tilted away from the viewer. Faces
are solid-shaded and hidden (back-facing) faces and markers are culled.

All features are toggleable and the camera (azimuth, elevation, roll,
perspective strength) is configurable, so the logo can be regenerated in
many variations from a single source. By default two files are written:
an icon-only SVG and an icon + 'ICON4Py' wordmark SVG.

Example:
    ./scripts/run generate-logo --azimuth 25 --elevation 18
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import sys
from typing import Annotated, Final

import drawsvg as draw
import numpy as np
import typer
from helpers import common


cli = typer.Typer(no_args_is_help=True, help=__doc__)

PHI: Final[float] = (1.0 + math.sqrt(5.0)) / 2.0


# --------------------------------------------------------------------------- #
# Colour palettes
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class Palette:
    """Colours used to render the logo.

    ``face_low`` and ``face_high`` are interpolated across the icosahedron by
    the (unrotated) height of each face centroid to give a subtle sky/ocean
    gradient before Lambert shading is applied. When ``per_face_hue`` is set,
    each face instead gets a distinct hue and the gradient colours are ignored.
    """

    face_low: tuple[int, int, int]
    face_high: tuple[int, int, int]
    circle: str
    circle_stroke: str
    square: str
    square_stroke: str
    edge: str
    wordmark: str
    per_face_hue: bool = False


PALETTES: Final[dict[str, Palette]] = {
    # Sky/ocean blues and greens evoking climate and weather modelling.
    "atmosphere": Palette(
        face_low=(31, 158, 138),  # sea green
        face_high=(47, 111, 176),  # sky blue
        circle="#ffc857",  # warm gold for cell centres
        circle_stroke="#9c6b10",
        square="#eef6ff",  # pale sky for edge midpoints
        square_stroke="#2f6fb0",
        edge="#16384f",
        wordmark="#1f3a4d",
    ),
    # Each face a distinct hue; playful, GT4Py-inspired energy.
    "vibrant": Palette(
        face_low=(0, 0, 0),
        face_high=(0, 0, 0),
        circle="#ffffff",
        circle_stroke="#222222",
        square="#111827",
        square_stroke="#ffffff",
        edge="#111827",
        wordmark="#1f2937",
        per_face_hue=True,
    ),
    # Neutral slate icosahedron with bright accents reserved for the markers.
    "mono": Palette(
        face_low=(70, 84, 99),
        face_high=(120, 134, 150),
        circle="#ff5d5d",
        circle_stroke="#7a1f1f",
        square="#5dd0ff",
        square_stroke="#0d3b52",
        edge="#1b2530",
        wordmark="#1b2530",
    ),
}


# --------------------------------------------------------------------------- #
# Icosahedron geometry
# --------------------------------------------------------------------------- #
def icosahedron() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the unit icosahedron as ``(vertices, edges, faces)``.

    ``vertices`` has shape ``(12, 3)`` (on the unit sphere), ``edges`` shape
    ``(30, 2)`` and ``faces`` shape ``(20, 3)``; the latter two index into
    ``vertices``. Faces are oriented counter-clockwise as seen from outside,
    so their geometric normal points outward.
    """
    patterns = ((0.0, 1.0, PHI), (1.0, PHI, 0.0), (PHI, 0.0, 1.0))
    coords = []
    for pattern in patterns:
        nonzero = [i for i, c in enumerate(pattern) if c != 0.0]
        for sa in (1.0, -1.0):
            for sb in (1.0, -1.0):
                v = list(pattern)
                v[nonzero[0]] *= sa
                v[nonzero[1]] *= sb
                coords.append(v)
    vertices = np.asarray(coords, dtype=float)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)

    edges, faces = _edges_and_faces(vertices)
    return vertices, edges, faces


def _edges_and_faces(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive edges and outward-oriented faces from the vertex positions."""
    n = len(vertices)
    dist = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=2)
    edge_len = dist[dist > 1e-9].min()
    adjacent = np.abs(dist - edge_len) < 1e-6

    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adjacent[i, j]]
    faces = [
        _orient_outward(vertices, i, j, k)
        for i in range(n)
        for j in range(i + 1, n)
        for k in range(j + 1, n)
        if adjacent[i, j] and adjacent[j, k] and adjacent[i, k]
    ]

    if len(edges) != 30 or len(faces) != 20:
        raise RuntimeError(
            f"Degenerate icosahedron: expected 30 edges and 20 faces, "
            f"got {len(edges)} edges and {len(faces)} faces."
        )
    return np.asarray(edges), np.asarray(faces)


def _orient_outward(vertices: np.ndarray, i: int, j: int, k: int) -> tuple[int, int, int]:
    """Order a face's vertices so its geometric normal points away from origin."""
    normal = np.cross(vertices[j] - vertices[i], vertices[k] - vertices[i])
    centroid = (vertices[i] + vertices[j] + vertices[k]) / 3.0
    return (i, j, k) if np.dot(normal, centroid) > 0 else (i, k, j)


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Outward unit normal of each face."""
    a, b, c = (vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]])
    normals = np.cross(b - a, c - a)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors spanning the plane with the given normal."""
    n = _normalize(normal)
    reference = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _normalize(np.cross(reference, n))
    v = np.cross(n, u)
    return u, v


def planar_disc(
    center: np.ndarray, normal: np.ndarray, radius: float, samples: int = 48
) -> np.ndarray:
    """Outline of a circle lying in the tangent plane at ``center``."""
    u, v = _tangent_basis(normal)
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    return center + radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)


def planar_square(
    center: np.ndarray, normal: np.ndarray, size: float, orient_dir: np.ndarray | None = None
) -> np.ndarray:
    """Corners of a square lying in the tangent plane at ``center``.

    ``orient_dir`` (e.g. the edge direction) is projected into the plane to
    align one pair of sides, so the square reads as deliberate rather than
    arbitrarily rotated.
    """
    n = _normalize(normal)
    if orient_dir is not None:
        u = orient_dir - n * np.dot(orient_dir, n)
        u = _normalize(u) if np.linalg.norm(u) > 1e-9 else _tangent_basis(n)[0]
        v = np.cross(n, u)
    else:
        u, v = _tangent_basis(n)
    half = size / 2.0
    corners = np.array(
        [-half * u - half * v, half * u - half * v, half * u + half * v, -half * u + half * v]
    )
    return center + corners


# --------------------------------------------------------------------------- #
# Camera / projection
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class View:
    """A perspective camera looking at the origin down the +z axis.

    ``rot`` rotates the model before projection; ``distance`` is the camera
    distance from the origin and controls the strength of the perspective
    (smaller is stronger). The absolute scale is fitted to the canvas later,
    so only the ratio of object size to ``distance`` matters.
    """

    rot: np.ndarray
    distance: float
    eye: np.ndarray


def make_view(azimuth: float, elevation: float, roll: float, distance: float) -> View:
    """Build a :class:`View` from camera angles (degrees) and distance."""
    az, el, ro = np.radians([azimuth, elevation, roll])
    rot_y = np.array(
        [[math.cos(az), 0.0, math.sin(az)], [0.0, 1.0, 0.0], [-math.sin(az), 0.0, math.cos(az)]]
    )
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(el), -math.sin(el)], [0.0, math.sin(el), math.cos(el)]]
    )
    rot_z = np.array(
        [[math.cos(ro), -math.sin(ro), 0.0], [math.sin(ro), math.cos(ro), 0.0], [0.0, 0.0, 1.0]]
    )
    rot = rot_z @ rot_x @ rot_y
    return View(rot=rot, distance=distance, eye=np.array([0.0, 0.0, distance]))


def project(view: View, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D screen coordinates (perspective divide).

    Returns the ``(M, 2)`` screen coordinates and the per-point depth (distance
    in front of the camera; larger is farther).
    """
    rotated = points @ view.rot.T
    depth = view.distance - rotated[:, 2]
    factor = 1.0 / np.clip(depth, 1e-6, None)
    screen = rotated[:, :2] * factor[:, None]
    return screen, depth


def front_facing(view: View, point: np.ndarray, normal: np.ndarray, min_cos: float = 0.0) -> bool:
    """Whether an outward-oriented feature faces the camera.

    ``normal`` must be a unit vector. ``min_cos`` is a cutoff on the cosine of
    the angle between the (rotated) normal and the direction to the camera:
    ``0.0`` keeps everything up to the silhouette, while a small positive value
    drops grazing features that would otherwise appear half-occluded at the rim.
    """
    point_rot = view.rot @ point
    normal_rot = view.rot @ normal
    to_eye = view.eye - point_rot
    cos = float(np.dot(normal_rot, to_eye) / np.linalg.norm(to_eye))
    return cos > min_cos


# --------------------------------------------------------------------------- #
# Shading
# --------------------------------------------------------------------------- #
def _hex(rgb: np.ndarray) -> str:
    r, g, b = (round(float(np.clip(c, 0, 255))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))[i]
    return np.array([r, g, b]) * 255.0


def face_base_color(palette: Palette, centroid_z: float, face_index: int) -> np.ndarray:
    """Base (unlit) RGB colour of a face."""
    if palette.per_face_hue:
        return _hsv_to_rgb((face_index / 20.0) % 1.0, 0.62, 0.92)
    t = (centroid_z + 1.0) / 2.0  # centroid lies on the unit sphere
    return (1.0 - t) * np.array(palette.face_low) + t * np.array(palette.face_high)


def shade(base: np.ndarray, normal_rot: np.ndarray, light: np.ndarray, ambient: float) -> str:
    """Apply Lambert shading (light is given in the camera/view frame)."""
    intensity = ambient + (1.0 - ambient) * max(0.0, float(np.dot(normal_rot, light)))
    return _hex(base * min(1.0, intensity))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Options:
    palette: Palette
    light: np.ndarray
    ambient: float
    draw_faces: bool
    # Marker sizes as a ratio of the sphere radius; 0 disables the marker.
    cell_size: float
    edge_size: float
    vertex_size: float
    line_width: float
    margin: float
    marker_facing: float


def _pixels(screen: np.ndarray, cx: float, cy: float, scale: float) -> list[float]:
    """Map screen coordinates to a flat ``[x0, y0, x1, y1, ...]`` pixel list."""
    return [coord for sx, sy in screen for coord in (cx + scale * sx, cy - scale * sy)]


def render_icosahedron(
    drawing: draw.Drawing, cx: float, cy: float, fit_radius: float, *, view: View, opt: Options
) -> None:
    """Draw the shaded, marker-annotated icosahedron centered at ``(cx, cy)``."""
    vertices, edges, faces = icosahedron()
    fnormals = face_normals(vertices, faces)

    # Fit the whole icosahedron (its vertices bound everything) into fit_radius.
    vert_screen, _ = project(view, vertices)
    extent = float(np.max(np.hypot(vert_screen[:, 0], vert_screen[:, 1])))
    scale = fit_radius * (1.0 - opt.margin) / extent

    def to_pixels(points: np.ndarray) -> list[float]:
        screen, _ = project(view, points)
        return _pixels(screen, cx, cy, scale)

    # --- Faces (painter's algorithm, back-facing culled) ------------------- #
    if opt.draw_faces:
        face_centroids = vertices[faces].mean(axis=1)
        centroids_rot = face_centroids @ view.rot.T
        face_depth = view.distance - centroids_rot[:, 2]
        for fi in np.argsort(-face_depth):
            normal_rot = view.rot @ fnormals[fi]
            if np.dot(normal_rot, view.eye - centroids_rot[fi]) <= 0.0:
                continue  # back-facing
            color = shade(
                face_base_color(opt.palette, float(face_centroids[fi, 2]), int(fi)),
                normal_rot,
                opt.light,
                opt.ambient,
            )
            drawing.append(
                draw.Lines(
                    *to_pixels(vertices[faces[fi]]),
                    close=True,
                    fill=color,
                    stroke=opt.palette.edge,
                    stroke_width=opt.line_width,
                    stroke_linejoin="round",
                )
            )
    else:
        _render_wireframe(drawing, vertices, edges, view=view, opt=opt, to_pixels=to_pixels)

    # --- Markers (drawn far-to-near, back-facing culled) ------------------- #
    markers: list[tuple[float, draw.DrawingBasicElement]] = []
    if opt.cell_size > 0.0:
        markers.extend(_cell_center_markers(vertices, faces, view, opt, to_pixels))
    if opt.edge_size > 0.0:
        markers.extend(_edge_midpoint_markers(vertices, edges, view, opt, to_pixels))
    if opt.vertex_size > 0.0:
        markers.extend(_vertex_markers(vertices, view, opt, to_pixels))
    for _, element in sorted(markers, key=lambda item: -item[0]):
        drawing.append(element)


def _render_wireframe(drawing, vertices, edges, *, view, opt, to_pixels) -> None:
    """Draw every edge whose midpoint faces the camera (used when faces off)."""
    for i, j in edges:
        midpoint = (vertices[i] + vertices[j]) / 2.0
        if not front_facing(view, midpoint, _normalize(midpoint)):
            continue
        pts = to_pixels(np.array([vertices[i], vertices[j]]))
        drawing.append(
            draw.Lines(*pts, close=False, stroke=opt.palette.edge, stroke_width=opt.line_width)
        )


def _cell_center_markers(vertices, faces, view, opt, to_pixels):
    centroids = vertices[faces].mean(axis=1)
    for centroid in centroids:
        normal = _normalize(centroid)
        if not front_facing(view, centroid, normal, opt.marker_facing):
            continue
        outline = planar_disc(centroid, normal, opt.cell_size)
        depth = view.distance - (view.rot @ centroid)[2]
        yield (
            depth,
            draw.Lines(
                *to_pixels(outline),
                close=True,
                fill=opt.palette.circle,
                stroke=opt.palette.circle_stroke,
                stroke_width=opt.line_width * 0.6,
            ),
        )


def _edge_midpoint_markers(vertices, edges, view, opt, to_pixels):
    for i, j in edges:
        midpoint = (vertices[i] + vertices[j]) / 2.0
        normal = _normalize(midpoint)
        if not front_facing(view, midpoint, normal, opt.marker_facing):
            continue
        outline = planar_square(
            midpoint, normal, opt.edge_size, orient_dir=vertices[j] - vertices[i]
        )
        depth = view.distance - (view.rot @ midpoint)[2]
        yield (
            depth,
            draw.Lines(
                *to_pixels(outline),
                close=True,
                fill=opt.palette.square,
                stroke=opt.palette.square_stroke,
                stroke_width=opt.line_width * 0.6,
            ),
        )


def _vertex_markers(vertices, view, opt, to_pixels):
    for vertex in vertices:
        normal = _normalize(vertex)
        if not front_facing(view, vertex, normal, opt.marker_facing):
            continue
        outline = planar_disc(vertex, normal, opt.vertex_size, samples=24)
        depth = view.distance - (view.rot @ vertex)[2]
        yield (
            depth,
            draw.Lines(
                *to_pixels(outline),
                close=True,
                fill=opt.palette.edge,
                stroke="none",
            ),
        )


def build_icon(width: int, height: int, view: View, opt: Options, background: str) -> draw.Drawing:
    """Build an icon-only drawing."""
    drawing = draw.Drawing(width, height, origin=(0, 0))
    if background != "none":
        drawing.append(draw.Rectangle(0, 0, width, height, fill=background))
    render_icosahedron(
        drawing, width / 2.0, height / 2.0, min(width, height) / 2.0, view=view, opt=opt
    )
    return drawing


def build_logo(
    icon_size: int, view: View, opt: Options, background: str, text: str
) -> draw.Drawing:
    """Build an icon + wordmark drawing."""
    width = icon_size
    text_band = int(icon_size * 0.32)
    height = icon_size + text_band
    drawing = draw.Drawing(width, height, origin=(0, 0))
    if background != "none":
        drawing.append(draw.Rectangle(0, 0, width, height, fill=background))
    render_icosahedron(drawing, width / 2.0, icon_size / 2.0, icon_size / 2.0, view=view, opt=opt)
    drawing.append(
        draw.Text(
            text,
            font_size=text_band * 0.55,
            x=width / 2.0,
            y=icon_size + text_band * 0.62,
            center=True,
            font_family="Helvetica, Arial, sans-serif",
            font_weight="bold",
            fill=opt.palette.wordmark,
        )
    )
    return drawing


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_vector(text: str) -> np.ndarray:
    parts = [p for p in text.replace(" ", "").split(",") if p]
    if len(parts) != 3:
        raise typer.BadParameter(f"Expected three comma-separated numbers, got '{text}'.")
    return np.array([float(p) for p in parts])


@cli.command(name="generate-logo")
def generate_logo(
    *,
    out_dir: Annotated[
        pathlib.Path, typer.Option("--out-dir", help="Directory for the generated SVG files.")
    ] = common.REPO_ROOT / "docs" / "logo",
    size: Annotated[int, typer.Option(help="Icon size in pixels (square).")] = 512,
    azimuth: Annotated[float, typer.Option(help="Camera azimuth in degrees.")] = 25.0,
    elevation: Annotated[float, typer.Option(help="Camera elevation in degrees.")] = 18.0,
    roll: Annotated[float, typer.Option(help="Camera roll in degrees.")] = 0.0,
    distance: Annotated[
        float, typer.Option(help="Camera distance; smaller means stronger perspective.")
    ] = 4.5,
    light: Annotated[
        str, typer.Option(help="Light direction 'x,y,z' in the camera frame.")
    ] = "-0.4,0.5,0.85",
    ambient: Annotated[float, typer.Option(help="Ambient light fraction in [0, 1].")] = 0.45,
    palette: Annotated[
        str, typer.Option(help=f"Colour palette: {', '.join(PALETTES)}.")
    ] = "atmosphere",
    faces: Annotated[
        bool, typer.Option("--faces/--wireframe", help="Solid shaded faces or wireframe.")
    ] = True,
    cell_size: Annotated[
        float,
        typer.Option(help="Cell-centre circle radius, as a ratio of the sphere radius (0 = off)."),
    ] = 0.07,
    edge_size: Annotated[
        float,
        typer.Option(help="Edge-midpoint square side, as a ratio of the sphere radius (0 = off)."),
    ] = 0.09,
    vertex_size: Annotated[
        float,
        typer.Option(help="Vertex dot radius, as a ratio of the sphere radius (0 = off)."),
    ] = 0.0,
    line_width: Annotated[float, typer.Option(help="Mesh edge stroke width in pixels.")] = 6,
    wordmark: Annotated[
        bool, typer.Option("--wordmark/--no-wordmark", help="Also write the icon + wordmark SVG.")
    ] = True,
    marker_facing: Annotated[
        float,
        typer.Option(help="Min. cosine for a marker to face the camera; drops grazing markers."),
    ] = 0.05,
    background: Annotated[
        str, typer.Option(help="Background colour, or 'none' for transparent.")
    ] = "none",
    text: Annotated[str, typer.Option(help="Wordmark text.")] = "ICON4Py",
) -> None:
    """Generate the ICON4Py logo and write the SVG file(s)."""
    if palette not in PALETTES:
        raise typer.BadParameter(f"Unknown palette '{palette}'; choose from {', '.join(PALETTES)}.")

    view = make_view(azimuth, elevation, roll, distance)
    opt = Options(
        palette=PALETTES[palette],
        light=_normalize(_parse_vector(light)),
        ambient=ambient,
        draw_faces=faces,
        cell_size=cell_size,
        edge_size=edge_size,
        vertex_size=vertex_size,
        line_width=line_width,
        margin=0.06,
        marker_facing=marker_facing,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    icon_path = out_dir / "icon4py_icon.svg"
    build_icon(size, size, view, opt, background).save_svg(icon_path)
    typer.echo(f"Wrote {icon_path}")

    if wordmark:
        logo_path = out_dir / "icon4py_logo.svg"
        build_logo(size, view, opt, background, text).save_svg(logo_path)
        typer.echo(f"Wrote {logo_path}")


if __name__ == "__main__":
    sys.exit(cli())
