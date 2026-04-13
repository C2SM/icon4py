# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive 3D visualization of StructuredDecomposer partitioning on ICON grids.

Usage:
    python scripts/visualize_decomposition_3d.py grid1.nc [grid2.nc ...]

Produces a single HTML file with two dropdowns:
  - Grid selector (if multiple grids provided)
  - Mode: "Cell index" or "N ranks" decomposition
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.express as px
import xarray as xr

from icon4py.model.common.decomposition.decomposer import StructuredDecomposer


def valid_rank_counts(grid_root: int, grid_level: int, max_ranks: int = 320):
    max_depth = StructuredDecomposer._quad_depth(grid_root) + grid_level
    counts = set()
    for d in range(max_depth + 1):
        total = 10 * (4**d)
        counts.update(r for r in range(1, min(total, max_ranks) + 1) if total % r == 0)
    return sorted(counts)


def lonlat_to_xyz(lon_rad, lat_rad):
    return (
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    )


def make_colors(n_ranks):
    if n_ranks <= 10:
        palette = px.colors.qualitative.D3
    elif n_ranks <= 26:
        palette = px.colors.qualitative.Alphabet
    else:
        palette = [
            f"hsl({int(i * 360 / n_ranks)}, {70 + (i % 3) * 10}%, {45 + (i % 5) * 5}%)"
            for i in range(n_ranks)
        ]
    return [palette[p % len(palette)] for p in range(n_ranks)]


def index_colors(n_cells, block_size):
    """Cyclic colormap: repeat the full Turbo color cycle every block_size cells."""
    within_block = np.arange(n_cells) % block_size
    t = within_block / block_size
    return px.colors.sample_colorscale("Turbo", t)


def precompute_grid(grid_file: str, max_ranks: int):
    """Precompute all data for one grid: vertices, triangles, and all colorings."""
    ds = xr.open_dataset(grid_file)
    grid_root = int(ds.attrs["grid_root"])
    grid_level = int(ds.attrs["grid_level"])
    n_cells = ds.sizes["cell"]
    block_size = 4**grid_level
    label = f"R{grid_root:02d}B{grid_level:02d}"

    decomposer = StructuredDecomposer(grid_root, grid_level)
    dummy_adj = np.zeros((n_cells, 3), dtype=np.int32)
    rank_options = valid_rank_counts(grid_root, grid_level, max_ranks)

    print(f"  {label}: {n_cells} cells, block_size={block_size}, ranks={rank_options}")

    # Vertex geometry
    voc = ds["vertex_of_cell"].values.T - 1
    vx, vy, vz = lonlat_to_xyz(ds["vlon"].values, ds["vlat"].values)

    # Precompute all face-color arrays
    modes = {}

    # Cell index mode
    modes["Cell index"] = {
        "colors": index_colors(n_cells, block_size),
        "hover": [f"Cell {c}" for c in range(n_cells)],
        "info": (
            f"<b>Cell index</b> — Turbo colormap cycling every {block_size} cells "
            f"(one sub-triangle block)"
        ),
    }

    # Decomposition modes
    for n_ranks in rank_options:
        partition = np.asarray(decomposer(dummy_adj, n_ranks))
        colors = make_colors(n_ranks)
        cells_per_rank = n_cells // n_ranks
        modes[f"{n_ranks} ranks"] = {
            "colors": [colors[p] for p in partition],
            "hover": [f"Cell {c}, Rank {partition[c]}" for c in range(n_cells)],
            "info": f"<b>{n_ranks} ranks</b> — {cells_per_rank} cells/rank",
        }

    # Precompute wireframe edges (deduplicated)
    # Offset vertices slightly outward so edges render above the mesh surface
    r = np.sqrt(vx**2 + vy**2 + vz**2)
    offset = 1.002  # 0.2% outward
    ox = vx * offset / r
    oy = vy * offset / r
    oz = vz * offset / r

    edges = set()
    for c in range(n_cells):
        v0, v1, v2 = voc[c]
        for a, b in [(v0, v1), (v1, v2), (v2, v0)]:
            edges.add((min(a, b), max(a, b)))

    # Build line coordinates with None separators for Scatter3d
    ex, ey, ez = [], [], []
    for a, b in edges:
        ex.extend([ox[a], ox[b], None])
        ey.extend([oy[a], oy[b], None])
        ez.extend([oz[a], oz[b], None])

    return {
        "label": label,
        "n_cells": n_cells,
        "vx": vx.tolist(),
        "vy": vy.tolist(),
        "vz": vz.tolist(),
        "i": voc[:, 0].tolist(),
        "j": voc[:, 1].tolist(),
        "k": voc[:, 2].tolist(),
        "ex": ex,
        "ey": ey,
        "ez": ez,
        "mode_names": list(modes.keys()),
        "mode_colors": [m["colors"] for m in modes.values()],
        "mode_hover": [m["hover"] for m in modes.values()],
        "mode_info": [m["info"] for m in modes.values()],
    }


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ICON Grid Decomposition</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family: sans-serif; margin: 0; padding: 10px; }}
  #controls {{ display: flex; gap: 20px; align-items: center; margin-bottom: 5px; }}
  #controls label {{ font-weight: bold; }}
  #controls select {{ font-size: 14px; padding: 4px 8px; }}
  #info {{ text-align: center; font-size: 14px; margin-top: 5px; }}
  #plot {{ width: 100%; height: calc(100vh - 80px); }}
</style>
</head>
<body>
<div id="controls">
  <label>Grid: <select id="gridSelect" onchange="updatePlot()">
    {grid_options}
  </select></label>
  <label>Mode: <select id="modeSelect" onchange="updatePlot()">
  </select></label>
  <label><input type="checkbox" id="edgesCheck" onchange="toggleEdges()"> Show edges</label>
</div>
<div id="plot"></div>
<div id="info"></div>
<script>
const GRIDS = {grids_json};

const layout = {{
  scene: {{
    xaxis: {{visible: false}},
    yaxis: {{visible: false}},
    zaxis: {{visible: false}},
    aspectmode: "data",
    camera: {{eye: {{x: 1.5, y: 0.3, z: 0.5}}}},
  }},
  margin: {{l: 0, r: 0, t: 10, b: 0}},
}};

let currentGrid = null;

function populateModes(gridIdx) {{
  const g = GRIDS[gridIdx];
  const sel = document.getElementById("modeSelect");
  sel.innerHTML = "";
  g.mode_names.forEach((name, i) => {{
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = name;
    // default to "10 ranks" if available, else first decomposition mode
    if (name === "10 ranks") opt.selected = true;
    sel.appendChild(opt);
  }});
  // If no "10 ranks", select second option (first decomposition) if available
  if (sel.value === "0" && g.mode_names.length > 1) {{
    sel.value = "1";
  }}
}}

function updatePlot() {{
  const gridIdx = parseInt(document.getElementById("gridSelect").value);
  const modeIdx = parseInt(document.getElementById("modeSelect").value);
  const g = GRIDS[gridIdx];

  if (currentGrid !== gridIdx) {{
    populateModes(gridIdx);
    currentGrid = gridIdx;
    // re-read mode after populating
    const newModeIdx = parseInt(document.getElementById("modeSelect").value);
    renderMesh(g, newModeIdx);
  }} else {{
    renderMesh(g, modeIdx);
  }}
}}

function renderMesh(g, modeIdx) {{
  const mesh = {{
    type: "mesh3d",
    x: g.vx, y: g.vy, z: g.vz,
    i: g.i, j: g.j, k: g.k,
    facecolor: g.mode_colors[modeIdx],
    flatshading: true,
    hoverinfo: "text",
    text: g.mode_hover[modeIdx],
    lighting: {{ambient: 0.6, diffuse: 0.4, specular: 0.1}},
    lightposition: {{x: 1000, y: 1000, z: 1000}},
  }};
  const edges = {{
    type: "scatter3d",
    mode: "lines",
    x: g.ex, y: g.ey, z: g.ez,
    line: {{color: "rgba(0,0,0,0.6)", width: 2}},
    hoverinfo: "skip",
    visible: document.getElementById("edgesCheck").checked,
  }};
  Plotly.react("plot", [mesh, edges], layout);
  document.getElementById("info").innerHTML = g.mode_info[modeIdx];
}}

function toggleEdges() {{
  const visible = document.getElementById("edgesCheck").checked;
  Plotly.restyle("plot", {{visible: visible}}, [1]);
}}

// Initial render
populateModes(0);
updatePlot();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Visualize ICON grid decomposition")
    parser.add_argument("grid_files", nargs="+", help="One or more ICON grid NetCDF files")
    parser.add_argument(
        "--max-ranks", type=int, default=320, help="Maximum rank count to include in dropdown"
    )
    parser.add_argument(
        "-o", "--output", default="structured_decomposition_3d.html", help="Output HTML file"
    )
    args = parser.parse_args()

    print(f"Precomputing {len(args.grid_files)} grid(s)...")
    grids = []
    for gf in args.grid_files:
        grids.append(precompute_grid(gf, args.max_ranks))

    grid_options = "\n    ".join(
        f'<option value="{i}">{g["label"]} ({g["n_cells"]} cells)</option>'
        for i, g in enumerate(grids)
    )

    html = HTML_TEMPLATE.format(
        grid_options=grid_options,
        grids_json=json.dumps(grids),
    )

    Path(args.output).write_text(html)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
