# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Generator for doubly periodic (plane torus) triangular ICON grids.

This is a partial port of the MPI-M GridGenerator, 'mo_create_torus_grid.f90', restricted to
what the icon4py grid reader consumes. The index and orientation logic follows the Fortran.
The intentional differences are:

- 'fit_resolution' (mo_create_torus_grid.f90:260-324) is not ported. It turns a requested
  domain extent into row and column counts and adjusts the domain height while doing so, which
  makes two grids of different resolution discretise slightly different domains. Taking
  (n_rows, n_cols, edge_length) directly means that (2*n_rows, 2*n_cols, edge_length/2) is the
  exact bisection of a grid, with bit-identical extents.
- The domain origin is (0, 0) rather than the Fortran's centred (-length/2, -height/2)
  (mo_create_torus_grid.f90:308-310), so that vertex_x.min() and vertex_y.min() are exactly
  0.0. icon4py's analytical initial conditions use those minima as the domain origin.
- Coordinates are wrapped with a plain periodic modulo instead of 'get_x'
  (mo_create_torus_grid.f90:1075-1085), which divides by the half extent but subtracts the full
  one and so places coordinates outside the domain box. The minimum image convention icon4py
  uses on a torus ('common/math/distance.py') shifts by at most one period and therefore
  requires all coordinates to lie within a single period.
- The vertex-side connectivities ('edges_of_vertex', 'cells_of_vertex', 'vertices_of_vertex'
  and hence 'edge_orientation') are ordered counter-clockwise. The Fortran leaves them in edge
  creation order because 'order_vertex_connectivity' is not called for a torus
  (mo_create_torus_grid.f90:378-385); the distributed MPI-M grid files are counter-clockwise.
- Latitudes are a linear function of the cartesian y coordinate. The Fortran offsets the
  latitude of an up-pointing cell by 2/3 of a row but its cartesian y by only 1/3
  (mo_create_torus_grid.f90:860-861, against :881-882 which uses -1/3 for both), so its 'clat'
  is not the image of 'cell_circumcenter_cartesian_y' under any single map. The distributed
  MPI-M grid files are linear, so this follows the files rather than the source.
"""

import math
import pathlib
import uuid
from typing import Any, Final

import netCDF4 as nc
import numpy as np


# sine of 60 degrees, the row spacing in units of the edge length (mo_create_torus_grid.f90:181)
_SIN60: Final = math.sqrt(0.75)

# earth radius; unused on a torus but written for fidelity (mo_create_torus_grid.f90:833)
_SPHERE_RADIUS: Final = 6371229.0

# 'icon.GeometryType.TORUS'. The grid manager reads this attribute with a truthiness check, so
# it has to be present and non-zero or the file is interpreted as an icosahedral grid.
_TORUS_GEOMETRY: Final = 2

# The Fortran maps the plane onto a synthetic lon/lat band (mo_create_torus_grid.f90:810-818).
# These are the defaults of its namelist variables (:162, :165), not the values MPI-M passed
# when generating the distributed torus files, which span 20 rather than 160 degrees. The band
# is cosmetic either way: on a torus icon4py works from the cartesian coordinates and pins the
# Coriolis parameter to zero ('grid/geometry.py', 'coriolis_parameter_on_edges_torus').
_LAT_CENTER_DEG: Final = 0.0
_LAT_LENGTH_DEG: Final = 160.0

# "Unordered interior" refinement markers, see 'grid_refinement._UNORDERED'. They must not be
# positive or icon4py classifies the grid as limited area.
_REFINE_CELLS: Final = -4
_REFINE_EDGES: Final = -8
_REFINE_VERTICES: Final = 0

# Dimensions of every variable we write, in the order they appear on disk. The two dimensional
# ones are stored (neighbours, entities): the reader transposes them back
# ('GridManager._read_geometry_fields', 'GridFile.variable(transpose=True)').
_DIMENSIONS: Final[dict[str, tuple[str, ...]]] = {
    **dict.fromkeys(
        ("vertex_of_cell", "edge_of_cell", "neighbor_cell_index", "orientation_of_normal"),
        ("nv", "cell"),
    ),
    **dict.fromkeys(
        ("edge_vertices", "adjacent_cell_of_edge", "edge_cell_distance", "edge_vert_distance"),
        ("nc", "edge"),
    ),
    **dict.fromkeys(
        ("edges_of_vertex", "cells_of_vertex", "vertices_of_vertex", "edge_orientation"),
        ("ne", "vertex"),
    ),
    **dict.fromkeys(
        (
            "cell_area",
            "clon",
            "clat",
            "refin_c_ctrl",
            "cell_circumcenter_cartesian_x",
            "cell_circumcenter_cartesian_y",
            "cell_circumcenter_cartesian_z",
        ),
        ("cell",),
    ),
    **dict.fromkeys(
        (
            "edge_length",
            "dual_edge_length",
            "edge_system_orientation",
            "elon",
            "elat",
            "refin_e_ctrl",
            "edge_middle_cartesian_x",
            "edge_middle_cartesian_y",
            "edge_middle_cartesian_z",
        ),
        ("edge",),
    ),
    **dict.fromkeys(
        (
            "dual_area",
            "vlon",
            "vlat",
            "refin_v_ctrl",
            "cartesian_x_vertices",
            "cartesian_y_vertices",
            "cartesian_z_vertices",
        ),
        ("vertex",),
    ),
}


class _Lattice:
    """
    The integer (column, row) lattice of the torus and the Fortran's index formulas.

    Indices are 1-based, as in the grid file. Entities come in two families: 'vertex',
    'right_edge', 'top_right_edge' and 'top_right_cell' live on rows [0, n_rows), while
    'down_right_edge' and 'down_right_cell' live on rows [1, n_rows]. Both are periodic in the
    column and sheared in the row: row y is staggered y*edge_length/2 to the right, so after
    n_rows rows the stagger is exactly n_rows/2 whole columns and moving up n_rows rows lands on
    the same lattice point as moving right n_rows/2 columns (mo_create_torus_grid.f90:598-604,
    :687-693, :712-717, :733-738). That shear is why n_rows must be even.
    """

    def __init__(self, n_rows: int, n_cols: int) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_cells = 2 * n_rows * n_cols
        self.n_edges = 3 * n_rows * n_cols
        self.n_vertices = n_rows * n_cols
        # the (column, row) pairs the Fortran loops over, flattened; indices grow fastest with
        # the row, then with the column, so this ordering is the entity ordering
        columns, rows = np.meshgrid(np.arange(n_cols), np.arange(n_rows), indexing="ij")
        self.columns = columns.ravel()
        self.rows_low = rows.ravel()
        self.rows_high = self.rows_low + 1

    def _low(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fold (x, y) onto the lattice for the entities living on rows [0, n_rows)."""
        shift = np.floor_divide(y, self.n_rows)
        return (x + shift * (self.n_rows // 2)) % self.n_cols, y - shift * self.n_rows

    def _high(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fold (x, y) onto the lattice for the entities living on rows [1, n_rows]."""
        shift = np.floor_divide(y - 1, self.n_rows)
        return (x + shift * (self.n_rows // 2)) % self.n_cols, y - shift * self.n_rows

    def vertex(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """mo_create_torus_grid.f90:695"""
        x, y = self._low(x, y)
        return x * self.n_rows + y + 1

    def right_edge(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Horizontal edge (x, y) -> (x+1, y), mo_create_torus_grid.f90:606."""
        x, y = self._low(x, y)
        return (x * self.n_rows + y) * 3 + 1

    def top_right_edge(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """+60 degree edge (x, y) -> (x, y+1), mo_create_torus_grid.f90:624."""
        x, y = self._low(x, y)
        return (x * self.n_rows + y) * 3 + 2

    def down_right_edge(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """+120 degree edge (x+1, y-1) -> (x, y), mo_create_torus_grid.f90:641."""
        x, y = self._high(x, y)
        return (x * self.n_rows + y) * 3

    def top_right_cell(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Up-pointing triangle with lower left corner (x, y), mo_create_torus_grid.f90:719."""
        x, y = self._low(x, y)
        return (x * self.n_rows + y) * 2 + 1

    def down_right_cell(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Down-pointing triangle with upper left corner (x, y), mo_create_torus_grid.f90:740."""
        x, y = self._high(x, y)
        return (x * self.n_rows + y) * 2


def _build_connectivity(lattice: _Lattice) -> dict[str, np.ndarray]:
    """The eight connectivities and the three orientation fields, 1-based and (neighbours, entities)."""
    # 'y' runs over the rows of the low family, [0, n_rows), 'y_up' over those of the high
    # family, [1, n_rows]; 'up_cell' and 'down_cell' are the up- and down-pointing triangles.
    # The Fortran's 'down_cell' and 'top_cell' helpers (mo_create_torus_grid.f90:746, :763) are
    # inlined below as 'up_cell(x, y-1)' and 'down_cell(x-1, y+1)'.
    x, y, y_up = lattice.columns, lattice.rows_low, lattice.rows_high
    vertex, up_cell, down_cell = lattice.vertex, lattice.top_right_cell, lattice.down_right_cell
    right, top_right, down_right = (
        lattice.right_edge,
        lattice.top_right_edge,
        lattice.down_right_edge,
    )

    # All three edge families are created with system orientation +1
    # (mo_create_torus_grid.f90:447, :463, :478). That fixes the tangent to point from
    # 'edge_vertices[0]' to 'edge_vertices[1]', and the primal normal, which is the tangent
    # rotated by 90 degrees counter-clockwise, to point from 'adjacent_cell_of_edge[0]' to
    # 'adjacent_cell_of_edge[1]' (:441-450, :456-466, :471-481).
    edge_vertices = np.empty((2, lattice.n_edges), dtype=np.int32)
    adjacent_cell_of_edge = np.empty((2, lattice.n_edges), dtype=np.int32)
    index = top_right(x, y) - 1
    edge_vertices[:, index] = (vertex(x, y), vertex(x, y + 1))
    adjacent_cell_of_edge[:, index] = (up_cell(x, y), down_cell(x - 1, y + 1))
    index = right(x, y) - 1
    edge_vertices[:, index] = (vertex(x, y), vertex(x + 1, y))
    adjacent_cell_of_edge[:, index] = (down_cell(x, y), up_cell(x, y))
    index = down_right(x, y_up) - 1
    edge_vertices[:, index] = (vertex(x + 1, y_up - 1), vertex(x, y_up))
    adjacent_cell_of_edge[:, index] = (down_cell(x, y_up), up_cell(x, y_up - 1))

    # Cells, mo_create_torus_grid.f90:498-518 and :533-553. 'vertex_of_cell' is
    # counter-clockwise and 'edge_of_cell[k]' spans 'vertex_of_cell[k]' and
    # 'vertex_of_cell[k+1]'; 'neighbor_cell_index[k]' is the cell across 'edge_of_cell[k]'.
    # 'orientation_of_normal[k]' is +1 exactly where this cell is
    # 'adjacent_cell_of_edge[0]' of 'edge_of_cell[k]', i.e. where the stored normal points out
    # of the cell. The two families have different sign patterns.
    # 'order_cell_connectivity' (mo_local_grid_geometry.f90:611-753) is a no-op for this grid.
    vertex_of_cell = np.empty((3, lattice.n_cells), dtype=np.int32)
    edge_of_cell = np.empty((3, lattice.n_cells), dtype=np.int32)
    neighbor_cell_index = np.empty((3, lattice.n_cells), dtype=np.int32)
    orientation_of_normal = np.empty((3, lattice.n_cells), dtype=np.int32)
    index = up_cell(x, y) - 1
    vertex_of_cell[:, index] = (vertex(x, y), vertex(x + 1, y), vertex(x, y + 1))
    edge_of_cell[:, index] = (right(x, y), down_right(x, y + 1), top_right(x, y))
    neighbor_cell_index[:, index] = (down_cell(x, y), down_cell(x, y + 1), down_cell(x - 1, y + 1))
    orientation_of_normal[:, index] = np.array([[-1], [-1], [1]])
    index = down_cell(x, y_up) - 1
    vertex_of_cell[:, index] = (vertex(x, y_up), vertex(x + 1, y_up - 1), vertex(x + 1, y_up))
    edge_of_cell[:, index] = (down_right(x, y_up), top_right(x + 1, y_up - 1), right(x, y_up))
    neighbor_cell_index[:, index] = (
        up_cell(x, y_up - 1),
        up_cell(x + 1, y_up - 1),
        up_cell(x, y_up),
    )
    orientation_of_normal[:, index] = np.array([[1], [-1], [1]])

    # Vertices, counter-clockwise starting east and turning by 60 degrees rather than in the
    # Fortran's edge creation order, see the module docstring. 'vertices_of_vertex[k]' is the
    # far endpoint of 'edges_of_vertex[k]', and 'cells_of_vertex[k]' the cell at 30 + 60*k
    # degrees.
    edges_of_vertex = np.empty((6, lattice.n_vertices), dtype=np.int32)
    vertices_of_vertex = np.empty((6, lattice.n_vertices), dtype=np.int32)
    cells_of_vertex = np.empty((6, lattice.n_vertices), dtype=np.int32)
    index = vertex(x, y) - 1
    edges_of_vertex[:, index] = (
        right(x, y),
        top_right(x, y),
        down_right(x - 1, y + 1),
        right(x - 1, y),
        top_right(x, y - 1),
        down_right(x, y),
    )
    vertices_of_vertex[:, index] = (
        vertex(x + 1, y),
        vertex(x, y + 1),
        vertex(x - 1, y + 1),
        vertex(x - 1, y),
        vertex(x, y - 1),
        vertex(x + 1, y - 1),
    )
    cells_of_vertex[:, index] = (
        up_cell(x, y),
        down_cell(x - 1, y + 1),
        up_cell(x - 1, y),
        down_cell(x - 1, y),
        up_cell(x, y - 1),
        down_cell(x, y),
    )

    return {
        "vertex_of_cell": vertex_of_cell,
        "edge_of_cell": edge_of_cell,
        "neighbor_cell_index": neighbor_cell_index,
        "orientation_of_normal": orientation_of_normal,
        "edge_vertices": edge_vertices,
        "adjacent_cell_of_edge": adjacent_cell_of_edge,
        "edges_of_vertex": edges_of_vertex,
        "cells_of_vertex": cells_of_vertex,
        "vertices_of_vertex": vertices_of_vertex,
        # +1 where this vertex is 'edge_vertices[0]' of the edge, -1 otherwise
        # (mo_create_torus_grid.f90:1115, :1121). With system orientation +1 everywhere that is
        # the east, north-east and north-west edges of the list above.
        "edge_orientation": np.tile(
            np.array([[1], [1], [1], [-1], [-1], [-1]], dtype=np.int32),
            (1, lattice.n_vertices),
        ),
        "edge_system_orientation": np.ones(lattice.n_edges, dtype=np.int32),
    }


def _build_coordinates(
    lattice: _Lattice,
    edge_length: float,
    *,
    domain_length: float,
    domain_height: float,
) -> dict[str, np.ndarray]:
    """Cartesian coordinates of vertices, cell circumcenters and edge midpoints, plus lon/lat."""
    x, y, y_up = lattice.columns, lattice.rows_low, lattice.rows_high
    y_step = edge_length * _SIN60

    def wrap_x(quarter_columns: np.ndarray) -> np.ndarray:
        # Every x offset below is a multiple of edge_length/4, so wrapping the integer count of
        # quarter columns makes the periodic wrap exact and the minimum exactly 0.0 for any
        # 'edge_length'. Wrapping in floating point could land one ulp below 'domain_length'.
        return (quarter_columns % (4 * lattice.n_cols)) * (edge_length / 4.0)

    vertex_x, vertex_y = np.empty(lattice.n_vertices), np.empty(lattice.n_vertices)
    cell_x, cell_y = np.empty(lattice.n_cells), np.empty(lattice.n_cells)
    edge_x, edge_y = np.empty(lattice.n_edges), np.empty(lattice.n_edges)

    # The trailing line numbers refer to mo_create_torus_grid.f90.
    # Each expression is the centroid of the entity's vertices on the unwrapped plane, wrapped
    # only at the end; wrapping the vertices first and averaging afterwards would put every
    # entity that straddles the seam in the middle of the domain. Equilateral triangles have
    # their circumcenter at the centroid, hence the 1/3 offsets.
    index = lattice.vertex(x, y) - 1
    vertex_x[index] = wrap_x(4 * x + 2 * y)  # :850
    vertex_y[index] = y_step * y  # :851
    index = lattice.top_right_cell(x, y) - 1
    cell_x[index] = wrap_x(4 * x + 2 * y + 2)  # :872
    cell_y[index] = y_step / 3.0 + y_step * y  # :873
    index = lattice.down_right_cell(x, y_up) - 1
    cell_x[index] = wrap_x(4 * (x + 1) + 2 * (y_up - 1))  # :892
    cell_y[index] = -y_step / 3.0 + y_step * y_up  # :893
    index = lattice.top_right_edge(x, y) - 1
    edge_x[index] = wrap_x(4 * x + 2 * y + 1)  # :915
    edge_y[index] = y_step * 0.5 + y_step * y  # :916
    index = lattice.down_right_edge(x, y_up) - 1
    edge_x[index] = wrap_x(4 * x + 2 * y_up + 1)  # :955
    edge_y[index] = -y_step * 0.5 + y_step * y_up  # :956
    index = lattice.right_edge(x, y) - 1
    edge_x[index] = wrap_x(4 * x + 2 * y + 2)  # :995
    edge_y[index] = y_step * y  # :996

    # icon4py works from the cartesian coordinates on a torus but reads lon/lat
    # unconditionally, so rescale the plane onto the Fortran's synthetic band
    # (mo_create_torus_grid.f90:810-818, :848-849).
    lat_center, lat_length = math.radians(_LAT_CENTER_DEG), math.radians(_LAT_LENGTH_DEG)

    def to_lon(cartesian_x: np.ndarray) -> np.ndarray:
        return 2.0 * math.pi * (cartesian_x / domain_length) - math.pi

    def to_lat(cartesian_y: np.ndarray) -> np.ndarray:
        return lat_center + lat_length * (cartesian_y / domain_height - 0.5)

    return {
        "cartesian_x_vertices": vertex_x,
        "cartesian_y_vertices": vertex_y,
        "cartesian_z_vertices": np.zeros(lattice.n_vertices),
        "cell_circumcenter_cartesian_x": cell_x,
        "cell_circumcenter_cartesian_y": cell_y,
        "cell_circumcenter_cartesian_z": np.zeros(lattice.n_cells),
        "edge_middle_cartesian_x": edge_x,
        "edge_middle_cartesian_y": edge_y,
        "edge_middle_cartesian_z": np.zeros(lattice.n_edges),
        "vlon": to_lon(vertex_x),
        "vlat": to_lat(vertex_y),
        "clon": to_lon(cell_x),
        "clat": to_lat(cell_y),
        "elon": to_lon(edge_x),
        "elat": to_lat(edge_y),
    }


def _build_grid(
    lattice: _Lattice, edge_length: float
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Assemble all NetCDF variables and global attributes of the grid file."""
    # Keep the Fortran's evaluation order (mo_create_torus_grid.f90:803-806, :831-832): the
    # results are bit-identical to the MPI-M files, and e.g. 'hexagon_area' is one ulp away
    # from 2*'triangle_area'.
    half_edge_length = edge_length * 0.5
    dual_edge_length = edge_length * (1.0 / math.sqrt(3.0))
    triangle_area = edge_length * edge_length * math.sqrt(0.1875)
    hexagon_area = dual_edge_length * edge_length * 1.5
    domain_length = edge_length * lattice.n_cols
    domain_height = (edge_length * _SIN60) * lattice.n_rows

    variables: dict[str, np.ndarray] = {
        **_build_connectivity(lattice),
        **_build_coordinates(
            lattice, edge_length, domain_length=domain_length, domain_height=domain_height
        ),
        "cell_area": np.full(lattice.n_cells, triangle_area),
        "dual_area": np.full(lattice.n_vertices, hexagon_area),
        "edge_length": np.full(lattice.n_edges, edge_length),
        "dual_edge_length": np.full(lattice.n_edges, dual_edge_length),
        "edge_cell_distance": np.full((2, lattice.n_edges), dual_edge_length * 0.5),
        "edge_vert_distance": np.full((2, lattice.n_edges), half_edge_length),
        "refin_c_ctrl": np.full(lattice.n_cells, _REFINE_CELLS, dtype=np.int32),
        "refin_e_ctrl": np.full(lattice.n_edges, _REFINE_EDGES, dtype=np.int32),
        "refin_v_ctrl": np.full(lattice.n_vertices, _REFINE_VERTICES, dtype=np.int32),
    }
    attributes: dict[str, Any] = {
        "title": "ICON grid description",
        "source": "icon4py.model.testing.torus_grid_generator",
        # a deterministic id keeps caches keyed on 'IconGrid.id' stable across regenerations,
        # while still distinguishing grids that differ in any of the three parameters
        "uuidOfHGrid": str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"icon4py-torus-{lattice.n_rows}x{lattice.n_cols}-{edge_length!r}",
            )
        ),
        # meaningless for a torus, but read unconditionally by 'GridManager'; these are the
        # values MPI-M writes for a torus
        "grid_level": np.int32(0),
        "grid_root": np.int32(2),
        "grid_geometry": np.int32(_TORUS_GEOMETRY),
        "grid_cell_type": np.int32(3),
        "domain_length": domain_length,
        "domain_height": domain_height,
        "sphere_radius": _SPHERE_RADIUS,
        "domain_cartesian_center": np.array([0.5 * domain_length, 0.5 * domain_height, 0.0]),
        "mean_edge_length": edge_length,
        "mean_dual_edge_length": dual_edge_length,
        "mean_cell_area": triangle_area,
        "mean_dual_cell_area": hexagon_area,
    }
    return variables, attributes


def _write_grid_file(
    out_file: pathlib.Path,
    *,
    dimensions: dict[str, int],
    variables: dict[str, np.ndarray],
    attributes: dict[str, Any],
) -> None:
    with nc.Dataset(str(out_file), "w", format="NETCDF4") as dataset:
        dataset.setncatts(attributes)
        for name, size in dimensions.items():
            dataset.createDimension(name, size)
        for name, data in variables.items():
            # never set a '_FillValue': 'GridFile.variable' reads through 'np.asarray', which
            # would substitute the fill value for every matching entry
            variable = dataset.createVariable(name, data.dtype, _DIMENSIONS[name])
            if name.endswith(("lon", "lat")):
                variable.units = "radian"
            elif data.dtype == np.float64:
                variable.units = "m2" if name.endswith("area") else "m"
            variable[...] = data


def generate_torus_grid(
    *,
    n_rows: int,
    n_cols: int,
    edge_length: float,
    out_file: pathlib.Path,
) -> pathlib.Path:
    """
    Write an ICON grid file for a doubly periodic mesh of equilateral triangles.

    The domain is 'n_cols * edge_length' wide and 'n_rows * edge_length * sqrt(3)/2' high, with
    its lower left corner at (0, 0). Doubling 'n_rows' and 'n_cols' while halving 'edge_length'
    bisects the mesh and leaves both extents bit-identical, which is what makes a family of
    these grids usable for a convergence study.

    Args:
        n_rows: number of rows, even and at least 4
        n_cols: number of columns, at least 3
        edge_length: triangle side length in meters
        out_file: path of the NetCDF file to write

    Returns:
        The path that was written.
    """
    # the row periodicity shifts the column by n_rows//2, which only closes the mesh for an
    # even number of rows; below 4 rows or 3 columns a vertex becomes its own neighbour and
    # 'vertices_of_vertex' has repeated entries
    if n_rows < 4 or n_rows % 2 != 0:
        raise ValueError(f"Invalid argument 'n_rows': expected an even number >= 4, got {n_rows}.")
    if n_cols < 3:
        raise ValueError(f"Invalid argument 'n_cols': expected a number >= 3, got {n_cols}.")
    if edge_length <= 0.0:
        raise ValueError(
            f"Invalid argument 'edge_length': expected a positive length, got {edge_length}."
        )

    lattice = _Lattice(n_rows, n_cols)
    variables, attributes = _build_grid(lattice, edge_length)
    dimensions = {
        "cell": lattice.n_cells,
        "vertex": lattice.n_vertices,
        "edge": lattice.n_edges,
        "nv": 3,
        "ne": 6,
        "nc": 2,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    _write_grid_file(out_file, dimensions=dimensions, variables=variables, attributes=attributes)
    return out_file
