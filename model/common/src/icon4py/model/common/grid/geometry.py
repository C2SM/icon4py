# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import functools
import math
from typing import Literal, Union

import xarray as xa
from gt4py import next as gtx
from gt4py.next import backend
from gt4py.next.ffront.fbuiltins import arccos, cos, neighbor_sum, sin, sqrt, where

import icon4py.model.common.grid.geometry_attributes as attrs
import icon4py.model.common.math.helpers as math_helpers
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.dimension import E2C, E2C2V, E2V, E2VDim, EdgeDim
from icon4py.model.common.grid import horizontal as h_grid, icon
from icon4py.model.common.math.helpers import (
    dot_product,
    normalize_cartesian_vector,
    spherical_to_cartesian_on_cells,
    spherical_to_cartesian_on_vertex,
)
from icon4py.model.common.states import factory, model, utils as state_utils
from icon4py.model.common.states.factory import ProgramFieldProvider
from icon4py.model.common.type_alias import wpfloat


"""


Edges:
: "elat" or "lat_edge_center" (DWD units radians), what is the difference between those two?
edge_center_lon: "elat" or "lat_edge_center" (DWD units radians), what is the difference between those two?
tangent_orientation: "edge_system_orientation" from grid file
edge_orientation: "orientation_of_normal"  from grid file
vertex_edge_orientation:
edge_vert_length:
v_dual_area or vertex_dual_area:

reading is done in mo_domimp_patches.f90, computation of derived fields in mo_grid_tools.f90, mo_intp_coeffs.f90

"""


class EdgeParams:
    def __init__(
        self,
        tangent_orientation=None,  # from grid file
        primal_edge_lengths=None,  # computed, see below (computed does not match, from grid file matches serialized)
        inverse_primal_edge_lengths=None,  # computed, inverse
        dual_edge_lengths=None,  # computed, see below (computed does not match, from grid file matches serialized)
        inverse_dual_edge_lengths=None,  # computed, inverse
        inverse_vertex_vertex_lengths=None,  # computed inverse , see below
        primal_normal_vert_x=None,  # computed
        primal_normal_vert_y=None,  # computed
        dual_normal_vert_x=None,  # computed
        dual_normal_vert_y=None,  # computed
        primal_normal_cell_x=None,  # computed
        dual_normal_cell_x=None,  # computed
        primal_normal_cell_y=None,  # computed
        dual_normal_cell_y=None,  # computed
        edge_areas=None,  # computed, verifies
        f_e=None,  # computed, verifies
        edge_center_lat=None,  # coordinate in gridfile - "lat_edge_center" units:radians (what is the difference to elat?)
        edge_center_lon=None,  # coordinate in gridfile - "lon_edge_center" units:radians (what is the difference to elon?
        primal_normal_x=None,  # from gridfile (computed in bridge code?
        primal_normal_y=None,  # from gridfile (computed in bridge code?)
    ):
        self.tangent_orientation: fa.EdgeField[float] = tangent_orientation
        """
        Orientation of vector product of the edge and the adjacent cell centers
             v3
            /  \
           /    \
          /  c1  \
         /    |   \
         v1---|--->v2
         \    |   /
          \   v  /
           \ c2 /
            \  /
            v4
        +1 or -1 depending on whether the vector product of
        (v2-v1) x (c2-c1) points outside (+) or inside (-) the sphere

        defined in ICON in mo_model_domain.f90:t_grid_edges%tangent_orientation
        """

        self.primal_edge_lengths: fa.EdgeField[float] = primal_edge_lengths
        """
        Length of the triangle edge.

        defined in ICON in mo_model_domain.f90:t_grid_edges%primal_edge_length
        """

        self.inverse_primal_edge_lengths: fa.EdgeField[float] = inverse_primal_edge_lengths
        """
        Inverse of the triangle edge length: 1.0/primal_edge_length.

        defined in ICON in mo_model_domain.f90:t_grid_edges%inv_primal_edge_length
        """

        self.dual_edge_lengths: fa.EdgeField[float] = dual_edge_lengths
        """
        Length of the hexagon/pentagon edge.
        vertices of the hexagon/pentagon are cell centers and its center
        is located at the common vertex.
        the dual edge bisects the primal edge othorgonally.

        defined in ICON in mo_model_domain.f90:t_grid_edges%dual_edge_length
        """

        self.inverse_dual_edge_lengths: fa.EdgeField[float] = inverse_dual_edge_lengths
        """
        Inverse of hexagon/pentagon edge length: 1.0/dual_edge_length.

        defined in ICON in mo_model_domain.f90:t_grid_edges%inv_dual_edge_length
        """

        self.inverse_vertex_vertex_lengths: fa.EdgeField[float] = inverse_vertex_vertex_lengths
        """
        Inverse distance between outer vertices of adjacent cells.

        v1--------
        |       /|
        |      / |
        |    e   |
        |  /     |
        |/       |
        --------v2

        inverse_vertex_vertex_length(e) = 1.0/|v2-v1|

        defined in ICON in mo_model_domain.f90:t_grid_edges%inv_vert_vert_length
        """

        self.primal_normal_vert: tuple[
            gtx.Field[[dims.ECVDim], float], gtx.Field[[dims.ECVDim], float]
        ] = (
            primal_normal_vert_x,
            primal_normal_vert_y,
        )
        """
        Normal of the triangle edge, projected onto the location of the
        four vertices of neighboring cells.

        defined in ICON in mo_model_domain.f90:t_grid_edges%primal_normal_vert
        and computed in ICON in mo_intp_coeffs.f90
        """

        self.dual_normal_vert: tuple[
            gtx.Field[[dims.ECVDim], float], gtx.Field[[dims.ECVDim], float]
        ] = (
            dual_normal_vert_x,
            dual_normal_vert_y,
        )
        """
        zonal (x) and meridional (y) components of vector tangent to the triangle edge,
        projected onto the location of the four vertices of neighboring cells.

        defined in ICON in mo_model_domain.f90:t_grid_edges%dual_normal_vert
        and computed in ICON in mo_intp_coeffs.f90
        """

        self.primal_normal_cell: tuple[
            gtx.Field[[dims.ECDim], float], gtx.Field[[dims.ECDim], float]
        ] = (
            primal_normal_cell_x,
            primal_normal_cell_y,
        )
        """
        zonal (x) and meridional (y) components of vector normal to the cell edge
        projected onto the location of neighboring cell centers.

        defined in ICON in mo_model_domain.f90:t_grid_edges%primal_normal_cell
        and computed in ICON in mo_intp_coeffs.f90
        """

        self.dual_normal_cell: tuple[
            gtx.Field[[dims.ECDim], float], gtx.Field[[dims.ECDim], float]
        ] = (
            dual_normal_cell_x,
            dual_normal_cell_y,
        )
        """
        zonal (x) and meridional (y) components of vector normal to the dual edge
        projected onto the location of neighboring cell centers.

        defined in ICON in mo_model_domain.f90:t_grid_edges%dual_normal_cell
        and computed in ICON in mo_intp_coeffs.f90
        """

        self.edge_areas: fa.EdgeField[float] = edge_areas
        """
        Area of the quadrilateral whose edges are the primal edge and
        the associated dual edge.

        defined in ICON in mo_model_domain.f90:t_grid_edges%area_edge
        and computed in ICON in mo_intp_coeffs.f90
        """

        self.f_e: fa.EdgeField[float] = f_e
        """
        Coriolis parameter at cell edges
        """

        self.edge_center: tuple[fa.EdgeField[float], fa.EdgeField[float]] = (
            edge_center_lat,
            edge_center_lon,
        )
        """
        Latitude and longitude at the edge center

        defined in ICON in mo_model_domain.f90:t_grid_edges%center
        """

        self.primal_normal: tuple[
            gtx.Field[[dims.ECDim], float], gtx.Field[[dims.ECDim], float]
        ] = (
            primal_normal_x,
            primal_normal_y,
        )
        """
        zonal (x) and meridional (y) components of vector normal to the cell edge

        defined in ICON in mo_model_domain.f90:t_grid_edges%primal_normal
        """


@dataclasses.dataclass(frozen=True)
class CellParams:
    #: Latitude at the cell center. The cell center is defined to be the circumcenter of a triangle.
    cell_center_lat: fa.CellField[float] = None
    #: Longitude at the cell center. The cell center is defined to be the circumcenter of a triangle.
    cell_center_lon: fa.CellField[float] = None
    #: Area of a cell, defined in ICON in mo_model_domain.f90:t_grid_cells%area
    area: fa.CellField[float] = None
    #: Mean area of a cell [m^2] = total surface area / numer of cells defined in ICON in in mo_model_domimp_patches.f90
    mean_cell_area: float = None
    length_rescale_factor: float = 1.0

    @classmethod
    def from_global_num_cells(
        cls,
        cell_center_lat: fa.CellField[float],
        cell_center_lon: fa.CellField[float],
        area: fa.CellField[float],
        global_num_cells: int,
        length_rescale_factor: float = 1.0,
    ):
        if global_num_cells == 0:
            # Compute from the area array (should be a torus grid)
            # TODO (Magdalena) this would not work for a distributed setup (at
            # least not for a sphere) for the torus it would because cell area
            # is constant.
            mean_cell_area = area.asnumpy().mean()
        else:
            mean_cell_area = compute_mean_cell_area_for_sphere(
                constants.EARTH_RADIUS, global_num_cells
            )
        return cls(
            cell_center_lat=cell_center_lat,
            cell_center_lon=cell_center_lon,
            area=area,
            mean_cell_area=mean_cell_area,
            length_rescale_factor=length_rescale_factor,
        )

    @functools.cached_property
    def characteristic_length(self):
        return math.sqrt(self.mean_cell_area)

    @functools.cached_property
    def mean_cell_area(self):
        return self.mean_cell_area


def compute_mean_cell_area_for_sphere(radius, num_cells):
    """
    Compute the mean cell area.

    Computes the mean cell area by dividing the sphere by the number of cells in the
    global grid.

    Args:
        radius: average earth radius, might be rescaled by a scaling parameter
        num_cells: number of cells on the global grid
    Returns: mean area of one cell [m^2]
    """
    return 4.0 * math.pi * radius**2 / num_cells


def edge_normals():
    """
    compute
     - primal_normal_x and primal_normal_y
        algorithm:
         for all edges compute
            compute primal_tangent: normalize(cartesian_coordinates(neighboring vertices of an edge)[0] - cartesian_coordinates(neighboring vertices of an edge)[1]
            cartesian coordinate of edge centers: spherical_to_cartesian_on_edges(edge_center_lat, edge_center_lon)
            take cross product aka outer product the above and primal_tangent
            normalize the result.


     - primal_normal_vert (x, y)
     - dual_normal_vert (x, y)
     - primal_normal_cell (x, y) - done

     algorithm:
        compute zonal and meridional component of  primal_normal at cell centers

           IF ( ilc(1) > 0 ) THEN ! Cells of outermost edge not in halo
        CALL cvec2gvec(primal_normal,tri%cells%center(ilc(1),ibc(1)),edges%primal_normal_cell(jl_e,jb_e,1))
        CALL cvec2gvec(dual_normal,tri%cells%center(ilc(1),ibc(1)),edges%dual_normal_cell(jl_e,jb_e,1))
      ELSE
        edges%primal_normal_cell(jl_e,jb_e,1)%v1 = -1._wp
        edges%primal_normal_cell(jl_e,jb_e,1)%v2 =  0._wp
        edges%dual_normal_cell(jl_e,jb_e,1)%v1   = -1._wp
        edges%dual_normal_cell(jl_e,jb_e,1)%v2   =  0._wp


     - dual_normal_cell (x, y)
        compute zonal and meridional component of  primal_normal at cell centers

    """


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_zonal_and_meridional_components_on_cells(
    lat: fa.CellField[ta.wpfloat],
    lon: fa.CellField[ta.wpfloat],
    x: fa.CellField[ta.wpfloat],
    y: fa.CellField[ta.wpfloat],
    z: fa.CellField[ta.wpfloat],
) -> tuple[fa.CellField[ta.wpfloat], fa.CellField[ta.wpfloat]]:
    cos_lat = cos(lat)
    sin_lat = sin(lat)
    cos_lon = cos(lon)
    sin_lon = sin(lon)
    u = cos_lon * y - sin_lon * x

    v = cos_lat * z - sin_lat * (cos_lon * x + sin_lon * y)
    norm = sqrt(u * u + v * v)
    return u / norm, v / norm


@gtx.field_operator
def compute_zonal_and_meridional_components_on_edges(
    lat: fa.EdgeField[ta.wpfloat],
    lon: fa.EdgeField[ta.wpfloat],
    x: fa.EdgeField[ta.wpfloat],
    y: fa.EdgeField[ta.wpfloat],
    z: fa.EdgeField[ta.wpfloat],
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    cos_lat = cos(lat)
    sin_lat = sin(lat)
    cos_lon = cos(lon)
    sin_lon = sin(lon)
    u = cos_lon * y - sin_lon * x

    v = cos_lat * z - sin_lat * (cos_lon * x + sin_lon * y)
    norm = sqrt(u * u + v * v)
    return u / norm, v / norm


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def edge_primal_normal(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    subtract_coeff: gtx.Field[gtx.Dims[EdgeDim, E2VDim], ta.wpfloat],
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    x, y, z = spherical_to_cartesian_on_vertex(vertex_lat, vertex_lon, 1.0)
    x = neighbor_sum(subtract_coeff * x(E2V), axis=E2VDim)
    y = neighbor_sum(subtract_coeff * y(E2V), axis=E2VDim)
    z = neighbor_sum(subtract_coeff * z(E2V), axis=E2VDim)
    return normalize_cartesian_vector(x, y, z)


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def primal_normals(
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    x: fa.EdgeField[ta.wpfloat],
    y: fa.EdgeField[ta.wpfloat],
    z: fa.EdgeField[ta.wpfloat],
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    """computes edges%primal_normal_cell, edges%primal_normal_vert"""
    cell_lat_1 = cell_lat(E2C[0])
    cell_lon_1 = cell_lon(E2C[0])
    u1_cell, v1_cell = compute_zonal_and_meridional_components_on_edges(
        cell_lat_1, cell_lon_1, x, y, z
    )
    cell_lat_2 = cell_lat(E2C[1])
    cell_lon_2 = cell_lon(E2C[1])
    u2_cell, v2_cell = compute_zonal_and_meridional_components_on_edges(
        cell_lat_2, cell_lon_2, x, y, z
    )
    vertex_lat_1 = vertex_lat(E2V[0])
    vertex_lon_1 = vertex_lon(E2V[0])
    u1_vertex, v1_vertex = compute_zonal_and_meridional_components_on_edges(
        vertex_lat_1, vertex_lon_1, x, y, z
    )
    vertex_lat_2 = vertex_lat(E2V[1])
    vertex_lon_2 = vertex_lon(E2V[1])
    u2_vertex, v2_vertex = compute_zonal_and_meridional_components_on_edges(
        vertex_lat_2, vertex_lon_2, x, y, z
    )
    return u1_cell, u2_cell, v1_cell, v2_cell, u1_vertex, u2_vertex, v1_vertex, v2_vertex


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def cell_center_arc_distance(
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    radius: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """Compute the length of dual edge.

    Distance between the cell center of edge adjacent cells. This is a edge of the dual grid and is
    orthogonal to the edge. dual_edge_length in ICON.
    """
    x, y, z = spherical_to_cartesian_on_cells(cell_lat, cell_lon, wpfloat(1.0))
    # xe, ye, ze = spherical_to_cartesian_on_edges(edge_lat, edge_lon, wpfloat(1.0))
    x0 = x(E2C[0])
    x1 = x(E2C[1])
    y0 = y(E2C[0])
    y1 = y(E2C[1])
    z0 = z(E2C[0])
    z1 = z(E2C[1])
    # (xi, yi, zi) are normalized by construction
    # arc1 = radius * arccos(dot_product(x0, xe, y0, ye, z0, ze))
    # arc2 = radius * arccos(dot_product(xe, x1, ye, y1, ze, z1))
    # arc = arc1 + arc2
    arc = radius * arccos(dot_product(x0, x1, y0, y1, z0, z1))
    return arc


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_arc_distance_of_far_edges_in_diamond(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    radius: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """Computes the length of a spherical edges
    - the direct edge length (primal_edge_length in ICON)ü
    - the length of the arc between the two far vertices in the diamond E2C2V (vertex_vertex_length in ICON)
    """
    x, y, z = spherical_to_cartesian_on_vertex(vertex_lat, vertex_lon, 1.0)
    x2 = x(E2C2V[2])
    x3 = x(E2C2V[3])
    y2 = y(E2C2V[2])
    y3 = y(E2C2V[3])
    z2 = z(E2C2V[2])
    z3 = z(E2C2V[3])
    # (xi, yi, zi) are normalized by construction

    far_vertex_vertex_length = radius * arccos(dot_product(x2, x3, y2, y3, z2, z3))
    return far_vertex_vertex_length


@gtx.field_operator
def compute_primal_edge_length(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    radius: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """Computes the length of a spherical edges

    Called edge_length in ICON.
    The computation is the same as for the arc length between the far vertices in the E2C2V diamond but
    and could be done using the E2C2V connectivity, but is has different bounds, as there are no skip values for the edge adjacent vertices.
    """
    x, y, z = spherical_to_cartesian_on_vertex(vertex_lat, vertex_lon, 1.0)
    x0 = x(E2V[0])
    x1 = x(E2V[1])
    y0 = y(E2V[0])
    y1 = y(E2V[1])
    z0 = z(E2V[0])
    z1 = z(E2V[1])
    # (xi, yi, zi) are normalized by construction

    edge_length = radius * arccos(dot_product(x0, x1, y0, y1, z0, z1))
    return edge_length


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_edge_length(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    radius: ta.wpfloat,
    edge_length: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    compute_primal_edge_length(
        vertex_lat,
        vertex_lon,
        radius,
        out=edge_length,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def _compute_dual_edge_length_and_far_vertex_distance_in_diamond(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    radius: ta.wpfloat,
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    far_vertex_distance = compute_arc_distance_of_far_edges_in_diamond(
        vertex_lat, vertex_lon, radius
    )
    dual_edge_length = cell_center_arc_distance(cell_lat, cell_lon, edge_lat, edge_lon, radius)
    return far_vertex_distance, dual_edge_length


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_dual_edge_length_and_far_vertex_distance_in_diamond(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    radius: ta.wpfloat,
    far_vertex_distance: fa.EdgeField[ta.wpfloat],
    dual_edge_length: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    _compute_dual_edge_length_and_far_vertex_distance_in_diamond(
        vertex_lat=vertex_lat,
        vertex_lon=vertex_lon,
        cell_lat=cell_lat,
        cell_lon=cell_lon,
        edge_lat=edge_lat,
        edge_lon=edge_lon,
        radius=radius,
        out=(far_vertex_distance, dual_edge_length),
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def _edge_area(
    owner_mask: fa.EdgeField[bool],
    primal_edge_length: fa.EdgeField[fa.wpfloat],
    dual_edge_length: fa.EdgeField[ta.wpfloat],
) -> fa.EdgeField[ta.wpfloat]:
    """compute the edge_area"""
    return where(owner_mask, primal_edge_length * dual_edge_length, 0.0)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def edge_area(
    owner_mask: fa.EdgeField[bool],
    primal_edge_length: fa.EdgeField[fa.wpfloat],
    dual_edge_length: fa.EdgeField[ta.wpfloat],
    edge_area: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    _edge_area(
        owner_mask,
        primal_edge_length,
        dual_edge_length,
        out=edge_area,
        domain={EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def _coriolis_parameter_on_edges(
    edge_center_lat: fa.EdgeField[ta.wpfloat],
    angular_velocity: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """Compute the coriolis force on edges."""
    return 2.0 * angular_velocity * sin(edge_center_lat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def coriolis_parameter_on_edges(
    edge_center_lat: fa.EdgeField[ta.wpfloat],
    angular_velocity: ta.wpfloat,
    coriolis_parameter: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    _coriolis_parameter_on_edges(
        edge_center_lat,
        angular_velocity,
        out=coriolis_parameter,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


class GridGeometry(state_utils.FieldSource):
    def __init__(
        self,
        grid: icon.IconGrid,
        decomposition_info: definitions.DecompositionInfo,
        backend: backend.Backend,
        coordinates: dict[dims.Dimension, dict[Literal["lat", "lon"], gtx.Field]],
        metadata: dict[str, model.FieldMetaData],
    ):
        self._backend = backend
        self._allocator = gtx.constructors.zeros.partial(allocator=backend)
        self._grid = grid
        self._decomposition_info = decomposition_info
        self._attrs = metadata
        self._geometry_type: icon.GeometryType = grid.global_properties.geometry_type
        self._edge_domain = h_grid.domain(dims.EdgeDim)
        self._edge_local_end = self._grid.end_index(self._edge_domain(h_grid.Zone.LOCAL))
        self._edge_local_start = self._grid.start_index(self._edge_domain(h_grid.Zone.LOCAL))
        self._edge_second_boundary_level_start = self._grid.start_index(
            self._edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )

        self._providers: dict[str, factory.FieldProvider] = {}
        coordinates = {
            attrs.CELL_LAT: coordinates[dims.CellDim]["lat"],
            attrs.CELL_LON: coordinates[dims.CellDim]["lon"],
            attrs.VERTEX_LAT: coordinates[dims.VertexDim]["lat"],
            attrs.EDGE_LON: coordinates[dims.EdgeDim]["lon"],
            attrs.EDGE_LAT: coordinates[dims.EdgeDim]["lat"],
            attrs.VERTEX_LON: coordinates[dims.VertexDim]["lon"],
            "edge_owner_mask": gtx.as_field(
                (dims.EdgeDim,), decomposition_info.owner_mask(dims.EdgeDim), dtype=bool
            ),
        }
        coodinate_provider = factory.PrecomputedFieldProvider(coordinates)
        self.register_provider(coodinate_provider)
        # TODO: remove if it works with the providers
        self._fields = coordinates

    def register_provider(self, provider: factory.FieldProvider):
        for dependency in provider.dependencies:
            if dependency not in self._providers.keys():
                raise ValueError(f"Dependency '{dependency}' not found in registered providers")

        for field in provider.fields:
            self._providers[field] = provider

    def __call__(self):
        # edge_length = self._allocator((dims.EdgeDim,), dtype = ta.wpfloat)
        # compute_edge_length.with_backend(self._backend)(vertex_lat=self._fields[attrs.VERTEX_LAT],
        #                                                 vertex_lon = self._fields[attrs.VERTEX_LON],
        #                                                 radius = self._grid.global_properties.length,
        #                                                 edge_length= edge_length,
        #                                                 horizontal_start = self._edge_local_start,
        #                                                 horizontal_end = self._edge_local_end,
        #                                                 )
        # self._fields[attrs.EDGE_LENGTH] = edge_length,
        #
        #
        # dual_edge_length = self._allocator((dims.EdgeDim,), dtype = ta.wpfloat)
        # vertex_vertex_length = self._allocator((dims.EdgeDim,), dtype = ta.wpfloat)
        # compute_dual_edge_length_and_far_vertex_distance_in_diamond.with_backend(self._backend)(
        #
        # )

        edge_length_provider = factory.ProgramFieldProvider(
            func=compute_edge_length,
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.LOCAL),
                )
            },
            fields={
                "edge_length": attrs.EDGE_LENGTH,
            },
            deps={
                "vertex_lat": attrs.VERTEX_LAT,
                "vertex_lon": attrs.VERTEX_LON,
            },
            params={"radius": self._grid.global_properties.length},
        )
        self.register_provider(edge_length_provider)
        name, meta = attrs.data_for_inverse(attrs.attrs[attrs.EDGE_LENGTH])
        self._attrs.update({name: meta})
        inverse_edge_length = ProgramFieldProvider(
            func=math_helpers.compute_inverse,
            deps={"f": attrs.EDGE_LENGTH},
            fields={"f_inverse": name},
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.LOCAL),
                )
            },
        )
        self.register_provider(inverse_edge_length)

        dual_length_provider = factory.ProgramFieldProvider(
            func=compute_dual_edge_length_and_far_vertex_distance_in_diamond,
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    self._edge_domain(h_grid.Zone.LOCAL),
                )
            },
            fields={
                "dual_edge_length": attrs.DUAL_EDGE_LENGTH,
                "far_vertex_distance": attrs.VERTEX_VERTEX_LENGTH,
            },
            deps={
                "vertex_lat": attrs.VERTEX_LAT,
                "vertex_lon": attrs.VERTEX_LON,
                "cell_lat": attrs.CELL_LAT,
                "cell_lon": attrs.CELL_LON,
                "edge_lat": attrs.EDGE_LAT,
                "edge_lon": attrs.EDGE_LON,
            },
            params={"radius": self._grid.global_properties.length},
        )
        self.register_provider(dual_length_provider)
        name, meta = attrs.data_for_inverse(attrs.attrs[attrs.DUAL_EDGE_LENGTH])
        self._attrs.update({name: meta})
        inverse_dual_length = ProgramFieldProvider(
            func=math_helpers.compute_inverse,
            deps={"f": attrs.DUAL_EDGE_LENGTH},
            fields={"f_inverse": name},
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.LOCAL),
                )
            },
        )
        self.register_provider(inverse_dual_length)

        name, meta = attrs.data_for_inverse(attrs.attrs[attrs.VERTEX_VERTEX_LENGTH])
        self._attrs.update({name: meta})
        inverse_far_edge_distance_provider = ProgramFieldProvider(
            func=math_helpers.compute_inverse,
            deps={"f": attrs.VERTEX_VERTEX_LENGTH},
            fields={"f_inverse": name},
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.LOCAL),
                )
            },
        )
        self.register_provider(inverse_far_edge_distance_provider)

        edge_areas = factory.ProgramFieldProvider(
            func=edge_area,
            deps={
                "owner_mask": "edge_owner_mask",
                "primal_edge_length": attrs.EDGE_LENGTH,
                "dual_edge_length": attrs.DUAL_EDGE_LENGTH,
            },
            fields={"edge_area": attrs.EDGE_AREA},
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
        )
        self.register_provider(edge_areas)
        coriolis_params = factory.ProgramFieldProvider(
            func=coriolis_parameter_on_edges,
            deps={"edge_center_lat": attrs.EDGE_LAT},
            params={"angular_velocity": constants.EARTH_ANGULAR_VELOCITY},
            fields={"coriolis_parameter": attrs.CORIOLIS_PARAMETER},
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
        )
        self.register_provider(coriolis_params)

        # normals:
        # 1. primal_normals: gridfile%zonal_normal_primal_edge - edges%primal_normal%v1, gridfile%meridional_normal_primal_edge - edges%primal_normal%v2,
        # 2. edges%primal_cart_normal (cartesian coordinates for primal_normal
        # 3. primal_normal_vert, primal_normal_cell

        # dual normals:
        # zonal_normal_dual_edge -> edges%dual_normal%v1, meridional_normal_dual_edge -> edges%dual_normal%v2

    def get(
        self, field_name: str, type_: state_utils.RetrievalType = state_utils.RetrievalType.FIELD
    ) -> Union[state_utils.FieldType, xa.DataArray, model.FieldMetaData]:
        if field_name not in self._providers.keys():
            raise ValueError(f"Field {field_name}: unknown geometry field")
        match type_:
            case state_utils.RetrievalType.METADATA:
                return self._attrs[field_name]
            case state_utils.RetrievalType.FIELD | state_utils.RetrievalType.DATA_ARRAY:
                provider = self._providers[field_name]
                if field_name not in provider.fields:
                    raise ValueError(
                        f"Field {field_name} not provided by f{provider.func.__name__}."
                    )

                buffer = provider(field_name, self, self._backend, self)
                return (
                    buffer
                    if type_ == state_utils.RetrievalType.FIELD
                    else state_utils.to_data_array(buffer, attrs=attrs[field_name])
                )
            case _:
                raise NotImplementedError("not yet implemented")

    @property
    def grid(self):
        return self._grid

    @property
    def vertical_grid(self):
        return None
