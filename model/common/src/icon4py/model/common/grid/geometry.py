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

from gt4py import next as gtx
from gt4py.next import backend
from gt4py.next.ffront.fbuiltins import arccos, cos, neighbor_sum, sin, sqrt, where

from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.dimension import E2C, E2C2V, E2V, E2CDim, E2VDim, EdgeDim
from icon4py.model.common.grid import icon
from icon4py.model.common.math.helpers import (
    dot_product,
    norm2,
    normalize_cartesian_vector,
    spherical_to_cartesian_on_cells,
    spherical_to_cartesian_on_vertex,
)
from icon4py.model.common.states import utils as state_utils
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
def primal_normals(
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    vertex_lat:fa.VertexField[ta.wpfloat],
    vertex_lon:fa.VertexField[ta.wpfloat],
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
    u1_cell, v1_cell = compute_zonal_and_meridional_components_on_edges(cell_lat_1, cell_lon_1, x, y, z)
    cell_lat_2 = cell_lat(E2C[1])
    cell_lon_2 = cell_lon(E2C[1])
    u2_cell, v2_cell = compute_zonal_and_meridional_components_on_edges(cell_lat_2, cell_lon_2, x, y, z)
    vertex_lat_1 = vertex_lat(E2V[0])
    vertex_lon_1 = vertex_lon(E2V[0])
    u1_vertex, v1_vertex = compute_zonal_and_meridional_components_on_edges(vertex_lat_1, vertex_lon_1, x, y, z)
    vertex_lat_2 = vertex_lat(E2V[1])
    vertex_lon_2 = vertex_lon(E2V[1])
    u2_vertex, v2_vertex = compute_zonal_and_meridional_components_on_edges(vertex_lat_2,
                                                                            vertex_lon_2, x, y, z)
    return u1_cell, u2_cell, v1_cell, v2_cell, u1_vertex, u2_vertex, v1_vertex, v2_vertex



@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def dual_edge_length(
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    subtract_coeff: gtx.Field[gtx.Dims[EdgeDim, E2CDim], ta.wpfloat],
    radius: ta.wpfloat,
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    x, y, z = spherical_to_cartesian_on_cells(cell_lat, cell_lon, wpfloat(1.0))

    # that is the "Bogensehne"
    dx = neighbor_sum(subtract_coeff * x(E2C), axis=E2CDim)
    dy = neighbor_sum(subtract_coeff * y(E2C), axis=E2CDim)
    dz = neighbor_sum(subtract_coeff * z(E2C), axis=E2CDim)
    tendon = radius * norm2(dx, dy, dz)

    x0 = x(E2C[0])
    x1 = x(E2C[1])
    y0 = y(E2C[0])
    y1 = y(E2C[1])
    z0 = z(E2C[0])
    z1 = z(E2C[1])
    # norms = norm2(x0, y0, z0) * norm2(x1, y1, z1) # == 1 by construction
    prod = dot_product(x0, x1, y0, y1, z0, z1) #/ norms
    arc = radius * arccos(prod)
    return arc, tendon


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
def primal_edge_length(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    subtract_coeff: gtx.Field[gtx.Dims[EdgeDim, E2VDim], ta.wpfloat],
    radius: ta.wpfloat,
) -> tuple[fa.EdgeField[ta.wpfloat],fa.EdgeField[ta.wpfloat] ]:
    x, y, z = spherical_to_cartesian_on_vertex(vertex_lat, vertex_lon, 1.0)
    dx = neighbor_sum(subtract_coeff * x(E2V), axis=E2VDim)
    dy = neighbor_sum(subtract_coeff * y(E2V), axis=E2VDim)
    dz = neighbor_sum(subtract_coeff * z(E2V), axis=E2VDim)

    # length of tenddon:
    tendon = radius * norm2(dx, dy, dz)
    # arc length
    x0 = x(E2V[0])
    x1 = x(E2V[1])

    y0 = y(E2V[0])
    y1 = y(E2V[1])
    z0 = z(E2V[0])
    z1 = z(E2V[1])

    #norms = norm2(x0, y0, z0) * norm2(x1, y1, z1) # norm is 1 by constructions
    prod = dot_product(x0, x1, y0, y1, z0, z1)
    arc = radius * arccos(prod)
    return arc, tendon


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def vertex_vertex_length(
    vertex_lat: fa.VertexField[fa.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    radius: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    x, y, z = spherical_to_cartesian_on_vertex(vertex_lat, vertex_lon, 1.0)
    x1 = x(E2C2V[2])
    x2 = x(E2C2V[3])
    y1 = y(E2C2V[2])
    y2 = y(E2C2V[3])
    z1 = z(E2C2V[2])
    z2 = z(E2C2V[3])
    norm = norm2(x1, y1, z1) * norm2(x2, y2, z2)

    alpha = dot_product(x1, x2, y1, y2, z1, z2) / norm
    return radius * arccos(alpha)


@gtx.field_operator
def edge_control_area(
    owner_mask: fa.EdgeField[bool],
    primal_egde_length: fa.EdgeField[fa.wpfloat],
    dual_edge_length: fa.EdgeField[ta.wpfloat],
) -> fa.EdgeField[ta.wpfloat]:
    """compute the edge_area"""
    return where(owner_mask, primal_egde_length * dual_edge_length, 0.0)


@gtx.field_operator
def coriolis_parameter_on_edges(
    edge_center_lat: fa.EdgeField[ta.wpfloat], angular_velocity: ta.wpfloat
) -> fa.EdgeField[ta.wpfloat]:
    """Compute the coriolis force on edges: f_e"""
    return 2.0 * angular_velocity * sin(edge_center_lat)


class GridGeometry(state_utils.FieldSource):

    def __init__(self, grid:icon.IconGrid, backend:backend.Backend):
        self._backend = backend
        self._grid = grid
        self._geometry_type:icon.GeometryType = grid.global_properties.geometry_type

    def get(self, field_mame:str, type:state_utils.RetrievalType):
        return 0
