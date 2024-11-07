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
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, TypeAlias, TypeVar

from gt4py import next as gtx
from gt4py.next import backend, backend as gtx_backend

import icon4py.model.common.grid.geometry_attributes as attrs
import icon4py.model.common.math.helpers as math_helpers
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import (
    geometry_program as func,
    grid_manager as gm,
    horizontal as h_grid,
    icon,
)
from icon4py.model.common.settings import xp
from icon4py.model.common.states import factory, model, utils as state_utils
from icon4py.model.common.states.factory import FieldProvider
from icon4py.model.common.states.model import FieldMetaData


class EdgeParams:
    def __init__(
        self,
        tangent_orientation=None,
        primal_edge_lengths=None,
        inverse_primal_edge_lengths=None,
        dual_edge_lengths=None,
        inverse_dual_edge_lengths=None,
        inverse_vertex_vertex_lengths=None,
        primal_normal_vert_x=None,
        primal_normal_vert_y=None,
        dual_normal_vert_x=None,
        dual_normal_vert_y=None,
        primal_normal_cell_x=None,
        dual_normal_cell_x=None,
        primal_normal_cell_y=None,
        dual_normal_cell_y=None,
        edge_areas=None,
        f_e=None,
        edge_center_lat=None,
        edge_center_lon=None,
        primal_normal_x=None,
        primal_normal_y=None,
    ):
        self.tangent_orientation: fa.EdgeField[float] = tangent_orientation
        r"""
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


InputGeometryFieldType: TypeAlias = Literal[attrs.CELL_AREA, attrs.TANGENT_ORIENTATION]


class GridGeometry(factory.FieldSource):
    def __init__(
        self,
        grid: icon.IconGrid,
        decomposition_info: definitions.DecompositionInfo,
        backend: gtx_backend.Backend,
        coordinates: gm.CoordinateDict,
        extra_fields: dict[InputGeometryFieldType, gtx.Field],
        metadata: dict[str, model.FieldMetaData],
    ):
        self._backend = backend
        self._allocator = gtx.constructors.zeros.partial(allocator=backend)
        self._grid = grid
        self._decomposition_info = decomposition_info
        self._attrs = metadata
        self._geometry_type: icon.GeometryType = grid.global_properties.geometry_type
        self._edge_domain = h_grid.domain(dims.EdgeDim)
        self._providers: dict[str, factory.FieldProvider] = {}

        (
            edge_orientation0_lat,
            edge_orientation0_lon,
            edge_orientation1_lat,
            edge_orientation1_lon,
        ) = create_auxiliary_coordinate_arrays_for_orientation(
            self._grid,
            coordinates[dims.CellDim]["lat"],
            coordinates[dims.CellDim]["lon"],
            coordinates[dims.EdgeDim]["lat"],
            coordinates[dims.EdgeDim]["lon"],
        )
        coordinates_ = {
            attrs.CELL_LAT: coordinates[dims.CellDim]["lat"],
            attrs.CELL_LON: coordinates[dims.CellDim]["lon"],
            attrs.VERTEX_LAT: coordinates[dims.VertexDim]["lat"],
            attrs.EDGE_LON: coordinates[dims.EdgeDim]["lon"],
            attrs.EDGE_LAT: coordinates[dims.EdgeDim]["lat"],
            attrs.VERTEX_LON: coordinates[dims.VertexDim]["lon"],
            "latitude_of_edge_cell_neighbor_0": edge_orientation0_lat,
            "longitude_of_edge_cell_neighbor_0": edge_orientation0_lon,
            "latitude_of_edge_cell_neighbor_1": edge_orientation1_lat,
            "longitude_of_edge_cell_neighbor_1": edge_orientation1_lon,
        }
        coodinate_provider = factory.PrecomputedFieldProvider(coordinates_)
        self.register_provider(coodinate_provider)

        input_fields_provider = factory.PrecomputedFieldProvider(
            {
                attrs.CELL_AREA: extra_fields[gm.GeometryName.CELL_AREA],
                attrs.TANGENT_ORIENTATION: extra_fields[gm.GeometryName.TANGENT_ORIENTATION],
                "edge_owner_mask": gtx.as_field(
                    (dims.EdgeDim,), decomposition_info.owner_mask(dims.EdgeDim), dtype=bool
                ),
            }
        )
        self.register_provider(input_fields_provider)
        self._register_computed_fields()

    def _register_computed_fields(self):
        edge_length_provider = factory.ProgramFieldProvider(
            func=func.compute_edge_length,
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.LOCAL),
                )
            },
            fields={
                "length": attrs.EDGE_LENGTH,
            },
            deps={
                "vertex_lat": attrs.VERTEX_LAT,
                "vertex_lon": attrs.VERTEX_LON,
            },
            params={"radius": self._grid.global_properties.radius},
        )
        self.register_provider(edge_length_provider)
        meta = attrs.metadata_for_inverse(attrs.attrs[attrs.EDGE_LENGTH])
        name = meta["standard_name"]
        self._attrs.update({name: meta})
        inverse_edge_length = self._inverse_field_provider(attrs.EDGE_LENGTH)
        self.register_provider(inverse_edge_length)

        dual_length_provider = factory.ProgramFieldProvider(
            func=func.compute_cell_center_arc_distance,
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.LOCAL),
                )
            },
            fields={
                "dual_edge_length": attrs.DUAL_EDGE_LENGTH,
            },
            deps={
                "edge_neighbor_0_lat": "latitude_of_edge_cell_neighbor_0",
                "edge_neighbor_0_lon": "longitude_of_edge_cell_neighbor_0",
                "edge_neighbor_1_lat": "latitude_of_edge_cell_neighbor_1",
                "edge_neighbor_1_lon": "longitude_of_edge_cell_neighbor_1",
            },
            params={"radius": self._grid.global_properties.radius},
        )
        self.register_provider(dual_length_provider)
        inverse_dual_edge_length = self._inverse_field_provider(attrs.DUAL_EDGE_LENGTH)
        self.register_provider(inverse_dual_edge_length)

        vertex_vertex_distance = factory.ProgramFieldProvider(
            func=func.compute_arc_distance_of_far_edges_in_diamond,
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    self._edge_domain(h_grid.Zone.LOCAL),
                )
            },
            fields={"far_vertex_distance": attrs.VERTEX_VERTEX_LENGTH},
            deps={
                "vertex_lat": attrs.VERTEX_LAT,
                "vertex_lon": attrs.VERTEX_LON,
            },
            params={"radius": self._grid.global_properties.radius},
        )
        self.register_provider(vertex_vertex_distance)

        inverse_far_edge_distance_provider = self._inverse_field_provider(
            attrs.VERTEX_VERTEX_LENGTH
        )
        self.register_provider(inverse_far_edge_distance_provider)

        edge_areas = factory.ProgramFieldProvider(
            func=func.compute_edge_area,
            deps={
                "owner_mask": "edge_owner_mask",
                "primal_edge_length": attrs.EDGE_LENGTH,
                "dual_edge_length": attrs.DUAL_EDGE_LENGTH,
            },
            fields={"area": attrs.EDGE_AREA},
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
        )
        self.register_provider(edge_areas)
        coriolis_params = factory.ProgramFieldProvider(
            func=func.compute_coriolis_parameter_on_edges,
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
        # 1. edges%primal_cart_normal (cartesian coordinates for primal_normal
        tangent_normal_coordinates = factory.ProgramFieldProvider(
            func=func.compute_cartesian_coordinates_of_edge_tangent_and_normal,
            deps={
                "vertex_lat": attrs.VERTEX_LAT,
                "vertex_lon": attrs.VERTEX_LON,
                "edge_lat": attrs.EDGE_LAT,
                "edge_lon": attrs.EDGE_LON,
                "edge_orientation": attrs.TANGENT_ORIENTATION,
            },
            fields={
                "tangent_x": attrs.EDGE_TANGENT_X,
                "tangent_y": attrs.EDGE_TANGENT_Y,
                "tangent_z": attrs.EDGE_TANGENT_Z,
                "normal_x": attrs.EDGE_NORMAL_X,
                "normal_y": attrs.EDGE_NORMAL_Y,
                "normal_z": attrs.EDGE_NORMAL_Z,
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
        )
        self.register_provider(tangent_normal_coordinates)
        # 2. primal_normals: gridfile%zonal_normal_primal_edge - edges%primal_normal%v1, gridfile%meridional_normal_primal_edge - edges%primal_normal%v2,
        normal_uv = factory.ProgramFieldProvider(
            func=math_helpers.compute_zonal_and_meridional_components_on_edges,
            deps={
                "lat": attrs.EDGE_LAT,
                "lon": attrs.EDGE_LON,
                "x": attrs.EDGE_NORMAL_X,
                "y": attrs.EDGE_NORMAL_Y,
                "z": attrs.EDGE_NORMAL_Z,
            },
            fields={
                "u": attrs.EDGE_NORMAL_U,
                "v": attrs.EDGE_NORMAL_V,
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
        )
        self.register_provider(normal_uv)

        # 3. primal_normal_vert, primal_normal_cell
        normal_vert = factory.ProgramFieldProvider(
            func=func.compute_zonal_and_meridional_component_of_edge_field_at_vertex,
            deps={
                "vertex_lat": attrs.VERTEX_LAT,
                "vertex_lon": attrs.VERTEX_LON,
                "x": attrs.EDGE_NORMAL_X,
                "y": attrs.EDGE_NORMAL_Y,
                "z": attrs.EDGE_NORMAL_Z,
            },
            fields={
                "u_vertex_1": "u_vertex_1",
                "v_vertex_1": "v_vertex_1",
                "u_vertex_2": "u_vertex_2",
                "v_vertex_2": "v_vertex_2",
                "u_vertex_3": "u_vertex_3",
                "v_vertex_3": "v_vertex_3",
                "u_vertex_4": "u_vertex_4",
                "v_vertex_4": "v_vertex_4",
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
        )
        normal_vert_wrapper = SparseFieldProviderWrapper(
            normal_vert,
            target_dims=attrs.attrs[attrs.EDGE_NORMAL_VERTEX_U]["dims"],
            fields=(attrs.EDGE_NORMAL_VERTEX_U, attrs.EDGE_NORMAL_VERTEX_V),
            pairs=(
                ("u_vertex_1", "u_vertex_2", "u_vertex_3", "u_vertex_4"),
                ("v_vertex_1", "v_vertex_2", "v_vertex_3", "v_vertex_4"),
            ),
        )
        self.register_provider(normal_vert_wrapper)
        normal_cell = factory.ProgramFieldProvider(
            func=func.compute_zonal_and_meridional_component_of_edge_field_at_cell_center,
            deps={
                "cell_lat": attrs.CELL_LAT,
                "cell_lon": attrs.CELL_LON,
                "x": attrs.EDGE_NORMAL_X,
                "y": attrs.EDGE_NORMAL_Y,
                "z": attrs.EDGE_NORMAL_Z,
            },
            fields={
                "u_cell_1": "u_cell_1",
                "v_cell_1": "v_cell_1",
                "u_cell_2": "u_cell_2",
                "v_cell_2": "v_cell_2",
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
        )
        normal_cell_wrapper = SparseFieldProviderWrapper(
            normal_cell,
            target_dims=attrs.attrs[attrs.EDGE_NORMAL_CELL_U]["dims"],
            fields=(attrs.EDGE_NORMAL_CELL_U, attrs.EDGE_NORMAL_CELL_V),
            pairs=(("u_cell_1", "u_cell_2"), ("v_cell_1", "v_cell_2")),
        )
        self.register_provider(normal_cell_wrapper)
        # 3. dual normals: the dual normals are the edge tangents
        tangent_vert = factory.ProgramFieldProvider(
            func=func.compute_zonal_and_meridional_component_of_edge_field_at_vertex,
            deps={
                "vertex_lat": attrs.VERTEX_LAT,
                "vertex_lon": attrs.VERTEX_LON,
                "x": attrs.EDGE_TANGENT_X,
                "y": attrs.EDGE_TANGENT_Y,
                "z": attrs.EDGE_TANGENT_Z,
            },
            fields={
                "u_vertex_1": "u_vertex_1",
                "v_vertex_1": "v_vertex_1",
                "u_vertex_2": "u_vertex_2",
                "v_vertex_2": "v_vertex_2",
                "u_vertex_3": "u_vertex_3",
                "v_vertex_3": "v_vertex_3",
                "u_vertex_4": "u_vertex_4",
                "v_vertex_4": "v_vertex_4",
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
        )
        tangent_vert_wrapper = SparseFieldProviderWrapper(
            tangent_vert,
            target_dims=attrs.attrs[attrs.EDGE_TANGENT_VERTEX_U]["dims"],
            fields=(attrs.EDGE_TANGENT_VERTEX_U, attrs.EDGE_TANGENT_VERTEX_V),
            pairs=(
                ("u_vertex_1", "u_vertex_2", "u_vertex_3", "u_vertex_4"),
                ("v_vertex_1", "v_vertex_2", "v_vertex_3", "v_vertex_4"),
            ),
        )
        self.register_provider(tangent_vert_wrapper)
        tangent_cell = factory.ProgramFieldProvider(
            func=func.compute_zonal_and_meridional_component_of_edge_field_at_cell_center,
            deps={
                "cell_lat": attrs.CELL_LAT,
                "cell_lon": attrs.CELL_LON,
                "x": attrs.EDGE_TANGENT_X,
                "y": attrs.EDGE_TANGENT_Y,
                "z": attrs.EDGE_TANGENT_Z,
            },
            fields={
                "u_cell_1": "u_cell_1",
                "v_cell_1": "v_cell_1",
                "u_cell_2": "u_cell_2",
                "v_cell_2": "v_cell_2",
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
        )
        tangent_cell_wrapper = SparseFieldProviderWrapper(
            tangent_cell,
            target_dims=attrs.attrs[attrs.EDGE_TANGENT_CELL_U]["dims"],
            fields=(attrs.EDGE_TANGENT_CELL_U, attrs.EDGE_TANGENT_CELL_V),
            pairs=(("u_cell_1", "u_cell_2"), ("v_cell_1", "v_cell_2")),
        )
        self.register_provider(tangent_cell_wrapper)

    def _inverse_field_provider(self, field_name: str):
        meta = attrs.metadata_for_inverse(attrs.attrs[field_name])
        name = meta["standard_name"]
        self._attrs.update({name: meta})
        provider = factory.ProgramFieldProvider(
            func=math_helpers.compute_inverse,
            deps={"f": field_name},
            fields={"f_inverse": name},
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.LOCAL),
                )
            },
        )
        return provider

    @property
    def providers(self) -> dict[str, FieldProvider]:
        return self._providers

    @property
    def metadata(self) -> dict[str, FieldMetaData]:
        return self._attrs

    @property
    def backend(self) -> backend.Backend:
        return self._backend

    @property
    def grid_provider(self):
        return self

    @property
    def grid(self):
        return self._grid

    @property
    def vertical_grid(self):
        return None


HorizontalD = TypeVar("HorizontalD", bound=gtx.Dimension)
SparseD = TypeVar("SparseD", bound=gtx.Dimension)


class SparseFieldProviderWrapper(factory.FieldProvider):
    def __init__(
        self,
        field_provider: factory.ProgramFieldProvider,
        target_dims: tuple[HorizontalD, SparseD],
        fields: Sequence[str],
        pairs: Sequence[tuple[str, ...]],
    ):
        self._wrapped_provider = field_provider
        self._fields = {name: None for name in fields}
        self._func = functools.partial(as_sparse_field, target_dims)
        self._pairs = pairs

    def __call__(
        self,
        field_name: str,
        field_src: Optional[factory.FieldSource],
        backend: Optional[gtx_backend.Backend],
        grid: Optional[factory.GridProvider],
    ):
        if not self._fields.get(field_name):
            # get the fields from the wrapped provider

            input_fields = []
            for p in self._pairs:
                t = tuple([self._wrapped_provider(name, field_src, backend, grid) for name in p])
                input_fields.append(t)
            sparse_fields = self.func(input_fields)
            self._fields = {k: sparse_fields[i] for i, k in enumerate(self.fields)}
        return self._fields[field_name]

    @property
    def dependencies(self) -> Sequence[str]:
        return self._wrapped_provider.dependencies

    @property
    def fields(self) -> Mapping[str, Any]:
        return self._fields

    @property
    def func(self) -> Callable:
        return self._func


def as_sparse_field(
    target_dims: tuple[HorizontalD, SparseD],
    data: Sequence[tuple[gtx.Field[gtx.Dims[HorizontalD], state_utils.ScalarType], ...]],
):
    assert len(target_dims) == 2
    assert target_dims[0].kind == gtx.DimensionKind.HORIZONTAL
    assert target_dims[1].kind == gtx.DimensionKind.LOCAL
    fields = []
    for t in data:
        buffers = list(b.ndarray for b in t)
        field = gtx.as_field(target_dims, data=(xp.vstack(buffers).T), dtype=buffers[0].dtype)
        fields.append(field)
    return fields


def create_auxiliary_coordinate_arrays_for_orientation(
    grid: icon.IconGrid,
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    e2c_table = grid.connectivities[dims.E2CDim]
    lat = cell_lat.ndarray[e2c_table]
    lon = cell_lon.ndarray[e2c_table]
    for i in (0, 1):
        boundary_edges = xp.where(e2c_table[:, i] == gm.GridFile.INVALID_INDEX)
        lat[boundary_edges, i] = edge_lat.ndarray[boundary_edges]
        lon[boundary_edges, i] = edge_lon.ndarray[boundary_edges]

    return (
        gtx.as_field((dims.EdgeDim,), lat[:, 0]),
        gtx.as_field((dims.EdgeDim,), lon[:, 0]),
        gtx.as_field((dims.EdgeDim,), lat[:, 1]),
        gtx.as_field((dims.EdgeDim,), lon[:, 1]),
    )
