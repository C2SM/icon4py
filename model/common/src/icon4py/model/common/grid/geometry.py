# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
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
    geometry_stencils as stencils,
    grid_manager as gm,
    horizontal as h_grid,
    icon,
)
from icon4py.model.common.settings import xp
from icon4py.model.common.states import factory, model, utils as state_utils
from icon4py.model.common.states.factory import FieldProvider
from icon4py.model.common.states.model import FieldMetaData


InputGeometryFieldType: TypeAlias = Literal[attrs.CELL_AREA, attrs.TANGENT_ORIENTATION]


class GridGeometry(factory.FieldSource):
    """
    Factory for the ICON grid geometry fields.

    Computes geometry fields from the grid geographical coordinates fo cells, egdes, vertices.
    Computations are triggered upon first request.

    Can be queried for geometry fields and metadata

    Examples:
        >>> geometry = GridGeometry(
        ...     grid, decomposition_info, backend, coordinates, extra_fields, geometry_attributes.attrs
        ... )
        GridGeometry for geometry_type=SPHERE grid=f2e06839-694a-cca1-a3d5-028e0ff326e0 : R9B4
        >>> geometry.get("edge_length")
        NumPyArrayField(_domain=Domain(dims=(Dimension(value='Edge', kind=<DimensionKind.HORIZONTAL: 'horizontal'>),), ranges=(UnitRange(0, 31558),)), _ndarray=array([3746.2669054 , 3746.2669066 , 3746.33418138, ..., 3736.61622936, 3792.41317057]))
        >>> geometry.get("edge_length", RetrievalType.METADATA)
        {'standard_name': 'edge_length',
        'long_name': 'edge length',
        'units': 'm',
        'dims': (Dimension(value='Edge', kind=<DimensionKind.HORIZONTAL: 'horizontal'>),),
        'icon_var_name': 't_grid_edges%primal_edge_length',
        'dtype': numpy.float64}
        >>> geometry.get("edge_length", RetrievalType.DATA_ARRAY)
        <xarray.DataArray (dim_0: 31558)> Size: 252kB
        array([3746.2669054 , 3746.2669066 , 3746.33418138, ..., 3889.53098062, 3736.61622936, 3792.41317057])
        Dimensions without coordinates: dim_0
        .Attributes:
        standard_name:  edge_length
        long_name:      edge length
        units:          m
        dims:           (Dimension(value='Edge', kind=<DimensionKind.HORIZONTAL: ...
        icon_var_name:  t_grid_edges%primal_edge_length
        dtype:          <class 'numpy.float64'>


    """

    def __init__(
        self,
        grid: icon.IconGrid,
        decomposition_info: definitions.DecompositionInfo,
        backend: gtx_backend.Backend,
        coordinates: gm.CoordinateDict,
        extra_fields: dict[InputGeometryFieldType, gtx.Field],
        metadata: dict[str, model.FieldMetaData],
    ):
        """
        Args:
            grid: IconGrid the grid topology
            decomposition_info: data structure containing owner masks for field dimensions
            backend: backend used for memory allocation and computation
            coordinates: dictionary containing geographical coordinates for grid cells, edges and vertices,
            extra_fields: fields that are not computed but directly read off the grid file,
                currently only the edge_system_orientation cell_area. Should eventually disappear.
            metadata: a dictionary of FieldMetaData for all fields computed in GridGeometry.

        """
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
            func=stencils.compute_edge_length,
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
            func=stencils.compute_cell_center_arc_distance,
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
            func=stencils.compute_arc_distance_of_far_edges_in_diamond,
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
            func=stencils.compute_edge_area,
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
            func=stencils.compute_coriolis_parameter_on_edges,
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
            func=stencils.compute_cartesian_coordinates_of_edge_tangent_and_normal,
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
            func=stencils.compute_zonal_and_meridional_component_of_edge_field_at_vertex,
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
            func=stencils.compute_zonal_and_meridional_component_of_edge_field_at_cell_center,
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
            func=stencils.compute_zonal_and_meridional_component_of_edge_field_at_vertex,
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
            func=stencils.compute_zonal_and_meridional_component_of_edge_field_at_cell_center,
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
            func=math_helpers.compute_inverse_on_edges,
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

    def __repr__(self):
        return f"{self.__class__.__name__} for geometry_type={self._geometry_type._name_} (grid={self._grid.id!r})"

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
    """
    Construct auxiliary arrays of geographical coordinates used in the computation of edge normal fields.

    The resulting fields are based on edges and contain geographical coordinates (lat, lon) that are
    - either the coordinates (lat, lon) of an edge's neighboring cell centers
    - or for boundary edges (that have no cell neighbor) the coordinates of the edge center

    Args:
        grid: icon grid
        cell_lat: latitude of cell centers
        cell_lon: longitude of cell centers
        edge_lat: latitude of edge centers
        edge_lon: longitude of edge centers

    Returns:
        latitude of first neighbor
        longitude of first neighbor
        latitude of second neighbor
        longitude of second neighbor
    """
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
