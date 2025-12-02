# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import gt4py.next.typing as gtx_typing
from gt4py import next as gtx
from typing_extensions import assert_never

import icon4py.model.common.math.helpers as math_helpers
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import (
    base,
    geometry_attributes as attrs,
    geometry_stencils as stencils,
    grid_manager as gm,
    gridfile,
    horizontal as h_grid,
    icon,
)
from icon4py.model.common.states import factory, model, utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils


log = logging.getLogger(__name__)


class GridGeometry(factory.FieldSource):
    """
    Factory for the ICON grid geometry fields.

    Computes geometry fields from the grid geographical coordinates fo cells, edges, vertices.
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
        backend: gtx_typing.Backend | None,
        coordinates: gm.CoordinateDict,
        extra_fields: gm.GeometryDict,
        metadata: dict[str, model.FieldMetaData],
    ) -> None:
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
        self._providers = {}
        self._backend = backend
        self._xp = data_alloc.import_array_ns(backend)
        self._allocator = gtx.constructors.zeros.partial(allocator=backend)
        self._grid = grid
        self._decomposition_info = decomposition_info
        self._attrs = metadata
        self._geometry_type: base.GeometryType = grid.global_properties.geometry_type
        self._edge_domain = h_grid.domain(dims.EdgeDim)
        log.info(
            f"initializing geometry for backend = '{self._backend_name()}' and grid = '{self._grid}'"
        )

        # Setup coordinates based on geometry type
        coordinates_ = {
            attrs.CELL_LON: coordinates[dims.CellDim]["lon"],
            attrs.CELL_LAT: coordinates[dims.CellDim]["lat"],
            attrs.EDGE_LON: coordinates[dims.EdgeDim]["lon"],
            attrs.EDGE_LAT: coordinates[dims.EdgeDim]["lat"],
            attrs.VERTEX_LON: coordinates[dims.VertexDim]["lon"],
            attrs.VERTEX_LAT: coordinates[dims.VertexDim]["lat"],
        }
        if self._geometry_type == base.GeometryType.TORUS:
            coordinates_[attrs.CELL_CENTER_X] = coordinates[dims.CellDim]["x"]
            coordinates_[attrs.CELL_CENTER_Y] = coordinates[dims.CellDim]["y"]
            coordinates_[attrs.CELL_CENTER_Z] = coordinates[dims.CellDim]["z"]
            coordinates_[attrs.EDGE_CENTER_X] = coordinates[dims.EdgeDim]["x"]
            coordinates_[attrs.EDGE_CENTER_Y] = coordinates[dims.EdgeDim]["y"]
            coordinates_[attrs.EDGE_CENTER_Z] = coordinates[dims.EdgeDim]["z"]
            coordinates_[attrs.VERTEX_X] = coordinates[dims.VertexDim]["x"]
            coordinates_[attrs.VERTEX_Y] = coordinates[dims.VertexDim]["y"]
            coordinates_[attrs.VERTEX_Z] = coordinates[dims.VertexDim]["z"]

        coordinate_provider = factory.PrecomputedFieldProvider(coordinates_)
        self.register_provider(coordinate_provider)

        # Setup input fields
        input_fields_provider = factory.PrecomputedFieldProvider(
            {
                # TODO(halungge): rescaled by grid_length_rescale_factor (mo_grid_tools.f90)
                attrs.EDGE_LENGTH: extra_fields[gridfile.GeometryName.EDGE_LENGTH],
                attrs.DUAL_EDGE_LENGTH: extra_fields[gridfile.GeometryName.DUAL_EDGE_LENGTH],
                attrs.EDGE_CELL_DISTANCE: extra_fields[gridfile.GeometryName.EDGE_CELL_DISTANCE],
                attrs.EDGE_VERTEX_DISTANCE: extra_fields[
                    gridfile.GeometryName.EDGE_VERTEX_DISTANCE
                ],
                attrs.CELL_AREA: extra_fields[gridfile.GeometryName.CELL_AREA],
                attrs.DUAL_AREA: extra_fields[gridfile.GeometryName.DUAL_AREA],
                attrs.TANGENT_ORIENTATION: extra_fields[gridfile.GeometryName.TANGENT_ORIENTATION],
                "edge_owner_mask": gtx.as_field(
                    (dims.EdgeDim,),
                    decomposition_info.owner_mask(dims.EdgeDim),
                    dtype=bool,
                    allocator=self._backend,
                ),
                attrs.CELL_NORMAL_ORIENTATION: extra_fields[
                    gridfile.GeometryName.CELL_NORMAL_ORIENTATION
                ],
                attrs.VERTEX_EDGE_ORIENTATION: extra_fields[
                    gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX
                ],
                "vertex_owner_mask": gtx.as_field(
                    (dims.VertexDim,),
                    decomposition_info.owner_mask(dims.VertexDim),
                    allocator=self._backend,
                    dtype=bool,
                ),
                "cell_owner_mask": gtx.as_field(
                    (dims.CellDim,),
                    decomposition_info.owner_mask(dims.CellDim),
                    allocator=self._backend,
                    dtype=bool,
                ),
            }
        )
        self.register_provider(input_fields_provider)
        self._register_computed_fields()

    @staticmethod
    def with_geometry_type(
        grid: icon.IconGrid,
        decomposition_info: definitions.DecompositionInfo,
        backend: gtx_typing.Backend | None,
        coordinates: gm.CoordinateDict,
        extra_fields: gm.GeometryDict,
        metadata: dict[str, model.FieldMetaData],
    ) -> "GridGeometry":
        return GridGeometry(
            grid,
            decomposition_info,
            backend,
            coordinates,
            extra_fields,
            metadata,
        )

    def _inverse_field_provider(self, field_name: str) -> factory.FieldProvider:
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

    def _register_computed_fields(self) -> None:
        """Register all computed geometry fields."""
        # Common fields for both geometries
        meta = attrs.metadata_for_inverse(attrs.attrs[attrs.EDGE_LENGTH])
        name = meta["standard_name"]
        self._attrs.update({name: meta})

        inverse_edge_length = self._inverse_field_provider(attrs.EDGE_LENGTH)
        self.register_provider(inverse_edge_length)

        inverse_dual_edge_length = self._inverse_field_provider(attrs.DUAL_EDGE_LENGTH)
        self.register_provider(inverse_dual_edge_length)

        # Cartesian coordinates for icosahedron geometry (the torus reads them
        # from the grid file)
        if self._geometry_type == base.GeometryType.ICOSAHEDRON:
            self._register_cartesian_coordinates_icosahedron()

        # vertex-vertex distance (geometry-specific)
        match self._geometry_type:
            case base.GeometryType.ICOSAHEDRON:
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
            case base.GeometryType.TORUS:
                vertex_vertex_distance = factory.ProgramFieldProvider(
                    func=stencils.compute_distance_of_far_edges_in_diamond_torus,
                    domain={
                        dims.EdgeDim: (
                            self._edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                            self._edge_domain(h_grid.Zone.LOCAL),
                        )
                    },
                    fields={"far_vertex_distance": attrs.VERTEX_VERTEX_LENGTH},
                    deps={
                        "vertex_x": attrs.VERTEX_X,
                        "vertex_y": attrs.VERTEX_Y,
                    },
                    params={
                        "domain_length": self._grid.global_properties.domain_length,
                        "domain_height": self._grid.global_properties.domain_height,
                    },
                )
            case _:
                assert_never(self._geometry_type)
        self.register_provider(vertex_vertex_distance)

        # Inverse of vertex-vertex distance
        inverse_far_edge_distance_provider = self._inverse_field_provider(
            attrs.VERTEX_VERTEX_LENGTH
        )
        self.register_provider(inverse_far_edge_distance_provider)

        # Edge areas
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

        # Coriolis parameter (geometry-specific)
        match self._geometry_type:
            case base.GeometryType.ICOSAHEDRON:
                coriolis_param = factory.ProgramFieldProvider(
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
            case base.GeometryType.TORUS:
                coriolis_param = factory.PrecomputedFieldProvider(
                    {
                        "coriolis_parameter": gtx.as_field(
                            (dims.EdgeDim,),
                            # TODO(jcanton): this should eventually come from
                            # the config: const * ones
                            self._xp.zeros(
                                self._grid.start_index(self._edge_domain(h_grid.Zone.END))
                                - self._grid.start_index(self._edge_domain(h_grid.Zone.LOCAL))
                            ),
                            dtype=ta.wpfloat,
                            allocator=self._backend,
                        )
                    }
                )
            case _:
                assert_never(self._geometry_type)
        self.register_provider(coriolis_param)

        # Tangent and normal coordinates (geometry-specific)
        match self._geometry_type:
            case base.GeometryType.ICOSAHEDRON:
                self._register_normals_and_tangents_icosahedron()
            case base.GeometryType.TORUS:
                self._register_normals_and_tangents_torus()
            case _:
                assert_never(self._geometry_type)

    def _register_normals_and_tangents_icosahedron(self) -> None:
        """Register normals and tangents specific to icosahedron geometry."""
        # 1. edges%primal_cart_normal (cartesian coordinates for primal_normal)
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

        dual_uv = factory.ProgramFieldProvider(
            func=math_helpers.compute_zonal_and_meridional_components_on_edges,
            deps={
                "lat": attrs.EDGE_LAT,
                "lon": attrs.EDGE_LON,
                "x": attrs.EDGE_TANGENT_X,
                "y": attrs.EDGE_TANGENT_Y,
                "z": attrs.EDGE_TANGENT_Z,
            },
            fields={
                "u": attrs.EDGE_DUAL_U,
                "v": attrs.EDGE_DUAL_V,
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
        )
        self.register_provider(dual_uv)

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

        # dual normals: the dual normals are the edge tangents
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

    def _register_normals_and_tangents_torus(self) -> None:
        """Register normals and tangents specific to torus geometry."""
        # 1. edges%primal_cart_normal (cartesian coordinates for primal_normal)
        tangent_normal_coordinates = factory.ProgramFieldProvider(
            func=stencils.compute_cartesian_coordinates_of_edge_tangent_and_normal_torus,
            deps={
                "vertex_x": attrs.VERTEX_X,
                "vertex_y": attrs.VERTEX_Y,
                "edge_x": attrs.EDGE_CENTER_X,
                "edge_y": attrs.EDGE_CENTER_Y,
                "edge_orientation": attrs.TANGENT_ORIENTATION,
            },
            fields={
                "tangent_x": attrs.EDGE_TANGENT_X,
                "tangent_y": attrs.EDGE_TANGENT_Y,
                "tangent_z": attrs.EDGE_TANGENT_Z,
                "tangent_u": attrs.EDGE_DUAL_U,
                "tangent_v": attrs.EDGE_DUAL_V,
                "normal_x": attrs.EDGE_NORMAL_X,
                "normal_y": attrs.EDGE_NORMAL_Y,
                "normal_z": attrs.EDGE_NORMAL_Z,
                "normal_u": attrs.EDGE_NORMAL_U,
                "normal_v": attrs.EDGE_NORMAL_V,
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
            params={
                "domain_length": self._grid.global_properties.domain_length,
                "domain_height": self._grid.global_properties.domain_height,
            },
        )
        self.register_provider(tangent_normal_coordinates)

        # primal_normal_vert, primal_normal_cell
        normal_vert_wrapper = SparseFieldProviderWrapper(
            tangent_normal_coordinates,
            target_dims=attrs.attrs[attrs.EDGE_NORMAL_VERTEX_U]["dims"],
            fields=(attrs.EDGE_NORMAL_VERTEX_U, attrs.EDGE_NORMAL_VERTEX_V),
            pairs=(
                (
                    attrs.EDGE_NORMAL_X,
                    attrs.EDGE_NORMAL_X,
                    attrs.EDGE_NORMAL_X,
                    attrs.EDGE_NORMAL_X,
                ),
                (
                    attrs.EDGE_NORMAL_Y,
                    attrs.EDGE_NORMAL_Y,
                    attrs.EDGE_NORMAL_Y,
                    attrs.EDGE_NORMAL_Y,
                ),
            ),
        )
        self.register_provider(normal_vert_wrapper)

        normal_cell_wrapper = SparseFieldProviderWrapper(
            tangent_normal_coordinates,
            target_dims=attrs.attrs[attrs.EDGE_NORMAL_CELL_U]["dims"],
            fields=(attrs.EDGE_NORMAL_CELL_U, attrs.EDGE_NORMAL_CELL_V),
            pairs=(
                (attrs.EDGE_NORMAL_X, attrs.EDGE_NORMAL_X),
                (attrs.EDGE_NORMAL_Y, attrs.EDGE_NORMAL_Y),
            ),
        )
        self.register_provider(normal_cell_wrapper)

        # dual normals: the dual normals are the edge tangents
        tangent_vert_wrapper = SparseFieldProviderWrapper(
            tangent_normal_coordinates,
            target_dims=attrs.attrs[attrs.EDGE_TANGENT_VERTEX_U]["dims"],
            fields=(attrs.EDGE_TANGENT_VERTEX_U, attrs.EDGE_TANGENT_VERTEX_V),
            pairs=(
                (
                    attrs.EDGE_TANGENT_X,
                    attrs.EDGE_TANGENT_X,
                    attrs.EDGE_TANGENT_X,
                    attrs.EDGE_TANGENT_X,
                ),
                (
                    attrs.EDGE_TANGENT_Y,
                    attrs.EDGE_TANGENT_Y,
                    attrs.EDGE_TANGENT_Y,
                    attrs.EDGE_TANGENT_Y,
                ),
            ),
        )
        self.register_provider(tangent_vert_wrapper)

        tangent_cell_wrapper = SparseFieldProviderWrapper(
            tangent_normal_coordinates,
            target_dims=attrs.attrs[attrs.EDGE_TANGENT_CELL_U]["dims"],
            fields=(attrs.EDGE_TANGENT_CELL_U, attrs.EDGE_TANGENT_CELL_V),
            pairs=(
                (attrs.EDGE_TANGENT_X, attrs.EDGE_TANGENT_X),
                (attrs.EDGE_TANGENT_Y, attrs.EDGE_TANGENT_Y),
            ),
        )
        self.register_provider(tangent_cell_wrapper)

    def _register_cartesian_coordinates_icosahedron(self) -> None:
        """Register Cartesian coordinate conversions for icosahedron geometry."""
        cartesian_vertices = factory.EmbeddedFieldOperatorProvider(
            func=math_helpers.geographical_to_cartesian_on_vertices.with_backend(self.backend),
            domain={
                dims.VertexDim: (
                    h_grid.vertex_domain(h_grid.Zone.LOCAL),
                    h_grid.vertex_domain(h_grid.Zone.END),
                )
            },
            fields={
                attrs.VERTEX_X: attrs.VERTEX_X,
                attrs.VERTEX_Y: attrs.VERTEX_Y,
                attrs.VERTEX_Z: attrs.VERTEX_Z,
            },
            deps={
                "lat": attrs.VERTEX_LAT,
                "lon": attrs.VERTEX_LON,
            },
        )
        self.register_provider(cartesian_vertices)
        cartesian_edge_centers = factory.EmbeddedFieldOperatorProvider(
            func=math_helpers.geographical_to_cartesian_on_edges.with_backend(self.backend),
            domain={
                dims.EdgeDim: (
                    h_grid.edge_domain(h_grid.Zone.LOCAL),
                    h_grid.edge_domain(h_grid.Zone.END),
                )
            },
            fields={
                attrs.EDGE_CENTER_X: attrs.EDGE_CENTER_X,
                attrs.EDGE_CENTER_Y: attrs.EDGE_CENTER_Y,
                attrs.EDGE_CENTER_Z: attrs.EDGE_CENTER_Z,
            },
            deps={
                "lat": attrs.EDGE_LAT,
                "lon": attrs.EDGE_LON,
            },
        )
        self.register_provider(cartesian_edge_centers)
        cartesian_cell_centers = factory.EmbeddedFieldOperatorProvider(
            func=math_helpers.geographical_to_cartesian_on_cells.with_backend(self.backend),
            domain={
                dims.CellDim: (
                    h_grid.cell_domain(h_grid.Zone.LOCAL),
                    h_grid.cell_domain(h_grid.Zone.END),
                )
            },
            fields={
                attrs.CELL_CENTER_X: attrs.CELL_CENTER_X,
                attrs.CELL_CENTER_Y: attrs.CELL_CENTER_Y,
                attrs.CELL_CENTER_Z: attrs.CELL_CENTER_Z,
            },
            deps={
                "lat": attrs.CELL_LAT,
                "lon": attrs.CELL_LON,
            },
        )
        self.register_provider(cartesian_cell_centers)

    def __repr__(self) -> str:
        geometry_name = self._geometry_type._name_ if self._geometry_type else ""
        return (
            f"{self.__class__.__name__} for geometry_type={geometry_name} (grid={self._grid.id!r})"
        )

    @property
    def metadata(self) -> dict[str, model.FieldMetaData]:
        return self._attrs

    @property
    def backend(self) -> gtx_typing.Backend:
        return self._backend

    @property
    def grid(self) -> icon.IconGrid:
        return self._grid

    @property
    def vertical_grid(self) -> None:
        return None


class SparseFieldProviderWrapper(factory.FieldProvider):
    def __init__(
        self,
        field_provider: factory.FieldProvider,
        target_dims: Sequence[gtx.Dimension],
        fields: Sequence[str],
        pairs: Sequence[tuple[str, ...]],
    ):
        assert len(target_dims) == 2
        assert target_dims[1].kind == gtx.DimensionKind.LOCAL
        self._wrapped_provider = field_provider
        self._fields = {name: None for name in fields}
        self._func = functools.partial(as_sparse_field, target_dims)
        self._pairs = pairs

    def __call__(
        self,
        field_name: str,
        field_src: factory.FieldSource | None,
        backend: gtx_typing.Backend | None,
        grid: factory.GridProvider,
    ) -> state_utils.GTXFieldType | None:
        if self._fields.get(field_name) is None:
            # get the fields from the wrapped provider
            input_fields = []
            for p in self._pairs:
                t = tuple([self._wrapped_provider(name, field_src, backend, grid) for name in p])
                input_fields.append(t)
            sparse_fields = self.func(input_fields, backend=backend)
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
    target_dims: tuple[gtx.Dimension, gtx.Dimension],
    data: Sequence[tuple[gtx.Field[gtx.Dims[gtx.Dimension], state_utils.ScalarType], ...]],
    backend: gtx_typing.Backend | None = None,
) -> Sequence[state_utils.GTXFieldType]:
    assert len(target_dims) == 2
    assert target_dims[0].kind == gtx.DimensionKind.HORIZONTAL
    assert target_dims[1].kind == gtx.DimensionKind.LOCAL
    on_gpu = device_utils.is_cupy_device(backend)
    xp = data_alloc.array_ns(on_gpu)
    fields = []
    for t in data:
        buffers = list(b.ndarray for b in t)
        field = gtx.as_field(
            target_dims, data=(xp.vstack(buffers).T), dtype=buffers[0].dtype, allocator=backend
        )
        fields.append(field)
    return fields


def create_auxiliary_coordinate_arrays_for_orientation(
    grid: icon.IconGrid,
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    allocator: gtx_typing.FieldBufferAllocationUtil | None,
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
    xp = data_alloc.array_ns(device_utils.is_cupy_device(allocator))
    e2c_table = grid.get_connectivity(dims.E2C).ndarray
    lat = cell_lat.ndarray[e2c_table]
    lon = cell_lon.ndarray[e2c_table]
    for i in (0, 1):
        boundary_edges = xp.where(e2c_table[:, i] == gridfile.GridFile.INVALID_INDEX)
        lat[boundary_edges, i] = edge_lat.ndarray[boundary_edges]
        lon[boundary_edges, i] = edge_lon.ndarray[boundary_edges]

    return (
        gtx.as_field((dims.EdgeDim,), lat[:, 0], allocator=allocator),
        gtx.as_field((dims.EdgeDim,), lon[:, 0], allocator=allocator),
        gtx.as_field((dims.EdgeDim,), lat[:, 1], allocator=allocator),
        gtx.as_field((dims.EdgeDim,), lon[:, 1], allocator=allocator),
    )
