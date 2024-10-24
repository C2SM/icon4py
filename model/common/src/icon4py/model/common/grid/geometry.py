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
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, TypeAlias, TypeVar, Union

import xarray as xa
from gt4py import next as gtx
from gt4py.next import backend as gtx_backend

import icon4py.model.common.grid.geometry_attributes as attrs
import icon4py.model.common.math.helpers as math_helpers
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
)
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import grid_manager as gm, horizontal as h_grid, icon
from icon4py.model.common.grid.geometry_program import (
    compute_cartesian_coordinates_of_edge_tangent_and_normal,
    compute_coriolis_parameter_on_edges,
    compute_dual_edge_length_and_far_vertex_distance_in_diamond,
    compute_edge_area,
    compute_edge_length,
    compute_edge_primal_normal_cell,
    compute_edge_primal_normal_vertex,
)
from icon4py.model.common.settings import xp
from icon4py.model.common.states import factory, model, utils as state_utils
from icon4py.model.common.states.factory import ProgramFieldProvider


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
        tangent_orientation=None,  # from grid file, computation still buggy
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
        primal_normal_x=None,  # from gridfile (computed in bridge code?)
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


InputGeometryFieldType: TypeAlias = Literal[attrs.CELL_AREA]


class GridGeometry(state_utils.FieldSource):
    def __init__(
        self,
        grid: icon.IconGrid,
        decomposition_info: definitions.DecompositionInfo,
        backend: gtx_backend.Backend,
        coordinates: dict[dims.Dimension, dict[Literal["lat", "lon"], gtx.Field]],
        fields: dict[InputGeometryFieldType, gtx.Field],
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
        }
        coodinate_provider = factory.PrecomputedFieldProvider(coordinates)
        self.register_provider(coodinate_provider)
        input_fields_provider = factory.PrecomputedFieldProvider(
            {
                attrs.CELL_AREA: fields[gm.GeometryName.CELL_AREA],
                attrs.TANGENT_ORIENTATION: fields[gm.GeometryName.TANGENT_ORIENTATION],
                "edge_owner_mask": gtx.as_field(
                    (dims.EdgeDim,), decomposition_info.owner_mask(dims.EdgeDim), dtype=bool
                ),
            }
        )
        self.register_provider(input_fields_provider)
        # TODO: remove if it works with the providers
        self._fields = coordinates

    def register_provider(self, provider: factory.FieldProvider):
        for dependency in provider.dependencies:
            if dependency not in self._providers.keys():
                raise ValueError(f"Dependency '{dependency}' not found in registered providers")

        for field in provider.fields:
            self._providers[field] = provider

    def __call__(self):
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
            func=compute_edge_area,
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
            func=compute_coriolis_parameter_on_edges,
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
        provider = ProgramFieldProvider(
            func=compute_cartesian_coordinates_of_edge_tangent_and_normal,
            deps={
                "cell_lat": attrs.CELL_LAT,
                "cell_lon": attrs.CELL_LON,
                "vertex_lat": attrs.VERTEX_LAT,
                "vertex_lon": attrs.VERTEX_LON,
                "edge_lat": attrs.EDGE_LAT,
                "edge_lon": attrs.EDGE_LON,
            },
            fields={
                "tangent_orientation": attrs.TANGENT_ORIENTATION,
                "tangent_x": attrs.EDGE_TANGENT_X,
                "tangent_y": attrs.EDGE_TANGENT_Y,
                "tangent_z": attrs.EDGE_TANGENT_Z,
                "normal_x": attrs.EDGE_NORMAL_X,
                "normal_y": attrs.EDGE_NORMAL_Y,
                "normal_z": attrs.EDGE_NORMAL_Z,
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
        )
        self.register_provider(provider)
        # 2. primal_normals: gridfile%zonal_normal_primal_edge - edges%primal_normal%v1, gridfile%meridional_normal_primal_edge - edges%primal_normal%v2,
        provider = ProgramFieldProvider(
            func=math_helpers.compute_zonal_and_meridional_components_on_edges,
            deps={
                "lat": attrs.EDGE_LAT,
                "lon": attrs.EDGE_LON,
                "x": attrs.EDGE_NORMAL_X,
                "y": attrs.EDGE_NORMAL_Y,
                "z": attrs.EDGE_NORMAL_Z,
            },
            fields={
                "u": attrs.EDGE_PRIMAL_NORMAL_U,
                "v": attrs.EDGE_PRIMAL_NORMAL_V,
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
        )
        self.register_provider(provider)

        # 3. primal_normal_vert, primal_normal_cell
        wrapped_provider = ProgramFieldProvider(
            func=compute_edge_primal_normal_vertex,
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
        provider = SparseFieldProviderWrapper(
            wrapped_provider,
            target_dims=attrs.attrs[attrs.EDGE_NORMAL_VERTEX_U]["dims"],
            fields=(attrs.EDGE_NORMAL_VERTEX_U, attrs.EDGE_NORMAL_VERTEX_V),
            pairs=(
                ("u_vertex_1", "u_vertex_2", "u_vertex_3", "u_vertex_4"),
                ("v_vertex_1", "v_vertex_2", "v_vertex_3", "v_vertex_4"),
            ),
        )
        self.register_provider(provider)
        wrapped_provider = ProgramFieldProvider(
            func=compute_edge_primal_normal_cell,
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
        provider = SparseFieldProviderWrapper(
            wrapped_provider,
            target_dims=attrs.attrs[attrs.EDGE_NORMAL_CELL_U]["dims"],
            fields=(attrs.EDGE_NORMAL_CELL_U, attrs.EDGE_NORMAL_CELL_V),
            pairs=(("u_cell_1", "u_cell_2"), ("v_cell_1", "v_cell_2")),
        )
        self.register_provider(provider)

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


HorizontalD = TypeVar("HorizontalD", bound=gtx.Dimension)
SparseD = TypeVar("SparseD", bound=gtx.Dimension)


class SparseFieldProviderWrapper(factory.FieldProvider):
    def __init__(
        self,
        field_provider: factory.ProgramFieldProvider,
        target_dims: tuple[HorizontalD, SparseD],
        fields: Sequence[str],
        pairs: Sequence[tuple[str, str]],
    ):
        self._wrapped_provider = field_provider
        self._fields = {name: None for name in fields}
        self._func = functools.partial(as_sparse_field, target_dims)
        self._pairs = pairs

    def __call__(
        self,
        field_name: str,
        field_src: Optional[state_utils.FieldSource],
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
        # TODO or values?
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
