# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
from collections.abc import Sequence
from typing import ClassVar, Final, Literal, TypeAlias

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import serialbox

import icon4py.model.common.decomposition.definitions as decomposition
import icon4py.model.common.field_type_aliases as fa
import icon4py.model.common.grid.states as grid_states
from icon4py.model.common import dimension as dims, model_backends, type_alias
from icon4py.model.common.grid import base, horizontal as h_grid, icon, utils as grid_utils
from icon4py.model.common.states import prognostic_state
from icon4py.model.common.utils import data_allocation as data_alloc, field_utils


log = logging.getLogger(__name__)

TimeIndex: TypeAlias = Literal[0, 1]
FourIndex: TypeAlias = Literal[0, 1, 2, 3]
TwoIndex: TypeAlias = Literal[0, 1]

#: ICON default indices for the tracers, see mo_advection_utils.f90
QV: Final[int] = 0
QC: Final[int] = 1
QI: Final[int] = 2
QR: Final[int] = 3
QS: Final[int] = 4
QG: Final[int] = 5


TracerIndex: TypeAlias = Literal[QV, QC, QI, QR, QS, QG]


class IconSavepoint:
    def __init__(
        self,
        sp: serialbox.Savepoint,
        ser: serialbox.Serializer,
        size: dict,
        backend: gtx_typing.Backend | None,
    ):
        self.savepoint = sp
        self.serializer = ser
        self.sizes = size
        self.log = logging.getLogger(__name__)
        self.backend = backend
        self.xp = data_alloc.import_array_ns(self.backend)

    def optionally_registered(*dims, dtype=type_alias.wpfloat):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                try:
                    name = func.__name__
                    return func(self, *args, **kwargs)
                except serialbox.SerialboxError:
                    log.warning(
                        f"{name}: field not registered in savepoint {self.savepoint.metainfo}"
                    )
                    if dims:
                        # We allocate a dummy field with size 1 in each dimension
                        # as a workaround for the lack of support for optional fields in gt4py.
                        shp = (1,) * len(dims)
                        return gtx.as_field(
                            dims, self.xp.zeros(shp, dtype=dtype), allocator=self.backend
                        )
                    else:
                        return None

            return wrapper

        return decorator

    def log_meta_info(self):
        self.log.info(self.savepoint.metainfo)

    def _get_field(
        self,
        name,
        *dimensions,
        dtype=float,
        slice_: int | slice | tuple[int | slice, ...] | None = None,
        transpose: None | Sequence[int] = None,
    ):
        # Note: slice is applied before transpose!
        buffer = self.xp.squeeze(self.serializer.read(name, self.savepoint).astype(dtype))
        if slice_ is not None:
            buffer = buffer[slice_]
        if transpose is not None:
            buffer = self.xp.transpose(buffer, axes=transpose)
        buffer = self._reduce_to_dim_size(buffer, dimensions)

        self.log.debug(f"{name} {buffer.shape}")
        return gtx.as_field(dimensions, buffer, allocator=self.backend)

    def _get_field_component(self, name: str, level: int, dims: tuple[gtx.Dimension, gtx]):
        buffer = self.serializer.read(name, self.savepoint).astype(float)
        buffer = self.xp.squeeze(buffer)[:, :, level]
        buffer = self._reduce_to_dim_size(buffer, dims)
        self.log.debug(f"{name} {buffer.shape}")
        return gtx.as_field(dims, buffer, allocator=self.backend)

    def _reduce_to_dim_size(self, buffer, dimensions):
        buffer_size = (
            self.sizes[d] if d.kind is gtx.DimensionKind.HORIZONTAL else s
            for s, d in zip(buffer.shape, dimensions, strict=False)
        )
        return buffer[tuple(map(slice, buffer_size))]

    def _get_field_from_ndarray(self, ar, *dimensions, dtype=float):
        ar = self._reduce_to_dim_size(ar, dimensions)
        return gtx.as_field(dimensions, ar, allocator=self.backend, dtype=dtype)

    def get_metadata(self, *names):
        metadata = self.savepoint.metainfo.to_dict()
        return {n: metadata[n] for n in names if n in metadata}

    def _read_int32_shift1(self, name: str):
        """
        Read a start indices field.

        use for start indices: the shift accounts for the zero based python
        values are converted to gtx.int32
        """
        return self._read_int32(name, offset=1)

    def _read_int32(self, name: str, offset=0):
        """
        Read an end indices field.

        use this for end indices: because FORTRAN slices  are inclusive [from:to] _and_ one based
        this accounts for being exclusive python exclusive bounds: [from:to)
        field values are convert to gtx.int32
        """
        return self._read(name, offset, dtype=gtx.int32)

    def _read_bool(self, name: str):
        return self._read(name, offset=0, dtype=bool)

    def _read(self, name: str, offset=0, dtype=int):
        return self.xp.asarray(
            self.xp.squeeze(self.serializer.read(name, self.savepoint) - offset).astype(dtype)
        )


class IconGridSavepoint(IconSavepoint):
    def __init__(
        self,
        *,
        sp: serialbox.Savepoint,
        ser: serialbox.Serializer,
        grid_id: str,
        size: dict,
        grid_params: icon.GridParams,
        backend: gtx_typing.Backend | None,
    ):
        super().__init__(sp, ser, size, backend)
        self._grid_id = grid_id
        self.grid_params = grid_params

    def verts_vertex_lat(self):
        """vertex latituted"""
        return self._get_field("verts_vertex_lat", dims.VertexDim)

    def verts_vertex_lon(self):
        """vertex longitude"""
        return self._get_field("verts_vertex_lon", dims.VertexDim)

    def verts_vertex_cart_x(self):
        """vertex cartesian x coordinate"""
        return self._get_field("verts_vertex_cart_x", dims.VertexDim)

    def verts_vertex_cart_y(self):
        """vertex cartesian y coordinate"""
        return self._get_field("verts_vertex_cart_y", dims.VertexDim)

    def verts_vertex_cart_z(self):
        """vertex cartesian z coordinate"""
        return self._get_field("verts_vertex_cart_z", dims.VertexDim)

    def primal_normal_v1(self):
        return self._get_field("primal_normal_v1", dims.EdgeDim)

    def primal_normal_v2(self):
        return self._get_field("primal_normal_v2", dims.EdgeDim)

    def dual_normal_v1(self):
        return self._get_field("dual_normal_v1", dims.EdgeDim)

    def dual_normal_v2(self):
        return self._get_field("dual_normal_v2", dims.EdgeDim)

    def edges_center_lat(self):
        """edge center latitude"""
        return self._get_field("edges_center_lat", dims.EdgeDim)

    def edges_center_lon(self):
        """edge center longitude"""
        return self._get_field("edges_center_lon", dims.EdgeDim)

    def edges_center_cart_x(self):
        """edge center cartesian x coordinate"""
        return self._get_field("edges_center_cart_x", dims.EdgeDim)

    def edges_center_cart_y(self):
        """edge center cartesian y coordinate"""
        return self._get_field("edges_center_cart_y", dims.EdgeDim)

    def edges_center_cart_z(self):
        """edge center cartesian z coordinate"""
        return self._get_field("edges_center_cart_z", dims.EdgeDim)

    def edge_vert_length(self):
        """length of edge midpoint to vertex"""
        return self._get_field("edge_vert_length", dims.EdgeDim, dims.E2C2VDim)

    def vct_a(self):
        return self._get_field("vct_a", dims.KDim)

    def vct_b(self):
        return self._get_field("vct_b", dims.KDim)

    def tangent_orientation(self):
        return self._get_field("tangent_orientation", dims.EdgeDim)

    def edge_orientation(self):
        return self._get_field("cells_edge_orientation", dims.CellDim, dims.C2EDim)

    def vertex_edge_orientation(self):
        return self._get_field("v_edge_orientation", dims.VertexDim, dims.V2EDim)

    def vertex_dual_area(self):
        return self._get_field("v_dual_area", dims.VertexDim)

    def inverse_primal_edge_lengths(self):
        return self._get_field("inv_primal_edge_length", dims.EdgeDim)

    def primal_edge_length(self):
        return self._get_field("primal_edge_length", dims.EdgeDim)

    def primal_cart_normal_x(self):
        return self._get_field("primal_cart_normal_x", dims.EdgeDim)

    def primal_cart_normal_y(self):
        return self._get_field("primal_cart_normal_y", dims.EdgeDim)

    def primal_cart_normal_z(self):
        return self._get_field("primal_cart_normal_z", dims.EdgeDim)

    def dual_cart_normal_x(self):
        return self._get_field("dual_cart_normal_x", dims.EdgeDim)

    def dual_cart_normal_y(self):
        return self._get_field("dual_cart_normal_y", dims.EdgeDim)

    def dual_cart_normal_z(self):
        return self._get_field("dual_cart_normal_z", dims.EdgeDim)

    def inv_vert_vert_length(self):
        return self._get_field("inv_vert_vert_length", dims.EdgeDim)

    def primal_normal_vert_x(self):
        return self._get_field("primal_normal_vert_x", dims.EdgeDim, dims.E2C2VDim)

    def primal_normal_vert_y(self):
        return self._get_field("primal_normal_vert_y", dims.EdgeDim, dims.E2C2VDim)

    def dual_normal_vert_y(self):
        return self._get_field("dual_normal_vert_y", dims.EdgeDim, dims.E2C2VDim)

    def dual_normal_vert_x(self):
        return self._get_field("dual_normal_vert_x", dims.EdgeDim, dims.E2C2VDim)

    def primal_normal_cell_x(self):
        return self._get_field("primal_normal_cell_x", dims.EdgeDim, dims.E2CDim)

    def primal_normal_cell_y(self):
        return self._get_field("primal_normal_cell_y", dims.EdgeDim, dims.E2CDim)

    def dual_normal_cell_x(self):
        return self._get_field("dual_normal_cell_x", dims.EdgeDim, dims.E2CDim)

    def dual_normal_cell_y(self):
        return self._get_field("dual_normal_cell_y", dims.EdgeDim, dims.E2CDim)

    def cell_areas(self):
        return self._get_field("cell_areas", dims.CellDim)

    def lat(self, dim: gtx.Dimension):
        match dim:
            case dims.CellDim:
                return self.cell_center_lat()
            case dims.EdgeDim:
                return self.edges_center_lat()
            case dims.VertexDim:
                return self.verts_vertex_lat()
            case _:
                raise ValueError

    def lon(self, dim: gtx.Dimension):
        match dim:
            case dims.CellDim:
                return self.cell_center_lon()
            case dims.EdgeDim:
                return self.edges_center_lon()
            case dims.VertexDim:
                return self.verts_vertex_lon()
            case _:
                raise ValueError

    def coordinates(self):
        coords = {
            dims.CellDim: {"lat": self.cell_center_lat(), "lon": self.cell_center_lon()},
            dims.EdgeDim: {"lat": self.edges_center_lat(), "lon": self.edges_center_lon()},
            dims.VertexDim: {"lat": self.verts_vertex_lat(), "lon": self.verts_vertex_lon()},
        }

        if self.grid_params.geometry_type == icon.GeometryType.TORUS:
            coords[dims.CellDim]["x"] = self.cell_center_cart_x()
            coords[dims.CellDim]["y"] = self.cell_center_cart_y()
            coords[dims.CellDim]["z"] = self.cell_center_cart_z()
            coords[dims.EdgeDim]["x"] = self.edges_center_cart_x()
            coords[dims.EdgeDim]["y"] = self.edges_center_cart_y()
            coords[dims.EdgeDim]["z"] = self.edges_center_cart_z()
            coords[dims.VertexDim]["x"] = self.verts_vertex_cart_x()
            coords[dims.VertexDim]["y"] = self.verts_vertex_cart_y()
            coords[dims.VertexDim]["z"] = self.verts_vertex_cart_z()

        return coords

    def cell_center_lat(self):
        return self._get_field("cell_center_lat", dims.CellDim)

    def cell_center_lon(self):
        return self._get_field("cell_center_lon", dims.CellDim)

    def cell_center_cart_x(self):
        """cell center cartesian x coordinate"""
        return self._get_field("cell_center_cart_x", dims.CellDim)

    def cell_center_cart_y(self):
        """cell center cartesian y coordinate"""
        return self._get_field("cell_center_cart_y", dims.CellDim)

    def cell_center_cart_z(self):
        """cell center cartesian z coordinate"""
        return self._get_field("cell_center_cart_z", dims.CellDim)

    def edge_center_lat(self):
        return self._get_field("edges_center_lat", dims.EdgeDim)

    def edge_center_lon(self):
        return self._get_field("edges_center_lon", dims.EdgeDim)

    def mean_cell_area(self):
        return self.serializer.read("mean_cell_area", self.savepoint).astype(float)[0]

    def edge_areas(self):
        return self._get_field("edge_areas", dims.EdgeDim)

    def inv_dual_edge_length(self):
        return self._get_field("inv_dual_edge_length", dims.EdgeDim)

    def dual_edge_length(self):
        return self._get_field("dual_edge_length", dims.EdgeDim)

    def edge_cell_length(self):
        """length of edge midpoint to cell center"""
        return self._get_field("edge_cell_length", dims.EdgeDim, dims.E2CDim)

    def cells_start_index(self):
        start_idx = self._read_int32("c_start_index")
        return self.xp.where(start_idx == 0, start_idx, start_idx - 1)

    def cells_end_index(self):
        return self._read_int32("c_end_index")

    def vertex_start_index(self):
        start_idx = self._read_int32("v_start_index")
        return self.xp.where(start_idx == 0, start_idx, start_idx - 1)

    def vertex_end_index(self):
        return self._read_int32("v_end_index")

    def edge_start_index(self):
        start_idx = self._read_int32("e_start_index")
        return self.xp.where(start_idx == 0, start_idx, start_idx - 1)

    def edge_end_index(self):
        # don't need to subtract 1, because FORTRAN slices  are inclusive [from:to] so the being
        # one off accounts for being exclusive [from:to)
        return self._read_int32("e_end_index")

    def start_index(self) -> dict[gtx.Dimension, data_alloc.NDArray]:
        return {
            dims.CellDim: self.cells_start_index(),
            dims.EdgeDim: self.edge_start_index(),
            dims.VertexDim: self.vertex_start_index(),
        }

    def end_index(self) -> dict[gtx.Dimension, data_alloc.NDArray]:
        return {
            dims.CellDim: self.cells_end_index(),
            dims.EdgeDim: self.edge_end_index(),
            dims.VertexDim: self.vertex_end_index(),
        }

    def nflatlev(self):
        return self._read_int32_shift1("nflatlev").item()

    def nflat_gradp(self):
        return self._read_int32_shift1("nflat_gradp").item()

    def v_owner_mask(self):
        return self._get_field("v_owner_mask", dims.VertexDim, dtype=bool)

    def c_owner_mask(self):
        return self._get_field("c_owner_mask", dims.CellDim, dtype=bool)

    def e_owner_mask(self):
        return self._get_field("e_owner_mask", dims.EdgeDim, dtype=bool)

    def f_e(self):
        return self._get_field("f_e", dims.EdgeDim)

    def print_connectivity_info(self, name: str, ar: data_alloc.NDArray):
        self.log.debug(f" connectivity {name} {ar.shape}")

    def c2e(self):
        return self._get_connectivity_array("c2e", dims.CellDim)

    def _get_connectivity_array(self, name: str, target_dim: gtx.Dimension, reverse: bool = False):
        if reverse:
            connectivity = self.xp.transpose(self._read_int32(name, offset=1))[
                : self.sizes[target_dim], :
            ]
        else:
            connectivity = self._read_int32(name, offset=1)[: self.sizes[target_dim], :]
        self.log.debug(f" connectivity {name} : {connectivity.shape}")
        return connectivity

    def c2e2c(self):
        return self._get_connectivity_array("c2e2c", dims.CellDim)

    def e2c2e(self):
        return self._get_connectivity_array("e2c2e", dims.EdgeDim)

    def c2e2c2e(self):
        if self._c2e2c2e() is None:
            return self.xp.zeros((self.sizes[dims.CellDim], 9), dtype=gtx.int32)
        else:
            return self._c2e2c2e()

    @IconSavepoint.optionally_registered()
    def _c2e2c2e(self):
        return self._get_connectivity_array("c2e2c2e", dims.CellDim, reverse=True)

    def e2c(self):
        return self._get_connectivity_array("e2c", dims.EdgeDim)

    def e2v(self):
        # array "e2v" is actually e2c2v
        v_ = self._get_connectivity_array("e2v", dims.EdgeDim)[:, 0:2]
        self.log.debug(f"real e2v {v_.shape}")
        return v_

    def e2c2v(self):
        # array "e2v" is actually e2c2v, that is hexagon or pentagon
        return self._get_connectivity_array("e2v", dims.EdgeDim)

    def v2e(self):
        return self._get_connectivity_array("v2e", dims.VertexDim)

    def v2c(self):
        return self._get_connectivity_array("v2c", dims.VertexDim)

    def c2v(self):
        return self._get_connectivity_array("c2v", dims.CellDim)

    def nrdmax(self):
        return gtx.int32(self._read_int32_shift1("nrdmax").item())

    def refin_ctrl(self, dim: gtx.Dimension):
        field_name = "refin_ctl"
        return gtx.as_field(
            (dim,),
            self._read_field_for_dim(field_name, self._read_int32, dim)[: self.num(dim)],
            allocator=self.backend,
        )

    def num(self, dim: gtx.Dimension):
        return self.sizes[dim]

    @staticmethod
    def _read_field_for_dim(field_name, read_func, dim: gtx.Dimension):
        match dim:
            case dims.CellDim:
                return read_func(f"c_{field_name}")
            case dims.EdgeDim:
                return read_func(f"e_{field_name}")
            case dims.VertexDim:
                return read_func(f"v_{field_name}")
            case _:
                raise NotImplementedError(
                    f"only {dims.CellDim, dims.EdgeDim, dims.VertexDim} are handled"
                )

    def owner_mask(self, dim: gtx.Dimension):
        return self.xp.squeeze(self._read_field_for_dim("owner_mask", self._read_bool, dim))

    def global_index(self, dim: gtx.Dimension):
        return self._read_field_for_dim("glb_index", self._read_int32_shift1, dim)

    def decomp_domain(self, dim):
        return self._read_field_for_dim("decomp_domain", self._read_int32, dim)

    def construct_decomposition_info(self) -> decomposition.DecompositionInfo:
        return (
            decomposition.DecompositionInfo()
            .set_dimension(*self._get_decomposition_fields(dims.CellDim))
            .set_dimension(*self._get_decomposition_fields(dims.EdgeDim))
            .set_dimension(*self._get_decomposition_fields(dims.VertexDim))
        )

    def _get_decomposition_fields(self, dim: gtx.Dimension):
        global_index = self.global_index(dim)
        mask = self.owner_mask(dim)[0 : self.num(dim)]
        halo_levels = self.decomp_domain(dim)[0 : self.num(dim)]
        return dim, global_index, mask, halo_levels

    def construct_icon_grid(
        self,
        backend: gtx_typing.Backend | None = None,
        keep_skip_values: bool = True,
        with_repeated_index: bool = True,
    ) -> icon.IconGrid:
        config = base.GridConfig(
            horizontal_config=base.HorizontalGridSize(
                num_vertices=self.num(dims.VertexDim),
                num_cells=self.num(dims.CellDim),
                num_edges=self.num(dims.EdgeDim),
            ),
            vertical_size=self.num(dims.KDim),
            limited_area=self.get_metadata("limited_area").get("limited_area"),
            distributed=self.construct_decomposition_info().is_distributed(),
            keep_skip_values=keep_skip_values,
        )

        if with_repeated_index:

            def potentially_revert_icon_index_transformation(ar):
                return ar
        else:
            potentially_revert_icon_index_transformation = (
                grid_utils.revert_repeated_index_to_invalid
            )

        c2e2c = self.c2e2c()
        e2c2e = potentially_revert_icon_index_transformation(self.e2c2e())
        c2e2c0 = self.xp.column_stack((self.xp.asarray(range(c2e2c.shape[0])), c2e2c))
        e2c2e0 = self.xp.column_stack((self.xp.asarray(range(e2c2e.shape[0])), e2c2e))

        constructor = functools.partial(
            h_grid.get_start_end_idx_from_icon_arrays,
            start_indices=self.start_index(),
            end_indices=self.end_index(),
        )
        c2e2c2e = potentially_revert_icon_index_transformation(self.c2e2c2e())
        v2e = potentially_revert_icon_index_transformation(self.v2e())

        start_index, end_index = icon.get_start_and_end_index(constructor)
        neighbor_tables = {
            dims.C2E: self.c2e(),
            dims.E2C: self.e2c(),
            dims.C2E2C: c2e2c,
            dims.C2E2CO: c2e2c0,
            dims.C2E2C2E: c2e2c2e,
            dims.E2C2E: e2c2e,
            dims.E2C2EO: e2c2e0,
            dims.E2V: self.e2v(),
            dims.V2E: v2e,
            dims.V2C: self.v2c(),
            dims.E2C2V: self.e2c2v(),
            dims.C2V: self.c2v(),
        }

        return icon.icon_grid(
            id_=self._grid_id,
            allocator=backend,
            config=config,
            neighbor_tables=neighbor_tables,
            grid_params=self.grid_params,
            start_index=start_index,
            end_index=end_index,
            refinement_control={
                dims.CellDim: self.refin_ctrl(dims.CellDim),
                dims.EdgeDim: self.refin_ctrl(dims.EdgeDim),
                dims.VertexDim: self.refin_ctrl(dims.VertexDim),
            },
        )

    def construct_edge_geometry(self) -> grid_states.EdgeParams:
        return grid_states.EdgeParams(
            tangent_orientation=self.tangent_orientation(),
            inverse_primal_edge_lengths=self.inverse_primal_edge_lengths(),
            inverse_dual_edge_lengths=self.inv_dual_edge_length(),
            inverse_vertex_vertex_lengths=self.inv_vert_vert_length(),
            primal_normal_vert_x=self.primal_normal_vert_x(),
            primal_normal_vert_y=self.primal_normal_vert_y(),
            dual_normal_vert_x=self.dual_normal_vert_x(),
            dual_normal_vert_y=self.dual_normal_vert_y(),
            primal_normal_cell_x=self.primal_normal_cell_x(),
            dual_normal_cell_x=self.dual_normal_cell_x(),
            primal_normal_cell_y=self.primal_normal_cell_y(),
            dual_normal_cell_y=self.dual_normal_cell_y(),
            edge_areas=self.edge_areas(),
            coriolis_frequency=self.f_e(),
            edge_center_lat=self.edge_center_lat(),
            edge_center_lon=self.edge_center_lon(),
            primal_normal_x=self.primal_normal_v1(),
            primal_normal_y=self.primal_normal_v2(),
        )

    def construct_cell_geometry(self) -> grid_states.CellParams:
        return grid_states.CellParams(
            cell_center_lat=self.cell_center_lat(),
            cell_center_lon=self.cell_center_lon(),
            area=self.cell_areas(),
            mean_cell_area=self.mean_cell_area(),
        )


class InterpolationSavepoint(IconSavepoint):
    def c_bln_avg(self):
        return self._get_field("c_bln_avg", dims.CellDim, dims.C2E2CODim)

    def c_intp(self):
        return self._get_field("c_intp", dims.VertexDim, dims.V2CDim)

    def c_lin_e(self):
        return self._get_field("c_lin_e", dims.EdgeDim, dims.E2CDim)

    def e_bln_c_s(self):
        return self._get_field("e_bln_c_s", dims.CellDim, dims.C2EDim)

    def e_flx_avg(self):
        return self._get_field("e_flx_avg", dims.EdgeDim, dims.E2C2EODim)

    def geofac_div(self):
        return self._get_field("geofac_div", dims.CellDim, dims.C2EDim)

    def geofac_grdiv(self):
        return self._get_field("geofac_grdiv", dims.EdgeDim, dims.E2C2EODim)

    def geofac_grg(self):
        grg = self.xp.squeeze(self.serializer.read("geofac_grg", self.savepoint))
        num_cells = self.sizes[dims.CellDim]
        return gtx.as_field(
            (dims.CellDim, dims.C2E2CODim), grg[:num_cells, :, 0], allocator=self.backend
        ), gtx.as_field(
            (dims.CellDim, dims.C2E2CODim), grg[:num_cells, :, 1], allocator=self.backend
        )

    def geofac_n2s(self):
        return self._get_field("geofac_n2s", dims.CellDim, dims.C2E2CODim)

    def geofac_rot(self):
        return self._get_field("geofac_rot", dims.VertexDim, dims.V2EDim)

    def nudgecoeff_e(self):
        return self._get_field("nudgecoeff_e", dims.EdgeDim)

    def pos_on_tplane_e_x(self):
        return self._get_field(
            "pos_on_tplane_e_x", dims.EdgeDim, dims.E2CDim, slice_=(slice(None), slice(0, 2))
        )

    def pos_on_tplane_e_y(self):
        return self._get_field(
            "pos_on_tplane_e_y", dims.EdgeDim, dims.E2CDim, slice_=(slice(None), slice(0, 2))
        )

    def rbf_vec_coeff_e(self):
        return self._get_field("rbf_vec_coeff_e", dims.EdgeDim, dims.E2C2EDim, transpose=(1, 0))

    @IconSavepoint.optionally_registered()
    def rbf_vec_coeff_c1(self):
        return self._get_field("rbf_vec_coeff_c1", dims.CellDim, dims.C2E2C2EDim, transpose=(1, 0))

    @IconSavepoint.optionally_registered()
    def rbf_vec_coeff_c2(self):
        return self._get_field("rbf_vec_coeff_c2", dims.CellDim, dims.C2E2C2EDim, transpose=(1, 0))

    def rbf_vec_coeff_v1(self):
        return self._get_field(
            "rbf_vec_coeff_v",
            dims.VertexDim,
            dims.V2EDim,
            slice_=(slice(None), 0, slice(None)),
            transpose=(1, 0),
        )

    def rbf_vec_coeff_v2(self):
        return self._get_field(
            "rbf_vec_coeff_v",
            dims.VertexDim,
            dims.V2EDim,
            slice_=(slice(None), 1, slice(None)),
            transpose=(1, 0),
        )

    def rbf_vec_idx_v(self):
        return self._get_field("rbf_vec_idx_v", dims.VertexDim, dims.V2EDim)

    def lsq_pseudoinv_1(self):
        return self._get_field("lsq_pseudoinv_1", dims.CellDim, dims.C2E2CDim)

    def lsq_pseudoinv_2(self):
        return self._get_field("lsq_pseudoinv_2", dims.CellDim, dims.C2E2CDim)


class MetricSavepoint(IconSavepoint):
    def d2dexdz2_fac1_mc(self):
        return self._get_field("d2dexdz2_fac1_mc", dims.CellDim, dims.KDim)

    def d2dexdz2_fac2_mc(self):
        return self._get_field("d2dexdz2_fac2_mc", dims.CellDim, dims.KDim)

    def d_exner_dz_ref_ic(self):
        return self._get_field("d_exner_dz_ref_ic", dims.CellDim, dims.KDim)

    def exner_exfac(self):
        return self._get_field("exner_exfac", dims.CellDim, dims.KDim)

    def exner_ref_mc(self):
        return self._get_field("exner_ref_mc", dims.CellDim, dims.KDim)

    def hmask_dd3d(self):
        return self._get_field("hmask_dd3d", dims.EdgeDim)

    def inv_ddqz_z_full(self):
        return self._get_field("inv_ddqz_z_full", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def ddqz_z_full(self):
        return self._get_field("ddqz_z_full", dims.CellDim, dims.KDim)

    def mask_prog_halo_c(self):
        return self._get_field("mask_prog_halo_c", dims.CellDim, dtype=bool)

    @IconSavepoint.optionally_registered()
    def pg_edgeidx(self):
        return self.xp.squeeze(self.serializer.read("pg_edgeidx", self.savepoint))

    @IconSavepoint.optionally_registered()
    def pg_vertidx(self):
        return self.xp.squeeze(self.serializer.read("pg_vertidx", self.savepoint))

    @IconSavepoint.optionally_registered()
    def pg_exdist(self):
        return self.xp.squeeze(self.serializer.read("pg_exdist", self.savepoint))

    def pg_exdist_dsl(self):
        pg_edgeidx = self.pg_edgeidx()
        pg_vertidx = self.pg_vertidx()
        pg_exdist = self.pg_exdist()
        domain = self.rho_ref_me().domain
        default_value = gtx.float64(0.0)
        if (pg_edgeidx is None) or (pg_vertidx is None) or (pg_exdist is None):
            # if any of the fields is missing, return a zero field with the correct shape
            return gtx.as_field(
                domain,
                self.xp.full(domain.shape, fill_value=default_value, dtype=gtx.float64),
                allocator=model_backends.get_allocator(self.backend),
            )
        else:
            return data_alloc.list2field(
                domain=domain,
                values=pg_exdist,
                indices=(
                    data_alloc.adjust_fortran_indices(pg_edgeidx),
                    data_alloc.adjust_fortran_indices(pg_vertidx),
                ),
                default_value=default_value,
                allocator=model_backends.get_allocator(self.backend),
            )

    def rayleigh_w(self):
        return self._get_field("rayleigh_w", dims.KDim)

    def rho_ref_mc(self):
        return self._get_field("rho_ref_mc", dims.CellDim, dims.KDim)

    def rho_ref_me(self):
        return self._get_field("rho_ref_me", dims.EdgeDim, dims.KDim)

    def scalfac_dd3d(self):
        return self._get_field("scalfac_dd3d", dims.KDim)

    def theta_ref_ic(self):
        return self._get_field("theta_ref_ic", dims.CellDim, dims.KDim)

    def z_ifc(self):
        return self._get_field("z_ifc", dims.CellDim, dims.KDim)

    def z_mc(self):
        return self._get_field("z_mc", dims.CellDim, dims.KDim)

    def theta_ref_me(self):
        return self._get_field("theta_ref_me", dims.EdgeDim, dims.KDim)

    def vwind_expl_wgt(self):
        return self._get_field("vwind_expl_wgt", dims.CellDim)

    def vwind_impl_wgt(self):
        return self._get_field("vwind_impl_wgt", dims.CellDim)

    def wgtfacq_c(self):
        # The Fortran array stores the surface levels in reversed order.
        wgtfacq_c_fortran = self._get_field("wgtfacq_c", dims.CellDim, dims.KDim)
        assert len(wgtfacq_c_fortran.domain[dims.KDim].unit_range) == 3
        nlev = self.sizes[dims.KDim]
        return field_utils.flip(
            wgtfacq_c_fortran(dims.KDim - (nlev - 3)),  # GT4Py embedded shift
            dims.KDim,
            allocator=model_backends.get_allocator(self.backend),
        )

    def zdiff_gradp(self):
        return self._get_field("zdiff_gradp", dims.EdgeDim, dims.E2CDim, dims.KDim)

    def vertoffset_gradp(self):
        # In Fortran `vertidx_gradp` contains `0`s in areas where the array is not used.
        # When we translate to offsets we just subtract the current index, therefore these values will be negative.
        # Since in Fortran accessing index `0` would be out-of-bounds, we should be safe.
        vertidx_gradp = data_alloc.adjust_fortran_indices(
            self._get_field("vertidx_gradp", dims.EdgeDim, dims.E2CDim, dims.KDim, dtype=gtx.int32)
        )
        return field_utils.index2offset(vertidx_gradp, dims.KDim, self.backend)

    def coeff1_dwdz(self):
        return self._get_field("coeff1_dwdz", dims.CellDim, dims.KDim)

    def coeff2_dwdz(self):
        return self._get_field("coeff2_dwdz", dims.CellDim, dims.KDim)

    def coeff_gradekin(self):
        return self._get_field("coeff_gradekin", dims.EdgeDim, dims.E2CDim)

    def ddqz_z_full_e(self):
        return self._get_field("ddqz_z_full_e", dims.EdgeDim, dims.KDim)

    def ddqz_z_half(self):
        return self._get_field("ddqz_z_half", dims.CellDim, dims.KDim)

    def ddxn_z_full(self):
        return self._get_field("ddxn_z_full", dims.EdgeDim, dims.KDim)

    def ddxt_z_full(self):
        return self._get_field("ddxt_z_full", dims.EdgeDim, dims.KDim)

    def theta_ref_mc(self):
        return self._get_field("theta_ref_mc", dims.CellDim, dims.KDim)

    def wgtfac_c(self):
        return self._get_field("wgtfac_c", dims.CellDim, dims.KDim)

    def wgtfac_e(self):
        return self._get_field("wgtfac_e", dims.EdgeDim, dims.KDim)

    def wgtfacq_e(self):
        # The Fortran array stores the surface levels in reversed order.
        wgtfacq_e_fortran = self._get_field("wgtfacq_e", dims.EdgeDim, dims.KDim)
        assert len(wgtfacq_e_fortran.domain[dims.KDim].unit_range) == 3
        nlev = self.sizes[dims.KDim]
        return field_utils.flip(
            wgtfacq_e_fortran(dims.KDim - (nlev - 3)),  # GT4Py embedded shift
            dims.KDim,
            allocator=model_backends.get_allocator(self.backend),
        )

    def geopot(self):
        return self._get_field("geopot", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def zd_cellidx(self):
        return self.xp.squeeze(self.serializer.read("zd_cellidx", self.savepoint))

    @IconSavepoint.optionally_registered()
    def zd_vertidx(self):
        # this is the k list (with fortran 1-based indexing) for the central point of the C2E2C stencil
        return self.xp.squeeze(self.serializer.read("zd_vertidx", self.savepoint))[0, :]

    @IconSavepoint.optionally_registered(dims.CellDim, dims.C2E2CDim, dims.KDim, dtype=gtx.int32)
    def zd_vertoffset(self):
        zd_cellidx = self.zd_cellidx()
        zd_vertidx = self.zd_vertidx()
        # these are the three k offsets for the C2E2C neighbors
        zd_vertoffset = (
            self.xp.squeeze(self.serializer.read("zd_vertidx", self.savepoint))[1:, :] - zd_vertidx
        )
        cell_c2e2c_k_domain = gtx.domain(
            {
                dims.CellDim: self.theta_ref_mc().domain[dims.CellDim].unit_range,
                dims.C2E2CDim: 3,
                dims.KDim: self.theta_ref_mc().domain[dims.KDim].unit_range,
            }
        )
        return data_alloc.list2field(
            domain=cell_c2e2c_k_domain,
            values=zd_vertoffset.T,
            indices=(
                data_alloc.adjust_fortran_indices(zd_cellidx),
                slice(None),
                data_alloc.adjust_fortran_indices(zd_vertidx),
            ),
            default_value=gtx.int32(0),
            allocator=model_backends.get_allocator(self.backend),
        )

    @IconSavepoint.optionally_registered(dims.CellDim, dims.C2E2CDim, dims.KDim)
    def zd_intcoef(self):
        zd_cellidx = self.zd_cellidx()
        zd_vertidx = self.zd_vertidx()
        zd_intcoef = self.xp.squeeze(self.serializer.read("zd_intcoef", self.savepoint))
        cell_c2e2c_k_domain = gtx.domain(
            {
                dims.CellDim: self.theta_ref_mc().domain[dims.CellDim].unit_range,
                dims.C2E2CDim: 3,
                dims.KDim: self.theta_ref_mc().domain[dims.KDim].unit_range,
            }
        )
        return data_alloc.list2field(
            domain=cell_c2e2c_k_domain,
            values=zd_intcoef.T,
            indices=(
                data_alloc.adjust_fortran_indices(zd_cellidx),
                slice(None),
                data_alloc.adjust_fortran_indices(zd_vertidx),
            ),
            default_value=gtx.float64(0.0),
            allocator=model_backends.get_allocator(self.backend),
        )

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def zd_diffcoef(self):
        zd_cellidx = self.zd_cellidx()
        zd_vertidx = self.zd_vertidx()
        zd_diffcoef = self.xp.squeeze(self.serializer.read("zd_diffcoef", self.savepoint))
        return data_alloc.list2field(
            domain=self.geopot().domain,
            values=zd_diffcoef,
            indices=(
                data_alloc.adjust_fortran_indices(zd_cellidx),
                data_alloc.adjust_fortran_indices(zd_vertidx),
            ),
            default_value=gtx.float64(0.0),
            allocator=model_backends.get_allocator(self.backend),
        )


class AdvectionInitSavepoint(IconSavepoint):
    def airmass_now(self):
        return self._get_field("airmass_now", dims.CellDim, dims.KDim)

    def airmass_new(self):
        return self._get_field("airmass_new", dims.CellDim, dims.KDim)

    def vn_traj(self):
        return self._get_field("vn_traj", dims.EdgeDim, dims.KDim)

    def mass_flx_me(self):
        return self._get_field("mass_flx_me", dims.EdgeDim, dims.KDim)

    def mass_flx_ic(self):
        return self._get_field("mass_flx_ic", dims.CellDim, dims.KDim)

    def grf_tend_tracer(self, ntracer: int):
        return self._get_field_component("grf_tend_tracers", ntracer, (dims.CellDim, dims.KDim))

    def tracer(self, ntracer: int):
        return self._get_field_component("tracers_now", ntracer, (dims.CellDim, dims.KDim))


class AdvectionExitSavepoint(IconSavepoint):
    def hfl_tracer(self, ntracer: int):
        return self._get_field_component("hfl_tracers", ntracer, (dims.EdgeDim, dims.KDim))

    def vfl_tracer(self, ntracer: int):
        return self._get_field_component("vfl_tracers", ntracer, (dims.CellDim, dims.KDim))

    def tracer(self, ntracer: int):
        return self._get_field_component("tracers", ntracer, (dims.CellDim, dims.KDim))


class IconDiffusionInitSavepoint(IconSavepoint):
    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def hdef_ic(self):
        return self._get_field("hdef_ic", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def div_ic(self):
        return self._get_field("div_ic", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def dwdx(self):
        return self._get_field("dwdx", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def dwdy(self):
        return self._get_field("dwdy", dims.CellDim, dims.KDim)

    def vn(self):
        return self._get_field("vn", dims.EdgeDim, dims.KDim)

    def theta_v(self):
        return self._get_field("theta_v", dims.CellDim, dims.KDim)

    def w(self):
        return self._get_field("w", dims.CellDim, dims.KDim)

    def exner(self):
        return self._get_field("exner", dims.CellDim, dims.KDim)

    def diff_multfac_smag(self):
        return self.xp.squeeze(self.serializer.read("diff_multfac_smag", self.savepoint))

    def enh_smag_fac(self):
        return self.xp.squeeze(self.serializer.read("enh_smag_fac", self.savepoint))

    def smag_limit(self):
        return self.xp.squeeze(self.serializer.read("smag_limit", self.savepoint))

    def diff_multfac_n2w(self):
        return self.xp.squeeze(self.serializer.read("diff_multfac_n2w", self.savepoint))

    def nudgezone_diff(self) -> int:
        return self.serializer.read("nudgezone_diff", self.savepoint)[0]

    def bdy_diff(self) -> int:
        return self.serializer.read("bdy_diff", self.savepoint)[0]

    def fac_bdydiff_v(self) -> int:
        return self.serializer.read("fac_bdydiff_v", self.savepoint)[0]

    def smag_offset(self):
        return self.serializer.read("smag_offset", self.savepoint)[0]

    def diff_multfac_w(self):
        return self.serializer.read("diff_multfac_w", self.savepoint)[0]

    def diff_multfac_vn(self):
        return self.serializer.read("diff_multfac_vn", self.savepoint)

    def rho(self):
        return self._get_field("rho", dims.CellDim, dims.KDim)

    def construct_prognostics(self) -> prognostic_state.PrognosticState:
        return prognostic_state.PrognosticState(
            w=self.w(),
            vn=self.vn(),
            exner=self.exner(),
            theta_v=self.theta_v(),
            rho=self.rho(),
        )


class IconDiffusionExitSavepoint(IconSavepoint):
    def vn(self):
        return self._get_field("vn", dims.EdgeDim, dims.KDim)

    def theta_v(self):
        return self._get_field("theta_v", dims.CellDim, dims.KDim)

    def w(self):
        return self._get_field("w", dims.CellDim, dims.KDim)

    def dwdx(self):
        return self._get_field("dwdx", dims.CellDim, dims.KDim)

    def dwdy(self):
        return self._get_field("dwdy", dims.CellDim, dims.KDim)

    def exner(self):
        return self._get_field("exner", dims.CellDim, dims.KDim)

    def div_ic(self):
        return self._get_field("div_ic", dims.CellDim, dims.KDim)

    def hdef_ic(self):
        return self._get_field("hdef_ic", dims.CellDim, dims.KDim)


class IconNonHydroInitSavepoint(IconSavepoint):
    def z_vt_ie(self):
        return self._get_field("z_vt_ie", dims.EdgeDim, dims.KDim)

    def z_kin_hor_e(self):
        return self._get_field("z_kin_hor_e", dims.EdgeDim, dims.KDim)

    def vn_ie(self):
        return self._get_field("vn_ie", dims.EdgeDim, dims.KDim)

    def vt(self):
        return self._get_field("vt", dims.EdgeDim, dims.KDim)

    def bdy_divdamp(self):
        return self._get_field("bdy_divdamp", dims.KDim)

    def divdamp_fac_o2(self):
        return self.serializer.read("divdamp_fac_o2", self.savepoint).astype(float)[0]

    def ddt_exner_phy(self):
        return self._get_field("ddt_exner_phy", dims.CellDim, dims.KDim)

    def ddt_vn_phy(self):
        return self._get_field("ddt_vn_phy", dims.EdgeDim, dims.KDim)

    def exner_now(self):
        return self._get_field("exner_now", dims.CellDim, dims.KDim)

    def exner_new(self):
        return self._get_field("exner_new", dims.CellDim, dims.KDim)

    def theta_v_now(self):
        return self._get_field("theta_v_now", dims.CellDim, dims.KDim)

    def theta_v_new(self):
        return self._get_field("theta_v_new", dims.CellDim, dims.KDim)

    def rho_now(self):
        return self._get_field("rho_now", dims.CellDim, dims.KDim)

    def rho_new(self):
        return self._get_field("rho_new", dims.CellDim, dims.KDim)

    def exner_pr(self):
        return self._get_field("exner_pr", dims.CellDim, dims.KDim)

    def grf_tend_rho(self):
        return self._get_field("grf_tend_rho", dims.CellDim, dims.KDim)

    def grf_tend_thv(self):
        return self._get_field("grf_tend_thv", dims.CellDim, dims.KDim)

    def grf_tend_vn(self):
        return self._get_field("grf_tend_vn", dims.EdgeDim, dims.KDim)

    def w_concorr_c(self):
        return self._get_field("w_concorr_c", dims.CellDim, dims.KDim)

    def ddt_vn_apc_pc(self, ntnd):
        return self._get_field_component("ddt_vn_apc_pc", ntnd, (dims.EdgeDim, dims.KDim))

    def ddt_w_adv_pc(self, ntnd):
        return self._get_field_component("ddt_w_adv_pc", ntnd, (dims.CellDim, dims.KDim))

    def grf_tend_w(self):
        return self._get_field("grf_tend_w", dims.CellDim, dims.KDim)

    def mass_fl_e(self):
        return self._get_field("mass_fl_e", dims.EdgeDim, dims.KDim)

    def mass_flx_me(self):
        return self._get_field("mass_flx_me", dims.EdgeDim, dims.KDim)

    def mass_flx_ic(self):
        return self._get_field("mass_flx_ic", dims.CellDim, dims.KDim)

    def rho_ic(self):
        return self._get_field("rho_ic", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def rho_incr(self):
        return self._get_field("rho_incr", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def exner_incr(self):
        return self._get_field("exner_incr", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def vn_incr(self):
        return self._get_field("vn_incr", dims.EdgeDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def exner_dyn_incr(self):
        return self._get_field("exner_dyn_incr", dims.CellDim, dims.KDim)

    def scal_divdamp_o2(self) -> float:
        return self.serializer.read("scal_divdamp_o2", self.savepoint)[0]

    def scal_divdamp(self) -> fa.KField[float]:
        return self._get_field("scal_divdamp", dims.KDim)

    def theta_v_ic(self):
        return self._get_field("theta_v_ic", dims.CellDim, dims.KDim)

    def vn_traj(self):
        return self._get_field("vn_traj", dims.EdgeDim, dims.KDim)

    def z_dwdz_dd(self):
        return self._get_field("z_dwdz_dd", dims.CellDim, dims.KDim)

    def z_graddiv_vn(self):
        return self._get_field("z_graddiv_vn", dims.EdgeDim, dims.KDim)

    def z_theta_v_e(self):
        return self._get_field("z_theta_v_e", dims.EdgeDim, dims.KDim)

    def z_rho_e(self):
        return self._get_field("z_rho_e", dims.EdgeDim, dims.KDim)

    def z_gradh_exner(self):
        return self._get_field("z_gradh_exner", dims.EdgeDim, dims.KDim)

    def z_w_expl(self):
        return self._get_field("z_w_expl", dims.CellDim, dims.KDim)

    def z_rho_expl(self):
        return self._get_field("z_rho_expl", dims.CellDim, dims.KDim)

    def z_exner_expl(self):
        return self._get_field("z_exner_expl", dims.CellDim, dims.KDim)

    def z_alpha(self):
        return self._get_field("z_alpha", dims.CellDim, dims.KDim)

    def z_beta(self):
        return self._get_field("z_beta", dims.CellDim, dims.KDim)

    def z_contr_w_fl_l(self):
        return self._get_field("z_contr_w_fl_l", dims.CellDim, dims.KDim)

    def z_q(self):
        return self._get_field("z_q", dims.CellDim, dims.KDim)

    def wgt_nnow_rth(self) -> float:
        return self.serializer.read("wgt_nnow_rth", self.savepoint)[0]

    def wgt_nnew_rth(self) -> float:
        return self.serializer.read("wgt_nnew_rth", self.savepoint)[0]

    def wgt_nnow_vel(self) -> float:
        return self.serializer.read("wgt_nnow_vel", self.savepoint)[0]

    def wgt_nnew_vel(self) -> float:
        return self.serializer.read("wgt_nnew_vel", self.savepoint)[0]

    def w_now(self):
        return self._get_field("w_now", dims.CellDim, dims.KDim)

    def w_new(self):
        return self._get_field("w_new", dims.CellDim, dims.KDim)

    def vn_now(self):
        return self._get_field("vn_now", dims.EdgeDim, dims.KDim)

    def vn_new(self):
        return self._get_field("vn_new", dims.EdgeDim, dims.KDim)


class NonHydroInitEdgeDiagnosticsUpdateVnSavepoint(IconSavepoint):
    def rho_ic(self):
        return self._get_field("rho_ic", dims.CellDim, dims.KDim)

    def vn(self):
        return self._get_field("vn_now", dims.EdgeDim, dims.KDim)

    def vt(self):
        return self._get_field("vt", dims.EdgeDim, dims.KDim)

    def z_rth_pr(self, ind: TwoIndex):
        return self._get_field_component("z_rth_pr", ind, (dims.CellDim, dims.KDim))

    def z_exner_ex_pr(self):
        return self._get_field("z_exner_ex_pr", dims.CellDim, dims.KDim)

    def z_dexner_dz_c(self, ntnd: TimeIndex):
        return self._get_field_component("z_dexner_dz_c", ntnd, (dims.CellDim, dims.KDim))

    def theta_v(self):
        return self._get_field("theta_v_now", dims.CellDim, dims.KDim)

    def theta_v_ic(self):
        return self._get_field("theta_v_ic", dims.CellDim, dims.KDim)

    def z_dwdz_dd(self):
        return self._get_field("z_dwdz_dd", dims.CellDim, dims.KDim)

    def ddt_vn_apc_ntl(self, ntnd):
        return self._get_field_component("ddt_vn_apc_pc", ntnd, (dims.EdgeDim, dims.KDim))

    def ddt_vn_phy(self):
        return self._get_field("ddt_vn_phy", dims.EdgeDim, dims.KDim)

    def vn_incr(self):
        return self._get_field("vn_now", dims.EdgeDim, dims.KDim)

    def bdy_divdamp(self):
        return self._get_field("bdy_divdamp", dims.KDim)

    def z_hydro_corr(self):
        return self._get_field("z_hydro_corr", dims.EdgeDim)

    def z_graddiv2_vn(self):
        return self._get_field("z_graddiv2_vn", dims.EdgeDim, dims.KDim)

    def scal_divdamp(self):
        return self._get_field("scal_divdamp", dims.KDim)

    def z_rho_e(self):
        return self._get_field("z_rho_e", dims.EdgeDim, dims.KDim)

    def z_theta_v_e(self):
        return self._get_field("z_theta_v_e", dims.EdgeDim, dims.KDim)

    def z_gradh_exner(self):
        return self._get_field("z_gradh_exner", dims.EdgeDim, dims.KDim)

    def z_graddiv_vn(self):
        return self._get_field("z_graddiv_vn", dims.EdgeDim, dims.KDim)


class NonHydroInitVerticallyImplicitSolverSavepoint(IconSavepoint):
    def mass_fl_e(self):
        return self._get_field("mass_fl_e", dims.EdgeDim, dims.KDim)

    def z_theta_v_fl_e(self):
        return self._get_field("z_theta_v_fl_e", dims.EdgeDim, dims.KDim)

    def z_flxdiv_mass(self):
        return self._get_field("z_flxdiv_mass", dims.CellDim, dims.KDim)

    def z_flxdiv_theta(self):
        return self._get_field("z_flxdiv_theta", dims.CellDim, dims.KDim)

    def z_w_expl(self):
        return self._get_field("z_w_expl", dims.CellDim, dims.KDim)

    def ddt_w_adv_pc(self, ntnd: TimeIndex):
        return self._get_field_component("ddt_w_adv_pc", ntnd, (dims.CellDim, dims.KDim))

    def z_th_ddz_exner_c(self):
        return self._get_field("z_th_ddz_exner_c", dims.CellDim, dims.KDim)

    def z_contr_w_fl_l(self):
        return self._get_field("z_contr_w_fl_l", dims.CellDim, dims.KDim)

    def rho_ic(self):
        return self._get_field("rho_ic", dims.CellDim, dims.KDim)

    def w_concorr_c(self):
        return self._get_field("w_concorr_c", dims.CellDim, dims.KDim)

    def exner_nnow(self):
        return self._get_field("exner_now", dims.CellDim, dims.KDim)

    def rho_nnow(self):
        return self._get_field("rho_now", dims.CellDim, dims.KDim)

    def theta_v_nnow(self):
        return self._get_field("theta_v_now", dims.CellDim, dims.KDim)

    def z_alpha(self):
        return self._get_field("z_alpha", dims.CellDim, dims.KDim)

    def z_beta(self):
        return self._get_field("z_beta", dims.CellDim, dims.KDim)

    def theta_v_ic(self):
        return self._get_field("theta_v_ic", dims.CellDim, dims.KDim)

    def z_q(self):
        return self._get_field("z_q", dims.CellDim, dims.KDim)

    def w(self):
        return self._get_field("w_now", dims.CellDim, dims.KDim)

    def z_rho_expl(self):
        return self._get_field("z_rho_expl", dims.CellDim, dims.KDim)

    def z_exner_expl(self):
        return self._get_field("z_exner_expl", dims.CellDim, dims.KDim)

    def exner_pr(self):
        return self._get_field("exner_pr", dims.CellDim, dims.KDim)

    def ddt_exner_phy(self):
        return self._get_field("ddt_exner_phy", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def rho_incr(self):
        return self._get_field("rho_now", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def exner_incr(self):
        return self._get_field("exner_now", dims.CellDim, dims.KDim)

    def z_raylfac(self):
        return self._get_field("z_raylfac", dims.KDim)

    def rho(self):
        return self._get_field("rho_now", dims.CellDim, dims.KDim)

    def exner(self):
        return self._get_field("exner_now", dims.CellDim, dims.KDim)

    def theta_v(self):
        return self._get_field("theta_v_now", dims.CellDim, dims.KDim)

    def z_dwdz_dd(self):
        return self._get_field("z_dwdz_dd", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def exner_dyn_incr(self):
        return self._get_field("exner_dyn_incr", dims.CellDim, dims.KDim)

    def mass_flx_ic(self):
        return self._get_field("mass_flx_ic", dims.CellDim, dims.KDim)

    def vol_flx_ic(self):
        return self._get_field("vol_flx_ic", dims.CellDim, dims.KDim)


class IconDycoreInit30To38Savepoint(IconSavepoint):
    def z_vn_avg(self):
        return self._get_field("z_vn_avg", dims.EdgeDim, dims.KDim)

    def z_graddiv_vn(self):
        return self._get_field("z_graddiv_vn", dims.EdgeDim, dims.KDim)

    def vn(self):
        return self._get_field("vn", dims.EdgeDim, dims.KDim)

    def vt(self):
        return self._get_field("vt", dims.EdgeDim, dims.KDim)

    def z_rho_e(self):
        return self._get_field("z_rho_e", dims.EdgeDim, dims.KDim)

    def z_theta_v_e(self):
        return self._get_field("z_theta_v_e", dims.EdgeDim, dims.KDim)

    def z_vt_ie(self):
        return self._get_field("z_vt_ie", dims.EdgeDim, dims.KDim)

    def vn_ie(self):
        return self._get_field("vn_ie", dims.EdgeDim, dims.KDim)

    def mass_fl_e(self):
        return self._get_field("mass_fl_e", dims.EdgeDim, dims.KDim)

    def z_theta_v_fl_e(self):
        return self._get_field("z_theta_v_fl_e", dims.EdgeDim, dims.KDim)

    def z_kin_hor_e(self):
        return self._get_field("z_kin_hor_e", dims.EdgeDim, dims.KDim)

    def z_w_concorr_me(self):
        return self._get_field("z_w_concorr_me", dims.EdgeDim, dims.KDim)


class IconDycoreExit30To38Savepoint(IconSavepoint):
    def z_vn_avg(self):
        return self._get_field("z_vn_avg", dims.EdgeDim, dims.KDim)

    def z_graddiv_vn(self):
        return self._get_field("z_graddiv_vn", dims.EdgeDim, dims.KDim)

    def vt(self):
        return self._get_field("vt", dims.EdgeDim, dims.KDim)

    def mass_fl_e(self):
        return self._get_field("mass_fl_e", dims.EdgeDim, dims.KDim)

    def z_theta_v_fl_e(self):
        return self._get_field("z_theta_v_fl_e", dims.EdgeDim, dims.KDim)

    def vn_ie(self):
        return self._get_field("vn_ie", dims.EdgeDim, dims.KDim)

    def z_vt_ie(self):
        return self._get_field("z_vt_ie", dims.EdgeDim, dims.KDim)

    def z_kin_hor_e(self):
        return self._get_field("z_kin_hor_e", dims.EdgeDim, dims.KDim)

    def z_w_concorr_me(self):
        return self._get_field("z_w_concorr_me", dims.EdgeDim, dims.KDim)

    def vn_traj(self):
        return self._get_field("vn_traj", dims.EdgeDim, dims.KDim)

    def mass_flx_me(self):
        return self._get_field("mass_flx_me", dims.EdgeDim, dims.KDim)


class IconNonHydroExitSavepoint(IconSavepoint):
    def z_exner_ex_pr(self):
        return self._get_field("z_exner_ex_pr", dims.CellDim, dims.KDim)  # KHalfDim

    def rho_ic(self):
        return self._get_field("rho_ic", dims.CellDim, dims.KDim)

    def z_rho_e(self):
        return self._get_field("z_rho_e", dims.EdgeDim, dims.KDim)

    def z_exner_expl(self):
        return self._get_field("z_exner_expl", dims.CellDim, dims.KDim)

    def z_rho_expl(self):
        return self._get_field("z_rho_expl", dims.CellDim, dims.KDim)

    def z_theta_v_e(self):
        return self._get_field("z_theta_v_e", dims.EdgeDim, dims.KDim)

    def theta_v_ic(self):
        return self._get_field("theta_v_ic", dims.CellDim, dims.KDim)

    def z_q(self):
        return self._get_field("z_q", dims.CellDim, dims.KDim)

    def z_graddiv_vn(self):
        return self._get_field("z_graddiv_vn", dims.EdgeDim, dims.KDim)

    def exner_pr(self):
        return self._get_field("exner_pr", dims.CellDim, dims.KDim)

    def z_kin_hor_e(self):
        return self._get_field("z_kin_hor_e", dims.EdgeDim, dims.KDim)

    def z_alpha(self):
        return self._get_field("z_alpha", dims.CellDim, dims.KDim)

    def z_beta(self):
        return self._get_field("z_beta", dims.CellDim, dims.KDim)

    def vn_new(self):
        return self._get_field("vn_new", dims.EdgeDim, dims.KDim)

    def theta_v_new(self):
        return self._get_field("theta_v_new", dims.CellDim, dims.KDim)

    def rho_new(self):
        return self._get_field("rho_new", dims.CellDim, dims.KDim)

    def exner_new(self):
        return self._get_field("exner_new", dims.CellDim, dims.KDim)

    def w_new(self):
        return self._get_field("w_new", dims.CellDim, dims.KDim)

    def z_vn_avg(self):
        return self._get_field("z_vn_avg", dims.EdgeDim, dims.KDim)

    def mass_fl_e(self):
        return self._get_field("mass_fl_e", dims.EdgeDim, dims.KDim)

    def mass_flx_ic(self):
        return self._get_field("mass_flx_ic", dims.CellDim, dims.KDim)

    def vol_flx_ic(self):
        return self._get_field("vol_flx_ic", dims.CellDim, dims.KDim)

    def mass_flx_me(self):
        return self._get_field("mass_flx_me", dims.EdgeDim, dims.KDim)

    def vn_traj(self):
        return self._get_field("vn_traj", dims.EdgeDim, dims.KDim)

    def exner_dyn_incr(self):
        return self._get_field("exner_dyn_incr", dims.CellDim, dims.KDim)

    def z_exner_ic(self):
        return self._get_field("z_exner_ic", dims.CellDim, dims.KDim)

    def z_dexner_dz_c(self, ntnd: TimeIndex):
        return self._get_field_component("z_dexner_dz_c", ntnd, (dims.CellDim, dims.KDim))

    def z_rth_pr(self, ind: TwoIndex):
        return self._get_field_component("z_rth_pr", ind, (dims.CellDim, dims.KDim))

    def z_grad_rth(self, ind: FourIndex):
        return self._get_field_component("z_grad_rth", ind, (dims.CellDim, dims.KDim))

    def z_th_ddz_exner_c(self):
        return self._get_field("z_th_ddz_exner_c", dims.CellDim, dims.KDim)

    def z_gradh_exner(self):
        return self._get_field("z_gradh_exner", dims.EdgeDim, dims.KDim)

    def z_hydro_corr(self):
        return self._get_field("z_hydro_corr", dims.EdgeDim, dims.KDim)

    def z_theta_v_pr_ic(self):
        return self._get_field("z_theta_v_pr_ic", dims.CellDim, dims.KDim)

    def vt(self):
        return self._get_field("vt", dims.EdgeDim, dims.KDim)

    def z_flxdiv_mass(self):
        return self._get_field("z_flxdiv_mass", dims.CellDim, dims.KDim)

    def z_w_expl(self):
        return self._get_field("z_w_expl", dims.CellDim, dims.KDim)

    def z_flxdiv_theta(self):
        return self._get_field("z_flxdiv_theta", dims.CellDim, dims.KDim)

    def z_contr_w_fl_l(self):
        return self._get_field("z_contr_w_fl_l", dims.CellDim, dims.KDim)

    def vn_ie(self):
        return self._get_field("vn_ie", dims.EdgeDim, dims.KDim)

    def z_vt_ie(self):
        return self._get_field("z_vt_ie", dims.EdgeDim, dims.KDim)

    def z_w_concorr_me(self):
        return self._get_field("z_w_concorr_me", dims.EdgeDim, dims.KDim)

    def w_concorr_c(self):
        return self._get_field("w_concorr_c", dims.CellDim, dims.KDim)

    def z_theta_v_fl_e(self):
        return self._get_field("z_theta_v_fl_e", dims.EdgeDim, dims.KDim)

    def z_dwdz_dd(self):
        return self._get_field("z_dwdz_dd", dims.CellDim, dims.KDim)


class NonHydroExitEdgeDiagnosticsUpdateVnSavepoint(IconSavepoint):
    def z_rho_e(self):
        return self._get_field("z_rho_e", dims.EdgeDim, dims.KDim)

    def z_theta_v_e(self):
        return self._get_field("z_theta_v_e", dims.EdgeDim, dims.KDim)

    def z_gradh_exner(self):
        return self._get_field("z_gradd_exner", dims.EdgeDim, dims.KDim)

    def vn(self):
        return self._get_field("vn_new", dims.EdgeDim, dims.KDim)

    def z_graddiv_vn(self):
        return self._get_field("z_graddiv_vn", dims.EdgeDim, dims.KDim)

    def z_graddiv2_vn(self):
        return self._get_field("z_graddiv2_vn", dims.EdgeDim, dims.KDim)


# TODO(halungge): rename?
class IconNonHydroFinalSavepoint(IconSavepoint):
    def theta_v_new(self):
        return self._get_field("theta_v", dims.CellDim, dims.KDim)

    def exner_new(self):
        return self._get_field("exner", dims.CellDim, dims.KDim)


class IconVelocityInitSavepoint(IconSavepoint):
    def cfl_w_limit(self) -> float:
        return self.serializer.read("cfl_w_limit", self.savepoint)[0]

    def vn_only(self) -> bool:
        return bool(self.serializer.read("vn_only", self.savepoint)[0])

    def max_vcfl_dyn(self):
        return self.serializer.read("max_vcfl_dyn", self.savepoint)[0]

    def scalfac_exdiff(self) -> float:
        return self.serializer.read("scalfac_exdiff", self.savepoint)[0]

    def ddt_vn_apc_pc(self, ntnd: TimeIndex):
        return self._get_field_component("ddt_vn_apc_pc", ntnd, (dims.EdgeDim, dims.KDim))

    def ddt_w_adv_pc(self, ntnd: TimeIndex):
        return self._get_field_component("ddt_w_adv_pc", ntnd, (dims.CellDim, dims.KDim))

    def vn(self):
        return self._get_field("vn", dims.EdgeDim, dims.KDim)

    def vn_ie(self):
        return self._get_field("vn_ie", dims.EdgeDim, dims.KDim)

    def vt(self):
        return self._get_field("vt", dims.EdgeDim, dims.KDim)

    def w(self):
        return self._get_field("w", dims.CellDim, dims.KDim)

    def z_vt_ie(self):
        return self._get_field("z_vt_ie", dims.EdgeDim, dims.KDim)

    def z_kin_hor_e(self):
        return self._get_field("z_kin_hor_e", dims.EdgeDim, dims.KDim)

    def z_w_concorr_me(self):
        return self._get_field("z_w_concorr_me", dims.EdgeDim, dims.KDim)

    def w_concorr_c(self):
        return self._get_field("w_concorr_c", dims.CellDim, dims.KDim)

    def lvn_only(self) -> bool:
        return bool(self.serializer.read("vn_only", self.savepoint)[0])

    def z_w_con_c_full(self):
        return self._get_field("z_w_con_c_full", dims.CellDim, dims.KDim)


class IconVelocityExitSavepoint(IconSavepoint):
    def max_vcfl_dyn(self):
        return self.serializer.read("max_vcfl_dyn", self.savepoint)[0]

    def ddt_vn_apc_pc(self, ntnd: TimeIndex):
        return self._get_field_component("ddt_vn_apc_pc", ntnd, (dims.EdgeDim, dims.KDim))

    def ddt_w_adv_pc(self, ntnd: TimeIndex):
        return self._get_field_component("ddt_w_adv_pc", ntnd, (dims.CellDim, dims.KDim))

    def vn(self):
        return self._get_field("vn", dims.EdgeDim, dims.KDim)

    def w(self):
        return self._get_field("w", dims.CellDim, dims.KDim)

    def vt(self):
        return self._get_field("vt", dims.EdgeDim, dims.KDim)

    def vn_ie(self):
        return self._get_field("vn_ie", dims.EdgeDim, dims.KDim)

    def w_concorr_c(self):
        return self._get_field("w_concorr_c", dims.CellDim, dims.KDim)

    def z_vt_ie(self):
        return self._get_field("z_vt_ie", dims.EdgeDim, dims.KDim)

    def z_w_concorr_me(self):
        return self._get_field("z_w_concorr_me", dims.EdgeDim, dims.KDim)

    def z_w_concorr_mc(self):
        return self._get_field("z_w_concorr_mc", dims.CellDim, dims.KDim)

    def z_v_grad_w(self):
        return self._get_field("z_v_grad_w", dims.EdgeDim, dims.KDim)

    def z_w_con_c(self):
        return self._get_field("z_w_con_c", dims.CellDim, dims.KDim)  # KhalfDim

    def z_w_con_c_full(self):
        return self._get_field("z_w_con_c_full", dims.CellDim, dims.KDim)

    def z_ekinh(self):
        return self._get_field("z_ekinh", dims.CellDim, dims.KDim)

    def cfl_clipping(self):
        return self._get_field("cfl_clipping", dims.CellDim, dims.KDim, dtype=bool)

    def vcfl_dsl(self):
        return self._get_field("vcfl_dsl", dims.CellDim, dims.KDim)

    def z_kin_hor_e(self):
        return self._get_field("z_kin_hor_e", dims.EdgeDim, dims.KDim)


class IconJabwExitSavepoint(IconSavepoint):
    def exner(self):
        return self._get_field("exner", dims.CellDim, dims.KDim)

    def rho(self):
        return self._get_field("rho", dims.CellDim, dims.KDim)

    def vn(self):
        return self._get_field("vn", dims.EdgeDim, dims.KDim)

    def w(self):
        return self._get_field("w", dims.CellDim, dims.KDim)

    def theta_v(self):
        return self._get_field("theta_v", dims.CellDim, dims.KDim)

    def pressure(self):
        return self._get_field("pressure", dims.CellDim, dims.KDim)

    def temperature(self):
        return self._get_field("temperature", dims.CellDim, dims.KDim)

    # TODO(): change field name
    def pressure_sfc(self):
        return self._get_field("surface_pressure", dims.CellDim)


class IconDiagnosticsInitSavepoint(IconSavepoint):
    def pressure(self):
        return self._get_field("pressure", dims.CellDim, dims.KDim)

    def temperature(self):
        return self._get_field("temperature", dims.CellDim, dims.KDim)

    def exner_pr(self):
        return self._get_field("exner_pr", dims.CellDim, dims.KDim)

    def pressure_ifc(self):
        return self._get_field("pressure_ifc", dims.CellDim, dims.KDim)

    def pressure_sfc(self):
        return self._get_field("pressure_sfc", dims.CellDim)

    def virtual_temperature(self):
        return self._get_field("virtual_temperature", dims.CellDim, dims.KDim)

    def zonal_wind(self):
        return self._get_field("u", dims.CellDim, dims.KDim)

    def meridional_wind(self):
        return self._get_field("v", dims.CellDim, dims.KDim)


class IconPrognosticsInitSavepoint(IconSavepoint):
    def exner_now(self):
        return self._get_field("exner_now", dims.CellDim, dims.KDim)

    def rho_now(self):
        return self._get_field("rho_now", dims.CellDim, dims.KDim)

    def vn_now(self):
        return self._get_field("vn_now", dims.EdgeDim, dims.KDim)

    def theta_v_now(self):
        return self._get_field("theta_v_now", dims.CellDim, dims.KDim)


class IconGraupelSavepoint(IconSavepoint):
    def temperature(self):
        return self._get_field("temperature", dims.CellDim, dims.KDim)

    def pressure(self):
        return self._get_field("pressure", dims.CellDim, dims.KDim)

    def rho(self):
        return self._get_field("rho", dims.CellDim, dims.KDim)

    def tracer(self, ntracer: TracerIndex):
        return self._get_field_component("tracers", ntracer, (dims.CellDim, dims.KDim))

    def ddt_tend_t(self):
        return self._get_field("ddt_tend_t", dims.CellDim, dims.KDim)

    def ddt_tend_qv(self):
        return self._get_field("ddt_tend_qv", dims.CellDim, dims.KDim)

    def ddt_tend_qc(self):
        return self._get_field("ddt_tend_qc", dims.CellDim, dims.KDim)

    def ddt_tend_qi(self):
        return self._get_field("ddt_tend_qi", dims.CellDim, dims.KDim)

    def ddt_tend_qr(self):
        return self._get_field("ddt_tend_qr", dims.CellDim, dims.KDim)

    def ddt_tend_qs(self):
        return self._get_field("ddt_tend_qs", dims.CellDim, dims.KDim)

    def rain_flux(self):
        return self._get_field("rain_gsp_rate", dims.CellDim)

    def snow_flux(self):
        return self._get_field("snow_gsp_rate", dims.CellDim)

    def graupel_flux(self):
        return self._get_field("graupel_gsp_rate", dims.CellDim)

    def ice_flux(self):
        return self._get_field("ice_gsp_rate", dims.CellDim)

    def qv(self):
        return self.tracer(QV)

    def qc(self):
        return self.tracer(QC)

    def qi(self):
        return self.tracer(QI)

    def qr(self):
        return self.tracer(QR)

    def qs(self):
        return self.tracer(QS)

    def qg(self):
        return self.tracer(QG)

    def qnc(self):
        return self._get_field("qnc", dims.CellDim)

    def dtime(self):
        return self.serializer.read("dtime", self.savepoint)[0]


class IconSatadExitSavepoint(IconSavepoint):
    def temperature(self):
        return self._get_field("temperature", dims.CellDim, dims.KDim)

    def tracer(self, ntracer: TracerIndex):
        return self._get_field_component("tracers", ntracer, (dims.CellDim, dims.KDim))

    def qv(self):
        return self.tracer(QV)

    def qc(self):
        return self.tracer(QC)

    def qi(self):
        return self.tracer(QI)

    def qr(self):
        return self.tracer(QR)

    def qs(self):
        return self.tracer(QS)

    def qg(self):
        return self.tracer(QG)

    def exner(self):
        return self._get_field("exner", dims.CellDim, dims.KDim)

    def virtual_temperature(self):
        return self._get_field("virtual_temperature", dims.CellDim, dims.KDim)

    def pressure(self):
        return self._get_field("pressure", dims.CellDim, dims.KDim)

    def pressure_ifc(self):
        return self._get_field("pressure_ifc", dims.CellDim, dims.KDim)

    def pressure_sfc(self):
        return self._get_field("pressure_sfc", dims.CellDim)


class IconSatadInitSavepoint(IconSatadExitSavepoint):
    def rho(self):
        return self._get_field("rho", dims.CellDim, dims.KDim)


class TopographySavepoint(IconSavepoint):
    def topo_c(self):
        return self._get_field("topography", dims.CellDim)

    def topo_smt_c(self):
        return self._get_field("smooth_topography", dims.CellDim)


class IconTurbulenceSavepoint(IconSavepoint):
    """
    Common part of every savepoint of ICON's NWP 1D turbulence schemes: the block index and the
    horizontal window that alone holds computed values.

    Every serialized name carries a scheme prefix ('td_' for SUB 'turbdiff', 'vd_' for SUB
    'vertdiff'): one serializer serves the whole icon4py archive and its field table is global,
    so an unprefixed 'u' or 'w' would collide with the rank-3 fields the dycore savepoints
    already register. The two prefixes also keep apart names that the two schemes share without
    sharing their meaning -- 'vertdiff's 'zaux' holds the tridiagonal coefficients where
    'turbdiff's holds thermodynamic factors, and its 'len_scale' is the diffusion momentum.
    The accessors drop the prefix, so a method name is the Fortran dummy-argument name unless
    the Fortran name lies, in which case the method is named after the quantity.

    ONLY THE COLUMNS 'ivstart:ivend' HOLD COMPUTED VALUES. The hooks write the whole 'nproma'
    slab, but the schemes only loop over 'ivstart:ivend' (2425..10700 of nproma=16224 for
    exp.mch_icon-ch2_small). Outside that window nothing is NaN, so a comparison that forgets
    to mask fails looking exactly like a physics bug. What is there depends on the array: the
    routine-local working sets hold untouched memory ('vertdiff's surface Exner factor is 0.894
    to 1.002 inside and drops to 0 below 'ivstart'), while the interface fields hold the
    un-diffused lateral-boundary rows, which are not the result of this call either and are not
    necessarily physical ('tke' reaches -0.026 below 'ivstart' against a floor of 0.01 inside).
    Mask every comparison with 'ivstart()' and 'ivend()'.
    """

    #: Prefix of the serialized names of the scheme this savepoint belongs to.
    _prefix: ClassVar[str] = "td_"

    def _read_scalar(self, name: str):
        """Read a serialized scalar; serialbox stores it as a length-one array."""
        return self.serializer.read(name, self.savepoint).item()

    def _has_field(self, name: str) -> bool:
        """Whether 'name' was serialized into this savepoint at all."""
        return name in self.serializer.fields_at_savepoint(self.savepoint)

    def raw_field(self, name: str):
        """
        Escape hatch: the serialized buffer of 'name' (with the scheme prefix) as a plain array,
        unsqueezed, untruncated and uninterpreted.

        Use it for a storage slot whose meaning this reader refuses to name, and mask it with
        'ivstart()' and 'ivend()' yourself. Everything the reader does know is in the named
        accessors; reaching for this one means you have taken the interpretation on yourself.
        """
        return self.xp.asarray(self.serializer.read(name, self.savepoint))

    def block(self) -> int:
        """The block index 'iblock' this savepoint was written for; one-based, as in ICON."""
        return self.savepoint.metainfo.to_dict()["block"]

    def ivstart(self) -> int:
        """
        First computed column, as a zero-based Python index.

        Fortran 'ivstart' is one-based and inclusive, so it is shifted by one here, following
        '_read_int32_shift1'. Together with 'ivend()' this is a half-open Python range:
        'field.ndarray[sp.ivstart() : sp.ivend()]' is exactly the computed part.
        """
        return self._read_scalar(f"{self._prefix}ivstart") - 1

    def ivend(self) -> int:
        """
        End of the computed columns, exclusive.

        Fortran 'ivend' is one-based and inclusive, which is the same integer as the zero-based
        exclusive bound, so unlike 'ivstart()' it is returned unshifted.
        """
        return self._read_scalar(f"{self._prefix}ivend")


class IconTurbdiffSavepoint(IconTurbulenceSavepoint):
    """
    Common part of the two savepoints of the NWP 1D turbulence scheme SUB 'turbdiff'
    (turb_diffusion.f90:281), written by 'serialize_turbdiff_entry' and
    'serialize_turbdiff_exit' in mo_icon4py_verification.f90.

    Every Fortran line number quoted in this class and its two subclasses is as of 'icon'
    commit 38e3720277, the commit that inserted the hooks and produced this capture. That is
    two lines further down in turb_diffusion.f90 than the numbers the port spec quotes, which
    are from the parent commit.

    Every serialized name carries a 'td_' prefix: one serializer serves the whole icon4py
    archive and its field table is global, so an unprefixed 'u' or 'w' would collide with the
    rank-3 fields the dycore savepoints already register. The accessors drop the prefix, so
    the method name is the Fortran dummy-argument name.

    A savepoint is selected by ('date', 'id', 'block'): 'turbdiff' is called once per block of
    'nproma' columns and per domain, and the hook writes one savepoint per call.

    ONLY THE COLUMNS 'ivstart:ivend' HOLD COMPUTED VALUES. The hook writes the whole 'nproma'
    slab, but 'turbdiff' only loops over 'ivstart:ivend' (2425..10700 of nproma=16224 for
    exp.mch_icon-ch2_small). Outside that window the memory is untouched rather than NaN and
    the values are plausible: 'tke' reaches -0.026 below 'ivstart' against a physical floor of
    0.01 inside. Mask every comparison with 'ivstart()' and 'ivend()'.

    The cell-field accessors return 'num_cells' columns, not 'nproma': 'IconSavepoint.
    _reduce_to_dim_size' truncates the horizontal axis to the grid size, which happens to cut
    exactly at 'ivend' here because the domain is one block and 'ivend == num_cells == 10700'.
    That removes the untouched tail but not the lateral-boundary rows below 'ivstart'. The
    truncation is only meaningful for a single-block capture ('nproma >= n_patch_cells', which
    is how exp.mch_icon-ch2_small is configured and why 'block' is always 1); with several
    blocks the row index is block-local and 'block()' would have to enter the mapping.
    """

    def gz0(self):
        """Roughness length times gravity [m2/s2]."""
        return self._get_field("td_gz0", dims.CellDim)

    def tvm(self):
        """Turbulent transfer velocity for momentum at the surface [m/s]."""
        return self._get_field("td_tvm", dims.CellDim)

    def tvh(self):
        """Turbulent transfer velocity for heat at the surface [m/s]."""
        return self._get_field("td_tvh", dims.CellDim)

    def tfm(self):
        """Laminar reduction factor for momentum [-]."""
        return self._get_field("td_tfm", dims.CellDim)

    def tfh(self):
        """Laminar reduction factor for scalars [-]."""
        return self._get_field("td_tfh", dims.CellDim)

    def tfv(self):
        """Laminar reduction factor for water vapour compared to heat [-]."""
        return self._get_field("td_tfv", dims.CellDim)

    def tke(self):
        """
        'q = SQRT(2*TKE)', the turbulent velocity and not the energy, on half levels [m/s].

        Serialized as '(nvec, ke1, ntim)'. The NWP interface runs with 'ntim == 1' -- the
        entry savepoint records it as 'td_ntim' -- so the trailing axis is a singleton and is
        squeezed away, leaving a (CellDim, KDim) field of 'ke1' levels.
        """
        return self._get_field("td_tke", dims.CellDim, dims.KDim)

    def tkvm(self):
        """Turbulent diffusion coefficient for momentum, on half levels [m2/s]."""
        return self._get_field("td_tkvm", dims.CellDim, dims.KDim)

    def tkvh(self):
        """Turbulent diffusion coefficient for heat and other scalars, on half levels [m2/s]."""
        return self._get_field("td_tkvh", dims.CellDim, dims.KDim)

    def tprn(self):
        """
        Turbulent Prandtl number on half levels [-].

        Beware: in a capture like exp.mch_icon-ch2_small this is a '(1, 1)' dummy and not a
        cell field. 'prm_diag%tprn' is only given the '(nproma, nlevp1, nblks_c)' shape when
        the namelist selects a TMod that produces it; otherwise mo_nwp_phy_state.f90:4706-4709
        allocates '(/1, 1, kblks/)', "shape for dummy array only". Check '.ndarray.shape'
        against 'ke1()' before comparing anything. The field is built from the raw buffer
        rather than through '_get_field' because squeezing a '(1, 1)' array leaves a scalar.
        """
        buffer = self.xp.asarray(self.serializer.read("td_tprn", self.savepoint), dtype=float)
        return self._get_field_from_ndarray(buffer, dims.CellDim, dims.KDim)

    def rcld(self):
        """
        Standard deviation of the local super-saturation, on half levels [-].

        Genuine scheme input at entry: the zeroing loop in 'turb_setup' sits behind
        'IF (iini > 0)' and the NWP interface calls with 'iini = 0'.
        """
        return self._get_field("td_rcld", dims.CellDim, dims.KDim)

    def zvari(self, component: int):
        """
        One component of the scheme's main scratch array, on half levels.

        'zvari(:,:,0:ndim)' with 'ndim = MAX(nmvar, naux) = 5' (mo_turbdiff_config.f90:91), so
        the lower bound is zero and 'component' is the Fortran third index unchanged, 0..5.
        Unlike 'zaux', no offset is needed.

        The meaning changes as the routine proceeds (turb_diffusion.f90:594). Component
        indices (mo_turbdiff_config.f90:62-77): 'u_m=1', 'v_m=2', 'tem=tet=tet_l=3',
        'vap=h2o_g=4', 'liq=5'; component 0 is the half-level pressure and later the
        acceleration of the circulation kinetic energy.

        AT THE ENTRY SAVEPOINT THIS ARRAY IS NOT YET DEFINED. 'zvari' is 'INTENT(OUT)' in
        'turbdiff' and section 0) is the first thing that writes it, so the entry savepoint --
        which sits after 'turb_setup' but before section 0) -- holds whatever the allocation
        left there. Verified against the archive: 'zvari(1)' at 'turbdiff-entry' is not 'u',
        while at 'turbdiff-0-exit' it is, exactly. Read the conserved variables from
        'turbdiff-0-exit' and not from here. At the exit savepoint the components hold the
        effective vertical gradients of the model variables, and 0 the effective gradient of
        the circulation kinetic energy.
        """
        return self._get_field_component("td_zvari", component, (dims.CellDim, dims.KDim))

    @IconSavepoint.optionally_registered(dims.CellDim)
    def tkred_sfc(self):
        """Reduction factor for the minimum diffusion coefficients near the surface [-]."""
        return self._get_field("td_tkred_sfc", dims.CellDim)

    @IconSavepoint.optionally_registered(dims.CellDim)
    def tkred_sfc_h(self):
        """As 'tkred_sfc', for scalars [-]."""
        return self._get_field("td_tkred_sfc_h", dims.CellDim)


class IconTurbdiffEntrySavepoint(IconTurbdiffSavepoint):
    """
    State at the start of section 0) of SUB 'turbdiff' (turb_diffusion.f90:937).

    The hook sits after SUB 'turb_setup' (turb_utilities.f90:249), so this savepoint is also
    the reference for turb_setup's own output, which is serialized explicitly: 'lini',
    'it_start', 'nvor', 'fr_tke', 'l_scal' and 'fc_min'.

    The optional accessors below are the dummy arguments that are 'OPTIONAL' in 'turbdiff'
    and were only written because 'PRESENT()' was true. Their device presence is a property
    of the caller, not of the scheme, so a capture from a different namelist may lack them.
    """

    def nvec(self) -> int:
        """'nproma': the horizontal extent of the slab, computed columns and untouched alike."""
        return self._read_scalar("td_nvec")

    def ke(self) -> int:
        """Number of main (full) levels."""
        return self._read_scalar("td_ke")

    def ke1(self) -> int:
        """Number of half levels, 'ke + 1'."""
        return self._read_scalar("td_ke1")

    def kcm(self) -> int:
        """Level index above the roughness layer; 'ke1 + 1' when the canopy is switched off."""
        return self._read_scalar("td_kcm")

    def iini(self) -> int:
        """Initialization type; the NWP interface always calls with 0 (no separate init)."""
        return self._read_scalar("td_iini")

    def ntur(self) -> int:
        """Time index of 'tke' to be updated."""
        return self._read_scalar("td_ntur")

    def nprv(self) -> int:
        """Time index of the previous 'tke'."""
        return self._read_scalar("td_nprv")

    def ntim(self) -> int:
        """Number of 'tke' time levels; 1 for the NWP interface, see 'tke()'."""
        return self._read_scalar("td_ntim")

    def dt_var(self) -> float:
        """Time step for the vertical diffusion of the model variables [s]."""
        return self._read_scalar("td_dt_var")

    def dt_tke(self) -> float:
        """Time step for the TKE equation [s]."""
        return self._read_scalar("td_dt_tke")

    def ltkeinp(self) -> bool:
        """TKE is taken as input rather than computed."""
        return self._read_scalar("td_ltkeinp")

    def l3dturb(self) -> bool:
        """3D turbulence; hardcoded '.FALSE.' at mo_nwp_turbdiff_interface.f90:584."""
        return self._read_scalar("td_l3dturb")

    def lrunsso(self) -> bool:
        """The SSO scheme runs, so 'ut_sso' and 'vt_sso' are meaningful."""
        return self._read_scalar("td_lrunsso")

    def lruncnv(self) -> bool:
        """The convection scheme runs, so 'tket_conv' is meaningful."""
        return self._read_scalar("td_lruncnv")

    def lrunscm(self) -> bool:
        """Single-column mode."""
        return self._read_scalar("td_lrunscm")

    def lini(self) -> bool:
        """Output of 'turb_setup': this call is an initialization call."""
        return self._read_scalar("td_lini")

    def it_start(self) -> int:
        """Output of 'turb_setup': first index of the security iteration loop."""
        return self._read_scalar("td_it_start")

    def nvor(self) -> int:
        """Output of 'turb_setup': time index of the 'tke' the iteration starts from."""
        return self._read_scalar("td_nvor")

    def fr_tke(self) -> float:
        """Output of 'turb_setup': '1 / dt_tke' [1/s]."""
        return self._read_scalar("td_fr_tke")

    def l_scal(self):
        """Output of 'turb_setup': effective horizontal scale for the length-scale limit [m]."""
        return self._get_field("td_l_scal", dims.CellDim)

    def fc_min(self):
        """Output of 'turb_setup': minimum value of the forcing of the TKE equation [1/s2]."""
        return self._get_field("td_fc_min", dims.CellDim)

    def l_hori(self):
        """Horizontal grid spacing used as the turbulent length scale [m]."""
        return self._get_field("td_l_hori", dims.CellDim)

    def hhl(self):
        """Height of the model half levels [m]; ICON passes 'p_metrics%z_ifc'."""
        return self._get_field("td_hhl", dims.CellDim, dims.KDim)

    def dp0(self):
        """Pressure thickness of the main layers [Pa]."""
        return self._get_field("td_dp0", dims.CellDim, dims.KDim)

    def trop_mask(self):
        """Mask for the tropics, used by the vertical smoothing of the length scale [-]."""
        return self._get_field("td_trop_mask", dims.CellDim)

    def innertrop_mask(self):
        """Mask for the inner tropics [-]."""
        return self._get_field("td_innertrop_mask", dims.CellDim)

    def l_pat(self):
        """Effective length scale of the near-surface thermal inhomogeneity pattern [m]."""
        return self._get_field("td_l_pat", dims.CellDim)

    def t_g(self):
        """Surface temperature (grid-mean over the tiles) [K]."""
        return self._get_field("td_t_g", dims.CellDim)

    def qv_s(self):
        """Specific humidity at the surface [kg/kg]."""
        return self._get_field("td_qv_s", dims.CellDim)

    def ps(self):
        """Surface pressure [Pa]."""
        return self._get_field("td_ps", dims.CellDim)

    def u(self):
        """Zonal wind at the mass centre, main levels [m/s]."""
        return self._get_field("td_u", dims.CellDim, dims.KDim)

    def v(self):
        """Meridional wind at the mass centre, main levels [m/s]."""
        return self._get_field("td_v", dims.CellDim, dims.KDim)

    def t(self):
        """Temperature, main levels [K]."""
        return self._get_field("td_t", dims.CellDim, dims.KDim)

    def qv(self):
        """Specific humidity, main levels [kg/kg]."""
        return self._get_field("td_qv", dims.CellDim, dims.KDim)

    def qc(self):
        """Specific cloud water content, main levels [kg/kg]."""
        return self._get_field("td_qc", dims.CellDim, dims.KDim)

    def prs(self):
        """Pressure, main levels [Pa]."""
        return self._get_field("td_prs", dims.CellDim, dims.KDim)

    def rhoh(self):
        """Air density, main levels [kg/m3]."""
        return self._get_field("td_rhoh", dims.CellDim, dims.KDim)

    def epr(self):
        """Exner pressure, main levels [-]."""
        return self._get_field("td_epr", dims.CellDim, dims.KDim)

    def tketens(self):
        """
        TKE tendency from advection, on half levels [m/s2].

        'tketens' is 'INTENT(INOUT)' in 'turbdiff', so the entry value is serialized under
        'td_tketens_in' to keep it apart from the exit value in the global field table.
        """
        return self._get_field("td_tketens_in", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def w(self):
        """Vertical wind on half levels [m/s]."""
        return self._get_field("td_w", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def hdef2(self):
        """
        Square of the horizontal deformation, half levels [1/s2].

        Declared 'REAL(KIND=vp)' in the scheme (turb_diffusion.f90:612), together with 'hdiv',
        'dwdx' and 'dwdy'; the default double build makes 'vp' equal to 'wp', so all four are
        read as 'wpfloat' like everything else.
        """
        return self._get_field("td_hdef2", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def hdiv(self):
        """Horizontal divergence, half levels [1/s]. 'REAL(vp)', see 'hdef2'."""
        return self._get_field("td_hdiv", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def dwdx(self):
        """Zonal derivative of the vertical wind, half levels [1/s]. 'REAL(vp)', see 'hdef2'."""
        return self._get_field("td_dwdx", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def dwdy(self):
        """Meridional derivative of the vertical wind, half levels [1/s]. 'REAL(vp)'."""
        return self._get_field("td_dwdy", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def tket_conv(self):
        """TKE forcing by convective buoyancy, half levels [m2/s3]."""
        return self._get_field("td_tket_conv", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def u_tens(self):
        """Zonal wind tendency at entry, main levels [m/s2]; see 'tketens' on the '_in' name."""
        return self._get_field("td_u_tens_in", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def v_tens(self):
        """Meridional wind tendency at entry, main levels [m/s2]."""
        return self._get_field("td_v_tens_in", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def t_tens(self):
        """Temperature tendency at entry, main levels [K/s]."""
        return self._get_field("td_t_tens_in", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def ut_sso(self):
        """Zonal wind tendency from the SSO scheme, main levels [m/s2]."""
        return self._get_field("td_ut_sso", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def vt_sso(self):
        """Meridional wind tendency from the SSO scheme, main levels [m/s2]."""
        return self._get_field("td_vt_sso", dims.CellDim, dims.KDim)


class IconTurbdiffExitSavepoint(IconTurbdiffSavepoint):
    """
    State at the end of SUB 'turbdiff' (turb_diffusion.f90:2549).

    The hook sits between the closing '!$ACC WAIT' and the '!$ACC END DATA' so that the
    '!$ACC CREATE' working set is still on the device: 'dicke', 'frh', 'frm', 'ftm', 'hlp',
    'shv', 'zaux', 'len_scale' and 'edr' are routine locals that never reach an interface, and
    they are serialized because without them a wrong value in a fused operator cannot be
    localised.

    'nvec', 'ke', 'ke1' and the control flags are only written by the entry savepoint; read
    them from there for the same ('date', 'id', 'block').
    """

    def rcld(self):
        """
        Standard deviation of the local super-saturation (SDSS), MAIN levels [-].

        Overrides the base accessor only to correct the staggering: 'solve_turb_budgets' leaves
        SDSS on half levels in section 3), and section 11) (turb_diffusion.f90:2559-2586)
        interpolates it back to main levels in place, which is what the exit savepoint sees.
        Verified against the archive: this differs from 'turbdiff-10-exit'.
        """
        return self._get_field("td_rcld", dims.CellDim, dims.KDim)

    def rhon(self):
        """
        Air density on half levels [kg/m3].

        The model-top level (index 0) is not written by 'turbdiff': it holds untouched memory
        even inside 'ivstart:ivend', around -0.025 in this capture. Masking to the horizontal
        window is not enough for this field.
        """
        return self._get_field("td_rhon", dims.CellDim, dims.KDim)

    def tketens(self):
        """TKE tendency on half levels at exit [m/s2]; the entry value is 'td_tketens_in'."""
        return self._get_field("td_tketens", dims.CellDim, dims.KDim)

    def dicke(self):
        """Layer thickness, later reused as a general work array, half levels [m]."""
        return self._get_field("td_dicke", dims.CellDim, dims.KDim)

    def frh(self):
        """Thermal forcing of the TKE equation (buoyancy production), half levels [1/s2]."""
        return self._get_field("td_frh", dims.CellDim, dims.KDim)

    def frm(self):
        """Dynamical forcing of the TKE equation (shear production), half levels [1/s2]."""
        return self._get_field("td_frm", dims.CellDim, dims.KDim)

    def ftm(self):
        """Dynamical forcing without the additional 3D-shear terms, half levels [1/s2]."""
        return self._get_field("td_ftm", dims.CellDim, dims.KDim)

    def hlp(self):
        """The scheme's general-purpose work array, half levels."""
        return self._get_field("td_hlp", dims.CellDim, dims.KDim)

    def shv(self):
        """Stability function of the scalar variables, half levels [-]."""
        return self._get_field("td_shv", dims.CellDim, dims.KDim)

    def zaux(self, component: int):
        """
        One component of the thermodynamic and vertical-diffusion scratch, on half levels.

        'zaux(nvec,ke1,ndim)' with 'ndim = 5' (turb_diffusion.f90:790) is ONE-based, unlike
        'zvari', so 'component' here is zero-based: 'component = i' is Fortran 'zaux(:,:,i+1)'.

        The array is reused. Sections 0) and 1) put 'exner', 'r_cpd', 'qst_t', 'g_tet' and
        'g_h2o' into 1..5 (turb_diffusion.f90:991-995, :1021-1023); the vertical diffusion
        later overwrites them with 'upd_prof', 'sav_prof', 'expl_mom' (:2095-2102) and
        'impl_mom', 'invs_mom' (:2368-2369), which is what this savepoint sees.
        """
        return self._get_field_component("td_zaux", component, (dims.CellDim, dims.KDim))

    def len_scale(self):
        """
        Raw contents of the 'len_scale' storage at the end of 'turbdiff', half levels.

        NOT A LENGTH SCALE. Section 9) points 'eff_flux' at this array
        (turb_diffusion.f90:2431) and 'calc_impl_vert_diff' fills it with the effective TKE
        flux densities, positive downward, of the semi-implicit vertical diffusion; nothing
        restores the length scale afterwards. Verified against the archive: the exit values are
        byte-identical to 'turbdiff-9-exit' and differ from 'turbdiff-8-exit', and the range
        grows from 1.2e-5..118 m (a plausible master length scale) to 3.3e-5..2165.

        Use 'eff_tke_flux()' for the quantity this actually is. For the mixing length, read
        'mixing_length()' from a section savepoint up to 'turbdiff-8-exit'.

        Serialized from the 'len_scale' pointer (turb_diffusion.f90:691, assigned :854-856),
        which targets either the caller's 'tur_len_scale' or the routine-local 'len_scale_tar';
        the latter here, because mo_nwp_turbdiff_interface.f90:309-315 nullifies the pointer
        unless 'ldiagnose_tke' is set.
        """
        return self._get_field("td_len_scale", dims.CellDim, dims.KDim)

    def eff_tke_flux(self):
        """
        Effective TKE flux density of the semi-implicit vertical diffusion, positive downward,
        half levels [m3/s3 * kg/m3]; the same storage as 'len_scale()', correctly named.
        """
        return self._get_field("td_len_scale", dims.CellDim, dims.KDim)

    def edr(self):
        """
        Eddy dissipation rate, half levels [m2/s3].

        Serialized from the 'ediss' pointer (turb_diffusion.f90:690, assigned :848-850), which
        targets either the caller's 'edr' or the routine-local 'diss_tar'; named 'td_edr'
        after the ICON interface variable.

        IN THIS CAPTURE IT IS NEVER WRITTEN. mo_nwp_turbdiff_interface.f90:309-315 sets
        'edr_ptr => NULL()' unless 'ldiagnose_tke', so 'ediss' targets the routine-local
        automatic array 'diss_tar', and 'solve_turb_budgets' fills it only under
        'lpres_edr .OR. ltmpcor' (turb_utilities.f90:1830) with
        'lpres_edr = lsrfshear .OR. ASSOCIATED(edr)' -- both false here. What is serialized is
        therefore uninitialised stack memory: byte-identical at all fifteen section savepoints
        and at the exit, different between timesteps, ranging over -37.8..1.2 with an exact
        zero at the model top -- values a positive-definite dissipation rate cannot take. Do
        not compare a Python EDR against it; re-capture with 'ldiagnose_tke = .TRUE.' instead.
        """
        return self._get_field("td_edr", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def ldoexpcor(self) -> bool | None:
        """Explicit correction of the vertical diffusion is applied."""
        return self._read_scalar("td_ldoexpcor")

    @IconSavepoint.optionally_registered()
    def ldocirflx(self) -> bool | None:
        """The circulation-term heat flux is applied."""
        return self._read_scalar("td_ldocirflx")

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def u_tens(self):
        """Zonal wind tendency at exit, main levels [m/s2]."""
        return self._get_field("td_u_tens", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def v_tens(self):
        """Meridional wind tendency at exit, main levels [m/s2]."""
        return self._get_field("td_v_tens", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def t_tens(self):
        """Temperature tendency at exit, main levels [K/s]."""
        return self._get_field("td_t_tens", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def tket_hshr(self):
        """TKE forcing by horizontal shear, half levels [m2/s3]."""
        return self._get_field("td_tket_hshr", dims.CellDim, dims.KDim)


#: The fifteen numbered sections of SUB 'turbdiff' (turb_diffusion.f90), in execution order.
#: 'turbdiff-<label>-exit' is simultaneously the exit of section <label> and the entry of the
#: next one; the exit of the last section 11) is 'turbdiff-exit'.
TURBDIFF_SECTIONS: Final[tuple[str, ...]] = (
    "0",  # conserved variables, cloud cover, thermodynamic factors, turbulent length scales
    "1a",  # vertical gradients
    "1b",  # TKE forcing functions
    "1c",  # initialisation of tke/tkvh/tkvm
    "2a",  # 3D shear complements and shear by the non-turbulent sub-grid flow
    "2b",  # roughness-layer corrections
    "2c",  # final preparations before the turbulence model
    "3",  # turbulent budgets (SUB 'solve_turb_budgets')
    "4",  # effective diffusion coefficients
    "5",  # TKE-source temperature tendencies
    "6",  # q-diffusion tendency and the circulation term
    "7",  # theta gradient of the circulation heat flux
    "8",  # circulation term as an additional TKE flux density
    "9",  # TKE profile update (semi-implicit vertical diffusion)
    "10",  # q tendencies
)


class IconTurbdiffSectionSavepoint(IconTurbulenceSavepoint):
    """
    State at the boundary between two of Raschendorfer's numbered sections of SUB 'turbdiff',
    written by 'serialize_turbdiff_section' in mo_icon4py_verification.f90 (icon f6803c9fae).

    One class for all fifteen boundaries, because the Fortran is one routine for all fifteen:
    the section label picks the savepoint name and every savepoint carries the same superset of
    40 possible fields (31 written for sections 0)..3), 33 from 4) on, once 'ldoexpcor' and
    'ldocirflx' are defined). The label is available as 'self.section'.

    WHY THE ACCESSORS ARE NOT NAMED AFTER THE FORTRAN VARIABLES
    -----------------------------------------------------------
    'turbdiff' reuses its working arrays for unrelated quantities as it proceeds, so a single
    accessor name spanning all fifteen sections would be a trap: it would return a physically
    different field depending on which savepoint you happened to load, and the comparison would
    fail looking exactly like a physics bug. Every reused slot therefore has one accessor per
    role, and each one refuses to answer outside the sections where its role holds:

        storage      accessor                     sections    quantity
        -----------  ---------------------------  ----------  ------------------------------
        len_scale    mixing_length()              0..8        turbulent master length scale
        len_scale    eff_tke_flux()               9,10        effective TKE flux (pos. down)
        dicke        layer_depth()                0           hhl(k)-hhl(k+1)
        dicke        disc_mom()                   1a..10      rho_n * dz / dt_tke
        rcld         cloud_cover()                0..2c       saturation fraction
        rcld         sdss()                       3..10       std. dev. of local supersat.
        frh          thermal_forcing()            1b..5       buoyancy production
        frh          cke_flux_density()           6..8        scaled CKE flux density
        frh          invs_fac()                   9,10        inversion factor of the solver
        frm          mech_forcing()               1b..5       shear production
        frm          cke_flux_at_main_levels()    6..8        interpolated CKE flux density
        tkvm/tkvh    tkvm()/tkvh()                all but 2c  diffusion coefficients [m2/s]
        tkvm/tkvh    stab_len_m()/stab_len_h()    2c          stability length scales [m]
        zaux(:,:,1)  exner_factor()               0..8        Exner factor
        zaux(:,:,1)  upd_prof()                   9,10        updated TKE profile
        zaux(:,:,2)  r_cpd()                      0..5        c_p/c_pd
        zaux(:,:,2)  sav_prof()                   6..10       saved TKE profile
        zaux(:,:,3)  dqsat_dt()                   0..5        dQ_sat/dT
        zaux(:,:,3)  expl_mom()                   6..10       explicit diffusion momentum
        zaux(:,:,4)  g_tet_l()                    0..8        buoyancy factor for tet_l
        zaux(:,:,4)  impl_mom()                   9,10        implicit diffusion momentum
        zaux(:,:,5)  g_h2o()                      0..8        buoyancy factor for h2o_g
        zaux(:,:,5)  invs_mom()                   9,10        inverted diffusion momentum
        zvari        conserved_variable(n)        0           quasi-conserved model variables
        zvari        vertical_gradient(n)         1a..2c      vertical gradients
        zvari        effective_gradient(n)        3..10       effective gradients
        tfm/tfh      tfm()/tfh()                  0..2c       Prandtl-layer resistance fraction
        tfv          tfv()                        0..2b       laminar reduction for vapour

    Three slots have a role that this reader refuses to name at all, because whether the
    rewrite happened is decided by a namelist switch that is not serialized. They raise, with
    the alternatives spelled out, and 'raw_field()' is the escape hatch:

      * 'tfm'/'tfh' from section 3) on. Section 3) saves the un-limited 'tkvm(:,ke)'/'tkvh(:,ke)'
        into them and section 4) replaces those with the LLDC drag-reduction factor and the LLDC
        shear-forcing -- but only under 'lsrfshear', which is neither serialized nor derivable.
        In this capture 'lsrfshear' is false and both slots keep their Prandtl-layer values
        unchanged through the whole scheme (asserted by the datatests); a different namelist
        would silently make them something else.
      * 'tfv' from section 2c) on, overwritten with the NTC shear-forcing only under
        "rsur_sher > 0" (false here).
      * 'frm' at sections 9) and 10). 'prep_impl_vert_diff' would overwrite it with 'scal_fac',
        but only under 'lprecondi' (turb_data 'lprecnd'), which is off here -- the values are
        still section 6)'s interpolated CKE flux density.

    'hlp' is genuine scratch with no stable meaning at all; 'hlp()' hands it over with a
    per-section table in its docstring rather than pretending otherwise.

    NOT WRITTEN ANYWHERE IN THIS CONFIGURATION, verified field by field over the fifteen
    savepoints: 'edr' (see 'edr()'), 'ftm', 'shv', 'tfm', 'tfh', 'tfv', 'tprn', 'u_tens',
    'v_tens' and 't_tens'. They are serialized because a different namelist writes them.

    A savepoint is selected by ('date', 'id', 'block'), as for 'turbdiff-entry'. The sizes
    ('nvec', 'ke', 'ke1') and the control flags are only in 'turbdiff-entry'; read them from
    there for the same ('date', 'id', 'block').
    """

    _prefix: ClassVar[str] = "td_"

    def __init__(
        self,
        sp: serialbox.Savepoint,
        ser: serialbox.Serializer,
        *,
        section: str,
        size: dict,
        backend: gtx_typing.Backend | None,
    ):
        super().__init__(sp, ser, size, backend)
        self._section = section

    @property
    def section(self) -> str:
        """The section label this savepoint is the exit of; one of 'TURBDIFF_SECTIONS'."""
        return self._section

    def _require(self, accessor: str, at: tuple[str, ...], because: str) -> None:
        if self._section not in at:
            raise ValueError(
                f"{accessor}() is the meaning of this storage at turbdiff sections "
                f"{', '.join(at)} only; this savepoint is 'turbdiff-{self._section}-exit', "
                f"where {because}"
            )

    # ------------------------------------------------------------------ control -------------

    def nvor(self) -> int:
        """
        Time index of the 'tke' the security iteration started from.

        Advanced by the iteration in section 3) ('nvor = ntur' at turb_diffusion.f90:1861), so
        it is serialized at every boundary rather than only at the entry. With 'it_end == 1',
        as in this capture, it never moves.
        """
        return self._read_scalar("td_nvor")

    @IconSavepoint.optionally_registered()
    def ldoexpcor(self) -> bool | None:
        """
        Explicit correction of the vertical diffusion for warm-cloud effects is applied.

        Set at the end of section 4), so it is absent from the savepoints of sections 0)..3):
        passing an undefined 'INTENT(OUT)' dummy on would be a standard violation, so the
        earlier hooks simply leave the two flags out. 'None' there.
        """
        return self._read_scalar("td_ldoexpcor")

    @IconSavepoint.optionally_registered()
    def ldocirflx(self) -> bool | None:
        """The circulation-term heat flux is applied; see 'ldoexpcor()' on the missing sections."""
        return self._read_scalar("td_ldocirflx")

    # ------------------------------------------------------- unambiguous half-level state ---

    def tke(self):
        """
        'q = SQRT(2*TKE)', the turbulent velocity and not the energy, on half levels [m/s].

        Written by section 3) and possibly modified by section 4) ('ltkeadapt' or
        'imode_tkemini == 3'; neither here). Serialized as '(nvec, ke1, ntim)' with
        'ntim == 1' for the NWP interface, so the trailing axis is squeezed away.
        """
        return self._get_field("td_tke", dims.CellDim, dims.KDim)

    def rhon(self):
        """
        Air density on half levels [kg/m3], written by section 0).

        The model-top level (index 0) is never written, even inside 'ivstart:ivend':
        'bound_level_interp' runs over 'k=2..ke' and 'adjust_satur_equil' supplies 'ke1'.
        """
        return self._get_field("td_rhon", dims.CellDim, dims.KDim)

    def tprn(self):
        """
        Turbulent Prandtl number on half levels [-].

        Beware: in a capture like exp.mch_icon-ch2_small this is a '(1, 1)' dummy and not a
        cell field, because 'prm_diag%tprn' is only given the full shape when the namelist
        selects a TMod that produces it (mo_nwp_phy_state.f90:4706-4709). Check
        '.ndarray.shape' against 'ke1()' from 'turbdiff-entry' before comparing anything.
        """
        buffer = self.xp.asarray(self.serializer.read("td_tprn", self.savepoint), dtype=float)
        return self._get_field_from_ndarray(buffer, dims.CellDim, dims.KDim)

    def tketens(self):
        """
        A q-tendency on half levels [m/s2] -- but not the same one throughout.

        Up to and including section 9) this is the entry value, the advection tendency that
        'turbdiff' receives as 'INTENT(INOUT)' and hands to 'solve_turb_budgets' as 'tvt'.
        Section 10) overwrites it with the diffusion tendency of 'q' that is the scheme's
        output. Both are [m/s2]; only the section tells them apart.
        """
        return self._get_field("td_tketens", dims.CellDim, dims.KDim)

    def edr(self):
        """
        Raw contents of the 'ediss' storage, half levels [m2/s3] -- NOT an eddy dissipation rate
        in this capture.

        mo_nwp_turbdiff_interface.f90:309-315 sets 'edr_ptr => NULL()' unless 'ldiagnose_tke',
        so 'ediss' targets the routine-local automatic array 'diss_tar', and
        'solve_turb_budgets' fills it only under 'lpres_edr .OR. ltmpcor'
        (turb_utilities.f90:1830), both false here. Verified against the archive: byte-identical
        at all fifteen section savepoints and at 'turbdiff-exit', different between timesteps,
        ranging over -37.8..1.2 -- values the positive-definite 'q**3/(d_m*l)' of
        turb_utilities.f90:1846 cannot take. Re-capture with 'ldiagnose_tke = .TRUE.' if you
        need the EDR.
        """
        return self._get_field("td_edr", dims.CellDim, dims.KDim)

    def ftm(self):
        """
        Mechanical forcing by the traditional (pure mean) shear, half levels [1/s2].

        Section 2a) saves 'frm' here before the non-turbulent contributions are added, but only
        under 'lssintact .OR. loutbms' (all levels) or "rsur_sher > 0" (level 'ke' only). None
        of the three holds in this capture, so the array is untouched memory at every section --
        verified byte-identical across all fifteen.
        """
        return self._get_field("td_ftm", dims.CellDim, dims.KDim)

    def shv(self):
        """
        TKE source by the raw circulation term, half levels [m2/s3].

        Written by section 6) under 'lcirflx .OR. loutthcrc'; undefined before section 6) in any
        case, and untouched in this capture ('lcirflx' is off and 'tket_nstc' is not passed, so
        'loutthcrc' is false -- which is why 'td_tket_nstc' is absent from the savepoint).
        """
        self._require(
            "shv", TURBDIFF_SECTIONS[TURBDIFF_SECTIONS.index("6") :], "it is not yet written"
        )
        return self._get_field("td_shv", dims.CellDim, dims.KDim)

    def hlp(self):
        """
        The scheme's general-purpose scratch array, half levels -- no stable meaning.

        Unlike the other reused slots this one is scratch by design (turb_diffusion.f90:781,
        "any 'help' variable"), so there is nothing to name it after. What is in it, by section:

            0        'auxil' of 'bound_level_interp'
            1a..1c   1 / ((hhl(k-1) - hhl(k+1)) / 2), a reciprocal layer depth [1/m]
            2a       TKE source by the separated horizontal shear mode [m2/s3], or
                     'ut_sso*u + vt_sso*v' on MAIN levels if the SSO block ran afterwards
            2b       wind speed on main levels, if the roughness layer is resolved (dead here)
            2c       vertical temperature gradient, under 'ltmpcor .AND. lcpfluc' (off here)
            5        length-scale-scaled temperature tendency under 'ltmpcor', or
                     |frh*tkvh| under 'ldocirflx' (both off here)
            8        'cur_prof', the virtual TKE profile whose diffusion tendency carries the
                     circulation term, under 'lcircterm' (on here)
            6,7,9,10 unchanged from the last section that wrote it

        Verified against the archive: it changes at 1a), 2a) and 8) and nowhere else.
        """
        return self._get_field("td_hlp", dims.CellDim, dims.KDim)

    # ------------------------------------------------- section-2a diagnostics (main levels) --

    def hor_scale(self):
        """
        Effective horizontal length scale of the separated horizontal shear mode, main
        levels [m]; written by section 2a).
        """
        self._require(
            "hor_scale",
            TURBDIFF_SECTIONS[TURBDIFF_SECTIONS.index("2a") :],
            "it is not yet written",
        )
        return self._get_field("td_hor_scale", dims.CellDim, dims.KDim)

    def xri(self):
        """
        '1/Ri**(2/3)', main levels [-]; written by section 2a) and consumed by section 4) for
        the Richardson-number-dependent minimum diffusion coefficients.
        """
        self._require(
            "xri", TURBDIFF_SECTIONS[TURBDIFF_SECTIONS.index("2a") :], "it is not yet written"
        )
        return self._get_field("td_xri", dims.CellDim, dims.KDim)

    def layr(self):
        """
        Uncorrected effective horizontal length scale 'a_hshr*akt/2 * l_hori' [m], one value per
        column; written by section 2a).
        """
        self._require(
            "layr", TURBDIFF_SECTIONS[TURBDIFF_SECTIONS.index("2a") :], "it is not yet written"
        )
        return self._get_field("td_layr", dims.CellDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def tket_hshr(self):
        """TKE forcing by the separated horizontal shear mode, half levels [m2/s3]; section 2a)."""
        return self._get_field("td_tket_hshr", dims.CellDim, dims.KDim)

    def lays(self, component: int):
        """
        One of the two surface-transfer ratios written by section 1a), one value per column
        [1/m]: 'component=0' is 'tvm/(tkvm(:,ke1)*tfm)' for momentum ('mom=1' in Fortran),
        'component=1' is 'tvh/(tkvh(:,ke1)*tfh)' for scalars ('sca=2').

        They convert the Prandtl-layer difference into the surface gradient of 'zvari'.
        """
        if component not in (0, 1):
            raise IndexError(
                f"'lays' has two components, 0 (momentum) and 1 (scalars): {component}"
            )
        self._require(
            "lays", TURBDIFF_SECTIONS[TURBDIFF_SECTIONS.index("1a") :], "it is not yet written"
        )
        return self._get_field("td_lays", dims.CellDim, slice_=(slice(None), component))

    # ------------------------------------------------------------- tendencies (main levels) --

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def u_tens(self):
        """
        Zonal wind tendency, main levels [m/s2].

        'turbdiff' only touches it in section 2b), the roughness-layer form drag, which is dead
        here ('c_big'/'c_sml' are not passed and 'kcm = ke+1'). Verified unchanged across all
        fifteen sections.
        """
        return self._get_field("td_u_tens", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def v_tens(self):
        """Meridional wind tendency, main levels [m/s2]; see 'u_tens()'."""
        return self._get_field("td_v_tens", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def t_tens(self):
        """
        Temperature tendency, main levels [K/s].

        'turbdiff' adds to it in section 2b) (dead here) and section 5), the dissipative heating
        and phase-diffusion term, which runs only under 'ltmpcor' (off here). Verified unchanged
        across all fifteen sections.
        """
        return self._get_field("td_t_tens", dims.CellDim, dims.KDim)

    # ------------------------------------------------------------------ diffusion coefficients

    def tkvm(self):
        """
        Turbulent diffusion coefficient for momentum, half levels [m2/s].

        Section 2c) divides it by 'tke' to obtain the stability length scale that the turbulence
        model wants, and section 3) multiplies it back; 'turbdiff-2c-exit' is therefore the one
        savepoint where this array is a LENGTH in metres, and 'stab_len_m()' reads it there.
        Section 4) applies the lower limits.
        """
        self._require(
            "tkvm",
            tuple(s for s in TURBDIFF_SECTIONS if s != "2c"),
            "section 2c) has divided it by 'tke' and it is a stability length scale; use "
            "stab_len_m()",
        )
        return self._get_field("td_tkvm", dims.CellDim, dims.KDim)

    def tkvh(self):
        """Turbulent diffusion coefficient for heat and other scalars, half levels [m2/s]."""
        self._require(
            "tkvh",
            tuple(s for s in TURBDIFF_SECTIONS if s != "2c"),
            "section 2c) has divided it by 'tke' and it is a stability length scale; use "
            "stab_len_h()",
        )
        return self._get_field("td_tkvh", dims.CellDim, dims.KDim)

    def stab_len_m(self):
        """
        Turbulent stability length scale for momentum, 'ls_m = S_m * len_scale', half levels [m].

        The same storage as 'tkvm()', divided by 'tke' at turb_diffusion.f90:1720-1724 as the
        input the turbulence model expects, and multiplied back at :1868-1875.
        """
        self._require("stab_len_m", ("2c",), "'tkvm' is a diffusion coefficient; use tkvm()")
        return self._get_field("td_tkvm", dims.CellDim, dims.KDim)

    def stab_len_h(self):
        """Turbulent stability length scale for scalars, half levels [m]; see 'stab_len_m()'."""
        self._require("stab_len_h", ("2c",), "'tkvh' is a diffusion coefficient; use tkvh()")
        return self._get_field("td_tkvh", dims.CellDim, dims.KDim)

    # ---------------------------------------------------------------------- the 'len_scale' slot

    def mixing_length(self):
        """
        Turbulent master length scale, half levels [m]; written by section 0) and adjusted by
        'solve_turb_budgets' in section 3).

        Section 9) points 'eff_flux' at this array and destroys the length scale, so this
        accessor stops at 'turbdiff-8-exit'.
        """
        self._require(
            "mixing_length",
            TURBDIFF_SECTIONS[: TURBDIFF_SECTIONS.index("9")],
            "section 9) has aliased the storage as 'eff_flux'; use eff_tke_flux()",
        )
        return self._get_field("td_len_scale", dims.CellDim, dims.KDim)

    def eff_tke_flux(self):
        """
        Effective TKE flux density of the semi-implicit vertical diffusion, positive downward,
        half levels; written into the 'len_scale' storage by 'calc_impl_vert_diff'
        (turb_diffusion.f90:2431-2443).

        Not a length scale, despite the serialized name. Note the staggering shift the diffusion
        routines impose: 'eff_flux(:,k)' refers to the (k-1)-th main level
        (turb_diffusion.f90:2470-2472).
        """
        self._require(
            "eff_tke_flux",
            ("9", "10"),
            "section 9) has not aliased the storage yet and it is still the master length "
            "scale; use mixing_length()",
        )
        return self._get_field("td_len_scale", dims.CellDim, dims.KDim)

    # -------------------------------------------------------------------------- the 'dicke' slot

    def layer_depth(self):
        """Depth of the model layers, 'hhl(k) - hhl(k+1)', main levels [m]; written by section 0)."""
        self._require(
            "layer_depth",
            ("0",),
            "section 1a) has overwritten it with the discretisation momentum; use disc_mom()",
        )
        return self._get_field("td_dicke", dims.CellDim, dims.KDim)

    def disc_mom(self):
        """
        Discretisation momentum of the TKE diffusion, 'rho_n * dz / dt_tke', half levels
        [kg/m2/s]; written by section 1a) over the layer depth.

        At 'imode_tkediff == 1' (diffusion in terms of 'q' rather than TKE) section 6)
        multiplies it by 'q' as well, so that it becomes the momentum of the q-equation
        (turb_diffusion.f90:2245); this capture runs 'imode_tkediff == 2' and the extra factor
        is absent, verified byte-identical between 'turbdiff-5-exit' and 'turbdiff-6-exit'.
        """
        self._require(
            "disc_mom", TURBDIFF_SECTIONS[1:], "it is still the layer depth; use layer_depth()"
        )
        return self._get_field("td_dicke", dims.CellDim, dims.KDim)

    # --------------------------------------------------------------------------- the 'rcld' slot

    def cloud_cover(self):
        """
        Saturation fraction (cloud cover) [-], as 'adjust_satur_equil' leaves it in section 0).

        Main levels, except that the 'bound_level_interp' call at turb_diffusion.f90:1071-1073
        interpolates levels 2..ke to half levels in place -- the same array is both 'bl' and
        'ml' there, which is what 'solve_turb_budgets' wants as input.
        """
        self._require(
            "cloud_cover",
            TURBDIFF_SECTIONS[: TURBDIFF_SECTIONS.index("3")],
            "section 3) has overwritten it with the standard deviation of the local "
            "super-saturation; use sdss()",
        )
        return self._get_field("td_rcld", dims.CellDim, dims.KDim)

    def sdss(self):
        """
        Standard deviation of the local super-saturation, half levels [-]; the output of
        'solve_turb_budgets' in section 3), which overwrites the cloud cover it was given.

        Section 11), after the last section savepoint, interpolates it back to main levels; that
        is what 'turbdiff-exit' holds.
        """
        self._require(
            "sdss",
            TURBDIFF_SECTIONS[TURBDIFF_SECTIONS.index("3") :],
            "'rcld' is still the cloud cover; use cloud_cover()",
        )
        return self._get_field("td_rcld", dims.CellDim, dims.KDim)

    # ---------------------------------------------------------------------------- the 'frh' slot

    def thermal_forcing(self):
        """
        Thermal forcing of the TKE equation (buoyancy production), half levels [1/s2];
        'zaux(:,:,4)*d_z(tet_l) + zaux(:,:,5)*d_z(h2o_g)', written by section 1b) and smoothed
        in place by section 2c).

        Level 'ke1' is the acceleration of the near-surface non-turbulent circulations rather
        than a forcing.
        """
        self._require(
            "thermal_forcing",
            ("1b", "1c", "2a", "2b", "2c", "3", "4", "5"),
            "it is either not yet written (0, 1a) or section 6) has overwritten it; see "
            "cke_flux_density() and invs_fac()",
        )
        return self._get_field("td_frh", dims.CellDim, dims.KDim)

    def cke_flux_density(self):
        """
        Flux density of circulation kinetic energy by the near-surface thermal inhomogeneity,
        scaled by the length scale, half levels; 'rho_n*tkvh*prss*len_scale' written by
        section 6) under 'lcircterm .OR. loutthcrc'.

        'frh/len_scale' is the TKE flux density in [kg/s3]; the scaling is what makes the linear
        interpolation onto main levels in 'cke_flux_at_main_levels()' legitimate.
        """
        self._require(
            "cke_flux_density",
            ("6", "7", "8"),
            "'frh' is either the thermal forcing (before 6) or the solver's inversion factor "
            "(from 9); see thermal_forcing() and invs_fac()",
        )
        return self._get_field("td_frh", dims.CellDim, dims.KDim)

    def invs_fac(self):
        """
        'invs_fac', the inversion factor of the tridiagonal TKE solve, half levels [-]; written
        into the 'frh' storage by 'prep_impl_vert_diff' in section 9).
        """
        self._require(
            "invs_fac",
            ("9", "10"),
            "'frh' has not been handed to the solver yet; see thermal_forcing() and "
            "cke_flux_density()",
        )
        return self._get_field("td_frh", dims.CellDim, dims.KDim)

    # ---------------------------------------------------------------------------- the 'frm' slot

    def mech_forcing(self):
        """
        Mechanical forcing of the TKE equation (shear production), half levels [1/s2].

        Section 1b) puts the pure single-column vertical shear here; section 2a) adds the 3D
        shear complements, the separated horizontal shear mode, the SSO wake production and the
        convective circulation; section 2c) smooths it vertically in place.
        """
        self._require(
            "mech_forcing",
            ("1b", "1c", "2a", "2b", "2c", "3", "4", "5"),
            "it is either not yet written (0, 1a) or section 6) has overwritten it; see "
            "cke_flux_at_main_levels()",
        )
        return self._get_field("td_frm", dims.CellDim, dims.KDim)

    def cke_flux_at_main_levels(self):
        """
        The CKE flux density of 'cke_flux_density()' interpolated onto main levels, written by
        section 6) under 'lcircterm .OR. loutthcrc'.

        Stops at section 8) on purpose: at section 9) 'prep_impl_vert_diff' may overwrite this
        storage with 'scal_fac' -- see the class docstring.
        """
        self._require(
            "cke_flux_at_main_levels",
            ("6", "7", "8"),
            "'frm' is either still the mechanical forcing or, from 9), ambiguous between this "
            "and the solver's 'scal_fac'",
        )
        return self._get_field("td_frm", dims.CellDim, dims.KDim)

    # -------------------------------------------------------------- the Prandtl-layer scalars --

    def tfm(self):
        """
        Prandtl-layer resistance fraction for momentum [-]: the laminar-layer part of the
        transfer resistance, as 'turbtran' left it.
        """
        self._require(
            "tfm",
            TURBDIFF_SECTIONS[: TURBDIFF_SECTIONS.index("3")],
            "section 3) may have saved the un-limited 'tkvm(:,ke)' here and section 4) the LLDC "
            "drag-reduction factor, both only under 'lsrfshear'",
        )
        return self._get_field("td_tfm", dims.CellDim)

    def tfh(self):
        """Prandtl-layer resistance fraction for scalars [-]; see 'tfm()'."""
        self._require(
            "tfh",
            TURBDIFF_SECTIONS[: TURBDIFF_SECTIONS.index("3")],
            "section 3) may have saved the un-limited 'tkvh(:,ke)' here and section 4) the LLDC "
            "shear-forcing, both only under 'lsrfshear'",
        )
        return self._get_field("td_tfh", dims.CellDim)

    def tfv(self):
        """
        Laminar reduction factor for water vapour compared to heat [-], as 'turbtran' left it.

        Section 2c) overwrites it with 'frm(:,ke) - ftm(:,ke)', the shear forcing by the
        non-turbulent circulations, under "rsur_sher > 0" -- off in this capture, where the
        field is exactly 1.0 everywhere and never changes.
        """
        self._require(
            "tfv",
            TURBDIFF_SECTIONS[: TURBDIFF_SECTIONS.index("2c")],
            'section 2c) may have overwritten it with the NTC shear-forcing, under "rsur_sher > 0"',
        )
        return self._get_field("td_tfv", dims.CellDim)

    # --------------------------------------------------------------------------- the 'zaux' slots

    def raw_zaux(self, component: int):
        """
        One component of the 'zaux' storage, half levels, uninterpreted.

        'zaux(nvec,ke1,ndim)' with 'ndim = 5' is ONE-based in Fortran, so 'component' here is
        zero-based: 'component = i' is Fortran 'zaux(:,:,i+1)'. Prefer the named accessors --
        'exner_factor()', 'r_cpd()', 'dqsat_dt()', 'g_tet_l()', 'g_h2o()' and their post-section-6
        replacements -- which refuse to answer where the meaning does not hold.
        """
        if not 0 <= component < 5:
            raise IndexError(f"'zaux' has five components, 0..4: {component}")
        return self._get_field_component("td_zaux", component, (dims.CellDim, dims.KDim))

    def _zaux_before_the_solver(
        self, accessor: str, component: int, at: tuple[str, ...], role: str
    ):
        self._require(accessor, at, f"the vertical diffusion has overwritten it with {role}")
        if self._section in ("6", "7", "8") and (
            self.ldocirflx() or self._has_field("td_tket_nstc")
        ):
            raise ValueError(
                f"{accessor}() cannot be trusted at 'turbdiff-{self._section}-exit' of this "
                "capture: section 6) points 'upd_prof' at 'zaux(:,:,1)' and fills it with the "
                "explicit circulation increments under 'lcirflx .OR. loutthcrc', and neither "
                "can be ruled out here -- 'td_ldocirflx' is True, or 'tket_nstc' is passed so "
                "that 'loutthcrc' may hold. Use raw_zaux(0) if you have settled which."
            )
        return self.raw_zaux(component)

    def exner_factor(self):
        """
        Exner factor on half levels [-], 'zaux(:,:,1)'; written by section 0) via
        'adjust_satur_equil' and 'bound_level_interp'.

        Section 6) points 'upd_prof' at this storage but only writes it under
        'lcirflx .OR. loutthcrc'; this accessor refuses whenever 'td_ldocirflx' says the former
        held. 'loutthcrc' additionally requires 'tket_nstc' to be passed, which is why its
        absence from the field list settles that half.
        """
        return self._zaux_before_the_solver(
            "exner_factor",
            0,
            TURBDIFF_SECTIONS[: TURBDIFF_SECTIONS.index("9")],
            "'upd_prof', the updated TKE profile",
        )

    def upd_prof(self):
        """
        Updated TKE profile after the semi-implicit diffusion, half levels; 'zaux(:,:,1)' as
        'calc_impl_vert_diff' leaves it in section 9), plus the roughness-layer volume term.

        A TKE profile at 'imode_tkediff == 2' (this capture) and a 'q' profile at 1.
        """
        self._require("upd_prof", ("9", "10"), "section 9) has not run yet; use exner_factor()")
        return self.raw_zaux(0)

    def r_cpd(self):
        """
        'c_p/c_pd', the specific-heat ratio of moist air, half levels [-]; 'zaux(:,:,2)' from
        section 0). Only interpolated to half levels under 'lcpfluc'.
        """
        return self._zaux_before_the_solver(
            "r_cpd",
            1,
            TURBDIFF_SECTIONS[: TURBDIFF_SECTIONS.index("6")],
            "'sav_prof', the saved TKE profile",
        )

    def sav_prof(self):
        """
        Saved TKE profile before the diffusion, half levels; 'zaux(:,:,2)' from section 6),
        '0.5*q**2' at 'imode_tkediff == 2' and 'q' at 1.

        Written inside 'IF (ldotkedif .OR. lcircterm)'. The presence of 'turbdiff-9-exit' in the
        archive proves that branch was taken.
        """
        self._require(
            "sav_prof",
            TURBDIFF_SECTIONS[TURBDIFF_SECTIONS.index("6") :],
            "section 6) has not run yet; use r_cpd()",
        )
        return self.raw_zaux(1)

    def dqsat_dt(self):
        """'dQ_sat/dT' on half levels [1/K]; 'zaux(:,:,3)' from section 0)."""
        return self._zaux_before_the_solver(
            "dqsat_dt",
            2,
            TURBDIFF_SECTIONS[: TURBDIFF_SECTIONS.index("6")],
            "'expl_mom', the explicit part of the diffusion momentum",
        )

    def expl_mom(self):
        """
        Explicit part of the TKE diffusion momentum, half levels; 'zaux(:,:,3)' from section 6),
        'rho_h * (K(k-1)+K(k))/2 / (hhl(k-1)-hhl(k))'.

        It refers to the FLUX levels, which for the TKE diffusion are the main levels: a flux
        level with a given index sits above the variable level with the same index
        (turb_diffusion.f90:2178-2181).
        """
        self._require(
            "expl_mom",
            TURBDIFF_SECTIONS[TURBDIFF_SECTIONS.index("6") :],
            "section 6) has not run yet; use dqsat_dt()",
        )
        return self.raw_zaux(2)

    def g_tet_l(self):
        """
        Buoyancy factor of the liquid-water potential temperature, half levels [m/s2 per K];
        'zaux(:,:,4)' from section 0), the 'g_tet' output of 'adjust_satur_equil'.
        """
        return self._zaux_before_the_solver(
            "g_tet_l",
            3,
            TURBDIFF_SECTIONS[: TURBDIFF_SECTIONS.index("9")],
            "'impl_mom', the implicit part of the diffusion momentum",
        )

    def impl_mom(self):
        """
        Implicit part of the TKE diffusion momentum, half levels; 'zaux(:,:,4)' as
        'prep_impl_vert_diff' leaves it in section 9).
        """
        self._require("impl_mom", ("9", "10"), "section 9) has not run yet; use g_tet_l()")
        return self.raw_zaux(3)

    def g_h2o(self):
        """
        Buoyancy factor of the total water content, half levels [m/s2 per (kg/kg)];
        'zaux(:,:,5)' from section 0), the 'g_h2o' output of 'adjust_satur_equil'.
        """
        return self._zaux_before_the_solver(
            "g_h2o",
            4,
            TURBDIFF_SECTIONS[: TURBDIFF_SECTIONS.index("9")],
            "'invs_mom', the inverted diffusion momentum",
        )

    def invs_mom(self):
        """
        Inverted diffusion momentum of the tridiagonal TKE solve, half levels; 'zaux(:,:,5)' as
        'prep_impl_vert_diff' leaves it in section 9).
        """
        self._require("invs_mom", ("9", "10"), "section 9) has not run yet; use g_h2o()")
        return self.raw_zaux(4)

    # -------------------------------------------------------------------------- the 'zvari' slot

    def raw_zvari(self, component: int):
        """
        One component of the 'zvari' storage, uninterpreted.

        'zvari(:,:,0:ndim)' with 'ndim = 5' has a ZERO lower bound in 'turbdiff', so 'component'
        is the Fortran third index unchanged, 0..5 -- unlike 'raw_zaux()'. The serializer's own
        dummy is rebased to 1:6 by the assumed-shape interface, which changes nothing about the
        buffer: serialized index 1 is Fortran index 0, which is Python index 0. Verified against
        the archive: 'zvari(1)' at 'turbdiff-0-exit' is byte-identical to 'u' at
        'turbdiff-entry', and 'zvari(2)' to 'v'.

        Component indices (mo_turbdiff_config.f90:62-77): 'u_m=1', 'v_m=2',
        'tem=tet=tet_l=3', 'vap=h2o_g=4', 'liq=5'. Component 0 is the half-level pressure and
        later the acceleration of the circulation kinetic energy.

        Prefer 'conserved_variable()', 'vertical_gradient()' and 'effective_gradient()'.
        """
        if not 0 <= component < 6:
            raise IndexError(f"'zvari' has six components, 0..5: {component}")
        return self._get_field_component("td_zvari", component, (dims.CellDim, dims.KDim))

    def conserved_variable(self, component: int):
        """
        One quasi-conserved model variable at main levels, with the Prandtl-layer lower boundary
        value at level 'ke1'; written by section 0).

        'u_m=1' and 'v_m=2' are the wind components straight from the interface, 'tet_l=3' the
        liquid-water potential temperature, 'h2o_g=4' the total water content and 'liq=5' the
        liquid water, all from 'adjust_satur_equil'. Component 0 is the half-level pressure, and
        only written when the raw circulation term is required; its model-top level is untouched.
        """
        self._require(
            "conserved_variable",
            ("0",),
            "section 1a) has replaced the variables with their vertical gradients; use "
            "vertical_gradient()",
        )
        return self.raw_zvari(component)

    def vertical_gradient(self, component: int):
        """
        Vertical gradient of one quasi-conserved model variable, half levels; written by section
        1a) over the variables themselves.

        Level 'ke1' is the Prandtl-layer gradient, '(value(ke) - surface value) * lays', and
        levels 2..ke are the centred differences '(value(k-1) - value(k)) / dz'. Level 1 is
        never written.
        """
        self._require(
            "vertical_gradient",
            ("1a", "1b", "1c", "2a", "2b", "2c"),
            "the gradients are either not yet formed (0) or section 3) has converted them into "
            "effective gradients; use conserved_variable() and effective_gradient()",
        )
        return self.raw_zvari(component)

    def effective_gradient(self, component: int):
        """
        Effective vertical gradient of one model variable, half levels; the output of
        'solve_turb_budgets' in section 3), after the turbulent flux conversion.

        The component meaning shifts with the conversion: 3 is now 'tet' and 4 'vap', not
        'tet_l' and 'h2o_g'. Component 0 is the effective gradient of the circulation kinetic
        energy per mass when the raw circulation term is on. Section 7) adds the circulation
        heat flux to component 3 under 'ldocirflx' (off here). This is the array 'vertdiff'
        receives as 'vd_zvari_in'; verified byte-identical between 'turbdiff-exit' and
        'vertdiff-entry'.
        """
        self._require(
            "effective_gradient",
            TURBDIFF_SECTIONS[TURBDIFF_SECTIONS.index("3") :],
            "'solve_turb_budgets' has not run yet; use vertical_gradient()",
        )
        return self.raw_zvari(component)


class IconVertdiffSavepoint(IconTurbulenceSavepoint):
    """
    Common part of the two savepoints of SUB 'vertdiff' (turb_vertdiff.f90:116), written by
    'serialize_vertdiff_entry' and 'serialize_vertdiff_exit' in mo_icon4py_verification.f90
    (icon f6803c9fae).

    'vertdiff' is one stage and gets an entry and an exit savepoint only. Both hooks sit inside
    its '!$ACC DATA' region, so its '!$ACC CREATE' working set is still readable at the exit.

    THE 'vd_' PREFIX IS NOT COSMETIC. 'vertdiff' shares argument names with 'turbdiff' without
    sharing their meaning, and its own workspace is a second set of aliases on top:

        vertdiff storage   quantity                    turbdiff's variable of that name
        -----------------  --------------------------  ---------------------------------
        len_scale          diff_mom                    turbulent master length scale
        zaux(:,:,1)        disc_mom                    Exner factor
        zaux(:,:,2)        expl_mom                    c_p/c_pd
        zaux(:,:,3)        impl_mom                    dQ_sat/dT
        zaux(:,:,4)        invs_mom                    buoyancy factor for tet_l
        zaux(:,:,5)        diff_dep                    buoyancy factor for h2o_g
        frh                invs_fac                    thermal forcing
        frm                scal_fac                    mechanical forcing
        dicke              dif_tend                    layer depth / discretisation momentum
        hlp                cur_prof                    scratch

    The exit accessors are therefore named after the quantity ('diff_mom()', 'disc_mom()',
    'invs_fac()', ...) and not after the storage.

    Only the columns 'ivstart:ivend' hold computed values; see 'IconTurbulenceSavepoint'.
    """

    _prefix: ClassVar[str] = "vd_"


class IconVertdiffEntrySavepoint(IconVertdiffSavepoint):
    """
    State at the start of the body of SUB 'vertdiff' (turb_vertdiff.f90:522), inside the
    '!$ACC DATA' region and after the pointer setup.

    The state that 'vertdiff' receives from 'turbdiff' is byte-identical to 'turbdiff-exit' for
    the same ('date', 'id', 'block') -- verified for 'zvari', 'tkvh' and 'rhon' -- because the
    two are called back to back from SUB 'nwp_turbdiff'.

    KNOWN GAP: the 'ndtr' passive tracers reach 'vertdiff' through 'ptr(:)', an array of POINTER
    components that a '!$ser accdata' directive cannot name, so they are not serialized. In this
    capture 'ndtr() == 0' and nothing is missing; assert that before relying on it.
    """

    def nvec(self) -> int:
        """'nproma': the horizontal extent of the slab, computed columns and untouched alike."""
        return self._read_scalar("vd_nvec")

    def ke(self) -> int:
        """Number of main (full) levels."""
        return self._read_scalar("vd_ke")

    def ke1(self) -> int:
        """Number of half levels, 'ke + 1'."""
        return self._read_scalar("vd_ke1")

    def kcm(self) -> int:
        """Level index of the upper canopy bound; 'ke1' when the canopy is switched off."""
        return self._read_scalar("vd_kcm")

    def kstart_cloud(self) -> int:
        """
        First level index at which cloud water is diffused; one-based, as in ICON.

        Vertical diffusion of 'qc' is artificially suppressed above this level.
        """
        return self._read_scalar("vd_kstart_cloud")

    def itndcon(self) -> int:
        """
        Mode of considering the explicit tendencies: 0 none, 1 on the r.h.s. of the implicit
        equation, 2 added to the profile beforehand, 3 corrected virtual profiles.
        """
        return self._read_scalar("vd_itndcon")

    def ndtr(self) -> int:
        """Number of passive tracers to be diffused; these are NOT serialized, see the class."""
        return self._read_scalar("vd_ndtr")

    def ntrac(self) -> int:
        """'UBOUND(ptr,1)', the declared length of the tracer vector, not the number in use."""
        return self._read_scalar("vd_ntrac")

    def ndiff(self) -> int:
        """Number of first-order variables, 'nmvar + ndtr'."""
        return self._read_scalar("vd_ndiff")

    def dt_var(self) -> float:
        """Time step for the ordinary prognostic variables [s]."""
        return self._read_scalar("vd_dt_var")

    def lentire(self) -> bool:
        """Calculate the entire vertical diffusion, rather than the non-gradient part only."""
        return self._read_scalar("vd_lentire")

    def lsfluse(self) -> bool:
        """Use the explicit surface heat-flux densities as the lower boundary condition."""
        return self._read_scalar("vd_lsfluse")

    def lqvcrst(self) -> bool:
        """Reset the qv-flux divergence; only meaningful when 'qv_conv' is present."""
        return self._read_scalar("vd_lqvcrst")

    def lrunscm(self) -> bool:
        """Single-column mode."""
        return self._read_scalar("vd_lrunscm")

    def ldoexpcor(self) -> bool:
        """Apply the explicit warm-cloud correction to the turbulent scalar fluxes."""
        return self._read_scalar("vd_ldoexpcor")

    def ldocirflx(self) -> bool:
        """Apply the circulation-term heat flux."""
        return self._read_scalar("vd_ldocirflx")

    def l3dflxout(self) -> bool:
        """3D output of the effective vertical fluxes of the dynamically active scalars."""
        return self._read_scalar("vd_l3dflxout")

    def ldogrdcor(self) -> bool:
        """
        Gradient correction required, 'ldoexpcor .OR. ldocirflx'.

        When false, 'igrdcon' is 0 and 'zvari' plays no part in the solve at all: the exit value
        is then purely the effective gradient that the semi-implicit procedure produced.
        """
        return self._read_scalar("vd_ldogrdcor")

    def hhl(self):
        """Height of the model half levels [m]."""
        return self._get_field("vd_hhl", dims.CellDim, dims.KDim)

    def t_g(self):
        """Weighted surface temperature [K]; the lower boundary value of 't'."""
        return self._get_field("vd_t_g", dims.CellDim)

    def qv_s(self):
        """Specific humidity at the surface [kg/kg]; the lower boundary value of 'qv'."""
        return self._get_field("vd_qv_s", dims.CellDim)

    def ps(self):
        """Surface pressure [Pa]."""
        return self._get_field("vd_ps", dims.CellDim)

    def u(self):
        """Zonal wind at the mass centre, main levels [m/s]; 'vd_u_in'."""
        return self._get_field("vd_u_in", dims.CellDim, dims.KDim)

    def v(self):
        """Meridional wind at the mass centre, main levels [m/s]; 'vd_v_in'."""
        return self._get_field("vd_v_in", dims.CellDim, dims.KDim)

    def t(self):
        """Temperature, main levels [K]; 'vd_t_in'."""
        return self._get_field("vd_t_in", dims.CellDim, dims.KDim)

    def qv(self):
        """Specific humidity, main levels [kg/kg]; 'vd_qv_in'."""
        return self._get_field("vd_qv_in", dims.CellDim, dims.KDim)

    def qc(self):
        """Specific cloud water content, main levels [kg/kg]; 'vd_qc_in'."""
        return self._get_field("vd_qc_in", dims.CellDim, dims.KDim)

    def prs(self):
        """Atmospheric pressure, main levels [Pa]."""
        return self._get_field("vd_prs", dims.CellDim, dims.KDim)

    def rhoh(self):
        """Air density, main levels [kg/m3]."""
        return self._get_field("vd_rhoh", dims.CellDim, dims.KDim)

    def epr(self):
        """Exner pressure, main levels [-]."""
        return self._get_field("vd_epr", dims.CellDim, dims.KDim)

    def rhon(self):
        """
        Air density on half levels [kg/m3]; 'vd_rhon_in'.

        'INTENT(INOUT)': 'vertdiff' rescales it, so the exit value differs. The model-top level
        is untouched, inherited from 'turbdiff'.
        """
        return self._get_field("vd_rhon_in", dims.CellDim, dims.KDim)

    def tvm(self):
        """Turbulent transfer velocity for momentum at the surface [m/s]."""
        return self._get_field("vd_tvm", dims.CellDim)

    def tvh(self):
        """Turbulent transfer velocity for heat and moisture at the surface [m/s]."""
        return self._get_field("vd_tvh", dims.CellDim)

    def tkvm(self):
        """Turbulent diffusion coefficient for momentum, half levels [m2/s]; 'vd_tkvm_in'."""
        return self._get_field("vd_tkvm_in", dims.CellDim, dims.KDim)

    def tkvh(self):
        """Turbulent diffusion coefficient for scalars, half levels [m2/s]; 'vd_tkvh_in'."""
        return self._get_field("vd_tkvh_in", dims.CellDim, dims.KDim)

    def zvari(self, component: int):
        """
        One component of the effective vertical gradients handed over by 'turbdiff', half levels;
        'vd_zvari_in'.

        Zero-based like 'turbdiff's, so 'component' is the Fortran third index unchanged, 0..5:
        'u_m=1', 'v_m=2', 'tet=3', 'vap=4', 'liq=5', and 0 the circulation kinetic energy.
        Byte-identical to 'IconTurbdiffExitSavepoint.zvari()'.
        """
        if not 0 <= component < 6:
            raise IndexError(f"'zvari' has six components, 0..5: {component}")
        return self._get_field_component("vd_zvari_in", component, (dims.CellDim, dims.KDim))

    def u_tens(self):
        """Zonal wind tendency before the diffusion increment, main levels [m/s2]."""
        return self._get_field("vd_u_tens_in", dims.CellDim, dims.KDim)

    def v_tens(self):
        """Meridional wind tendency before the diffusion increment, main levels [m/s2]."""
        return self._get_field("vd_v_tens_in", dims.CellDim, dims.KDim)

    def t_tens(self):
        """Temperature tendency before the diffusion increment, main levels [K/s]."""
        return self._get_field("vd_t_tens_in", dims.CellDim, dims.KDim)

    def qv_tens(self):
        """Specific-humidity tendency before the diffusion increment, main levels [1/s]."""
        return self._get_field("vd_qv_tens_in", dims.CellDim, dims.KDim)

    def qc_tens(self):
        """Cloud-water tendency before the diffusion increment, main levels [1/s]."""
        return self._get_field("vd_qc_tens_in", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def dp0(self):
        """Pressure thickness of the layers [Pa]; 'OPTIONAL' and not passed by the NWP interface."""
        return self._get_field("vd_dp0", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim)
    def shfl_s(self):
        """Sensible heat flux at the surface [W/m2], positive downward; 'vd_shfl_s_in'."""
        return self._get_field("vd_shfl_s_in", dims.CellDim)

    @IconSavepoint.optionally_registered(dims.CellDim)
    def qvfl_s(self):
        """Water-vapour flux at the surface [kg/m2/s], positive downward; 'vd_qvfl_s_in'."""
        return self._get_field("vd_qvfl_s_in", dims.CellDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def qv_conv(self):
        """qv-flux convergence [1/s]; 'OPTIONAL' and not passed by the NWP interface."""
        return self._get_field("vd_qv_conv_in", dims.CellDim, dims.KDim)


class IconVertdiffExitSavepoint(IconVertdiffSavepoint):
    """
    State at the end of SUB 'vertdiff' (turb_vertdiff.f90:928), inside the '!$ACC DATA' region.

    THE WORKSPACE IS THE LAST VARIABLE'S. 'vertdiff' loops over variable types and over the
    variables of each type, reusing 'zaux', 'len_scale', 'frh', 'frm', 'dicke' and 'hlp' for
    every one of them. What this savepoint holds is whatever the LAST variable of the LAST type
    left there -- 'ivtype()' and the loop counters say which. The per-variable outputs ('u',
    'v', 't', 'qv', 'qc' and their tendencies) are of course complete.

    'nvec', 'ke', 'ke1' and the control flags are only written by the entry savepoint; read them
    from there for the same ('date', 'id', 'block').
    """

    def ncorr(self) -> int:
        """Start index of the variables that get a gradient correction; one-based."""
        return self._read_scalar("vd_ncorr")

    def mcorr(self) -> int:
        """
        End index of the variables that get a gradient correction; one-based and inclusive.

        'ncorr > mcorr' means the range is empty and no variable was gradient-corrected.
        """
        return self._read_scalar("vd_mcorr")

    def igrdcon(self) -> int:
        """
        Mode of considering the non-gradient flux contributions: 0 none, 1 correction only,
        2 corrected profile from the effective gradients, 3 add the contribution in 'zvari'.

        Only assigned inside 'IF (lentire .OR. ...)' (turb_vertdiff.f90:600), so it is defined
        here only because the NWP interface hard-codes 'lentire = .TRUE.'.
        """
        return self._read_scalar("vd_igrdcon")

    def ivtype(self) -> int:
        """Variable type of the last iteration: 1 momentum, 2 scalars."""
        return self._read_scalar("vd_ivtype")

    def u(self):
        """
        Zonal wind at the mass centre, main levels [m/s].

        Unchanged from the entry whenever the tendency fields are present, which they are here:
        the diffusion increment goes into 'u_tens()' instead.
        """
        return self._get_field("vd_u", dims.CellDim, dims.KDim)

    def v(self):
        """Meridional wind at the mass centre, main levels [m/s]; see 'u()'."""
        return self._get_field("vd_v", dims.CellDim, dims.KDim)

    def t(self):
        """Temperature, main levels [K]; see 'u()'."""
        return self._get_field("vd_t", dims.CellDim, dims.KDim)

    def qv(self):
        """Specific humidity, main levels [kg/kg]; see 'u()'."""
        return self._get_field("vd_qv", dims.CellDim, dims.KDim)

    def qc(self):
        """Specific cloud water content, main levels [kg/kg]; see 'u()'."""
        return self._get_field("vd_qc", dims.CellDim, dims.KDim)

    def rhon(self):
        """Air density on half levels [kg/m3]; rescaled by 'vertdiff', so it differs from entry."""
        return self._get_field("vd_rhon", dims.CellDim, dims.KDim)

    def tkvm(self):
        """Turbulent diffusion coefficient for momentum, half levels [m2/s]."""
        return self._get_field("vd_tkvm", dims.CellDim, dims.KDim)

    def tkvh(self):
        """Turbulent diffusion coefficient for scalars, half levels [m2/s]."""
        return self._get_field("vd_tkvh", dims.CellDim, dims.KDim)

    def zvari(self, component: int):
        """
        Effective vertical gradient of one model variable as the semi-implicit procedure
        produced it, half levels; 'component' is the Fortran third index, 0..5.
        """
        if not 0 <= component < 6:
            raise IndexError(f"'zvari' has six components, 0..5: {component}")
        return self._get_field_component("vd_zvari", component, (dims.CellDim, dims.KDim))

    def u_tens(self):
        """Zonal wind tendency including the diffusion increment, main levels [m/s2]."""
        return self._get_field("vd_u_tens", dims.CellDim, dims.KDim)

    def v_tens(self):
        """Meridional wind tendency including the diffusion increment, main levels [m/s2]."""
        return self._get_field("vd_v_tens", dims.CellDim, dims.KDim)

    def t_tens(self):
        """Temperature tendency including the diffusion increment, main levels [K/s]."""
        return self._get_field("vd_t_tens", dims.CellDim, dims.KDim)

    def qv_tens(self):
        """Specific-humidity tendency including the diffusion increment, main levels [1/s]."""
        return self._get_field("vd_qv_tens", dims.CellDim, dims.KDim)

    def qc_tens(self):
        """Cloud-water tendency including the diffusion increment, main levels [1/s]."""
        return self._get_field("vd_qc_tens", dims.CellDim, dims.KDim)

    def eprs(self):
        """
        Surface Exner factor [-], one value per column.

        Declared '(nvec, ke1:ke1)' in Fortran, so it is serialized as a '(nvec, 1)' slab; the
        singleton level is squeezed away and this is a cell field.
        """
        return self._get_field("vd_eprs", dims.CellDim)

    def diff_mom(self):
        """
        Diffusion momentum of the last diffused variable, half levels; the 'len_scale' storage
        (turb_vertdiff.f90:369), 'rho * K / dz' on the flux levels.
        """
        return self._get_field("vd_len_scale", dims.CellDim, dims.KDim)

    def raw_zaux(self, component: int):
        """
        One component of 'vertdiff's 'zaux', half levels, uninterpreted.

        ONE-based in Fortran, so 'component' here is zero-based: 'component = i' is
        'zaux(:,:,i+1)'. Prefer 'disc_mom()', 'expl_mom()', 'impl_mom()', 'invs_mom()' and
        'diff_dep()'; these are NOT turbdiff's 'zaux' components of the same index.
        """
        if not 0 <= component < 5:
            raise IndexError(f"'zaux' has five components, 0..4: {component}")
        return self._get_field_component("vd_zaux", component, (dims.CellDim, dims.KDim))

    def disc_mom(self):
        """Discretisation momentum 'disc_mom', half levels; 'zaux(:,:,1)'."""
        return self.raw_zaux(0)

    def expl_mom(self):
        """Explicit part of the diffusion momentum 'expl_mom', half levels; 'zaux(:,:,2)'."""
        return self.raw_zaux(1)

    def impl_mom(self):
        """Implicit part of the diffusion momentum 'impl_mom', half levels; 'zaux(:,:,3)'."""
        return self.raw_zaux(2)

    def invs_mom(self):
        """Inverted diffusion momentum 'invs_mom', half levels; 'zaux(:,:,4)'."""
        return self.raw_zaux(3)

    def diff_dep(self):
        """Diffusion depth 'diff_dep', half levels [m]; 'zaux(:,:,5)'."""
        return self.raw_zaux(4)

    def invs_fac(self):
        """Inversion factor 'invs_fac' of the tridiagonal solve, half levels; the 'frh' storage."""
        return self._get_field("vd_frh", dims.CellDim, dims.KDim)

    def scal_fac(self):
        """Scaling factor 'scal_fac' of the tridiagonal solve, half levels; the 'frm' storage."""
        return self._get_field("vd_frm", dims.CellDim, dims.KDim)

    def dif_tend(self):
        """
        Diffusion tendency 'dif_tend' of the last diffused variable, half levels; the 'dicke'
        storage.
        """
        return self._get_field("vd_dicke", dims.CellDim, dims.KDim)

    def cur_prof(self):
        """
        Current profile 'cur_prof' of the last diffused variable of the last variable type, half
        levels; the 'hlp' storage.
        """
        return self._get_field("vd_hlp", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim)
    def shfl_s(self):
        """Sensible heat flux at the surface [W/m2], positive downward."""
        return self._get_field("vd_shfl_s", dims.CellDim)

    @IconSavepoint.optionally_registered(dims.CellDim)
    def qvfl_s(self):
        """Water-vapour flux at the surface [kg/m2/s], positive downward."""
        return self._get_field("vd_qvfl_s", dims.CellDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def qv_conv(self):
        """qv-flux convergence [1/s]; 'OPTIONAL' and not passed by the NWP interface."""
        return self._get_field("vd_qv_conv", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim)
    def umfl_s(self):
        """u-momentum flux at the surface [N/m2]; 'OPTIONAL' and not passed by the NWP interface."""
        return self._get_field("vd_umfl_s", dims.CellDim)

    @IconSavepoint.optionally_registered(dims.CellDim)
    def vmfl_s(self):
        """v-momentum flux at the surface [N/m2]; 'OPTIONAL' and not passed by the NWP interface."""
        return self._get_field("vd_vmfl_s", dims.CellDim)


class IconSerialDataProvider:
    def __init__(
        self,
        *,
        backend: gtx_typing.Backend | None,
        fname_prefix,
        path=".",
        do_print=False,
        mpi_rank=0,
    ):
        self.rank = mpi_rank
        self.serializer: serialbox.Serializer = None
        self.file_path: str = path
        self.fname = f"{fname_prefix}_rank{self.rank!s}"
        self.log = logging.getLogger(__name__)
        self._init_serializer(do_print)
        self.backend = backend

    def _init_serializer(self, do_print: bool):
        if not self.fname:
            self.log.warning(" WARNING: no filename! closing serializer")
        self.serializer = serialbox.Serializer(
            serialbox.OpenModeKind.Read, self.file_path, self.fname
        )
        if do_print:
            self.print_info()

    def print_info(self):
        self.log.info(f"SAVEPOINTS: {self.serializer.savepoint_list()}")
        self.log.info(f"FIELDNAMES: {self.serializer.fieldnames()}")

    @functools.cached_property
    def grid_size(self):
        sp = self._get_icon_grid_savepoint()
        grid_sizes = {
            dims.CellDim: self.serializer.read("num_cells", savepoint=sp).astype(gtx.int32)[0],
            dims.EdgeDim: self.serializer.read("num_edges", savepoint=sp).astype(gtx.int32)[0],
            dims.VertexDim: self.serializer.read("num_vert", savepoint=sp).astype(gtx.int32)[0],
            dims.KDim: sp.metainfo.to_dict()["nlev"],
        }
        return grid_sizes

    def from_savepoint_grid(self, grid_id: str, grid_params: icon.GridParams) -> IconGridSavepoint:
        savepoint = self._get_icon_grid_savepoint()
        return IconGridSavepoint(
            sp=savepoint,
            ser=self.serializer,
            grid_id=grid_id,
            size=self.grid_size,
            grid_params=grid_params,
            backend=self.backend,
        )

    def _get_icon_grid_savepoint(self):
        savepoint = self.serializer.savepoint["icon-grid"].id[1].as_savepoint()
        return savepoint

    def from_savepoint_diffusion_init(
        self,
        linit: bool,
        date: str,
    ) -> IconDiffusionInitSavepoint:
        savepoint = (
            self.serializer.savepoint["diffusion-init"].linit[linit].date[date].as_savepoint()
        )
        return IconDiffusionInitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_velocity_init(
        self, istep: int, date: str, substep: int
    ) -> IconVelocityInitSavepoint:
        savepoint = (
            self.serializer.savepoint["velocity-tendencies-init"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return IconVelocityInitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_nonhydro_init(
        self, istep: int, date: str, substep: int
    ) -> IconNonHydroInitSavepoint:
        savepoint = (
            self.serializer.savepoint["solve-nonhydro-init"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return IconNonHydroInitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init(
        self, istep: int, date: str, substep: int
    ) -> NonHydroInitEdgeDiagnosticsUpdateVnSavepoint:
        savepoint = (
            self.serializer.savepoint["solve-nonhydro-14to28-init_1to13-exit"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return NonHydroInitEdgeDiagnosticsUpdateVnSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_vertically_implicit_dycore_solver_init(
        self, istep: int, date: str, substep: int
    ) -> NonHydroInitVerticallyImplicitSolverSavepoint:
        savepoint = (
            self.serializer.savepoint["solve-nonhydro-41to60-init"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return NonHydroInitVerticallyImplicitSolverSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_30_to_38_init(
        self, istep: int, date: str, substep: int
    ) -> IconDycoreInit30To38Savepoint:
        savepoint = (
            self.serializer.savepoint["solve-nonhydro-30to38-init"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return IconDycoreInit30To38Savepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_interpolation_savepoint(self) -> InterpolationSavepoint:
        savepoint = self.serializer.savepoint["interpolation-state"].as_savepoint()
        return InterpolationSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_metrics_savepoint(self) -> MetricSavepoint:
        savepoint = self.serializer.savepoint["metric-state"].as_savepoint()
        return MetricSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_topography_savepoint(self) -> TopographySavepoint:
        savepoint = self.serializer.savepoint["smooth-topo-savepoint"].as_savepoint()
        return TopographySavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_advection_init_savepoint(self, size: dict, date: str) -> AdvectionInitSavepoint:
        savepoint = self.serializer.savepoint["advection-init"].id[1].date[date].as_savepoint()
        return AdvectionInitSavepoint(savepoint, self.serializer, size=size, backend=self.backend)

    def from_advection_exit_savepoint(self, size: dict, date: str) -> AdvectionExitSavepoint:
        savepoint = self.serializer.savepoint["advection-exit"].id[1].date[date].as_savepoint()
        return AdvectionExitSavepoint(savepoint, self.serializer, size=size, backend=self.backend)

    def from_savepoint_diffusion_exit(self, linit: bool, date: str) -> IconDiffusionExitSavepoint:
        savepoint = (
            self.serializer.savepoint["diffusion-exit"].linit[linit].date[date].as_savepoint()
        )
        return IconDiffusionExitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_velocity_exit(
        self, istep: int, date: str, substep: int
    ) -> IconVelocityExitSavepoint:
        savepoint = (
            self.serializer.savepoint["velocity-tendencies-exit"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return IconVelocityExitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_30_to_38_exit(
        self, istep: int, date: str, substep: int
    ) -> IconDycoreExit30To38Savepoint:
        savepoint = (
            self.serializer.savepoint["solve-nonhydro-30to38-exit"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return IconDycoreExit30To38Savepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_nonhydro_exit(
        self, istep: int, date: str, substep: int
    ) -> IconNonHydroExitSavepoint:
        savepoint = (
            self.serializer.savepoint["solve-nonhydro-exit"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return IconNonHydroExitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_exit(
        self, istep: int, date: str, substep: int
    ) -> NonHydroExitEdgeDiagnosticsUpdateVnSavepoint:
        savepoint = (
            self.serializer.savepoint["solve-nonhydro-14to28-exit"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return NonHydroExitEdgeDiagnosticsUpdateVnSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_nonhydro_step_final(
        self, date: str, substep: int
    ) -> IconNonHydroFinalSavepoint:
        savepoint = (
            self.serializer.savepoint["solve-nonhydro-final"]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return IconNonHydroFinalSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_jabw_exit(self) -> IconJabwExitSavepoint:
        savepoint = self.serializer.savepoint["jabw-initial-state-exit"].id[1].as_savepoint()
        return IconJabwExitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_prognostics_initial(self) -> IconPrognosticsInitSavepoint:
        savepoint = (
            self.serializer.savepoint["prognostics"].id[1].location["initial-state"].as_savepoint()
        )
        return IconPrognosticsInitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_diagnostics_initial(self) -> IconDiagnosticsInitSavepoint:
        savepoint = (
            self.serializer.savepoint["diagnostics"].id[1].location["initial-state"].as_savepoint()
        )
        return IconDiagnosticsInitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_weisman_klemp_graupel_entry(self, date: str) -> IconGraupelSavepoint:
        savepoint = self.serializer.savepoint["microphysics-init"].date[date].as_savepoint()
        return IconGraupelSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_weisman_klemp_graupel_exit(self, date: str) -> IconGraupelSavepoint:
        savepoint = self.serializer.savepoint["microphysics-exit"].date[date].as_savepoint()
        return IconGraupelSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_satad_init(self, location: str, date: str) -> IconSatadInitSavepoint:
        savepoint = (
            self.serializer.savepoint["satad-init"].location[location].date[date].as_savepoint()
        )
        return IconSatadInitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_satad_exit(self, location: str, date: str) -> IconSatadExitSavepoint:
        savepoint = (
            self.serializer.savepoint["satad-exit"].date[date].location[location].as_savepoint()
        )
        return IconSatadExitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_turbdiff_entry(
        self, date: str, block: int = 1
    ) -> IconTurbdiffEntrySavepoint:
        """
        Load data from the ICON savepoint at the start of section 0) of SUB 'turbdiff'
        (turb_diffusion.f90:937), i.e. after SUB 'turb_setup'.

        metadata to select a unique savepoint:
        - date: <iso_string> of the model timestep
        - id: the domain 'jg'; 1, as in every other reader here
        - block: the one-based block index 'iblock'. 'turbdiff' is called once per block of
          'nproma' columns, so a capture only has block 1 when 'nproma >= n_patch_cells'.
        """
        savepoint = (
            self.serializer.savepoint["turbdiff-entry"].id[1].date[date].block[block].as_savepoint()
        )
        return IconTurbdiffEntrySavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_turbdiff_exit(self, date: str, block: int = 1) -> IconTurbdiffExitSavepoint:
        """
        Load data from the ICON savepoint at the end of SUB 'turbdiff'
        (turb_diffusion.f90:2549), inside the '!$ACC DATA' region.

        metadata to select a unique savepoint: see 'from_savepoint_turbdiff_entry'.
        """
        savepoint = (
            self.serializer.savepoint["turbdiff-exit"].id[1].date[date].block[block].as_savepoint()
        )
        return IconTurbdiffExitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_turbdiff_section(
        self, section: str, date: str, block: int = 1
    ) -> IconTurbdiffSectionSavepoint:
        """
        Load data from one of the fifteen intermediate savepoints of SUB 'turbdiff', at the
        boundary between Raschendorfer's numbered sections.

        - section: the label of the section this savepoint is the EXIT of, one of
          'TURBDIFF_SECTIONS' ('0', '1a', '1b', '1c', '2a', '2b', '2c', '3' .. '10'). It is also
          the entry of the next section. The exit of the last section 11) is 'turbdiff-exit',
          read with 'from_savepoint_turbdiff_exit'.
        - date: <iso_string> of the model timestep
        - id: the domain 'jg'; 1, as in every other reader here
        - block: the one-based block index 'iblock'; see 'from_savepoint_turbdiff_entry'.

        'turbdiff-9-exit' is conditional. Its hook could not be lifted out of
        'IF (ldotkedif .OR. lcircterm)' (turb_diffusion.f90:2404), because section 10)'s own
        banner is inside it and no position further out is still the exit of 9). A run with
        'c_diff = 0' and no circulation term therefore has no such savepoint, and this raises
        with that explanation rather than passing a serialbox error on.
        """
        if section not in TURBDIFF_SECTIONS:
            raise ValueError(
                f"unknown turbdiff section {section!r}; expected one of "
                f"{', '.join(TURBDIFF_SECTIONS)}"
            )
        name = f"turbdiff-{section}-exit"
        if not any(sp.name == name for sp in self.serializer.savepoint_list()):
            hint = (
                " Section 9)'s hook sits inside 'IF (ldotkedif .OR. lcircterm)', so a run with"
                " 'c_diff = 0' and no circulation term does not write it at all."
                if section == "9"
                else ""
            )
            raise ValueError(f"no savepoint {name!r} in this archive.{hint}")
        savepoint = self.serializer.savepoint[name].id[1].date[date].block[block].as_savepoint()
        return IconTurbdiffSectionSavepoint(
            savepoint,
            self.serializer,
            section=section,
            size=self.grid_size,
            backend=self.backend,
        )

    def from_savepoint_vertdiff_entry(
        self, date: str, block: int = 1
    ) -> IconVertdiffEntrySavepoint:
        """
        Load data from the ICON savepoint at the start of the body of SUB 'vertdiff'
        (turb_vertdiff.f90:522), inside its '!$ACC DATA' region.

        metadata to select a unique savepoint: see 'from_savepoint_turbdiff_entry'. 'vertdiff'
        is called from SUB 'nwp_turbdiff' immediately after 'turbdiff' for the same block, so
        the ('date', 'id', 'block') of the two schemes line up.
        """
        savepoint = (
            self.serializer.savepoint["vertdiff-entry"].id[1].date[date].block[block].as_savepoint()
        )
        return IconVertdiffEntrySavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_vertdiff_exit(self, date: str, block: int = 1) -> IconVertdiffExitSavepoint:
        """
        Load data from the ICON savepoint at the end of SUB 'vertdiff' (turb_vertdiff.f90:928),
        inside its '!$ACC DATA' region.

        metadata to select a unique savepoint: see 'from_savepoint_vertdiff_entry'.
        """
        savepoint = (
            self.serializer.savepoint["vertdiff-exit"].id[1].date[date].block[block].as_savepoint()
        )
        return IconVertdiffExitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )
