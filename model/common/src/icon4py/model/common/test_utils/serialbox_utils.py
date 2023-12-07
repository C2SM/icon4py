# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import logging

import numpy as np
import serialbox as ser
from gt4py.next.common import Dimension, DimensionKind
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.iterator.embedded import np_as_located_field

from icon4py.model.atmosphere.diffusion.diffusion import VectorTuple
from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
    PrognosticState,
)
from icon4py.model.common import dimension
from icon4py.model.common.decomposition.definitions import DecompositionInfo
from icon4py.model.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CECDim,
    CEDim,
    CellDim,
    E2C2VDim,
    E2CDim,
    E2VDim,
    ECVDim,
    EdgeDim,
    KDim,
    V2EDim,
    VertexDim,
)
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams, HorizontalGridSize
from icon4py.model.common.grid.icon_grid import GridConfig, IconGrid, VerticalGridSize
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field


class IconSavepoint:
    def __init__(self, sp: ser.Savepoint, ser: ser.Serializer, size: dict):
        self.savepoint = sp
        self.serializer = ser
        self.sizes = size
        self.log = logging.getLogger((__name__))

    def log_meta_info(self):
        self.log.info(self.savepoint.metainfo)

    def _get_field(self, name, *dimensions, dtype=float):
        buffer = np.squeeze(self.serializer.read(name, self.savepoint).astype(dtype))
        buffer_size = (
            self.sizes[d] if d.kind is DimensionKind.HORIZONTAL else s
            for s, d in zip(buffer.shape, dimensions)
        )
        buffer = buffer[tuple(map(slice, buffer_size))]

        self.log.debug(f"{name} {buffer.shape}")
        return np_as_located_field(*dimensions)(buffer)

    def get_metadata(self, *names):
        metadata = self.savepoint.metainfo.to_dict()
        return {n: metadata[n] for n in names if n in metadata}

    def _read_int32_shift1(self, name: str):
        """
        Read a start indices field.

        use for start indices: the shift accounts for the zero based python
        values are converted to int32
        """
        return self._read_int32(name, offset=1)

    def _read_int32(self, name: str, offset=0):
        """
        Read an end indices field.

        use this for end indices: because FORTRAN slices  are inclusive [from:to] _and_ one based
        this accounts for being exclusive python exclusive bounds: [from:to)
        field values are convert to int32
        """
        return self._read(name, offset, dtype=int32)

    def _read_bool(self, name: str):
        return self._read(name, offset=0, dtype=bool)

    def _read(self, name: str, offset=0, dtype=int):
        return (self.serializer.read(name, self.savepoint) - offset).astype(dtype)


class IconGridSavePoint(IconSavepoint):
    def vct_a(self):
        return self._get_field("vct_a", KDim)

    def tangent_orientation(self):
        return self._get_field("tangent_orientation", EdgeDim)

    def edge_orientation(self):
        return self._get_field("cells_edge_orientation", CellDim, C2EDim)

    def inverse_primal_edge_lengths(self):
        return self._get_field("inv_primal_edge_length", EdgeDim)

    def inv_vert_vert_length(self):
        return self._get_field("inv_vert_vert_length", EdgeDim)

    def primal_normal_vert_x(self):
        return self._get_field("primal_normal_vert_x", VertexDim, E2C2VDim)

    def primal_normal_vert_y(self):
        return self._get_field("primal_normal_vert_y", VertexDim, E2C2VDim)

    def dual_normal_vert_y(self):
        return self._get_field("dual_normal_vert_y", VertexDim, E2C2VDim)

    def dual_normal_vert_x(self):
        return self._get_field("dual_normal_vert_x", VertexDim, E2C2VDim)

    def cell_areas(self):
        return self._get_field("cell_areas", CellDim)

    def edge_areas(self):
        return self._get_field("edge_areas", EdgeDim)

    def inv_dual_edge_length(self):
        return self._get_field("inv_dual_edge_length", EdgeDim)

    def edge_cell_length(self):
        return self._get_field("edge_cell_length", EdgeDim, E2CDim)

    def cells_start_index(self):
        return self._read_int32_shift1("c_start_index")

    def cells_end_index(self):
        return self._read_int32("c_end_index")

    def vertex_start_index(self):
        return self._read_int32_shift1("v_start_index")

    def vertex_end_index(self):
        return self._read_int32("v_end_index")

    def edge_start_index(self):
        return self._read_int32_shift1("e_start_index")

    def edge_end_index(self):
        # don't need to subtract 1, because FORTRAN slices  are inclusive [from:to] so the being
        # one off accounts for being exclusive [from:to)
        return self.serializer.read("e_end_index", self.savepoint)

    def c_owner_mask(self):
        return self._get_field("c_owner_mask", CellDim, dtype=bool)

    def e_owner_mask(self):
        return self._get_field("e_owner_mask", EdgeDim, dtype=bool)

    def print_connectivity_info(self, name: str, ar: np.ndarray):
        self.log.debug(f" connectivity {name} {ar.shape}")

    def c2e(self):
        return self._get_connectivity_array("c2e", CellDim)

    def _get_connectivity_array(self, name: str, target_dim: Dimension):
        connectivity = self._read_int32(name, offset=1)[: self.sizes[target_dim], :]
        self.log.debug(f" connectivity {name} : {connectivity.shape}")
        return connectivity

    def c2e2c(self):
        return self._get_connectivity_array("c2e2c", CellDim)

    def e2c(self):
        return self._get_connectivity_array("e2c", EdgeDim)

    def e2v(self):
        # array "e2v" is actually e2c2v
        v_ = self._get_connectivity_array("e2v", EdgeDim)[:, 0:2]
        self.log.debug(f"real e2v {v_.shape}")
        return v_

    def e2c2v(self):
        # array "e2v" is actually e2c2v, that is hexagon or pentagon
        return self._get_connectivity_array("e2v", EdgeDim)

    def v2e(self):
        return self._get_connectivity_array("v2e", VertexDim)

    def v2c(self):
        return self._get_connectivity_array("v2c", VertexDim)

    def c2v(self):
        return self._get_connectivity_array("c2v", CellDim)

    def nrdmax(self):
        return self._read_int32_shift1("nrdmax")

    def refin_ctrl(self, dim: Dimension):
        field_name = "refin_ctl"
        return self._read_field_for_dim(field_name, self._read_int32, dim)

    def num(self, dim: Dimension):
        return self.sizes[dim]

    def _read_field_for_dim(self, field_name, read_func, dim: Dimension):
        match (dim):
            case dimension.CellDim:
                return read_func(f"c_{field_name}")
            case dimension.EdgeDim:
                return read_func(f"e_{field_name}")
            case dimension.VertexDim:
                return read_func(f"v_{field_name}")
            case _:
                raise NotImplementedError(
                    f"only {dimension.CellDim, dimension.EdgeDim, dimension.VertexDim} are handled"
                )

    def owner_mask(self, dim: Dimension):
        field_name = "owner_mask"
        mask = self._read_field_for_dim(field_name, self._read_bool, dim)
        return np.squeeze(mask)

    def global_index(self, dim: Dimension):
        field_name = "glb_index"
        return self._read_field_for_dim(field_name, self._read_int32_shift1, dim)

    def decomp_domain(self, dim):
        field_name = "decomp_domain"
        return self._read_field_for_dim(field_name, self._read_int32, dim)

    def construct_decomposition_info(self):
        return (
            DecompositionInfo(klevels=self.num(KDim))
            .with_dimension(*self._get_decomp_fields(CellDim))
            .with_dimension(*self._get_decomp_fields(EdgeDim))
            .with_dimension(*self._get_decomp_fields(VertexDim))
        )

    def _get_decomp_fields(self, dim: Dimension):
        global_index = self.global_index(dim)
        mask = self.owner_mask(dim)[0 : self.num(dim)]
        return dim, global_index, mask

    def construct_icon_grid(self) -> IconGrid:

        cell_starts = self.cells_start_index()
        cell_ends = self.cells_end_index()
        vertex_starts = self.vertex_start_index()
        vertex_ends = self.vertex_end_index()
        edge_starts = self.edge_start_index()
        edge_ends = self.edge_end_index()
        config = GridConfig(
            horizontal_config=HorizontalGridSize(
                num_vertices=self.num(VertexDim),
                num_cells=self.num(CellDim),
                num_edges=self.num(EdgeDim),
            ),
            vertical_config=VerticalGridSize(num_lev=self.num(KDim)),
        )
        c2e2c = self.c2e2c()
        c2e2c0 = np.column_stack(((np.asarray(range(c2e2c.shape[0]))), c2e2c))
        grid = (
            IconGrid()
            .with_config(config)
            .with_start_end_indices(VertexDim, vertex_starts, vertex_ends)
            .with_start_end_indices(EdgeDim, edge_starts, edge_ends)
            .with_start_end_indices(CellDim, cell_starts, cell_ends)
            .with_connectivities(
                {
                    C2EDim: self.c2e(),
                    E2CDim: self.e2c(),
                    C2E2CDim: c2e2c,
                    C2E2CODim: c2e2c0,
                }
            )
            .with_connectivities({E2VDim: self.e2v(), V2EDim: self.v2e(), E2C2VDim: self.e2c2v()})
        )
        return grid

    def construct_edge_geometry(self) -> EdgeParams:
        primal_normal_vert: VectorTuple = (
            as_1D_sparse_field(self.primal_normal_vert_x(), ECVDim),
            as_1D_sparse_field(self.primal_normal_vert_y(), ECVDim),
        )
        dual_normal_vert: VectorTuple = (
            as_1D_sparse_field(self.dual_normal_vert_x(), ECVDim),
            as_1D_sparse_field(self.dual_normal_vert_y(), ECVDim),
        )
        return EdgeParams(
            tangent_orientation=self.tangent_orientation(),
            inverse_primal_edge_lengths=self.inverse_primal_edge_lengths(),
            inverse_dual_edge_lengths=self.inv_dual_edge_length(),
            inverse_vertex_vertex_lengths=self.inv_vert_vert_length(),
            primal_normal_vert_x=primal_normal_vert[0],
            primal_normal_vert_y=primal_normal_vert[1],
            dual_normal_vert_x=dual_normal_vert[0],
            dual_normal_vert_y=dual_normal_vert[1],
            edge_areas=self.edge_areas(),
        )

    def construct_cell_geometry(self) -> CellParams:
        return CellParams(area=self.cell_areas())


class InterpolationSavepoint(IconSavepoint):
    def geofac_grg(self):
        grg = np.squeeze(self.serializer.read("geofac_grg", self.savepoint))
        num_cells = self.sizes[CellDim]
        return np_as_located_field(CellDim, C2E2CODim)(grg[:num_cells, :, 0]), np_as_located_field(
            CellDim, C2E2CODim
        )(grg[:num_cells, :, 1])

    def zd_intcoef(self):
        return self._get_field("vcoef", CellDim, C2E2CDim, KDim)

    def e_bln_c_s(self):
        return self._get_field("e_bln_c_s", CellDim, C2EDim)

    def geofac_div(self):
        return self._get_field("geofac_div", CellDim, C2EDim)

    def geofac_n2s(self):
        return self._get_field("geofac_n2s", CellDim, C2E2CODim)

    def rbf_vec_coeff_v1(self):
        return self._get_field("rbf_vec_coeff_v1", VertexDim, V2EDim)

    def rbf_vec_coeff_v2(self):
        return self._get_field("rbf_vec_coeff_v2", VertexDim, V2EDim)

    def nudgecoeff_e(self):
        return self._get_field("nudgecoeff_e", EdgeDim)

    def construct_interpolation_state_for_diffusion(
        self,
    ) -> DiffusionInterpolationState:
        grg = self.geofac_grg()
        return DiffusionInterpolationState(
            e_bln_c_s=as_1D_sparse_field(self.e_bln_c_s(), CEDim),
            rbf_coeff_1=self.rbf_vec_coeff_v1(),
            rbf_coeff_2=self.rbf_vec_coeff_v2(),
            geofac_div=as_1D_sparse_field(self.geofac_div(), CEDim),
            geofac_n2s=self.geofac_n2s(),
            geofac_grg_x=grg[0],
            geofac_grg_y=grg[1],
            nudgecoeff_e=self.nudgecoeff_e(),
        )

    def c_lin_e(self):
        return self._get_field("c_lin_e", EdgeDim, E2CDim)


class MetricSavepoint(IconSavepoint):
    def construct_metric_state_for_diffusion(self) -> DiffusionMetricState:
        return DiffusionMetricState(
            mask_hdiff=self.mask_diff(),
            theta_ref_mc=self.theta_ref_mc(),
            wgtfac_c=self.wgtfac_c(),
            zd_intcoef=self.zd_intcoef(),
            zd_vertoffset=self.zd_vertoffset(),
            zd_diffcoef=self.zd_diffcoef(),
        )

    def zd_diffcoef(self):
        return self._get_field("zd_diffcoef", CellDim, KDim)

    def zd_intcoef(self):
        return self._read_and_reorder_sparse_field("vcoef", CellDim)

    def _read_and_reorder_sparse_field(self, name: str, horizontal_dim: Dimension, sparse_size=3):
        ser_input = np.squeeze(self.serializer.read(name, self.savepoint))[
            : self.sizes[horizontal_dim], :, :
        ]
        if ser_input.shape[1] != sparse_size:
            ser_input = np.moveaxis((ser_input), 1, -1)

        return self._linearize_first_2dims(
            ser_input, sparse_size=sparse_size, target_dims=(CECDim, KDim)
        )

    def _linearize_first_2dims(
        self, data: np.ndarray, sparse_size: int, target_dims: tuple[Dimension, ...]
    ):
        old_shape = data.shape
        assert old_shape[1] == sparse_size
        return np_as_located_field(*target_dims)(
            data.reshape(old_shape[0] * old_shape[1], old_shape[2])
        )

    def zd_vertoffset(self):
        return self._read_and_reorder_sparse_field("zd_vertoffset", CellDim)

    def zd_vertidx(self):
        return np.squeeze(self.serializer.read("zd_vertidx", self.savepoint))

    def zd_indlist(self):
        return np.squeeze(self.serializer.read("zd_indlist", self.savepoint))

    def theta_ref_mc(self):
        return self._get_field("theta_ref_mc", CellDim, KDim)

    def wgtfac_c(self):
        return self._get_field("wgtfac_c", CellDim, KDim)

    def wgtfac_e(self):
        return self._get_field("wgtfac_e", EdgeDim, KDim)

    def mask_diff(self):
        return self._get_field("mask_hdiff", CellDim, KDim, dtype=bool)


class IconDiffusionInitSavepoint(IconSavepoint):
    def hdef_ic(self):
        return self._get_field("hdef_ic", CellDim, KDim)

    def div_ic(self):
        return self._get_field("div_ic", CellDim, KDim)

    def dwdx(self):
        return self._get_field("dwdx", CellDim, KDim)

    def dwdy(self):
        return self._get_field("dwdy", CellDim, KDim)

    def vn(self):
        return self._get_field("vn", EdgeDim, KDim)

    def theta_v(self):
        return self._get_field("theta_v", CellDim, KDim)

    def w(self):
        return self._get_field("w", CellDim, KDim)

    def exner(self):
        return self._get_field("exner", CellDim, KDim)

    def diff_multfac_smag(self):
        return np.squeeze(self.serializer.read("diff_multfac_smag", self.savepoint))

    def smag_limit(self):
        return np.squeeze(self.serializer.read("smag_limit", self.savepoint))

    def diff_multfac_n2w(self):
        return np.squeeze(self.serializer.read("diff_multfac_n2w", self.savepoint))

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

    def construct_prognostics(self) -> PrognosticState:
        return PrognosticState(
            w=self.w(),
            vn=self.vn(),
            exner_pressure=self.exner(),
            theta_v=self.theta_v(),
        )

    def construct_diagnostics_for_diffusion(self) -> DiffusionDiagnosticState:
        return DiffusionDiagnosticState(
            hdef_ic=self.hdef_ic(),
            div_ic=self.div_ic(),
            dwdx=self.dwdx(),
            dwdy=self.dwdy(),
        )


class IconDiffusionExitSavepoint(IconSavepoint):
    def vn(self):
        return self._get_field("x_vn", EdgeDim, KDim)

    def theta_v(self):
        return self._get_field("x_theta_v", CellDim, KDim)

    def w(self):
        return self._get_field("x_w", CellDim, KDim)

    def dwdx(self):
        return self._get_field("x_dwdx", CellDim, KDim)

    def dwdy(self):
        return self._get_field("x_dwdy", CellDim, KDim)

    def exner(self):
        return self._get_field("x_exner", CellDim, KDim)

    def z_temp(self):
        return self._get_field("x_z_temp", CellDim, KDim)

    def div_ic(self):
        return self._get_field("x_div_ic", CellDim, KDim)

    def hdef_ic(self):
        return self._get_field("x_hdef_ic", CellDim, KDim)


class IconSerialDataProvider:
    def __init__(self, fname_prefix, path=".", do_print=False, mpi_rank=0):
        self.rank = mpi_rank
        self.serializer: ser.Serializer = None
        self.file_path: str = path
        self.fname = f"{fname_prefix}_rank{str(self.rank)}"
        self.log = logging.getLogger(__name__)
        self._init_serializer(do_print)
        self.grid_size = self._grid_size()

    def _init_serializer(self, do_print: bool):
        if not self.fname:
            self.log.warning(" WARNING: no filename! closing serializer")
        self.serializer = ser.Serializer(ser.OpenModeKind.Read, self.file_path, self.fname)
        if do_print:
            self.print_info()

    def print_info(self):
        self.log.info(f"SAVEPOINTS: {self.serializer.savepoint_list()}")
        self.log.info(f"FIELDNAMES: {self.serializer.fieldnames()}")

    def _grid_size(self):
        sp = self._get_icon_grid_savepoint()
        grid_sizes = {
            CellDim: self.serializer.read("num_cells", savepoint=sp).astype(int32)[0],
            EdgeDim: self.serializer.read("num_edges", savepoint=sp).astype(int32)[0],
            VertexDim: self.serializer.read("num_vert", savepoint=sp).astype(int32)[0],
            KDim: sp.metainfo.to_dict()["nlev"],
        }
        return grid_sizes

    def from_savepoint_grid(self) -> IconGridSavePoint:
        savepoint = self._get_icon_grid_savepoint()
        return IconGridSavePoint(savepoint, self.serializer, size=self.grid_size)

    def _get_icon_grid_savepoint(self):
        savepoint = self.serializer.savepoint["icon-grid"].id[1].as_savepoint()
        return savepoint

    def from_savepoint_diffusion_init(
        self,
        linit: bool,
        date: str,
    ) -> IconDiffusionInitSavepoint:
        savepoint = (
            self.serializer.savepoint["call-diffusion-init"].linit[linit].date[date].as_savepoint()
        )
        return IconDiffusionInitSavepoint(savepoint, self.serializer, size=self.grid_size)

    def from_interpolation_savepoint(self) -> InterpolationSavepoint:
        savepoint = self.serializer.savepoint["interpolation_state"].as_savepoint()
        return InterpolationSavepoint(savepoint, self.serializer, size=self.grid_size)

    def from_metrics_savepoint(self) -> MetricSavepoint:
        savepoint = self.serializer.savepoint["metric_state"].as_savepoint()
        return MetricSavepoint(savepoint, self.serializer, size=self.grid_size)

    def from_savepoint_diffusion_exit(self, linit: bool, date: str) -> IconDiffusionExitSavepoint:
        savepoint = (
            self.serializer.savepoint["call-diffusion-exit"].linit[linit].date[date].as_savepoint()
        )
        return IconDiffusionExitSavepoint(savepoint, self.serializer, size=self.grid_size)
