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
from gt4py.next import as_field
from gt4py.next.common import Dimension, DimensionKind
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.diffusion.diffusion import VectorTuple
from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.atmosphere.dycore.state_utils.diagnostic_state import DiagnosticState
from icon4py.model.atmosphere.dycore.state_utils.interpolation_state import InterpolationState
from icon4py.model.atmosphere.dycore.state_utils.metric_state import MetricStateNonHydro
from icon4py.model.common import dimension
from icon4py.model.common.decomposition.definitions import DecompositionInfo
from icon4py.model.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CECDim,
    CEDim,
    CellDim,
    E2C2EDim,
    E2C2EODim,
    E2C2VDim,
    E2CDim,
    E2VDim,
    ECDim,
    ECVDim,
    EdgeDim,
    KDim,
    V2CDim,
    V2EDim,
    VertexDim,
)
from icon4py.model.common.grid.base import GridConfig, VerticalGridSize
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams, HorizontalGridSize
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, flatten_first_two_dims


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
        buffer = self._reduce_to_dim_size(buffer, dimensions)

        self.log.debug(f"{name} {buffer.shape}")
        return as_field(dimensions, buffer)

    def _reduce_to_dim_size(self, buffer, dimensions):
        buffer_size = (
            self.sizes[d] if d.kind is DimensionKind.HORIZONTAL else s
            for s, d in zip(buffer.shape, dimensions)
        )
        buffer = buffer[tuple(map(slice, buffer_size))]
        return buffer

    def _get_field_from_ndarray(self, ar, *dimensions, dtype=float):
        ar = self._reduce_to_dim_size(ar, dimensions)
        return as_field(dimensions, ar)

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

    def inverse_primal_edge_lengths(self):
        return self._get_field("inv_primal_edge_length", EdgeDim)

    def inv_vert_vert_length(self):
        return self._get_field("inv_vert_vert_length", EdgeDim)

    def primal_normal_vert_x(self):
        return self._get_field("primal_normal_vert_x", EdgeDim, E2C2VDim)

    def primal_normal_vert_y(self):
        return self._get_field("primal_normal_vert_y", EdgeDim, E2C2VDim)

    def dual_normal_vert_y(self):
        return self._get_field("dual_normal_vert_y", EdgeDim, E2C2VDim)

    def dual_normal_vert_x(self):
        return self._get_field("dual_normal_vert_x", EdgeDim, E2C2VDim)

    def primal_normal_cell_x(self):
        return self._get_field("primal_normal_cell_x", EdgeDim, E2CDim)

    def primal_normal_cell_y(self):
        return self._get_field("primal_normal_cell_y", EdgeDim, E2CDim)

    def dual_normal_cell_x(self):
        return self._get_field("dual_normal_cell_x", EdgeDim, E2CDim)

    def dual_normal_cell_y(self):
        return self._get_field("dual_normal_cell_y", EdgeDim, E2CDim)

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

    def nflatlev(self):
        return self._read_int32_shift1("nflatlev")[0]

    def nflat_gradp(self):
        return self._read_int32_shift1("nflat_gradp")[0]

    def edge_end_index(self):
        # don't need to subtract 1, because FORTRAN slices  are inclusive [from:to] so the being
        # one off accounts for being exclusive [from:to)
        return self.serializer.read("e_end_index", self.savepoint)

    def c_owner_mask(self):
        return self._get_field("c_owner_mask", CellDim, dtype=bool)

    def e_owner_mask(self):
        return self._get_field("e_owner_mask", EdgeDim, dtype=bool)

    def f_e(self):
        return self._get_field("f_e", EdgeDim)

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

    def e2c2e(self):
        return self._get_connectivity_array("e2c2e", EdgeDim)

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
        e2c2e = self.e2c2e()
        c2e2c0 = np.column_stack(((np.asarray(range(c2e2c.shape[0]))), c2e2c))
        e2c2e0 = np.column_stack(((np.asarray(range(e2c2e.shape[0]))), e2c2e))
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
                    E2C2EDim: e2c2e,
                    E2C2EODim: e2c2e0,
                }
            )
            .with_connectivities(
                {
                    E2VDim: self.e2v(),
                    V2EDim: self.v2e(),
                    V2CDim: self.v2c(),
                    E2C2VDim: self.e2c2v(),
                }
            )
        )

        grid.update_size_connectivities(
            {
                ECVDim: grid.size[EdgeDim] * grid.size[E2C2VDim],
                CEDim: grid.size[CellDim] * grid.size[C2EDim],
                ECDim: grid.size[EdgeDim] * grid.size[E2CDim],
            }
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

        primal_normal_cell: VectorTuple = (
            as_1D_sparse_field(self.primal_normal_cell_x(), ECDim),
            as_1D_sparse_field(self.primal_normal_cell_y(), ECDim),
        )

        dual_normal_cell: VectorTuple = (
            as_1D_sparse_field(self.dual_normal_cell_x(), ECDim),
            as_1D_sparse_field(self.dual_normal_cell_y(), ECDim),
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
            primal_normal_cell_x=primal_normal_cell[0],
            dual_normal_cell_x=dual_normal_cell[0],
            primal_normal_cell_y=primal_normal_cell[1],
            dual_normal_cell_y=dual_normal_cell[1],
            edge_areas=self.edge_areas(),
            f_e=self.f_e(),
        )

    def construct_cell_geometry(self) -> CellParams:
        return CellParams(area=self.cell_areas())


class InterpolationSavepoint(IconSavepoint):
    def c_intp(self):
        return self._get_field("c_intp", VertexDim, V2CDim)

    def c_lin_e(self):
        return self._get_field("c_lin_e", EdgeDim, E2CDim)

    def e_bln_c_s(self):
        return self._get_field("e_bln_c_s", CellDim, C2EDim)

    def e_flx_avg(self):
        return self._get_field("e_flx_avg", EdgeDim, E2C2EODim)

    def geofac_div(self):
        return self._get_field("geofac_div", CellDim, C2EDim)

    def geofac_grdiv(self):
        return self._get_field("geofac_grdiv", EdgeDim, E2C2EODim)

    def geofac_grg(self):
        grg = np.squeeze(self.serializer.read("geofac_grg", self.savepoint))
        num_cells = self.sizes[CellDim]
        return as_field((CellDim, C2E2CODim), grg[:num_cells, :, 0]), as_field(
            (CellDim, C2E2CODim), grg[:num_cells, :, 1]
        )

    def zd_intcoef(self):
        return self._get_field("vcoef", CellDim, C2E2CDim, KDim)

    def geofac_n2s(self):
        return self._get_field("geofac_n2s", CellDim, C2E2CODim)

    def geofac_rot(self):
        return self._get_field("geofac_rot", VertexDim, V2EDim)

    def nudgecoeff_e(self):
        return self._get_field("nudgecoeff_e", EdgeDim)

    def pos_on_tplane_e_x(self):
        field = self._get_field("pos_on_tplane_e_x", EdgeDim, E2CDim)
        return as_1D_sparse_field(field[:, 0:2], ECDim)

    def pos_on_tplane_e_y(self):
        field = self._get_field("pos_on_tplane_e_y", EdgeDim, E2CDim)
        return as_1D_sparse_field(field[:, 0:2], ECDim)

    def rbf_vec_coeff_e(self):
        buffer = np.squeeze(
            self.serializer.read("rbf_vec_coeff_e", self.savepoint).astype(float)
        ).transpose()
        return as_field((EdgeDim, E2C2EDim), buffer)

    def rbf_vec_coeff_v1(self):
        return self._get_field("rbf_vec_coeff_v1", VertexDim, V2EDim)

    def rbf_vec_coeff_v2(self):
        return self._get_field("rbf_vec_coeff_v2", VertexDim, V2EDim)

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

    def construct_interpolation_state_for_nonhydro(self) -> InterpolationState:
        grg = self.geofac_grg()
        return InterpolationState(
            c_lin_e=self.c_lin_e(),
            c_intp=self.c_intp(),
            e_flx_avg=self.e_flx_avg(),
            geofac_grdiv=self.geofac_grdiv(),
            geofac_rot=self.geofac_rot(),
            pos_on_tplane_e_1=self.pos_on_tplane_e_x(),
            pos_on_tplane_e_2=self.pos_on_tplane_e_y(),
            rbf_vec_coeff_e=self.rbf_vec_coeff_e(),
            e_bln_c_s=as_1D_sparse_field(self.e_bln_c_s(), CEDim),
            rbf_coeff_1=self.rbf_vec_coeff_v1(),
            rbf_coeff_2=self.rbf_vec_coeff_v2(),
            geofac_div=as_1D_sparse_field(self.geofac_div(), CEDim),
            geofac_n2s=self.geofac_n2s(),
            geofac_grg_x=grg[0],
            geofac_grg_y=grg[1],
            nudgecoeff_e=self.nudgecoeff_e(),
        )


class MetricSavepoint(IconSavepoint):
    def bdy_halo_c(self):
        return self._get_field("bdy_halo_c", CellDim, dtype=bool)

    def d2dexdz2_fac1_mc(self):
        return self._get_field("d2dexdz2_fac1_mc", CellDim, KDim)

    def d2dexdz2_fac2_mc(self):
        return self._get_field("d2dexdz2_fac2_mc", CellDim, KDim)

    def d_exner_dz_ref_ic(self):
        return self._get_field("d_exner_dz_ref_ic", CellDim, KDim)

    def exner_exfac(self):
        return self._get_field("exner_exfac", CellDim, KDim)

    def exner_ref_mc(self):
        return self._get_field("exner_ref_mc", CellDim, KDim)

    def hmask_dd3d(self):
        return self._get_field("hmask_dd3d", EdgeDim)

    def inv_ddqz_z_full(self):
        return self._get_field("inv_ddqz_z_full", CellDim, KDim)

    def ipeidx_dsl(self):
        return self._get_field("ipeidx_dsl", EdgeDim, KDim, dtype=bool)

    def mask_prog_halo_c(self):
        return self._get_field("mask_prog_halo_c", CellDim, dtype=bool)

    def pg_exdist(self):
        return self._get_field("pg_exdist_dsl", EdgeDim, KDim)

    def rayleigh_w(self):
        return self._get_field("rayleigh_w", KDim)

    def rho_ref_mc(self):
        return self._get_field("rho_ref_mc", CellDim, KDim)

    def rho_ref_me(self):
        return self._get_field("rho_ref_me", EdgeDim, KDim)

    def scalfac_dd3d(self):
        return self._get_field("scalfac_dd3d", KDim)

    def theta_ref_ic(self):
        return self._get_field("theta_ref_ic", CellDim, KDim)

    def theta_ref_me(self):
        return self._get_field("theta_ref_me", EdgeDim, KDim)

    def vwind_expl_wgt(self):
        return self._get_field("vwind_expl_wgt", CellDim)

    def vwind_impl_wgt(self):
        return self._get_field("vwind_impl_wgt", CellDim)

    def wgtfacq_c_dsl(self):
        return self._get_field("wgtfacq_c_dsl", CellDim, KDim)

    def wgtfacq_c(self):
        return self._get_field("wgtfacq_c", CellDim, KDim)

    def zdiff_gradp(self):
        field = self._get_field("zdiff_gradp_dsl", EdgeDim, E2CDim, KDim)
        return flatten_first_two_dims(ECDim, KDim, field=field)

    def vertoffset_gradp(self):
        field = self._get_field("vertoffset_gradp_dsl", EdgeDim, E2CDim, KDim, dtype=int32)
        return flatten_first_two_dims(ECDim, KDim, field=field)

    def coeff1_dwdz(self):
        return self._get_field("coeff1_dwdz", CellDim, KDim)

    def coeff2_dwdz(self):
        return self._get_field("coeff2_dwdz", CellDim, KDim)

    def coeff_gradekin(self):
        field = self._get_field("coeff_gradekin", EdgeDim, E2CDim)
        return as_1D_sparse_field(field, ECDim)

    def ddqz_z_full_e(self):
        return self._get_field("ddqz_z_full_e", EdgeDim, KDim)

    def ddqz_z_half(self):
        return self._get_field("ddqz_z_half", CellDim, KDim)

    def ddxn_z_full(self):
        return self._get_field("ddxn_z_full", EdgeDim, KDim)

    def ddxt_z_full(self):
        return self._get_field("ddxt_z_full", EdgeDim, KDim)

    def mask_hdiff(self):
        return self._get_field("mask_hdiff", CellDim, KDim, dtype=bool)

    def theta_ref_mc(self):
        return self._get_field("theta_ref_mc", CellDim, KDim)

    def wgtfac_c(self):
        return self._get_field("wgtfac_c", CellDim, KDim)

    def wgtfac_e(self):
        return self._get_field("wgtfac_e", EdgeDim, KDim)

    def wgtfacq_e_dsl(
        self, k_level
    ):  # TODO: @abishekg7 Simplify this after serialized data is fixed
        ar = np.squeeze(self.serializer.read("wgtfacq_e", self.savepoint))
        k = k_level - 3
        ar = np.pad(ar[:, ::-1], ((0, 0), (k, 0)), "constant", constant_values=(0.0,))
        return self._get_field_from_ndarray(ar, EdgeDim, KDim)

    def zd_diffcoef(self):
        return self._get_field("zd_diffcoef", CellDim, KDim)

    def zd_intcoef(self):
        return self._read_and_reorder_sparse_field("vcoef")

    def _read_and_reorder_sparse_field(self, name: str, sparse_size=3):
        ser_input = np.squeeze(self.serializer.read(name, self.savepoint))[:, :, :]
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
        return as_field(target_dims, data.reshape(old_shape[0] * old_shape[1], old_shape[2]))

    def zd_vertoffset(self):
        return self._read_and_reorder_sparse_field("zd_vertoffset")

    def zd_vertidx(self):
        return np.squeeze(self.serializer.read("zd_vertidx", self.savepoint))

    def zd_indlist(self):
        return np.squeeze(self.serializer.read("zd_indlist", self.savepoint))

    def construct_nh_metric_state(self, num_k_lev) -> MetricStateNonHydro:
        return MetricStateNonHydro(
            bdy_halo_c=self.bdy_halo_c(),
            mask_prog_halo_c=self.mask_prog_halo_c(),
            rayleigh_w=self.rayleigh_w(),
            exner_exfac=self.exner_exfac(),
            exner_ref_mc=self.exner_ref_mc(),
            wgtfac_c=self.wgtfac_c(),
            wgtfacq_c_dsl=self.wgtfacq_c_dsl(),
            inv_ddqz_z_full=self.inv_ddqz_z_full(),
            rho_ref_mc=self.rho_ref_mc(),
            theta_ref_mc=self.theta_ref_mc(),
            vwind_expl_wgt=self.vwind_expl_wgt(),
            d_exner_dz_ref_ic=self.d_exner_dz_ref_ic(),
            ddqz_z_half=self.ddqz_z_half(),
            theta_ref_ic=self.theta_ref_ic(),
            d2dexdz2_fac1_mc=self.d2dexdz2_fac1_mc(),
            d2dexdz2_fac2_mc=self.d2dexdz2_fac2_mc(),
            rho_ref_me=self.rho_ref_me(),
            theta_ref_me=self.theta_ref_me(),
            ddxn_z_full=self.ddxn_z_full(),
            zdiff_gradp=self.zdiff_gradp(),
            vertoffset_gradp=self.vertoffset_gradp(),
            ipeidx_dsl=self.ipeidx_dsl(),
            pg_exdist=self.pg_exdist(),
            ddqz_z_full_e=self.ddqz_z_full_e(),
            ddxt_z_full=self.ddxt_z_full(),
            wgtfac_e=self.wgtfac_e(),
            wgtfacq_e_dsl=self.wgtfacq_e_dsl(num_k_lev),
            vwind_impl_wgt=self.vwind_impl_wgt(),
            hmask_dd3d=self.hmask_dd3d(),
            scalfac_dd3d=self.scalfac_dd3d(),
            coeff1_dwdz=self.coeff1_dwdz(),
            coeff2_dwdz=self.coeff2_dwdz(),
            coeff_gradekin=self.coeff_gradekin(),
        )

    def construct_metric_state_for_diffusion(self) -> DiffusionMetricState:
        return DiffusionMetricState(
            mask_hdiff=self.mask_hdiff(),
            theta_ref_mc=self.theta_ref_mc(),
            wgtfac_c=self.wgtfac_c(),
            zd_intcoef=self.zd_intcoef(),
            zd_vertoffset=self.zd_vertoffset(),
            zd_diffcoef=self.zd_diffcoef(),
        )


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

    def rho(self):
        return self._get_field("rho", CellDim, KDim)

    def construct_prognostics(self) -> PrognosticState:
        return PrognosticState(
            w=self.w(),
            vn=self.vn(),
            exner=self.exner(),
            theta_v=self.theta_v(),
            rho=self.rho(),
        )

    def construct_diagnostics_for_diffusion(self) -> DiffusionDiagnosticState:
        return DiffusionDiagnosticState(
            hdef_ic=self.hdef_ic(),
            div_ic=self.div_ic(),
            dwdx=self.dwdx(),
            dwdy=self.dwdy(),
        )

    def construct_diagnostics(self) -> DiagnosticState:
        return DiagnosticState(
            hdef_ic=self.hdef_ic(),
            div_ic=self.div_ic(),
            dwdx=self.dwdx(),
            dwdy=self.dwdy(),
            vt=None,
            vn_ie=None,
            w_concorr_c=None,
            ddt_w_adv_pc_before=None,
            ddt_vn_apc_pc_before=None,
            ntnd=None,
        )


class IconNonHydroInitSavepoint(IconSavepoint):
    def bdy_divdamp(self):
        return self._get_field("bdy_divdamp", KDim)

    def ddt_exner_phy(self):
        return self._get_field("ddt_exner_phy", CellDim, KDim)

    def ddt_vn_phy(self):
        return self._get_field("ddt_vn_phy", EdgeDim, KDim)

    def exner_now(self):
        return self._get_field("exner_now", CellDim, KDim)

    def exner_new(self):
        return self._get_field("exner_new", CellDim, KDim)

    def theta_v_now(self):
        return self._get_field("theta_v_now", CellDim, KDim)

    def theta_v_new(self):
        return self._get_field("theta_v_new", CellDim, KDim)

    def rho_now(self):
        return self._get_field("rho_now", CellDim, KDim)

    def rho_new(self):
        return self._get_field("rho_new", CellDim, KDim)

    def exner_pr(self):
        return self._get_field("exner_pr", CellDim, KDim)

    def grf_tend_rho(self):
        return self._get_field("grf_tend_rho", CellDim, KDim)

    def grf_tend_thv(self):
        return self._get_field("grf_tend_thv", CellDim, KDim)

    def grf_tend_vn(self):
        return self._get_field("grf_tend_vn", EdgeDim, KDim)

    def ddt_vn_adv_ntl(self, ntl):
        buffer = np.squeeze(self.serializer.read("ddt_vn_adv_ntl", self.savepoint).astype(float))
        return as_field((EdgeDim, KDim), buffer[:, :, ntl - 1])

    def ddt_w_adv_ntl(self, ntl):
        buffer = np.squeeze(self.serializer.read("ddt_w_adv_ntl", self.savepoint).astype(float))
        return as_field((CellDim, KDim), buffer[:, :, ntl - 1])

    def grf_tend_w(self):
        return self._get_field("grf_tend_w", CellDim, KDim)

    def mass_fl_e(self):
        return self._get_field("mass_fl_e", EdgeDim, KDim)

    def mass_flx_me(self):
        return self._get_field("prep_adv_mass_flx_me", EdgeDim, KDim)

    def mass_flx_ic(self):
        return self._get_field("prep_adv_mass_flx_ic", CellDim, KDim)

    def rho_ic(self):
        return self._get_field("rho_ic", CellDim, KDim)

    def rho_incr(self):
        return self._get_field("rho_incr", CellDim, KDim)

    def exner_incr(self):
        return self._get_field("exner_incr", CellDim, KDim)

    def vn_incr(self):
        return self._get_field("vn_incr", EdgeDim, KDim)

    def scal_divdamp(self) -> float:
        return self.serializer.read("scal_divdamp", self.savepoint)[0]

    def scal_divdamp_o2(self) -> float:
        return self.serializer.read("scal_divdamp_o2", self.savepoint)[0]

    def theta_v_ic(self):
        return self._get_field("theta_v_ic", CellDim, KDim)

    def vn_traj(self):
        return self._get_field("prep_adv_vn_traj", EdgeDim, KDim)

    def z_dwdz_dd(self):
        return self._get_field("z_dwdz_dd", CellDim, KDim)

    def z_graddiv_vn(self):
        return self._get_field("z_graddiv_vn", EdgeDim, KDim)

    def z_theta_v_e(self):
        return self._get_field("z_theta_v_e", EdgeDim, KDim)

    def z_rho_e(self):
        return self._get_field("z_rho_e", EdgeDim, KDim)

    def z_gradh_exner(self):
        return self._get_field("z_gradh_exner", EdgeDim, KDim)

    def z_w_expl(self):
        return self._get_field("z_w_expl", CellDim, KDim)

    def z_rho_expl(self):
        return self._get_field("z_rho_expl", CellDim, KDim)

    def z_exner_expl(self):
        return self._get_field("z_exner_expl", CellDim, KDim)

    def z_alpha(self):
        return self._get_field("z_alpha", CellDim, KDim)

    def z_beta(self):
        return self._get_field("z_beta", CellDim, KDim)

    def z_contr_w_fl_l(self):
        return self._get_field("z_contr_w_fl_l", CellDim, KDim)

    def z_q(self):
        return self._get_field("z_q", CellDim, KDim)

    def wgt_nnow_rth(self) -> float:
        return self.serializer.read("wgt_nnow_rth", self.savepoint)[0]

    def wgt_nnew_rth(self) -> float:
        return self.serializer.read("wgt_nnew_rth", self.savepoint)[0]

    def wgt_nnow_vel(self) -> float:
        return self.serializer.read("wgt_nnow_vel", self.savepoint)[0]

    def wgt_nnew_vel(self) -> float:
        return self.serializer.read("wgt_nnew_vel", self.savepoint)[0]

    def w_now(self):
        return self._get_field("w_now", CellDim, KDim)

    def w_new(self):
        return self._get_field("w_new", CellDim, KDim)

    def vn_now(self):
        return self._get_field("vn_now", EdgeDim, KDim)

    def vn_new(self):
        return self._get_field("vn_new", EdgeDim, KDim)


class IconVelocityInitSavepoint(IconSavepoint):
    def cfl_w_limit(self) -> float:
        return self.serializer.read("cfl_w_limit", self.savepoint)[0]

    def ddt_vn_apc_pc(self, ntnd):
        buffer = np.squeeze(self.serializer.read("ddt_vn_apc_pc", self.savepoint).astype(float))

        return as_field((EdgeDim, KDim), buffer[:, :, ntnd - 1])

    def ddt_w_adv_pc(self, ntnd):
        buffer = np.squeeze(self.serializer.read("ddt_w_adv_pc", self.savepoint).astype(float))
        return as_field((CellDim, KDim), buffer[:, :, ntnd - 1])

    def scalfac_exdiff(self) -> float:
        return self.serializer.read("scalfac_exdiff", self.savepoint)[0]

    def vn(self):
        return self._get_field("vn", EdgeDim, KDim)

    def vn_ie(self):
        return self._get_field("vn_ie", EdgeDim, KDim)

    def vt(self):
        return self._get_field("vt", EdgeDim, KDim)

    def w(self):
        return self._get_field("w", CellDim, KDim)

    def z_vt_ie(self):
        return self._get_field("z_vt_ie", EdgeDim, KDim)

    def z_kin_hor_e(self):
        return self._get_field("z_kin_hor_e", EdgeDim, KDim)

    def z_w_concorr_me(self):
        return self._get_field("z_w_concorr_me", EdgeDim, KDim)

    def w_concorr_c(self):
        return self._get_field("w_concorr_c", CellDim, KDim)


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


class IconExitSavepoint(IconSavepoint):
    def rho_new(self):
        return self._get_field("x_rho_new", CellDim, KDim)

    def rho_now(self):
        return self._get_field("x_rho_now", CellDim, KDim)

    def exner_now(self):
        return self._get_field("x_exner_now", CellDim, KDim)

    def theta_v_now(self):
        return self._get_field("x_theta_v_now", CellDim, KDim)

    def ddt_vn_apc_pc(self, ntnd):
        buffer = np.squeeze(self.serializer.read("x_ddt_vn_apc_pc", self.savepoint).astype(float))[
            :, :, ntnd - 1
        ]
        dims = (EdgeDim, KDim)
        buffer = self._reduce_to_dim_size(buffer, dims)
        return as_field((EdgeDim, KDim), buffer)

    def ddt_vn_apc_pc_19(self, ntnd):
        buffer = np.squeeze(
            self.serializer.read("x_ddt_vn_apc_pc_19", self.savepoint).astype(float)
        )
        return as_field((EdgeDim, KDim), buffer[:, :, ntnd - 1])

    def ddt_w_adv_pc(self, ntnd):
        buffer = np.squeeze(self.serializer.read("x_ddt_w_adv_pc", self.savepoint).astype(float))[
            :, :, ntnd - 1
        ]
        dims = (CellDim, KDim)
        buffer = self._reduce_to_dim_size(buffer, dims)
        return as_field(dims, buffer)

    def ddt_w_adv_pc_16(self, ntnd):
        buffer = np.squeeze(self.serializer.read("x_ddt_w_adv_pc_16", self.savepoint).astype(float))
        return as_field((CellDim, KDim), buffer[:, :, ntnd - 1])

    def ddt_w_adv_pc_17(self, ntnd):
        buffer = np.squeeze(self.serializer.read("x_ddt_w_adv_pc_17", self.savepoint).astype(float))
        return as_field((CellDim, KDim, buffer[:, :, ntnd - 1]))

    def scalfac_exdiff(self) -> float:
        return self.serializer.read("scalfac_exdiff", self.savepoint)[0]

    def vn_ie(self):
        return self._get_field("x_vn_ie", EdgeDim, KDim)

    def vt(self):
        return self._get_field("x_vt", EdgeDim, KDim)

    def z_kin_hor_e(self):
        return self._get_field("x_z_kin_hor_e", EdgeDim, KDim)

    def z_ekinh(self):
        return self._get_field("x_z_ekinh", CellDim, KDim)

    def z_vt_ie(self):
        return self._get_field("x_z_vt_ie", EdgeDim, KDim)

    def z_v_grad_w(self):
        return self._get_field("x_z_v_grad_w", EdgeDim, KDim)

    def z_w_v(self):
        return self._get_field("x_z_w_v", VertexDim, KDim)

    def z_w_concorr_me(self):
        return self._get_field("x_z_w_concorr_me", EdgeDim, KDim)

    def z_w_concorr_mc(self):
        return self._get_field("x_z_w_concorr_mc", CellDim, KDim)

    def z_w_con_c_full(self):
        return self._get_field("x_z_w_con_c_full", CellDim, KDim)

    def z_w_con_c(self):
        return self._get_field("x_z_w_con_c", CellDim, KDim)

    def cfl_clipping(self):
        return self._get_field("x_cfl_clipping", CellDim, KDim, dtype=bool)

    def vcfl(self):
        return self._get_field("x_vcfl_dsl", CellDim, KDim)

    def exner_new(self):
        return self._get_field("x_exner_new", CellDim, KDim)

    def z_exner_ex_pr(self):
        return self._get_field("x_z_exner_ex_pr", CellDim, KDim)

    def z_exner_ic(self):
        return self._get_field("x_z_exner_ic", CellDim, KDim)

    def exner_pr(self):
        return self._get_field("x_exner_pr", CellDim, KDim)

    def mass_fl_e(self):
        return self._get_field("x_mass_fl_e", EdgeDim, KDim)

    def z_theta_v_fl_e(self):
        return self._get_field("x_z_theta_v_fl_e", EdgeDim, KDim)

    def mass_flx_me(self):
        return self._get_field("x_prep_adv_mass_flx_me", EdgeDim, KDim)

    def vn_traj(self):
        return self._get_field("x_prep_adv_vn_traj", EdgeDim, KDim)

    def rho_ic(self):
        return self._get_field("x_rho_ic", CellDim, KDim)

    def theta_v_ic(self):
        return self._get_field("x_theta_v_ic", CellDim, KDim)

    def theta_v_new(self):
        return self._get_field("x_theta_v_new", CellDim, KDim)

    def vn_new(self):
        return self._get_field("x_vn_new", EdgeDim, KDim)

    def w_concorr_c(self):
        return self._get_field("x_w_concorr_c", CellDim, KDim)

    def w_new(self):
        return self._get_field("x_w_new", CellDim, KDim)

    def z_dexner_dz_c(self, ntnd):
        buffer = np.squeeze(self.serializer.read("x_z_dexner_dz_c", self.savepoint).astype(float))[
            :, :, ntnd - 1
        ]
        buffer = self._reduce_to_dim_size(buffer, (CellDim, KDim))
        return as_field((CellDim, KDim), buffer)

    def z_rth_pr(self, ind):
        buffer = np.squeeze(self.serializer.read("x_z_rth_pr", self.savepoint).astype(float))[
            :, :, ind - 1
        ]
        buffer = self._reduce_to_dim_size(buffer, (CellDim, KDim))
        return as_field((CellDim, KDim), buffer)

    def z_th_ddz_exner_c(self):
        return self._get_field("x_z_th_ddz_exner_c", CellDim, KDim)

    def z_gradh_exner(self):
        return self._get_field("x_z_gradh_exner", EdgeDim, KDim)

    def z_hydro_corr(self):
        return self._get_field("x_z_hydro_corr", EdgeDim, KDim)

    def z_flxdiv_mass(self):
        return self._get_field("x_z_flxdiv_mass", CellDim, KDim)

    def z_flxdiv_theta(self):
        return self._get_field("x_z_flxdiv_theta", CellDim, KDim)

    def z_contr_w_fl_l(self):
        return self._get_field("x_z_contr_w_fl", CellDim, KDim)

    def z_w_expl(self):
        return self._get_field("x_z_w_expl", CellDim, KDim)

    def z_alpha(self):
        return self._get_field("x_z_alpha", CellDim, KDim)

    def z_beta(self):
        return self._get_field("x_z_beta", CellDim, KDim)

    def z_q(self):
        return self._get_field("x_z_q", CellDim, KDim)

    def z_rho_expl(self):
        return self._get_field("x_z_rho_expl", CellDim, KDim)

    def z_exner_expl(self):
        return self._get_field("x_z_exner_expl", CellDim, KDim)

    def z_theta_v_pr_ic(self):
        return self._get_field("x_z_theta_v_pr_ic", CellDim, KDim)

    def z_rho_e(self):
        return self._get_field("x_z_rho_e", EdgeDim, KDim)

    def z_theta_v_e(self):
        return self._get_field("x_z_theta_v_e", EdgeDim, KDim)

    def z_vn_avg(self):
        return self._get_field("x_z_vn_avg", EdgeDim, KDim)

    def z_graddiv_vn(self):
        return self._get_field("x_z_graddiv_vn", EdgeDim, KDim)

    def z_grad_rth(self, ind):
        buffer = np.squeeze(self.serializer.read("x_z_grad_rth", self.savepoint).astype(float))[
            :, :, ind - 1
        ]

        dims = (CellDim, KDim)
        return as_field(dims, self._reduce_to_dim_size(buffer, dims))

    def z_gradh_exner_18(self):
        return self._get_field("x_z_gradh_exner_18", EdgeDim, KDim)

    def z_gradh_exner_19(self):
        return self._get_field("x_z_gradh_exner_19", EdgeDim, KDim)

    def z_gradh_exner_20(self):
        return self._get_field("x_z_gradh_exner_20", EdgeDim, KDim)

    def z_gradh_exner_22(self):
        return self._get_field("x_z_gradh_exner_22", EdgeDim, KDim)

    def vn_new_23(self):
        return self._get_field("x_vn_new_23", EdgeDim, KDim)

    def vn_new_24(self):
        return self._get_field("x_vn_new_24", EdgeDim, KDim)

    def vn_new_26(self):
        return self._get_field("x_vn_new_26", EdgeDim, KDim)

    def vn_new_27(self):
        return self._get_field("x_vn_new_27", EdgeDim, KDim)

    def vn_new_29(self):
        return self._get_field("x_vn_new_29", EdgeDim, KDim)

    def z_rho_e_01(self):
        return self._get_field("x_z_rho_e_01", EdgeDim, KDim)

    def z_theta_v_e_01(self):
        return self._get_field("x_z_theta_v_e_01", EdgeDim, KDim)

    def z_rho_e_00(self):
        return self._get_field("x_z_rho_e_00", EdgeDim, KDim)

    def z_theta_v_e_00(self):
        return self._get_field("x_z_theta_v_e_00", EdgeDim, KDim)

    def w_new_52(self):
        return self._get_field("x_w_new_52", CellDim, KDim)

    def w_new_53(self):
        return self._get_field("x_w_new_53", CellDim, KDim)

    def exner_new_55(self):
        return self._get_field("x_exner_new_55", CellDim, KDim)

    def theta_v_new_55(self):
        return self._get_field("x_theta_v_new_55", CellDim, KDim)

    def rho_new_55(self):
        return self._get_field("x_rho_new_55", CellDim, KDim)

    def w_10_start(self):
        return self._get_field("w_10_start", CellDim, KDim)

    def z_graddiv_vn_26_start(self):
        return self._get_field("z_graddiv_vn_26_start", EdgeDim, KDim)

    def z_graddiv2_vn_27_start(self):
        return self._get_field("z_graddiv2_vn_27_start", EdgeDim, KDim)

    def x_vn_new_26_end(self):
        return self._get_field("x_vn_new_26_end", EdgeDim, KDim)

    def x_vn_new_27_end(self):
        return self._get_field("x_vn_new_27_end", EdgeDim, KDim)

    def vn_26_start(self):
        return self._get_field("vn_26_start", EdgeDim, KDim)

    def z_dwdz_dd(self):
        return self._get_field("x_z_dwdz_dd", CellDim, KDim)


class IconNHFinalExitSavepoint(IconSavepoint):
    def theta_v_new(self):
        return self._get_field("x_theta_v", CellDim, KDim)

    def exner_new(self):
        return self._get_field("x_exner", CellDim, KDim)


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

    def from_savepoint_velocity_init(
        self, istep: int, vn_only: bool, date: str, jstep: int
    ) -> IconVelocityInitSavepoint:
        savepoint = (
            self.serializer.savepoint["call-velocity-tendencies"]
            .istep[istep]
            .vn_only[vn_only]
            .date[date]
            .jstep[jstep]
            .as_savepoint()
        )
        return IconVelocityInitSavepoint(savepoint, self.serializer, size=self.grid_size)

    def from_savepoint_nonhydro_init(
        self, istep: int, date: str, jstep: int
    ) -> IconNonHydroInitSavepoint:
        savepoint = (
            self.serializer.savepoint["solve_nonhydro"]
            .istep[istep]
            .date[date]
            .jstep[jstep]
            .as_savepoint()
        )
        return IconNonHydroInitSavepoint(savepoint, self.serializer, size=self.grid_size)

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

    def from_savepoint_velocity_exit(
        self, istep: int, vn_only: bool, date: str, jstep: int
    ) -> IconExitSavepoint:
        savepoint = (
            self.serializer.savepoint["call-velocity-tendencies"]
            .istep[istep]
            .vn_only[vn_only]
            .date[date]
            .jstep[jstep]
            .as_savepoint()
        )
        return IconExitSavepoint(savepoint, self.serializer, size=self.grid_size)

    def from_savepoint_nonhydro_exit(self, istep: int, date: str, jstep: int) -> IconExitSavepoint:
        savepoint = (
            self.serializer.savepoint["solve_nonhydro"]
            .istep[istep]
            .date[date]
            .jstep[jstep]
            .as_savepoint()
        )
        return IconExitSavepoint(savepoint, self.serializer, size=self.grid_size)

    def from_savepoint_nonhydro_step_exit(self, date: str, jstep: int) -> IconNHFinalExitSavepoint:
        savepoint = (
            self.serializer.savepoint["solve_nonhydro_step"].date[date].jstep[jstep].as_savepoint()
        )
        return IconNHFinalExitSavepoint(savepoint, self.serializer, size=self.grid_size)
