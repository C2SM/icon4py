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
from gt4py.next.common import Dimension
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.iterator.embedded import np_as_located_field

from atm_dyn_iconam.tests.test_utils.helpers import as_1D_sparse_field
from icon4py.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CellDim,
    E2C2EDim,
    E2C2EODim,
    E2C2VDim,
    E2CDim,
    ECDim,
    EdgeDim,
    KDim,
    E2VDim,
    KHalfDim,
    V2CDim,
    V2EDim,
    VertexDim,
)
from icon4py.state_utils.diagnostic_state import DiagnosticState
from icon4py.diffusion.diffusion import VectorTuple
from icon4py.state_utils.horizontal import (
    CellParams,
    EdgeParams,
    HorizontalMeshSize,
)
from icon4py.state_utils.icon_grid import IconGrid, MeshConfig, VerticalMeshConfig
from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.state_utils.metric_state import MetricState
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.state_utils.utils import as_1D_sparse_field


class IconSavepoint:
    def __init__(self, sp: ser.Savepoint, ser: ser.Serializer):
        self.savepoint = sp
        self.serializer = ser
        self.log = logging.getLogger((__name__))

    def log_meta_info(self):
        self.log.info(self.savepoint.metainfo)

    def _get_field(self, name, *dimensions, dtype=float):
        buffer = np.squeeze(self.serializer.read(name, self.savepoint).astype(dtype))
        self.log.debug(f"{name} {buffer.shape}")
        return np_as_located_field(*dimensions)(buffer)

    def get_metadata(self, *names):
        metadata = self.savepoint.metainfo.to_dict()
        return {n: metadata[n] for n in names if n in metadata}

    def _read_int32_shift1(self, name: str):
        """
        Read a index field and shift it by -1.

        use for start indeces: the shift accounts for the zero based python
        values are converted to int32
        """
        return (self.serializer.read(name, self.savepoint) - 1).astype(int32)

    def _read_int32(self, name: str):
        """
        Read a int field by name.

        use this for end indices: because FORTRAN slices  are inclusive [from:to] _and_ one based
        this accounts for being exclusive python exclusive bounds: [from:to)
        field values are convert to int32
        """
        return self.serializer.read(name, self.savepoint).astype(int32)

    def read_int(self, name: str):
        buffer = self.serializer.read(name, self.savepoint).astype(int)
        self.log.debug(f"{name} {buffer.shape}")
        return buffer


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
        return self._get_field("primal_normal_vert_x", VertexDim, E2C2VDim)

    def primal_normal_vert_y(self):
        return self._get_field("primal_normal_vert_y", VertexDim, E2C2VDim)

    def dual_normal_vert_y(self):
        return self._get_field("dual_normal_vert_y", VertexDim, E2C2VDim)

    def dual_normal_vert_x(self):
        return self._get_field("dual_normal_vert_x", VertexDim, E2C2VDim)

    def primal_normal_cell_x(self):
        return self._get_field("primal_normal_cell_x", ECDim)

    def dual_normal_cell_x(self):
        return self._get_field("dual_normal_cell_x", ECDim)

    def primal_normal_cell_y(self):
        return self._get_field("primal_normal_cell_y", ECDim)

    def dual_normal_cell_y(self):
        return self._get_field("dual_normal_cell_y", ECDim)

    def owner_mask(self):
        return self._get_field("owner_mask", CellDim, dtype=bool)

    def f_e(self):
        return self._get_field("f_e", EdgeDim)

    def cell_areas(self):
        return self._get_field("cell_areas", CellDim)

    def edge_areas(self):
        return self._get_field("edge_areas", EdgeDim)

    def inv_dual_edge_length(self):
        return self._get_field("inv_dual_edge_length", EdgeDim)

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

    def print_connectivity_info(self, name: str, ar: np.ndarray):
        self.log.debug(f" connectivity {name} {ar.shape}")

    def refin_ctrl(self, dim: Dimension):
        if dim == CellDim:
            return self.serializer.read("c_refin_ctl", self.savepoint)
        elif dim == EdgeDim:
            return self.serializer.read("e_refin_ctl", self.savepoint)
        elif dim == VertexDim:
            return self.serializer.read("v_refin_ctl", self.savepoint)
        else:
            return None

    def c2e(self):
        return self._get_connectiviy_array("c2e")

    def _get_connectiviy_array(self, name: str):
        connectivity = self.serializer.read(name, self.savepoint) - 1
        self.log.debug(f" connectivity {name} : {connectivity.shape}")
        return connectivity

    def c2e2c(self):
        return self._get_connectiviy_array("c2e2c")

    def e2c(self):
        return self._get_connectiviy_array("e2c")

    def e2v(self):
        # array "e2v" is actually e2c2v
        v_ = self._get_connectiviy_array("e2v")[:, 0:2]
        self.log.debug(f"real e2v {v_.shape}")
        return v_

    def e2c2v(self):
        # array "e2v" is actually e2c2v
        return self._get_connectiviy_array("e2v")

    def v2e(self):
        return self._get_connectiviy_array("v2e")

    def v2c(self):
        return self._get_connectivity_array("v2c")

    def construct_icon_grid(self) -> IconGrid:
        sp_meta = self.get_metadata(
            "nproma", "nlev", "num_vert", "num_cells", "num_edges"
        )
        cell_starts = self.cells_start_index()
        cell_ends = self.cells_end_index()
        vertex_starts = self.vertex_start_index()
        vertex_ends = self.vertex_end_index()
        edge_starts = self.edge_start_index()
        edge_ends = self.edge_end_index()
        config = MeshConfig(
            HorizontalMeshSize(
                num_vertices=sp_meta["nproma"],  # or rather "num_vert"
                num_cells=sp_meta["nproma"],  # or rather "num_cells"
                num_edges=sp_meta["nproma"],  # or rather "num_edges"
            ),
            VerticalMeshConfig(num_lev=sp_meta["nlev"]),
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
            .with_connectivities(
                {E2VDim: self.e2v(), V2EDim: self.v2e(), E2C2VDim: self.e2c2v()}
            )
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

class IconNonHydroInitSavepoint(IconSavepoint):
    def exner_exfac(self):
        return self._get_field("exner_exfac", CellDim, KDim)

    def exner_ref_mc(self):
        return self._get_field("exner_ref_mc", CellDim, KDim)

    def wgtfacq_c(self):
        return self._get_field("wgtfacq_c", CellDim, KDim)

    def inv_ddqz_z_full(self):
        return self._get_field("inv_ddqz_z_full", CellDim, KDim)

    def rho_ref_mc(self):
        return self._get_field("rho_ref_mc", CellDim, KDim)

    def theta_ref_mc(self):
        return self._get_field("theta_ref_mc", CellDim, KDim)

    def vwind_expl_wgt(self):
        return self._get_field("vwind_expl_wgt", CellDim)

    def d_exner_dz_ref_ic(self):
        return self._get_field("d_exner_dz_ref_ic", CellDim, KDim)

    def ddqz_z_half(self):
        return self._get_field("ddqz_z_half", CellDim, KDim)  # half KDim ?

    def theta_ref_ic(self):
        return self._get_field("theta_ref_ic", CellDim, KDim)

    def d2dexdz2_fac1_mc(self):
        return self._get_field("d2dexdz2_fac1_mc", CellDim, KDim)

    def d2dexdz2_fac2_mc(self):
        return self._get_field("d2dexdz2_fac2_mc", CellDim, KDim)

    def vwind_impl_wgt(self):
        return self._get_field("vwind_impl_wgt", CellDim)

    def bdy_halo_c(self):
        return self._get_field("bdy_halo_c", CellDim, dtype=bool)

    def ipeidx_dsl(self):
        return self._get_field("ipeidx_dsl", EdgeDim, KDim, dtype=bool)

    def pg_exdist(self):
        return self._get_field("pg_exdist", EdgeDim, KDim)

    def hmask_dd3d(self):
        return self._get_field("hmask_dd3d", EdgeDim)

    def scalfac_dd3d(self):
        return self._get_field("scalfac_dd3d", KDim)

    def rayleigh_w(self):
        return self._get_field("rayleigh_w", KDim)

    def rho_ref_me(self):
        return self._get_field("rho_ref_me", EdgeDim, KDim)

    def theta_ref_me(self):
        return self._get_field("theta_ref_me", EdgeDim, KDim)

    def mask_prog_halo_c(self):
        return self._get_field("mask_prog_halo_c", CellDim, KDim)

    def mask_hdiff(self):
        return self._get_field("mask_hdiff", CellDim, KDim)

    def pos_on_tplane_e(self):
        return self._get_field("pos_on_tplane_e", ECDim)

    def e_flx_avg(self):
        return self._get_field("e_flx_avg", EdgeDim, E2C2EODim)

    def rho(self):
        return self._get_field("rho", CellDim, KDim)

    def exner(self):
        return self._get_field("exner", CellDim, KDim)

    def theta_v_ic(self):
        return self._get_field("theta_v_ic", CellDim, KDim)

    def exner_pr(self):
        return self._get_field("exner_pr", CellDim, KDim)

    def rho_ic(self):
        return self._get_field("rho_ic", CellDim, KDim)

    def ddt_exner_phy(self):
        return self._get_field("ddt_exner_phy", CellDim, KDim)

    def grf_tend_rho(self):
        return self._get_field("grf_tend_rho", CellDim, KDim)

    def grf_tend_thv(self):
        return self._get_field("grf_tend_thv", CellDim, KDim)

    def grf_tend_w(self):
        return self._get_field("grf_tend_w", CellDim, KDim)

    def mass_fl_e(self):
        return self._get_field("mass_fl_e", EdgeDim, KDim)

    def ddt_vn_phy(self):
        return self._get_field("ddt_vn_phy", EdgeDim, KDim)

    def grf_tend_vn(self):
        return self._get_field("grf_tend_vn", EdgeDim, KDim)

    def rho_incr(self):
        return self._get_field("rho_incr", EdgeDim, KDim)

    def vn_incr(self):
        return self._get_field("vn_incr", EdgeDim, KDim)

    def exner_incr(self):
        return self._get_field("exner_incr", EdgeDim, KDim)

    def vn_traj(self):
        return self._get_field("vn_traj", EdgeDim, KDim)

    def mass_flx_me(self):
        return self._get_field("mass_flx_me", EdgeDim, KDim)

    def bdy_divdamp(self):
        return self._get_field("bdy_divdamp", KDim)

    def scal_divdamp_o2(self) -> float:
        return self.serializer.read("scal_divdamp_o2", self.savepoint)[0]

class IconVelocityInitSavepoint(IconSavepoint):
    def c_lin_e(self):
        return self._get_field("c_lin_e", EdgeDim, E2CDim)

    def cfl_w_limit(self) -> float:
        return self.serializer.read("cfl_w_limit", self.savepoint)[0]

    def coeff2_dwdz(self):
        return self._get_field("coeff2_dwdz", CellDim, KDim)

    def coeff1_dwdz(self):
        return self._get_field("coeff1_dwdz", CellDim, KDim)

    def ddqz_z_full_e(self):
        return self._get_field("ddqz_z_full_e", EdgeDim, KDim)

    def coeff_gradekin(self):
        return self._get_field("coeff_gradekin", ECDim)

    def ddqz_z_half(self):
        return self._get_field("ddqz_z_half", CellDim, KDim)

    def ddt_w_adv_pc_before(self):
        return self._get_field("ddt_w_adv_pc", CellDim, KDim)

    def ddt_vn_apc_pc_before(self):
        return self._get_field("ddt_vn_apc_pc", EdgeDim, KDim)

    def ddxt_z_full(self):
        return self._get_field("ddxt_z_full", EdgeDim, KDim)

    def ddxn_z_full(self):
        return self._get_field("ddxn_z_full", EdgeDim, KDim)

    def geofac_grdiv(self):
        return self._get_field("geofac_grdiv", EdgeDim, E2C2EODim)

    def rbf_vec_coeff_e(self):
        buffer = np.squeeze(
            self.serializer.read("rbf_vec_coeff_e", self.savepoint).astype(float)
        ).transpose()
        return np_as_located_field(ECDim, E2C2EDim)(buffer)

    def c_intp(self):
        return self._get_field("c_intp", VertexDim, V2CDim)

    def vn(self):
        return self._get_field("vn", EdgeDim, KDim)

    def geofac_rot(self):
        return self._get_field("geofac_rot", VertexDim, V2EDim)

    def scalfac_exdiff(self) -> float:
        return self.serializer.read("scalfac_exdiff", self.savepoint)[0]

    def vn_ie(self):
        return self._get_field("vn_ie", EdgeDim, KDim)

    def vt(self):
        return self._get_field("vt", EdgeDim, KDim)

    def e_bln_c_s(self):
        return self._get_field("e_bln_c_s", CellDim, C2EDim)

    def w_concorr_c(self):
        return self._get_field("w_concorr_c", CellDim, KDim)

    def wgtfac_e(self):
        return self._get_field("wgtfac_e", EdgeDim, KDim)

    def wgtfacq_e(self):
        return self._get_field("wgtfacq_e", EdgeDim, KDim)

    def geofac_n2s(self):
        return self._get_field("geofac_n2s", CellDim, C2E2CODim)

    def z_w_concorr_me(self):
        return self._get_field("z_w_concorr_me", EdgeDim, KDim)

    def wgtfac_c(self):
        return self._get_field("wgtfac_c", CellDim, KDim)

    def z_kin_hor_e(self):
        return self._get_field("z_kin_hor_e", EdgeDim, KDim)

    def z_vt_ie(self):
        return self._get_field("z_vt_ie", EdgeDim, KDim)

    def w(self):
        return self._get_field("w", CellDim, KDim)

class InterpolationSavepoint(IconSavepoint):
    def geofac_grg(self):
        grg = np.squeeze(self.serializer.read("geofac_grg", self.savepoint))
        return np_as_located_field(CellDim, C2E2CODim)(
            grg[:, :, 0]
        ), np_as_located_field(CellDim, C2E2CODim)(grg[:, :, 1])

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
        return self._get_field("geofac_grg", CellDim, C2E2CODim)

    def geofac_n2s(self):
        return self._get_field("geofac_n2s", CellDim, C2E2CODim)

    def geofac_rot(self):
        return self._get_field("geofac_rot", VertexDim, V2EDim)

    def nudgecoeff_e(self):
        return self._get_field("nudgecoeff_e", EdgeDim)

    def pos_on_tplane_e(self):
        return self._get_field("pos_on_tplane_e", ECDim)

    def rbf_vec_blk_v(self):
        return self._get_field("rbf_vec_blk_v", CellDim, KDim)

    def rbf_vec_coeff_e(self):
        buffer = np.squeeze(
            self.serializer.read("rbf_vec_coeff_e", self.savepoint).astype(float)
        ).transpose()
        return np_as_located_field(EdgeDim, E2C2EDim)(buffer)

    def rbf_vec_coeff_v1(self):
        return self._get_field("rbf_vec_coeff_v1", CellDim, KDim)

    def rbf_vec_coeff_v2(self):
        return self._get_field("rbf_vec_coeff_v2", CellDim, KDim)

    def rbf_vec_idx_v(self):
        return self._get_field("rbf_vec_idx_v", CellDim, KDim)

    def construct_interpolation_state(self) -> InterpolationState:
        grg = self.geofac_grg()
        return InterpolationState(
            e_bln_c_s=self.e_bln_c_s(),
            rbf_coeff_1=self.rbf_vec_coeff_v1(),
            rbf_coeff_2=self.rbf_vec_coeff_v2(),
            _geofac_div=self.geofac_div(),
            geofac_n2s=self.geofac_n2s(),
            geofac_grg_x=grg[0],
            geofac_grg_y=grg[1],
            nudgecoeff_e=self.nudgecoeff_e(),
        )


class MetricSavepoint(IconSavepoint):
    def construct_metric_state(self) -> MetricState:
        return MetricState(
            mask_hdiff=self.mask_diff(),
            theta_ref_mc=self.theta_ref_mc(),
            wgtfac_c=self.wgtfac_c(),
            zd_intcoef=self.zd_intcoef(),
            zd_vertidx=self.zd_vertoffset(),
            zd_diffcoef=self.zd_diffcoef(),
        )

    def zd_diffcoef(self):
        return self._get_field("zd_diffcoef", CellDim, KDim)

    def zd_intcoef(self):
        return self._get_field("vcoef", CellDim, C2E2CDim, KDim)

    def zd_vertidx(self):
        return self._get_field("zd_vertidx", CellDim, C2E2CDim, dtype=int)

    def zd_vertoffset(self):
        return self._get_field("zd_vertoffset", CellDim, C2E2CDim, KDim, dtype=int)

    def theta_ref_mc(self):
        return self._get_field("theta_ref_mc", CellDim, KDim)

    def wgtfac_c(self):
        return self._get_field("wgtfac_c", CellDim, KDim)

    def wgtfac_e(self):
        return self._get_field("wgtfac_e", EdgeDim, KDim)

    def mask_diff(self):
        return self._get_field("mask_hdiff", CellDim, KDim, dtype=int)

    def coeff1_dwdz(self):
        return self._get_field("coeff1_dwdz", CellDim, KDim)

    def coeff2_dwdz(self):
        return self._get_field("coeff2_dwdz", CellDim, KDim)

    def coeff_gradekin(self):
        return self._get_field("coeff_gradekin", ECDim)

    def d2dexdz2_fac1_mc(self):
        return self._get_field("d2dexdz2_fac1_mc", CellDim, KDim)

    def d2dexdz2_fac2_mc(self):
        return self._get_field("d2dexdz2_fac2_mc", CellDim, KDim)

    def ddqz_z_full_e(self):
        return self._get_field("ddqz_z_full_e", EdgeDim, KDim)

    def ddqz_z_half(self):
        return self._get_field("ddqz_z_half", CellDim, KDim)

    def ddxn_z_full(self):
        return self._get_field("ddxn_z_full", EdgeDim, KDim)

    def ddxt_z_full(self):
        return self._get_field("ddxt_z_full", EdgeDim, KDim)

    def exner_exfac(self):
        return self._get_field("exner_exfac", CellDim, KDim)

    def exner_ref_mc(self):
        return self._get_field("exner_ref_mc", CellDim, KDim)

    def hmask_dd3d(self):
        return self._get_field("hmask_dd3d", EdgeDim)

    def inv_ddqz_z_full(self):
        return self._get_field("inv_ddqz_z_full", CellDim, KDim)

    def mask_hdiff(self):
        return self._get_field("mask_hdiff", CellDim, KDim)

    def mask_prog_halo_c_dsl_low_refin(self):
        return self._get_field("mask_prog_halo_c_dsl_low_refin", CellDim)

    def pg_edgeidx_dsl(self):
        return self._get_field("pg_edgeidx_dsl", CellDim, KDim)

    def pg_exdist_dsl(self):
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

    def theta_ref_mc(self):
        return self._get_field("theta_ref_mc", CellDim, KDim)

    def theta_ref_me(self):
        return self._get_field("theta_ref_me", EdgeDim, KDim)

    def vcoef(self):
        return self._get_field("vcoef", CellDim, KDim)  # check

    def vwind_expl_wgt(self):
        return self._get_field("vwind_expl_wgt", CellDim)

    def vwind_impl_wgt(self):
        return self._get_field("vwind_impl_wgt", CellDim)

    def bdy_halo_c(self):
        return self._get_field("bdy_halo_c", CellDim, dtype=bool)

    def wgtfac_c(self):
        return self._get_field("wgtfac_c", CellDim, KDim)

    def wgtfac_e(self):
        return self._get_field("wgtfac_e", EdgeDim, KDim)

    def wgtfacq_c(self):
        return self._get_field("wgtfacq_c", CellDim, KDim)

    def wgtfacq_c_dsl(self):
        return self._get_field("wgtfacq_c_dsl", CellDim, KDim)

    def wgtfacq_e(self):
        return self._get_field("wgtfacq_e", EdgeDim, KDim)

    def wgtfacq_e_dsl(self, k_level): #TODO: @abishekg7 Simplify this after serialized data is fixed
        ar = np.squeeze(self.serializer.read("wgtfacq_e", self.savepoint))
        k = k_level - 3
        ar = np.pad(ar[:,::-1], ((0,0),(k,0)), 'constant', constant_values=(0.0,))
        return self._get_field_from_ndarray(ar, EdgeDim, KDim)


    def zd_diffcoef(self):
        return self._get_field("zd_diffcoef", CellDim, KDim)

    def zd_indlist(self):
        return self._get_field("zd_indlist", CellDim, KDim)

    def zd_vertidx(self):
        return self._get_field("zd_vertidx", CellDim, KDim)

    def zd_vertoffset(self):
        return self._get_field("zd_vertoffset", CellDim, KDim)

    def zdiff_gradp_dsl(self):
        buffer = self._get_field("zdiff_gradp_dsl", CellDim, KDim, ECDim)
        np_array_reshape = np.asarray(buffer).reshape(
            (buffer.shape[0] * buffer.shape[2], buffer.shape[1])
        )
        return np_as_located_field(ECDim, KDim)(np_array_reshape)

    def d_exner_dz_ref_ic(self):
        return self._get_field("d_exner_dz_ref_ic", CellDim, KDim)

    def ipeidx_dsl(self):
        return self._get_field("ipeidx_dsl", EdgeDim, KDim, dtype=bool)

    def mask_prog_halo_c(self):
        return self._get_field("mask_prog_halo_c", CellDim, dtype=bool)


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

    def construct_diagnostics(self) -> DiagnosticState:
        return DiagnosticState(
            hdef_ic=self.hdef_ic(),
            div_ic=self.div_ic(),
            dwdx=self.dwdx(),
            dwdy=self.dwdy(),
        )

class IconExitSavepoint(IconSavepoint):
    def vn(self):
        return self._get_field("x_vn", EdgeDim, KDim)

    def theta_v(self):
        return self._get_field("x_theta_v", CellDim, KDim)

    def w(self):
        return self._get_field("x_w", CellDim, KDim)

    def exner(self):
        return self._get_field("x_exner", CellDim, KDim)

    def ddt_vn_apc_pc(self):
        return self._get_field("x_ddt_vn_apc_pc", EdgeDim, KDim)

    def ddt_w_adv_pc(self):
        return self._get_field("x_ddt_w_adv_pc", CellDim, KDim)

    def vn_ie(self):
        return self._get_field("x_vn_ie", EdgeDim, KDim)

    def vt(self):
        return self._get_field("x_vt", EdgeDim, KDim)

    def z_kin_hor_e(self):
        return self._get_field("x_z_kin_hor_e", EdgeDim, KDim)

    def z_vt_ie(self):
        return self._get_field("x_z_vt_ie", EdgeDim, KDim)

    def z_w_concorr_me(self):
        return self._get_field("x_z_w_concorr_me", EdgeDim, KDim)

    def exner_new(self):
        return self._get_field("x_exner_new", CellDim, KDim)

    def exner_now(self):
        return self._get_field("x_exner_now", CellDim, KDim)

    def mass_fl_e(self):
        return self._get_field("x_mass_fl_e", EdgeDim, KDim)

    def prep_adv_mass_flx_me(self):
        return self._get_field("x_prep_adv_mass_flx_me", EdgeDim, KDim)

    def prep_adv_vn_traj(self):
        return self._get_field("x_prep_adv_vn_traj", EdgeDim, KDim)

    def rho_ic(self):
        return self._get_field("x_rho_ic", CellDim, KDim)

    def theta_v_ic(self):
        return self._get_field("x_theta_v_ic", CellDim, KDim)

    def theta_v_new(self):
        return self._get_field("x_theta_v_new", CellDim, KDim)

    def vn_ie(self):
        return self._get_field("x_vn_ie", EdgeDim, KDim)

    def vn_new(self):
        return self._get_field("x_vn_new", EdgeDim, KDim)

    def w_concorr_c(self):
        return self._get_field("x_w_concorr_c", CellDim, KDim)

    def w_new(self):
        return self._get_field("x_w_new", CellDim, KDim)


class IconSerialDataProvider:
    def __init__(self, fname_prefix, path=".", do_print=False):
        self.rank = 0
        self.serializer: ser.Serializer = None
        self.file_path: str = path
        self.fname = f"{fname_prefix}_rank{str(self.rank)}"
        self.log = logging.getLogger(__name__)
        self._init_serializer(do_print)

    def _init_serializer(self, do_print: bool):
        if not self.fname:
            self.log.warning(" WARNING: no filename! closing serializer")
        self.serializer = ser.Serializer(
            ser.OpenModeKind.Read, self.file_path, self.fname
        )
        if do_print:
            self.print_info()

    def print_info(self):
        self.log.info(f"SAVEPOINTS: {self.serializer.savepoint_list()}")
        self.log.info(f"FIELDNAMES: {self.serializer.fieldnames()}")

    def from_savepoint_grid(self) -> IconGridSavePoint:
        savepoint = self.serializer.savepoint["icon-grid"].id[1].as_savepoint()
        return IconGridSavePoint(savepoint, self.serializer)

    def from_savepoint_diffusion_init(
        self, linit: bool, date: str
    ) -> IconDiffusionInitSavepoint:
        savepoint = (
            self.serializer.savepoint["call-diffusion-init"]
            .linit[linit]
            .date[date]
            .as_savepoint()
        )
        return IconDiffusionInitSavepoint(savepoint, self.serializer)

    def from_interpolation_savepoint(self) -> InterpolationSavepoint:
        savepoint = self.serializer.savepoint["interpolation_state"].as_savepoint()
        return InterpolationSavepoint(savepoint, self.serializer)

    def from_metrics_savepoint(self) -> MetricSavepoint:
        savepoint = self.serializer.savepoint["metric_state"].as_savepoint()
        return MetricSavepoint(savepoint, self.serializer)

    def from_savepoint_diffusion_exit(
        self, linit: bool, date: str
    ) -> IconExitSavepoint:
        savepoint = (
            self.serializer.savepoint["call-diffusion-exit"]
            .linit[linit]
            .date[date]
            .as_savepoint()
        )
        return IconExitSavepoint(savepoint, self.serializer)

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
        return IconVelocityInitSavepoint(savepoint, self.serializer)

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
        return IconNonHydroInitSavepoint(savepoint, self.serializer)
