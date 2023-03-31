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
from typing import Optional

from gt4py.next.ffront.fbuiltins import Field

import icon4py.velocity.velocity_advection_program as velocity_prog
from icon4py.common.dimension import (
    C2E2CODim,
    C2EDim,
    CellDim,
    E2C2EDim,
    E2C2EODim,
    E2CDim,
    E2VDim,
    EdgeDim,
    KDim,
    V2CDim,
    V2EDim,
    VertexDim,
)
from icon4py.state_utils.diagnostic_state import DiagnosticState
from icon4py.state_utils.horizontal import HorizontalMarkerIndex
from icon4py.state_utils.icon_grid import IconGrid, VerticalModelParams
from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.state_utils.metric_state import MetricState
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.state_utils.utils import _allocate
from icon4py.velocity.z_fields import ZFields


class VelocityAdvection:
    def __init__(self, run_program=True):
        self._initialized = False
        self._run_program = run_program
        self.grid: Optional[IconGrid] = None
        self.interpolation_state = None
        self.metric_state = None
        self.vertical_params: Optional[VerticalModelParams] = None

        self.cfl_w_limit: Optional[float] = None
        self.scalfac_exdiff: Optional[float] = None

    def init(
        self,
        grid: IconGrid,
        metric_state: MetricState,
        interpolation_state: InterpolationState,
        vertical_params: VerticalModelParams,
    ):

        self.grid = grid
        self.metric_state: MetricState = metric_state
        self.interpolation_state: InterpolationState = interpolation_state
        self.vertical_params = vertical_params

        self._allocate_local_fields()

        self.cfl_w_limit = 0.65
        self.scalfac_exdiff = 0.05

        self._initialized = True

    @property
    def initialized(self):
        return self._initialized

    def _allocate_local_fields(self):
        self.z_w_v = _allocate(VertexDim, KDim, mesh=self.grid)
        self.z_v_grad_w = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_ekinh = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_w_concorr_mc = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_w_con_c = _allocate(CellDim, KDim, mesh=self.grid)
        self.zeta = _allocate(VertexDim, KDim, mesh=self.grid)
        self.z_w_con_c_full = _allocate(CellDim, KDim, mesh=self.grid)
        self.cfl_clipping = _allocate(CellDim, KDim, mesh=self.grid, dtype=bool)
        self.pre_levelmask = _allocate(CellDim, KDim, mesh=self.grid, dtype=bool)
        self.levelmask = _allocate(KDim, mesh=self.grid, dtype=bool)
        self.vcfl = _allocate(CellDim, KDim, mesh=self.grid)

    def run_predictor_step(
        self,
        vn_only: bool,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        z_fields: ZFields,
        inv_dual_edge_length: Field[[EdgeDim], float],
        inv_primal_edge_length: Field[[EdgeDim], float],
        dtime: float,
        tangent_orientation: Field[[EdgeDim], float],
        cfl_w_limit: float,
        scalfac_exdiff: float,
        cell_areas: Field[[CellDim], float],
        owner_mask: Field[[CellDim], bool],
        f_e: Field[[EdgeDim], float],
        area_edge: Field[[EdgeDim], float],
    ):

        (
            edge_startindex_nudging,
            edge_endindex_nudging,
            edge_startindex_local,
            edge_endindex_local,
            cell_startindex_nudging,
            cell_endindex_nudging,
            cell_startindex_local,
            cell_endindex_local,
            vert_startindex_local,
            vert_endindex_local,
        ) = self.init_dimensions_boundaries()

        self.cfl_w_limit = self.cfl_w_limit / dtime
        self.scalfac_exdiff = self.scalfac_exdiff / (
            dtime * (0.85 - self.cfl_w_limit * dtime)
        )

        if not vn_only:
            velocity_prog.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_vn_only(
                prognostic_state.w,
                self.interpolation_state.c_intp,
                self.z_w_v,
                vert_startindex_local - 1,
                self.grid.n_lev(),
                offset_provider={
                    "V2C": self.grid.get_v2c_connectivity(),
                    "V2CDim": V2CDim,
                },
            )

        velocity_prog.mo_math_divrot_rot_vertex_ri_dsl(
            prognostic_state.vn,
            self.interpolation_state.geofac_rot,
            self.zeta,
            vert_startindex_local - 1,
            self.grid.n_lev(),
            offset_provider={"V2E": self.grid.get_v2e_connectivity(), "V2EDim": V2EDim},
        )

        velocity_prog.predictor_tendencies_1_2(
            prognostic_state.vn,
            self.interpolation_state.rbf_vec_coeff_e,
            diagnostic_state.vt,
            self.metric_state.wgtfac_e,
            diagnostic_state.vn_ie,
            z_fields.z_kin_hor_e,
            edge_startindex_local - 2,
            self.vertical_params.nflatlev,
            self.grid.n_lev(),
            offset_provider={
                "E2C2E": self.grid.get_e2c2e_connectivity(),
                "E2C2EDim": E2C2EDim,
                "Koff": KDim,
            },
        )

        if not vn_only:
            velocity_prog.predictor_tendencies_3(
                diagnostic_state.vt,
                self.metric_state.wgtfac_e,
                z_fields.z_vt_ie,
                edge_startindex_local - 2,
                self.grid.n_lev(),
                offset_provider={"Koff": KDim},
            )
        velocity_prog.predictor_tendencies_4_5_6(
            prognostic_state.vn,
            diagnostic_state.vt,
            diagnostic_state.vn_ie,
            z_fields.z_vt_ie,
            z_fields.z_kin_hor_e,
            self.metric_state.ddxn_z_full,
            self.metric_state.ddxt_z_full,
            z_fields.z_w_concorr_me,
            self.metric_state.wgtfacq_e,
            edge_startindex_local - 2,
            self.vertical_params.nflatlev,
            self.grid.n_lev(),
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "C2EDim": C2EDim,
                "Koff": KDim,
            },
        )

        if not vn_only:
            velocity_prog.advector_tendencies_7(
                diagnostic_state.vn_ie,
                inv_dual_edge_length,
                prognostic_state.w,
                z_fields.z_vt_ie,
                inv_primal_edge_length,
                tangent_orientation,
                self.z_w_v,
                self.z_v_grad_w,
                edge_startindex_local - 1,
                self.grid.n_lev(),
                offset_provider={
                    "E2C": self.grid.get_e2c_connectivity(),
                    "E2V": self.grid.get_e2v_connectivity(),
                },
            )

        velocity_prog.advector_tendencies_8(
            z_fields.z_kin_hor_e,
            self.interpolation_state.e_bln_c_s,
            self.z_ekinh,
            cell_startindex_local - 1,
            self.grid.n_lev(),
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "C2EDim": C2EDim,
            },
        )

        velocity_prog.predictor_tendencies_9_10(
            z_fields.z_w_concorr_me,
            self.interpolation_state.e_bln_c_s,
            self.z_w_concorr_mc,
            self.metric_state.wgtfac_c,
            diagnostic_state.w_concorr_c,
            cell_startindex_local - 1,
            self.vertical_params.nflatlev,
            self.grid.n_lev(),
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "C2EDim": C2EDim,
                "Koff": KDim,
            },
        )

        velocity_prog.advector_tendencies_11_to_20(
            diagnostic_state.vt,
            diagnostic_state.vn_ie,
            z_fields.z_kin_hor_e,
            prognostic_state.w,
            self.z_v_grad_w,
            self.interpolation_state.e_bln_c_s,
            self.z_ekinh,
            diagnostic_state.w_concorr_c,
            self.z_w_con_c,
            self.metric_state.ddqz_z_half,
            self.cfl_clipping,
            self.pre_levelmask,
            self.vcfl,
            cfl_w_limit,
            dtime,
            self.z_w_con_c_full,
            self.metric_state.coeff1_dwdz,
            self.metric_state.coeff2_dwdz,
            diagnostic_state.ddt_w_adv_pc,
            self.metric_state.coeff_gradekin,
            self.zeta,
            f_e,
            self.interpolation_state.c_lin_e,
            self.metric_state.ddqz_z_full_e,
            diagnostic_state.ddt_vn_apc_pc,
            prognostic_state.vn,
            inv_primal_edge_length,
            tangent_orientation,
            cell_areas,
            self.interpolation_state.geofac_n2s,
            owner_mask,
            self.levelmask,
            scalfac_exdiff,
            area_edge,
            self.interpolation_state.geofac_grdiv,
            cell_endindex_nudging,
            cell_startindex_local - 1,
            cell_endindex_local,
            edge_startindex_nudging + 1,
            edge_endindex_local,
            self.vertical_params.nflatlev,
            self.grid.n_lev(),
            self.grid.n_lev() + 1,
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
                "E2V": self.grid.get_e2v_connectivity(),
                "C2E": self.grid.get_c2e_connectivity(),
                "C2EDim": C2EDim,
                "E2VDim": E2VDim,
                "E2CDim": E2CDim,
                "E2EC": self.grid.get_e2ec_connectivity(),
                "C2E2CO": self.grid.get_c2e2co_connectivity(),
                "C2E2CODim": C2E2CODim,
                "E2C2EO": self.grid.get_e2c2eo_connectivity(),
                "E2C2EODim": E2C2EODim,
                "Koff": KDim,
            },
        )

    def run_advector_step(
        self,
        vn_only: bool,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        z_fields: ZFields,
        inv_dual_edge_length: Field[[EdgeDim], float],
        inv_primal_edge_length: Field[[EdgeDim], float],
        dtime: float,
        tangent_orientation: Field[[EdgeDim], float],
        cfl_w_limit: float,
        scalfac_exdiff: float,
        cell_areas: Field[[CellDim], float],
        owner_mask: Field[[CellDim], bool],
        f_e: Field[[EdgeDim], float],
        area_edge: Field[[EdgeDim], float],
    ):

        (
            edge_startindex_nudging,
            edge_endindex_nudging,
            edge_startindex_local,
            edge_endindex_local,
            cell_startindex_nudging,
            cell_endindex_nudging,
            cell_startindex_local,
            cell_endindex_local,
            vert_startindex_local,
            vert_endindex_local,
        ) = self.init_dimensions_boundaries()

        self.cfl_w_limit = self.cfl_w_limit / dtime
        self.scalfac_exdiff = self.scalfac_exdiff / (
            dtime * (0.85 - self.cfl_w_limit * dtime)
        )

        if not vn_only:
            velocity_prog.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_vn_only(
                prognostic_state.w,
                self.interpolation_state.c_intp,
                self.z_w_v,
                vert_startindex_local - 1,
                self.grid.n_lev(),
                offset_provider={
                    "V2C": self.grid.get_v2c_connectivity(),
                    "V2CDim": V2CDim,
                },
            )

        velocity_prog.mo_math_divrot_rot_vertex_ri_dsl(
            prognostic_state.vn,
            self.interpolation_state.geofac_rot,
            self.zeta,
            vert_startindex_local - 1,
            self.grid.n_lev(),
            offset_provider={"V2E": self.grid.get_v2e_connectivity(), "V2EDim": V2EDim},
        )

        if not vn_only:
            velocity_prog.advector_tendencies_7(
                diagnostic_state.vn_ie,
                inv_dual_edge_length,
                prognostic_state.w,
                z_fields.z_vt_ie,
                inv_primal_edge_length,
                tangent_orientation,
                self.z_w_v,
                self.z_v_grad_w,
                edge_startindex_local - 1,
                self.grid.n_lev(),
                offset_provider={
                    "E2C": self.grid.get_e2c_connectivity(),
                    "E2V": self.grid.get_e2v_connectivity(),
                },
            )

        velocity_prog.advector_tendencies_8(
            z_fields.z_kin_hor_e,
            self.interpolation_state.e_bln_c_s,
            self.z_ekinh,
            cell_startindex_local - 1,
            self.grid.n_lev(),
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "C2EDim": C2EDim,
            },
        )

        velocity_prog.advector_tendencies_11_to_20(
            diagnostic_state.vt,
            diagnostic_state.vn_ie,
            z_fields.z_kin_hor_e,
            prognostic_state.w,
            self.z_v_grad_w,
            self.interpolation_state.e_bln_c_s,
            self.z_ekinh,
            diagnostic_state.w_concorr_c,
            self.z_w_con_c,
            self.metric_state.ddqz_z_half,
            self.cfl_clipping,
            self.pre_levelmask,
            self.vcfl,
            cfl_w_limit,
            dtime,
            self.z_w_con_c_full,
            self.metric_state.coeff1_dwdz,
            self.metric_state.coeff2_dwdz,
            diagnostic_state.ddt_w_adv_pc,
            self.metric_state.coeff_gradekin,
            self.zeta,
            f_e,
            self.interpolation_state.c_lin_e,
            self.metric_state.ddqz_z_full_e,
            diagnostic_state.ddt_vn_apc_pc,
            prognostic_state.vn,
            inv_primal_edge_length,
            tangent_orientation,
            cell_areas,
            self.interpolation_state.geofac_n2s,
            owner_mask,
            self.levelmask,
            scalfac_exdiff,
            area_edge,
            self.interpolation_state.geofac_grdiv,
            cell_endindex_nudging,
            cell_startindex_local - 1,
            cell_endindex_local,
            edge_startindex_nudging + 1,
            edge_endindex_local,
            self.vertical_params.nflatlev,
            self.grid.n_lev(),
            self.grid.n_lev() + 1,
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
                "E2V": self.grid.get_e2v_connectivity(),
                "C2E": self.grid.get_c2e_connectivity(),
                "C2EDim": C2EDim,
                "E2VDim": E2VDim,
                "E2CDim": E2CDim,
                "E2EC": self.grid.get_e2ec_connectivity(),
                "C2E2CO": self.grid.get_c2e2co_connectivity(),
                "C2E2CODim": C2E2CODim,
                "E2C2EO": self.grid.get_e2c2eo_connectivity(),
                "E2C2EODim": E2C2EODim,
                "Koff": KDim,
            },
        )

    def init_dimensions_boundaries(self):
        (
            edge_startindex_nudging,
            edge_endindex_nudging,
        ) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim),
            HorizontalMarkerIndex.nudging(EdgeDim),
        )

        (edge_startindex_local, edge_endindex_local,) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.local(EdgeDim),
            HorizontalMarkerIndex.local(EdgeDim),
        )

        (
            cell_startindex_nudging,
            cell_endindex_nudging,
        ) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim),
            HorizontalMarkerIndex.nudging(CellDim),
        )

        (cell_startindex_local, cell_endindex_local,) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.local(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        (vert_startindex_local, vert_endindex_local,) = self.grid.get_indices_from_to(
            VertexDim,
            HorizontalMarkerIndex.local(VertexDim),
            HorizontalMarkerIndex.local(VertexDim),
        )
        return (
            edge_startindex_nudging,
            edge_endindex_nudging,
            edge_startindex_local,
            edge_endindex_local,
            cell_startindex_nudging,
            cell_endindex_nudging,
            cell_startindex_local,
            cell_endindex_local,
            vert_startindex_local,
            vert_endindex_local,
        )
