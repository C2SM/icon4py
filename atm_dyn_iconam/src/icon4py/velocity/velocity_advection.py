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
from icon4py.atm_dyn_iconam.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.atm_dyn_iconam.mo_math_divrot_rot_vertex_ri_dsl import (
    mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_01 import (
    mo_velocity_advection_stencil_01,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_02 import (
    mo_velocity_advection_stencil_02,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_03 import (
    mo_velocity_advection_stencil_03,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_07 import (
    mo_velocity_advection_stencil_07,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_08 import (
    mo_velocity_advection_stencil_08,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_15 import (
    mo_velocity_advection_stencil_15,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_18 import (
    mo_velocity_advection_stencil_18,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_19 import (
    mo_velocity_advection_stencil_19,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_20 import (
    mo_velocity_advection_stencil_20,
)
from icon4py.common.dimension import CellDim, EdgeDim, KDim, VertexDim
from icon4py.state_utils.diagnostic_state import DiagnosticState
from icon4py.state_utils.horizontal import HorizontalMarkerIndex
from icon4py.state_utils.icon_grid import IconGrid, VerticalModelParams
from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.state_utils.metric_state import MetricState
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.state_utils.utils import _allocate, _allocate_indices


class VelocityAdvection:
    def __init__(
        self,
        grid: IconGrid,
        metric_state: MetricState,
        interpolation_state: InterpolationState,
        vertical_params: VerticalModelParams,
        run_program=True,
    ):
        self._initialized = False
        self._run_program = run_program
        self.grid: IconGrid = grid
        self.metric_state: MetricState = metric_state
        self.interpolation_state: InterpolationState = interpolation_state
        self.vertical_params = vertical_params

        self.cfl_w_limit: Optional[float] = 0.65
        self.scalfac_exdiff: Optional[float] = 0.05
        self._allocate_local_fields()

        self._initialized = True

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
        self.k_field = _allocate_indices(KDim, mesh=self.grid)

    def run_predictor_step(
        self,
        vn_only: bool,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        z_w_concorr_me: Field[[EdgeDim, KDim], float],
        z_kin_hor_e: Field[[EdgeDim, KDim], float],
        z_vt_ie: Field[[EdgeDim, KDim], float],
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
            edge_startindex_interior,
            edge_endindex_interior,
            cell_startindex_nudging,
            cell_endindex_nudging,
            cell_startindex_interior,
            cell_endindex_interior,
            vert_startindex_interior,
            vert_endindex_interior,
        ) = self.init_dimensions_boundaries()

        self.cfl_w_limit = self.cfl_w_limit / dtime
        self.scalfac_exdiff = self.scalfac_exdiff / (
            dtime * (0.85 - self.cfl_w_limit * dtime)
        )

        if not vn_only:
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                prognostic_state.w,
                self.interpolation_state.c_intp,
                self.z_w_v,
                horizontal_start=2,
                horizontal_end=vert_startindex_interior - 1,
                vertical_start=1,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "V2C": self.grid.get_v2c_connectivity(),
                },
            )

        mo_math_divrot_rot_vertex_ri_dsl(
            prognostic_state.vn,
            self.interpolation_state.geofac_rot,
            self.zeta,
            horizontal_start=2,
            horizontal_end=vert_startindex_interior - 1,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "V2E": self.grid.get_v2e_connectivity(),
            },
        )

        # mo_velocity_advection_stencil_01(
        #     prognostic_state.vn,
        #     self.interpolation_state.rbf_vec_coeff_e,
        #     diagnostic_state.vt,
        #     horizontal_start=4, # TODO: @nfarabullini: input data for rbf_vec_coeff_e has a start of 4 which does not work for the lower bound. Problem with the input data?
        #     horizontal_end=edge_startindex_interior - 1,
        #     vertical_start=0,
        #     vertical_end=self.grid.n_lev(),
        #     offset_provider={
        #         "E2C2E": self.grid.get_e2c2e_connectivity(),
        #     }
        # )

        mo_velocity_advection_stencil_02(
            self.metric_state.wgtfac_e,
            prognostic_state.vn,
            diagnostic_state.vt,
            diagnostic_state.vn_ie,
            z_kin_hor_e,
            horizontal_start=4,
            horizontal_end=edge_startindex_interior - 2,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "Koff": KDim,
            },
        )

        if not vn_only:
            mo_velocity_advection_stencil_03(
                diagnostic_state.vt,
                self.metric_state.wgtfac_e,
                z_vt_ie,
                horizontal_start=4,
                horizontal_end=edge_startindex_interior - 2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={"Koff": KDim},
            )

        velocity_prog.fused_stencils_4_5_6(
            prognostic_state.vn,
            diagnostic_state.vt,
            diagnostic_state.vn_ie,
            z_vt_ie,
            z_kin_hor_e,
            self.metric_state.ddxn_z_full,
            self.metric_state.ddxt_z_full,
            z_w_concorr_me,
            self.metric_state.wgtfacq_e,
            self.k_field,
            edge_startindex_interior - 2,
            self.vertical_params.nflatlev,
            self.grid.n_lev(),
            self.grid.n_lev(),
            offset_provider={
                "Koff": KDim,
            },
        )

        # if not vn_only:
        #     mo_velocity_advection_stencil_07(
        #         diagnostic_state.vn_ie,
        #         inv_dual_edge_length,
        #         prognostic_state.w,
        #         z_vt_ie,
        #         inv_primal_edge_length,
        #         tangent_orientation,
        #         self.z_w_v,
        #         self.z_v_grad_w,
        #         horizontal_start=6,
        #         horizontal_end=edge_startindex_interior - 1,
        #         vertical_start=0,
        #         vertical_end=self.grid.n_lev(),
        #         offset_provider={
        #             "E2C": self.grid.get_e2c_connectivity(),
        #             "E2V": self.grid.get_e2v_connectivity(),
        #         },
        #     )

        mo_velocity_advection_stencil_08(
            z_kin_hor_e,
            self.interpolation_state.e_bln_c_s,
            self.z_ekinh,
            horizontal_start=3,
            horizontal_end=cell_startindex_interior - 1,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={"C2E": self.grid.get_c2e_connectivity()},
        )

        velocity_prog.fused_stencils_9_10(
            z_w_concorr_me,
            self.interpolation_state.e_bln_c_s,
            self.z_w_concorr_mc,
            self.metric_state.wgtfac_c,
            diagnostic_state.w_concorr_c,
            self.k_field,
            cell_startindex_interior - 1,
            self.vertical_params.nflatlev,
            self.grid.n_lev(),
            self.grid.n_lev(),
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "Koff": KDim,
            },
        )

        velocity_prog.fused_stencils_11_to_13(
            prognostic_state.w,
            diagnostic_state.w_concorr_c,
            self.z_w_con_c,
            self.k_field,
            cell_endindex_interior - 1,
            self.vertical_params.nflatlev,
            self.grid.n_lev(),
            self.grid.n_lev() + 1,
            self.grid.n_lev(),
            offset_provider={},
        )

        velocity_prog.fused_stencil_14(
            self.z_w_con_c,
            self.metric_state.ddqz_z_half,
            self.cfl_clipping,
            self.pre_levelmask,
            self.vcfl,
            cfl_w_limit,
            dtime,
            cell_endindex_interior - 1,
            self.grid.n_lev(),
            max(3, self.vertical_params.index_of_damping_layer - 2),
            offset_provider={},
        )

        mo_velocity_advection_stencil_15(
            self.z_w_con_c,
            self.z_w_con_c_full,
            horizontal_start=4,
            horizontal_end=cell_endindex_interior - 1,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        velocity_prog.fused_stencils_16_to_17(
            prognostic_state.w,
            self.z_v_grad_w,
            self.interpolation_state.e_bln_c_s,
            self.z_w_con_c,
            self.metric_state.coeff1_dwdz,
            self.metric_state.coeff2_dwdz,
            diagnostic_state.ddt_w_adv_pc,
            horizontal_start=cell_startindex_nudging,
            horizontal_end=cell_endindex_interior,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "Koff": KDim,
            },
        )

        mo_velocity_advection_stencil_18(
            self.levelmask,
            self.cfl_clipping,
            owner_mask,
            self.z_w_con_c,
            self.metric_state.ddqz_z_half,
            cell_areas,
            self.interpolation_state.geofac_n2s,
            prognostic_state.w,
            diagnostic_state.ddt_w_adv_pc,
            scalfac_exdiff,
            cfl_w_limit,
            dtime,
            horizontal_start=cell_startindex_nudging,
            horizontal_end=cell_endindex_interior,
            vertical_start=max(3, self.vertical_params.index_of_damping_layer - 2),
            vertical_end=self.grid.n_lev() - 4,
            offset_provider={
                "C2E2CO": self.grid.get_c2e2co_connectivity(),
            },
        )

        mo_velocity_advection_stencil_19(
            z_kin_hor_e,
            self.metric_state.coeff_gradekin,
            self.z_ekinh,
            self.zeta,
            diagnostic_state.vt,
            f_e,
            self.interpolation_state.c_lin_e,
            self.z_w_con_c_full,
            diagnostic_state.vn_ie,
            self.metric_state.ddqz_z_full_e,
            diagnostic_state.ddt_vn_apc_pc,
            horizontal_start=edge_startindex_nudging + 1,
            horizontal_end=edge_endindex_interior,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
                "E2V": self.grid.get_e2v_connectivity(),
                "E2EC": self.grid.get_e2ec_connectivity(),
                "Koff": KDim,
            },
        )

        mo_velocity_advection_stencil_20(
            self.levelmask,
            self.interpolation_state.c_lin_e,
            self.z_w_con_c_full,
            self.metric_state.ddqz_z_full_e,
            area_edge,
            tangent_orientation,
            inv_primal_edge_length,
            self.zeta,
            self.interpolation_state.geofac_grdiv,
            prognostic_state.vn,
            diagnostic_state.ddt_vn_apc_pc,
            cfl_w_limit,
            scalfac_exdiff,
            dtime,
            horizontal_start=edge_startindex_nudging + 1,
            horizontal_end=edge_endindex_interior,
            vertical_start=max(3, self.vertical_params.index_of_damping_layer - 2),
            vertical_end=self.grid.n_lev() - 4,
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
                "E2V": self.grid.get_e2v_connectivity(),
                "E2C2EO": self.grid.get_e2c2eo_connectivity(),
                "Koff": KDim,
            },
        )

    def run_corrector_step(
        self,
        vn_only: bool,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        z_kin_hor_e: Field[[EdgeDim, KDim], float],
        z_vt_ie: Field[[EdgeDim, KDim], float],
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
            edge_startindex_interior,
            edge_endindex_interior,
            cell_startindex_nudging,
            cell_endindex_nudging,
            cell_startindex_interior,
            cell_endindex_interior,
            vert_startindex_interior,
            vert_endindex_interior,
        ) = self.init_dimensions_boundaries()

        self.cfl_w_limit = self.cfl_w_limit / dtime
        self.scalfac_exdiff = self.scalfac_exdiff / (
            dtime * (0.85 - self.cfl_w_limit * dtime)
        )

        if not vn_only:
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                prognostic_state.w,
                self.interpolation_state.c_intp,
                self.z_w_v,
                horizontal_start=2,
                horizontal_end=vert_startindex_interior - 1,
                vertical_start=1,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "V2C": self.grid.get_v2c_connectivity(),
                },
            )

        mo_math_divrot_rot_vertex_ri_dsl(
            prognostic_state.vn,
            self.interpolation_state.geofac_rot,
            self.zeta,
            horizontal_start=2,
            horizontal_end=vert_startindex_interior - 1,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "V2E": self.grid.get_v2e_connectivity(),
            },
        )

        # if not vn_only:
        #     mo_velocity_advection_stencil_07(
        #         diagnostic_state.vn_ie,
        #         inv_dual_edge_length,
        #         prognostic_state.w,
        #         z_vt_ie,
        #         inv_primal_edge_length,
        #         tangent_orientation,
        #         self.z_w_v,
        #         self.z_v_grad_w,
        #         horizontal_start=6,
        #         horizontal_end=edge_startindex_interior - 1,
        #         vertical_start=0,
        #         vertical_end=self.grid.n_lev(),
        #         offset_provider={
        #             "E2C": self.grid.get_e2c_connectivity(),
        #             "E2V": self.grid.get_e2v_connectivity(),
        #         },
        #     )

        mo_velocity_advection_stencil_08(
            z_kin_hor_e,
            self.interpolation_state.e_bln_c_s,
            self.z_ekinh,
            horizontal_start=3,
            horizontal_end=cell_startindex_interior - 1,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={"C2E": self.grid.get_c2e_connectivity()},
        )

        velocity_prog.fused_stencils_11_to_13(
            prognostic_state.w,
            diagnostic_state.w_concorr_c,
            self.z_w_con_c,
            self.k_field,
            cell_endindex_interior - 1,
            self.vertical_params.nflatlev,
            self.grid.n_lev(),
            self.grid.n_lev() + 1,
            self.grid.n_lev(),
            offset_provider={},
        )

        velocity_prog.fused_stencil_14(
            self.z_w_con_c,
            self.metric_state.ddqz_z_half,
            self.cfl_clipping,
            self.pre_levelmask,
            self.vcfl,
            cfl_w_limit,
            dtime,
            cell_endindex_interior - 1,
            self.grid.n_lev(),
            max(3, self.vertical_params.index_of_damping_layer - 2),
            offset_provider={},
        )

        mo_velocity_advection_stencil_15(
            self.z_w_con_c,
            self.z_w_con_c_full,
            horizontal_start=4,
            horizontal_end=cell_endindex_interior - 1,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        velocity_prog.fused_stencils_16_to_17(
            prognostic_state.w,
            self.z_v_grad_w,
            self.interpolation_state.e_bln_c_s,
            self.z_w_con_c,
            self.metric_state.coeff1_dwdz,
            self.metric_state.coeff2_dwdz,
            diagnostic_state.ddt_w_adv_pc,
            horizontal_start=cell_startindex_nudging,
            horizontal_end=cell_endindex_interior,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "Koff": KDim,
            },
        )

        mo_velocity_advection_stencil_18(
            self.levelmask,
            self.cfl_clipping,
            owner_mask,
            self.z_w_con_c,
            self.metric_state.ddqz_z_half,
            cell_areas,
            self.interpolation_state.geofac_n2s,
            prognostic_state.w,
            diagnostic_state.ddt_w_adv_pc,
            scalfac_exdiff,
            cfl_w_limit,
            dtime,
            horizontal_start=cell_startindex_nudging,
            horizontal_end=cell_endindex_interior,
            vertical_start=max(3, self.vertical_params.index_of_damping_layer - 2),
            vertical_end=self.grid.n_lev() - 4,
            offset_provider={
                "C2E2CO": self.grid.get_c2e2co_connectivity(),
            },
        )

        mo_velocity_advection_stencil_19(
            z_kin_hor_e,
            self.metric_state.coeff_gradekin,
            self.z_ekinh,
            self.zeta,
            diagnostic_state.vt,
            f_e,
            self.interpolation_state.c_lin_e,
            self.z_w_con_c_full,
            diagnostic_state.vn_ie,
            self.metric_state.ddqz_z_full_e,
            diagnostic_state.ddt_vn_apc_pc,
            horizontal_start=edge_startindex_nudging + 1,
            horizontal_end=edge_endindex_interior,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
                "E2V": self.grid.get_e2v_connectivity(),
                "E2EC": self.grid.get_e2ec_connectivity(),
                "Koff": KDim,
            },
        )

        mo_velocity_advection_stencil_20(
            self.levelmask,
            self.interpolation_state.c_lin_e,
            self.z_w_con_c_full,
            self.metric_state.ddqz_z_full_e,
            area_edge,
            tangent_orientation,
            inv_primal_edge_length,
            self.zeta,
            self.interpolation_state.geofac_grdiv,
            prognostic_state.vn,
            diagnostic_state.ddt_vn_apc_pc,
            cfl_w_limit,
            scalfac_exdiff,
            dtime,
            horizontal_start=edge_startindex_nudging + 1,
            horizontal_end=edge_endindex_interior,
            vertical_start=max(3, self.vertical_params.index_of_damping_layer - 2),
            vertical_end=self.grid.n_lev() - 4,
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
                "E2V": self.grid.get_e2v_connectivity(),
                "E2C2EO": self.grid.get_e2c2eo_connectivity(),
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

        (
            edge_startindex_interior,
            edge_endindex_interior,
        ) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.interior(EdgeDim),
            HorizontalMarkerIndex.interior(EdgeDim),
        )

        (
            cell_startindex_nudging,
            cell_endindex_nudging,
        ) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim),
            HorizontalMarkerIndex.nudging(CellDim),
        )

        (
            cell_startindex_interior,
            cell_endindex_interior,
        ) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.interior(CellDim),
            HorizontalMarkerIndex.interior(CellDim),
        )

        (
            vert_startindex_interior,
            vert_endindex_interior,
        ) = self.grid.get_indices_from_to(
            VertexDim,
            HorizontalMarkerIndex.interior(VertexDim),
            HorizontalMarkerIndex.interior(VertexDim),
        )
        return (
            edge_startindex_nudging,
            edge_endindex_nudging,
            edge_startindex_interior,
            edge_endindex_interior,
            cell_startindex_nudging,
            cell_endindex_nudging,
            cell_startindex_interior,
            cell_endindex_interior,
            vert_startindex_interior,
            vert_endindex_interior,
        )
