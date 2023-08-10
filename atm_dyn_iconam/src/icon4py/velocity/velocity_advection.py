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

from gt4py.next.common import Field
from gt4py.next.iterator.builtins import int32
from gt4py.next.program_processors.runners.gtfn_cpu import run_gtfn

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
from icon4py.grid.horizontal import HorizontalMarkerIndex
from icon4py.grid.icon_grid import IconGrid
from icon4py.grid.vertical import VerticalModelParams
from icon4py.state_utils.diagnostic_state import DiagnosticState
from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.state_utils.metric_state import MetricStateNonHydro
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.state_utils.utils import _allocate, _allocate_indices


class VelocityAdvection:
    def __init__(
        self,
        grid: IconGrid,
        metric_state: MetricStateNonHydro,
        interpolation_state: InterpolationState,
        vertical_params: VerticalModelParams,
        run_program=True,
    ):
        self._initialized = False
        self._run_program = run_program
        self.grid: IconGrid = grid
        self.metric_state: MetricStateNonHydro = metric_state
        self.interpolation_state: InterpolationState = interpolation_state
        self.vertical_params = vertical_params

        self.cfl_w_limit: Optional[float] = 0.65
        self.scalfac_exdiff: Optional[float] = 0.05
        self._allocate_local_fields()

        self._initialized = True

    def init(
        self,
        grid: IconGrid,
        metric_state: MetricStateNonHydro,
        interpolation_state: InterpolationState,
        vertical_params: VerticalModelParams,
    ):
        self.grid = grid
        self.metric_state: MetricStateNonHydro = metric_state
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
        self.z_w_v = _allocate(VertexDim, KDim, is_halfdim=True, mesh=self.grid)
        self.z_v_grad_w = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_ekinh = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_w_concorr_mc = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_w_con_c = _allocate(CellDim, KDim, is_halfdim=True, mesh=self.grid)
        self.zeta = _allocate(VertexDim, KDim, mesh=self.grid)
        self.z_w_con_c_full = _allocate(CellDim, KDim, mesh=self.grid)
        self.cfl_clipping = _allocate(CellDim, KDim, mesh=self.grid, dtype=bool)
        self.pre_levelmask = _allocate(CellDim, KDim, mesh=self.grid, dtype=bool)
        self.levelmask = _allocate(KDim, mesh=self.grid, dtype=bool)
        self.vcfl = _allocate(CellDim, KDim, mesh=self.grid)
        self.k_field = _allocate_indices(KDim, mesh=self.grid, is_halfdim=True)

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

        self.cfl_w_limit = self.cfl_w_limit / dtime
        self.scalfac_exdiff = self.scalfac_exdiff / (
            dtime * (0.85 - self.cfl_w_limit * dtime)
        )

        (indices_0_1, indices_0_2) = self.grid.get_indices_from_to(
            VertexDim,
            HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1,
            HorizontalMarkerIndex.local(VertexDim) - 1,
        )

        (indices_1_1, indices_1_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4,
            HorizontalMarkerIndex.local(EdgeDim) - 2,
        )

        (indices_2_1, indices_2_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6,
            HorizontalMarkerIndex.local(EdgeDim) - 1,
        )

        (indices_3_1, indices_3_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.lateral_boundary(CellDim) + 3,
            HorizontalMarkerIndex.local(CellDim) - 1,
        )

        (indices_4_1, indices_4_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        (indices_5_1, indices_5_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim) + 1,
            HorizontalMarkerIndex.local(EdgeDim),
        )

        if not vn_only:
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl.with_backend(
                run_gtfn
            )(
                p_cell_in=prognostic_state.w,
                c_intp=self.interpolation_state.c_intp,
                p_vert_out=self.z_w_v,
                horizontal_start=indices_0_1,
                horizontal_end=indices_0_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "V2C": self.grid.get_v2c_connectivity(),
                },
            )

        mo_math_divrot_rot_vertex_ri_dsl.with_backend(run_gtfn)(
            vec_e=prognostic_state.vn,
            geofac_rot=self.interpolation_state.geofac_rot,
            rot_vec=self.zeta,
            horizontal_start=indices_0_1,
            horizontal_end=indices_0_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "V2E": self.grid.get_v2e_connectivity(),
            },
        )

        mo_velocity_advection_stencil_01.with_backend(run_gtfn)(
            vn=prognostic_state.vn,
            rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
            vt=diagnostic_state.vt,
            horizontal_start=indices_1_1,
            horizontal_end=indices_1_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "E2C2E": self.grid.get_e2c2e_connectivity(),
            },
        )

        mo_velocity_advection_stencil_02.with_backend(run_gtfn)(
            wgtfac_e=self.metric_state.wgtfac_e,
            vn=prognostic_state.vn,
            vt=diagnostic_state.vt,
            vn_ie=diagnostic_state.vn_ie,
            z_kin_hor_e=z_kin_hor_e,
            horizontal_start=indices_1_1,
            horizontal_end=indices_1_2,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "Koff": KDim,
            },
        )

        if not vn_only:
            mo_velocity_advection_stencil_03.with_backend(run_gtfn)(
                wgtfac_e=self.metric_state.wgtfac_e,
                vt=diagnostic_state.vt,
                z_vt_ie=z_vt_ie,
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=1,
                vertical_end=self.grid.n_lev(),
                offset_provider={"Koff": KDim},
            )

        velocity_prog.fused_stencils_4_5_6.with_backend(run_gtfn)(
            vn=prognostic_state.vn,
            vt=diagnostic_state.vt,
            vn_ie=diagnostic_state.vn_ie,
            z_vt_ie=z_vt_ie,
            z_kin_hor_e=z_kin_hor_e,
            ddxn_z_full=self.metric_state.ddxn_z_full,
            ddxt_z_full=self.metric_state.ddxt_z_full,
            z_w_concorr_me=z_w_concorr_me,
            wgtfacq_e_dsl=self.metric_state.wgtfacq_e_dsl,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.n_lev(),
            horizontal_start=indices_1_1,
            horizontal_end=indices_1_2,
            vertical_start=int32(0),
            vertical_end=self.grid.n_lev() + 1,
            offset_provider={
                "Koff": KDim,
            },
        )

        if not vn_only:
            mo_velocity_advection_stencil_07.with_backend(run_gtfn)(
                vn_ie=diagnostic_state.vn_ie,
                inv_dual_edge_length=inv_dual_edge_length,
                w=prognostic_state.w,
                z_vt_ie=z_vt_ie,
                inv_primal_edge_length=inv_primal_edge_length,
                tangent_orientation=tangent_orientation,
                z_w_v=self.z_w_v,
                z_v_grad_w=self.z_v_grad_w,
                horizontal_start=indices_2_1,
                horizontal_end=indices_2_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "E2C": self.grid.get_e2c_connectivity(),
                    "E2V": self.grid.get_e2v_connectivity(),
                },
            )

        mo_velocity_advection_stencil_08.with_backend(run_gtfn)(
            z_kin_hor_e=z_kin_hor_e,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            z_ekinh=self.z_ekinh,
            horizontal_start=indices_3_1,
            horizontal_end=indices_3_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "C2CE": self.grid.get_c2ce_connectivity(),
            },
        )

        velocity_prog.fused_stencils_9_10.with_backend(run_gtfn)(
            z_w_concorr_me=z_w_concorr_me,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            local_z_w_concorr_mc=self.z_w_concorr_mc,
            wgtfac_c=self.metric_state.wgtfac_c,
            w_concorr_c=diagnostic_state.w_concorr_c,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.n_lev(),
            horizontal_start=indices_3_1,
            horizontal_end=indices_3_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "C2CE": self.grid.get_c2ce_connectivity(),
                "Koff": KDim,
            },
        )

        velocity_prog.fused_stencils_11_to_13.with_backend(run_gtfn)(
            w=prognostic_state.w,
            w_concorr_c=diagnostic_state.w_concorr_c,
            local_z_w_con_c=self.z_w_con_c,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.n_lev(),
            horizontal_start=indices_3_1,
            horizontal_end=indices_3_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev() + 1,
            offset_provider={},
        )

        velocity_prog.fused_stencil_14.with_backend(run_gtfn)(
            ddqz_z_half=self.metric_state.ddqz_z_half,
            local_z_w_con_c=self.z_w_con_c,
            local_cfl_clipping=self.cfl_clipping,
            local_pre_levelmask=self.pre_levelmask,
            local_vcfl=self.vcfl,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=indices_3_1,
            horizontal_end=indices_3_2,
            vertical_start=int32(
                max(3, self.vertical_params.index_of_damping_layer - 2)
            ),
            vertical_end=int32(self.grid.n_lev() - 3),
            offset_provider={},
        )

        mo_velocity_advection_stencil_15.with_backend(run_gtfn)(
            z_w_con_c=self.z_w_con_c,
            z_w_con_c_full=self.z_w_con_c_full,
            horizontal_start=indices_3_1,
            horizontal_end=indices_3_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        velocity_prog.fused_stencils_16_to_17.with_backend(run_gtfn)(
            w=prognostic_state.w,
            local_z_v_grad_w=self.z_v_grad_w,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            local_z_w_con_c=self.z_w_con_c,
            coeff1_dwdz=self.metric_state.coeff1_dwdz,
            coeff2_dwdz=self.metric_state.coeff2_dwdz,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc,
            horizontal_start=indices_4_1,
            horizontal_end=indices_4_2,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "C2CE": self.grid.get_c2ce_connectivity(),
                "Koff": KDim,
            },
        )

        mo_velocity_advection_stencil_18.with_backend(run_gtfn)(
            levelmask=self.levelmask,
            cfl_clipping=self.cfl_clipping,
            owner_mask=owner_mask,
            z_w_con_c=self.z_w_con_c,
            ddqz_z_half=self.metric_state.ddqz_z_half,
            area=cell_areas,
            geofac_n2s=self.interpolation_state.geofac_n2s,
            w=prognostic_state.w,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=indices_4_1,
            horizontal_end=indices_4_2,
            vertical_start=int32(
                max(3, self.vertical_params.index_of_damping_layer - 2)
            ),
            vertical_end=int32(self.grid.n_lev() - 4),
            offset_provider={
                "C2E2CO": self.grid.get_c2e2co_connectivity(),
            },
        )

        mo_velocity_advection_stencil_19.with_backend(run_gtfn)(
            z_kin_hor_e=z_kin_hor_e,
            coeff_gradekin=self.metric_state.coeff_gradekin,
            z_ekinh=self.z_ekinh,
            zeta=self.zeta,
            vt=diagnostic_state.vt,
            f_e=f_e,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            vn_ie=diagnostic_state.vn_ie,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            ddt_vn_adv=diagnostic_state.ddt_vn_apc_pc,
            horizontal_start=indices_5_1,
            horizontal_end=indices_5_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
                "E2V": self.grid.get_e2v_connectivity(),
                "E2EC": self.grid.get_e2ec_connectivity(),
                "Koff": KDim,
            },
        )

        mo_velocity_advection_stencil_20.with_backend(run_gtfn)(
            levelmask=self.levelmask,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            area_edge=area_edge,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            zeta=self.zeta,
            geofac_grdiv=self.interpolation_state.geofac_grdiv,
            vn=prognostic_state.vn,
            ddt_vn_adv=diagnostic_state.ddt_vn_apc_pc,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
            horizontal_start=indices_5_1,
            horizontal_end=indices_5_2,
            vertical_start=int32(
                max(3, self.vertical_params.index_of_damping_layer - 2)
            ),
            vertical_end=int32(self.grid.n_lev() - 4),
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

        self.cfl_w_limit = self.cfl_w_limit / dtime
        self.scalfac_exdiff = self.scalfac_exdiff / (
            dtime * (0.85 - self.cfl_w_limit * dtime)
        )

        (indices_0_1, indices_0_2) = self.grid.get_indices_from_to(
            VertexDim,
            HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1,
            HorizontalMarkerIndex.local(VertexDim) - 1,
        )

        (indices_1_1, indices_1_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4,
            HorizontalMarkerIndex.local(EdgeDim) - 2,
        )

        (indices_2_1, indices_2_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6,
            HorizontalMarkerIndex.local(EdgeDim) - 1,
        )

        (indices_3_1, indices_3_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.lateral_boundary(CellDim) + 3,
            HorizontalMarkerIndex.local(CellDim) - 1,
        )

        (indices_4_1, indices_4_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        (indices_5_1, indices_5_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim) + 1,
            HorizontalMarkerIndex.local(EdgeDim),
        )

        if not vn_only:
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl.with_backend(
                run_gtfn
            )(
                p_cell_in=prognostic_state.w,
                c_intp=self.interpolation_state.c_intp,
                p_vert_out=self.z_w_v,
                horizontal_start=indices_0_1,
                horizontal_end=indices_0_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "V2C": self.grid.get_v2c_connectivity(),
                },
            )

        mo_math_divrot_rot_vertex_ri_dsl.with_backend(run_gtfn)(
            vec_e=prognostic_state.vn,
            geofac_rot=self.interpolation_state.geofac_rot,
            rot_vec=self.zeta,
            horizontal_start=indices_0_1,
            horizontal_end=indices_0_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "V2E": self.grid.get_v2e_connectivity(),
            },
        )

        if not vn_only:
            mo_velocity_advection_stencil_07.with_backend(run_gtfn)(
                vn_ie=diagnostic_state.vn_ie,
                inv_dual_edge_length=inv_dual_edge_length,
                w=prognostic_state.w,
                z_vt_ie=z_vt_ie,
                inv_primal_edge_length=inv_primal_edge_length,
                tangent_orientation=tangent_orientation,
                z_w_v=self.z_w_v,
                z_v_grad_w=self.z_v_grad_w,
                horizontal_start=indices_2_1,
                horizontal_end=indices_2_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "E2C": self.grid.get_e2c_connectivity(),
                    "E2V": self.grid.get_e2v_connectivity(),
                },
            )


        mo_velocity_advection_stencil_08.with_backend(run_gtfn)(
            z_kin_hor_e=z_kin_hor_e,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            z_ekinh=self.z_ekinh,
            horizontal_start=indices_3_1,
            horizontal_end=indices_3_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "C2CE": self.grid.get_c2ce_connectivity(),
            },
        )

        velocity_prog.fused_stencils_11_to_13.with_backend(run_gtfn)(
            w=prognostic_state.w,
            w_concorr_c=diagnostic_state.w_concorr_c,
            local_z_w_con_c=self.z_w_con_c,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.n_lev(),
            horizontal_start=indices_3_1,
            horizontal_end=indices_3_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )

        velocity_prog.fused_stencil_14.with_backend(run_gtfn)(
            ddqz_z_half=self.metric_state.ddqz_z_half,
            local_z_w_con_c=self.z_w_con_c,
            local_cfl_clipping=self.cfl_clipping,
            local_pre_levelmask=self.pre_levelmask,
            local_vcfl=self.vcfl,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=indices_3_1,
            horizontal_end=indices_3_2,
            vertical_start=int32(
                max(3, self.vertical_params.index_of_damping_layer - 2)
            ),
            vertical_end=int32(self.grid.n_lev() - 3),
            offset_provider={},
        )

        mo_velocity_advection_stencil_15.with_backend(run_gtfn)(
            z_w_con_c=self.z_w_con_c,
            z_w_con_c_full=self.z_w_con_c_full,
            horizontal_start=indices_3_1,
            horizontal_end=indices_3_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        velocity_prog.fused_stencils_16_to_17.with_backend(run_gtfn)(
            w=prognostic_state.w,
            local_z_v_grad_w=self.z_v_grad_w,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            local_z_w_con_c=self.z_w_con_c,
            coeff1_dwdz=self.metric_state.coeff1_dwdz,
            coeff2_dwdz=self.metric_state.coeff2_dwdz,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc,
            horizontal_start=indices_4_1,
            horizontal_end=indices_4_2,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "C2CE": self.grid.get_c2ce_connectivity(),
                "Koff": KDim,
            },
        )

        mo_velocity_advection_stencil_18.with_backend(run_gtfn)(
            levelmask=self.levelmask,
            cfl_clipping=self.cfl_clipping,
            owner_mask=owner_mask,
            z_w_con_c=self.z_w_con_c,
            ddqz_z_half=self.metric_state.ddqz_z_half,
            area=cell_areas,
            geofac_n2s=self.interpolation_state.geofac_n2s,
            w=prognostic_state.w,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=indices_4_1,
            horizontal_end=indices_4_2,
            vertical_start=int32(
                max(3, self.vertical_params.index_of_damping_layer - 2)
            ),
            vertical_end=int32(self.grid.n_lev() - 4),
            offset_provider={
                "C2E2CO": self.grid.get_c2e2co_connectivity(),
            },
        )

        mo_velocity_advection_stencil_19.with_backend(run_gtfn)(
            z_kin_hor_e=z_kin_hor_e,
            coeff_gradekin=self.metric_state.coeff_gradekin,
            z_ekinh=self.z_ekinh,
            zeta=self.zeta,
            vt=diagnostic_state.vt,
            f_e=f_e,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            vn_ie=diagnostic_state.vn_ie,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            ddt_vn_adv=diagnostic_state.ddt_vn_apc_pc,
            horizontal_start=indices_5_1,
            horizontal_end=indices_5_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
                "E2V": self.grid.get_e2v_connectivity(),
                "E2EC": self.grid.get_e2ec_connectivity(),
                "Koff": KDim,
            },
        )

        mo_velocity_advection_stencil_20.with_backend(run_gtfn)(
            levelmask=self.levelmask,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            area_edge=area_edge,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            zeta=self.zeta,
            geofac_grdiv=self.interpolation_state.geofac_grdiv,
            vn=prognostic_state.vn,
            ddt_vn_adv=diagnostic_state.ddt_vn_apc_pc,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
            horizontal_start=indices_5_1,
            horizontal_end=indices_5_2,
            vertical_start=int32(
                max(3, self.vertical_params.index_of_damping_layer - 2)
            ),
            vertical_end=int32(self.grid.n_lev() - 4),
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
                "E2V": self.grid.get_e2v_connectivity(),
                "E2C2EO": self.grid.get_e2c2eo_connectivity(),
                "Koff": KDim,
            },
        )
