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

from functional.common import Dimension
from functional.ffront.fbuiltins import Field
from functional.iterator.embedded import StridedNeighborOffsetProvider
from z_fields import ZFields

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_01 import (
    mo_velocity_advection_stencil_01,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_02 import (
    mo_velocity_advection_stencil_02,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_03 import (
    mo_velocity_advection_stencil_03,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_04 import (
    mo_velocity_advection_stencil_04,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_05 import (
    mo_velocity_advection_stencil_05,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_06 import (
    mo_velocity_advection_stencil_06,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_07 import (
    mo_velocity_advection_stencil_07,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_08 import (
    mo_velocity_advection_stencil_08,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_09 import (
    mo_velocity_advection_stencil_09,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_10 import (
    mo_velocity_advection_stencil_10,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_11 import (
    mo_velocity_advection_stencil_11,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_12 import (
    mo_velocity_advection_stencil_12,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_13 import (
    mo_velocity_advection_stencil_13,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_14 import (
    mo_velocity_advection_stencil_14,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_15 import (
    mo_velocity_advection_stencil_15,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_16 import (
    mo_velocity_advection_stencil_16,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_17 import (
    mo_velocity_advection_stencil_17,
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
from icon4py.common.dimension import (
    C2E2CODim,
    C2EDim,
    CellDim,
    E2C2EDim,
    E2C2EODim,
    E2CDim,
    ECDim,
    EdgeDim,
    KDim,
    VertexDim,
)
from icon4py.diffusion.horizontal import HorizontalMarkerIndex
from icon4py.state_utils.diagnostic_state import DiagnosticState
from icon4py.state_utils.icon_grid import IconGrid, VerticalModelParams
from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.state_utils.metric_state import MetricState
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.state_utils.utils import bool_field, set_zero_w_k, zero_field
from icon4py.velocity.velocity_advection_program import velocity_advection_run


class VelocityAdvectionConfig:
    def __init__(self, lextra_diffu: bool = True):
        self.lextra_diffu: bool = lextra_diffu


class VelocityAdvection:
    def __init__(self, run_program=True):

        self._initialized = False

        self._run_program = run_program
        self.cfl_w_limit: Optional[float] = None
        self.scalfac_exdiff: Optional[float] = None
        self.cfl_w_limit: Optional[float] = None
        self.scalfac_exdiff: Optional[float] = None

        self.grid: Optional[IconGrid] = None
        self.interpolation_state = None
        self.metric_state = None

    def init(
        self,
        config: VelocityAdvectionConfig,
        grid: IconGrid,
        metric_state: MetricState,
        interpolation_state: InterpolationState,
        dtime: float,
    ):

        self.grid = grid
        self.metric_state: MetricState = metric_state
        self.interpolation_state: InterpolationState = interpolation_state

        self._allocate_local_fields()

        if config.lextra_diffu:
            self.cfl_w_limit = 0.65 / dtime
            self.scalfac_exdiff = 0.05 / (dtime * (0.85 - self.cfl_w_limit * dtime))
        else:
            self.cfl_w_limit = 0.85 / dtime
            self.scalfac_exdiff = 0.0

        self._initialized = True

    @property
    def initialized(self):
        return self._initialized

    def _allocate_local_fields(self):
        def _allocate(*dims: Dimension):
            return zero_field(self.grid, *dims)

        def _allocate_bool(*dims: Dimension):
            return bool_field(self.grid, *dims)

        self.z_w_v = _allocate(VertexDim, KDim)
        self.z_v_grad_w = _allocate(EdgeDim, KDim)
        self.z_ekinh = _allocate(CellDim, KDim)
        self.z_w_concorr_mc = _allocate(CellDim, KDim)
        self.z_w_con_c = _allocate(CellDim, KDim)
        self.zeta = _allocate(VertexDim, KDim)
        self.z_w_con_c_full = _allocate(CellDim, KDim)
        self.cfl_clipping = _allocate(CellDim, KDim)
        self.pre_levelmask = _allocate(CellDim, KDim)
        self.levelmask = _allocate_bool(KDim)
        self.vcfl = _allocate(CellDim, KDim)

    def time_step(
        self,
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
            edge_endindex_nudging_minus1,
            edge_endindex_local_minus2,
        ) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim) - 1,
            HorizontalMarkerIndex.local(EdgeDim) - 2,
        )

        cell_startindex_nudging, _ = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim),
            None,
        )

        cell_endindex_local, cell_endindex_local_minus1 = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.local(CellDim),
            HorizontalMarkerIndex.local(CellDim) - 1,
        )

        (
            edge_endindex_local_minus1,
            edge_endindex_local,
        ) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.local(EdgeDim) - 1,
            HorizontalMarkerIndex.local(EdgeDim),
        )

        edge_startindex_nudging_plus_one, _ = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.local(EdgeDim) + 1,
            None,
        )

        if not self._run_program:
            self._do_velocity_advection_step(
                diagnostic_state,
                prognostic_state,
                z_fields,
                inv_dual_edge_length,
                inv_primal_edge_length,
                dtime,
                tangent_orientation,
                cfl_w_limit,
                scalfac_exdiff,
                cell_areas,
                owner_mask,
                f_e,
                area_edge,
            )
        else:
            velocity_advection_run(
                vn=prognostic_state.vn,
                rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
                vt=diagnostic_state.vt,
                wgtfac_e=self.metric_state.wgtfac_e,
                vn_ie=diagnostic_state.vn_ie,
                z_kin_hor_e=z_fields.z_kin_hor_e,
                z_vt_ie=z_fields.z_vt_ie,
                ddxn_z_full=self.metric_state.ddxn_z_full,
                ddxt_z_full=self.metric_state.ddxt_z_full,
                z_w_concorr_me=z_fields.z_w_concorr_me,
                wgtfacq_e=self.metric_state.wgtfacq_e,
                inv_dual_edge_length=inv_dual_edge_length,
                w=prognostic_state.w,
                inv_primal_edge_length=inv_primal_edge_length,
                tangent_orientation=tangent_orientation,
                local_z_w_v=self.z_w_v,
                local_z_v_grad_w=self.z_v_grad_w,
                e_bln_c_s=self.interpolation_state.e_bln_c_s,
                local_z_ekinh=self.z_ekinh,
                local_z_w_concorr_mc=self.z_w_concorr_mc,
                wgtfac_c=self.metric_state.wgtfac_c,
                w_concorr_c=diagnostic_state.w_concorr_c,
                local_z_w_con_c=self.z_w_con_c,
                ddqz_z_half=self.metric_state.ddqz_z_half,
                local_cfl_clipping=self.cfl_clipping,
                local_pre_levelmask=self.pre_levelmask,
                local_vcfl=self.vcfl,
                cfl_w_limit=cfl_w_limit,
                dtime=dtime,
                local_z_w_con_c_full=self.z_w_con_c_full,
                coeff1_dwdz=self.metric_state.coeff1_dwdz,
                coeff2_dwdz=self.metric_state.coeff2_dwdz,
                ddt_w_adv=diagnostic_state.ddt_w_adv_pc,
                local_levelmask=self.levelmask,
                owner_mask=owner_mask,
                area=cell_areas,
                geofac_n2s=self.interpolation_state.geofac_n2s,
                scalfac_exdiff=scalfac_exdiff,
                coeff_gradekin=self.metric_state.coeff_gradekin,
                local_zeta=self.zeta,
                f_e=f_e,
                c_lin_e=self.interpolation_state.c_lin_e,
                ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
                ddt_vn_adv=diagnostic_state.ddt_vn_apc_pc,
                area_edge=area_edge,
                geofac_grdiv=self.interpolation_state.geofac_grdiv,
                cell_startindex_nudging=cell_startindex_nudging,
                cell_endindex_local_minus1=cell_endindex_local_minus1,
                cell_endindex_local=cell_endindex_local,
                edge_startindex_nudging_plus_one=edge_startindex_nudging_plus_one,
                edge_endindex_local=edge_endindex_local,
                edge_endindex_local_minus2=edge_endindex_local_minus2,
                edge_endindex_local_minus1=edge_endindex_local_minus1,
                nflatlev_startindex=VerticalModelParams.nflatlev,
                nrdmax_startindex=VerticalModelParams.rayleigh_damping_height,
                nlev=self.grid.n_lev(),
                nlevp1=self.grid.n_lev() + 1,
                offset_provider={
                    "E2C2E": self.grid.get_e2c2e_connectivity(),
                    "E2C2EDim": E2C2EDim,
                    "E2C": self.grid.get_e2c_connectivity(),
                    "E2V": self.grid.get_e2v_connectivity(),
                    "C2E": self.grid.get_c2e_connectivity(),
                    "C2EDim": C2EDim,
                    "C2E2CO": self.grid.get_c2e2cO_connectivity(),
                    "C2E2CODim": C2E2CODim,
                    "E2CDim": E2CDim,
                    "E2C2EO": self.grid.get_e2c2eO_connectivity(),
                    "E2C2EODim": E2C2EODim,
                    "E2EC": StridedNeighborOffsetProvider(EdgeDim, ECDim, 2),
                    "Koff": KDim,
                },
            )

    def _do_velocity_advection_step(
        self,
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

        klevels = self.grid.n_lev()
        nflatlev = VerticalModelParams.nflatlev
        nlevp1 = klevels + 1
        nrdmax = VerticalModelParams.rayleigh_damping_height

        (edge_start_nudging_plus_one, edge_end_local) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim) + 1,
            HorizontalMarkerIndex.local(EdgeDim),
        )

        (edge_end_local_minus1, edge_end_local_minus2,) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.local(EdgeDim) - 1,
            HorizontalMarkerIndex.local(EdgeDim) - 2,
        )

        cell_start_nudging, _ = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        (cell_end_local_minus1, cell_end_local) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.local(CellDim) - 1,
            HorizontalMarkerIndex.interior(CellDim),
        )

        mo_velocity_advection_stencil_01(
            vn=prognostic_state.vn,
            rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
            vt=diagnostic_state.vt,
            horizontal_start=5,
            horizontal_end=edge_end_local_minus2,
            vertical_start=0,
            vertical_end=klevels,
            offset_provider={
                "E2C2E": self.grid.get_e2c2e_connectivity(),
                "E2C2EDim": E2C2EDim,
            },
        )

        mo_velocity_advection_stencil_02(
            wgtfac_e=self.metric_state.wgtfac_e,
            vn=prognostic_state.vn,
            vt=diagnostic_state.vt,
            vn_ie=diagnostic_state.vn_ie,
            z_kin_hor_e=z_fields.z_kin_hor_e,
            horizontal_start=5,
            horizontal_end=edge_end_local_minus2,
            vertical_start=1,
            vertical_end=klevels,
            offset_provider={"Koff": KDim},
        )

        mo_velocity_advection_stencil_03(
            wgtfac_e=self.metric_state.wgtfac_e,
            vt=diagnostic_state.vt,
            z_vt_ie=z_fields.z_vt_ie,
            horizontal_start=5,
            horizontal_end=edge_end_local_minus2,
            vertical_start=1,
            vertical_end=klevels,
            offset_provider={"Koff": KDim},
        )

        mo_velocity_advection_stencil_04(
            vn=prognostic_state.vn,
            ddxn_z_full=self.metric_state.ddxn_z_full,
            ddxt_z_full=self.metric_state.ddxt_z_full,
            vt=diagnostic_state.vt,
            z_w_concorr_me=z_fields.z_w_concorr_me,
            horizontal_start=5,
            horizontal_end=edge_end_local_minus2,
            vertical_start=nflatlev,
            vertical_end=klevels,
            offset_provider={},
        )

        mo_velocity_advection_stencil_05(
            vn=prognostic_state.vn,
            vt=diagnostic_state.vt,
            vn_ie=diagnostic_state.vn_ie,
            z_vt_ie=z_fields.z_vt_ie,
            z_kin_hor_e=z_fields.z_kin_hor_e,
            horizontal_start=5,
            horizontal_end=edge_end_local_minus2,
            vertical_start=0,
            vertical_end=0,
            offset_provider={},
        )

        mo_velocity_advection_stencil_06(
            wgtfacq_e=self.metric_state.wgtfacq_e,
            vn=prognostic_state.vn,
            vn_ie=diagnostic_state.vn_ie,
            horizontal_start=5,
            horizontal_end=edge_end_local_minus2,
            vertical_start=klevels,
            vertical_end=klevels,
            offset_provider={"Koff": KDim},
        )

        mo_velocity_advection_stencil_07(
            vn_ie=diagnostic_state.vn_ie,
            inv_dual_edge_length=inv_dual_edge_length,
            w=prognostic_state.w,
            z_vt_ie=z_fields.z_vt_ie,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
            z_w_v=self.z_w_v,
            z_v_grad_w=self.z_v_grad_w,
            horizontal_start=7,
            horizontal_end=edge_end_local_minus1,
            vertical_start=0,
            vertical_end=klevels,
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
                "E2V": self.grid.get_e2v_connectivity(),
            },
        )

        mo_velocity_advection_stencil_08(
            z_kin_hor_e=z_fields.z_kin_hor_e,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            z_ekinh=self.z_ekinh,
            horizontal_start=4,
            horizontal_end=cell_end_local_minus1,
            vertical_start=0,
            vertical_end=klevels,
            offset_provider={"C2E": self.grid.get_c2e_connectivity(), "C2EDim": C2EDim},
        )

        mo_velocity_advection_stencil_09(
            z_w_concorr_me=z_fields.z_w_concorr_me,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            z_w_concorr_mc=self.z_w_concorr_mc,
            horizontal_start=4,
            horizontal_end=cell_end_local_minus1,
            vertical_start=nflatlev,
            vertical_end=klevels,
            offset_provider={"C2E": self.grid.get_c2e_connectivity(), "C2EDim": C2EDim},
        )

        mo_velocity_advection_stencil_10(
            z_w_concorr_mc=self.z_w_concorr_mc,
            wgtfac_c=self.metric_state.wgtfac_c,
            w_concorr_c=diagnostic_state.w_concorr_c,
            horizontal_start=4,
            horizontal_end=cell_end_local_minus1,
            vertical_start=nflatlev + 1,
            vertical_end=klevels,
            offset_provider={"Koff": KDim},
        )

        mo_velocity_advection_stencil_11(
            w=prognostic_state.w,
            z_w_con_c=self.z_w_con_c,
            horizontal_start=4,
            horizontal_end=cell_end_local_minus1,
            vertical_start=0,
            vertical_end=klevels,
            offset_provider={},
        )

        mo_velocity_advection_stencil_12(
            z_w_con_c=self.z_w_con_c,
            horizontal_start=4,
            horizontal_end=cell_end_local_minus1,
            vertical_start=nlevp1,
            vertical_end=nlevp1,
            offset_provider={},
        )

        mo_velocity_advection_stencil_13(
            w_concorr_c=diagnostic_state.w_concorr_c,
            z_w_con_c=self.z_w_con_c,
            horizontal_start=4,
            horizontal_end=cell_end_local_minus1,
            vertical_start=nflatlev + 1,
            vertical_end=klevels,
            offset_provider={},
        )

        set_zero_w_k(self.levelmask, offset_provider={})
        set_zero_w_k(self.cfl_clipping, offset_provider={})
        set_zero_w_k(self.pre_levelmask, offset_provider={})
        set_zero_w_k(self.vcfl, offset_provider={})

        mo_velocity_advection_stencil_14(
            ddqz_z_half=self.metric_state.ddqz_z_half,
            z_w_con_c=self.z_w_con_c,
            cfl_clipping=self.cfl_clipping,
            pre_levelmask=self.pre_levelmask,
            vcfl=self.vcfl,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=4,
            horizontal_end=cell_end_local_minus1,
            vertical_start=max(3, nrdmax - 2),
            vertical_end=klevels - 3,
            offset_provider={},
        )

        mo_velocity_advection_stencil_15(
            z_w_con_c=self.z_w_con_c,
            z_w_con_c_full=self.z_w_con_c_full,
            horizontal_start=4,
            horizontal_end=cell_end_local_minus1,
            vertical_start=0,
            vertical_end=klevels,
            offset_provider={"Koff": KDim},
        )

        mo_velocity_advection_stencil_16(
            z_w_con_c=self.z_w_con_c,
            w=prognostic_state.w,
            coeff1_dwdz=self.metric_state.coeff1_dwdz,
            coeff2_dwdz=self.metric_state.coeff2_dwdz,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc,
            horizontal_start=cell_start_nudging,
            horizontal_end=cell_end_local,
            vertical_start=1,
            vertical_end=klevels,
            offset_provider={"Koff": KDim},
        )

        mo_velocity_advection_stencil_17(
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            z_v_grad_w=self.z_v_grad_w,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc,
            horizontal_start=cell_start_nudging,
            horizontal_end=cell_end_local,
            vertical_start=1,
            vertical_end=klevels,
            offset_provider={"C2E": self.grid.get_c2e_connectivity(), "C2EDim": C2EDim},
        )

        # 18 does not have bounds??
        mo_velocity_advection_stencil_18(
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
            offset_provider={
                "C2E2CO": self.grid.get_c2e2cO_connectivity(),
                "C2E2CODim": C2E2CODim,
            },
        )

        mo_velocity_advection_stencil_19(
            z_kin_hor_e=z_fields.z_kin_hor_e,
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
            horizontal_start=edge_start_nudging_plus_one,
            horizontal_end=edge_end_local,
            vertical_start=0,
            vertical_end=klevels,
            offset_provider={
                "E2V": self.grid.get_e2v_connectivity(),
                "E2C": self.grid.get_e2c_connectivity(),
                "E2CDim": E2CDim,
                "E2EC": StridedNeighborOffsetProvider(EdgeDim, ECDim, 2),
                "Koff": KDim,
            },
        )

        mo_velocity_advection_stencil_20(
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
            d_time=dtime,
            horizontal_start=edge_start_nudging_plus_one,
            horizontal_end=edge_end_local,
            vertical_start=max(3, nrdmax - 2),
            vertical_end=klevels - 4,
            offset_provider={
                "Koff": KDim,
                "E2C": self.grid.get_e2c_connectivity(),
                "E2CDim": E2CDim,
                "E2C2EO": self.grid.get_e2c2eO_connectivity(),
                "E2C2EODim": E2C2EODim,
                "E2V": self.grid.get_e2v_connectivity(),
            },
        )
