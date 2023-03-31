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

import icon4py.nh_solve.solve_nonhydro_program as nhsolve_prog
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

from icon4py.state_utils.diagnostic_state import DiagnosticState
from icon4py.state_utils.icon_grid import VerticalModelParams
from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.state_utils.metric_state import MetricState
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.velocity.velocity_advection import VelocityAdvection
from icon4py.velocity.z_fields import ZFields



class NonHydrostaticConfig:
    """
    Contains necessary parameter to configure a nonhydro run.

    Encapsulates namelist parameters and derived parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    TODO: [ag] to be read from config
    TODO: [ag] handle dependencies on other namelists (see below...)
    """

    def __init__(
        self,
        itime_scheme: int = 5,
        ndyn_substeps: int = 5,
        divdamp_order: int = 1,
        lhdiff_rcf: bool = False,
        lextra_diffu: bool = False,
        divdamp_fac: float = 15.0,
        divdamp_fac2: float = 15.0,
        divdamp_fac3: float = 15.0,
        divdamp_fac4: float = 15.0,
        divdamp_z: float = 15.0,
        divdamp_z2: float = 15.0,
    ):

        # parameters from namelist diffusion_nml
        self.itime_scheme: int = itime_scheme

        self._validate()

    def _validate(self):
        """Apply consistency checks and validation on configuration parameters."""
        if self.diffusion_type != 5:
            raise NotImplementedError(
                "Only diffusion type 5 = `Smagorinsky diffusion with fourth-order background "
                "diffusion` is implemented"
            )

        if self.diffusion_type < 0:
            self.apply_to_temperature = False
            self.apply_to_horizontal_wind = False
            self.apply_to_vertical_wind = False
        else:
            self.apply_to_temperature = True
            self.apply_to_horizontal_wind = True

        if not self.apply_zdiffusion_t:
            raise NotImplementedError(
                "zdiffu_t = False is not implemented (leaves out stencil_15)"
            )



class NonHydrostaticParams:
    """Calculates derived quantities depending on the NonHydrostaticConfig."""

    def __init__(self, config: NonHydrostaticConfig):

        self.K2: Final[float] = (
            1.0 / (config.hdiff_efdt_ratio * 8.0)
            if config.hdiff_efdt_ratio > 0.0
            else 0.0
        )


class SolveNonhydro:
    def __init__(self, run_program=True):


    def init(
        self,
        grid: IconGrid,
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        metric_state: MetricState,
        interpolation_state: InterpolationState,
        vertical_params: VerticalModelParams,
    ):
        """
       Initialize NonHydrostatic granule with configuration.

       calculates all local fields that are used in diffusion within the time loop
       """
        self.config: NonHydrostaticConfig = config
        self.params: NonHydrostaticParams = params
        self.grid = grid
        self.vertical_params = vertical_params
        self.metric_state: MetricState = metric_state
        self.interpolation_state: InterpolationState = interpolation_state

        self._allocate_local_fields()


    @property
    def initialized(self):
        return self._initialized

    def _allocate_local_fields(self):



    def initial_step(self):


    def time_step(
        self,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        dtime: float,
        tangent_orientation: Field[[EdgeDim], float],
        inverse_primal_edge_lengths: Field[[EdgeDim], float],
        inverse_dual_edge_length: Field[[EdgeDim], float],
        inverse_vert_vert_lengths: Field[[EdgeDim], float],
        primal_normal_vert: VectorTuple[Field[[ECVDim], float], Field[[ECVDim], float]],
        dual_normal_vert: VectorTuple[Field[[ECVDim], float], Field[[ECVDim], float]],
        edge_areas: Field[[EdgeDim], float],
        cell_areas: Field[[CellDim], float],
    ):
        """
        Do one diffusion step within regular time loop.

        runs a diffusion step for the parameter linit=False, within regular time loop.
        """

        if not self._run_program:
            self._do_nonhydrosolve_step(
                diagnostic_state=diagnostic_state,
                prognostic_state=prognostic_state,
                dtime=dtime,
                tangent_orientation=tangent_orientation,
                inverse_primal_edge_lengths=inverse_primal_edge_lengths,
                inverse_dual_edge_length=inverse_dual_edge_length,
                inverse_vertex_vertex_lengths=inverse_vert_vert_lengths,
                primal_normal_vert=primal_normal_vert,
                dual_normal_vert=dual_normal_vert,
                edge_areas=edge_areas,
                cell_areas=cell_areas,
                diff_multfac_vn=self.diff_multfac_vn,
                smag_limit=self.smag_limit,
                smag_offset=self.smag_offset,
            )
        else:
            print("Not implemented")

    def run_predictor_step(
        self,
        vn_only: bool,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        z_fields: ZFields,
        dtime: float,
    ):
        if LAM:
            set_zero_c_k(self.z_rth_pr_1, offset_provider={})
            set_zero_c_k(self.z_rth_pr_2, offset_provider={})
            #_mo_solve_nonhydro_stencil_01()

        nhsolve_prog.nhsolve_predictor_tendencies_2_3(
            exner_exfac,
            exner,
            exner_ref_mc,
            exner_pr,
            z_exner_ex_pr,
            edge_startindex_local - 2,
            self.vertical_params.nflatlev,
            self.grid.n_lev(),
            offset_provider={
                "Koff": KDim,
            },
        )
        #_mo_solve_nonhydro_stencil_02()
        #_mo_solve_nonhydro_stencil_03()

        if (igradp_method <= 3):
            nhsolve_prog.nhsolve_predictor_tendencies_4_5_6(
                self.metric_state.wgtfacq_c,
                z_exner_ex_pr,
                z_exner_ic,
                self.metric_state.wgtfac_e,
                inv_ddqz_z_full,
                z_dexner_dz_c_1,
                edge_startindex_local - 2,
                self.vertical_params.nflatlev,
                self.grid.n_lev(),
                offset_provider={
                    "Koff": KDim,
                },
            )
            #_mo_solve_nonhydro_stencil_04()
            #_mo_solve_nonhydro_stencil_05()
            #_mo_solve_nonhydro_stencil_06()

            if (nflatlev == 1):
                print("Not implemented")

        nhsolve_prog.nhsolve_predictor_tendencies_4_5_6(
            prognostic_state.rho,

        )
        #_mo_solve_nonhydro_stencil_07()
        #_mo_solve_nonhydro_stencil_08()
        #_mo_solve_nonhydro_stencil_09()

        if (l_open_ubc and not l_vert_nested):
            print("Not implemented")

        nhsolve_prog.nhsolve_predictor_tendencies_11_lower_upper(
        )
        #_mo_solve_nonhydro_stencil_11_lower()
        #_mo_solve_nonhydro_stencil_11_upper()

        if (igradp_method <= 3):
            mo_solve_nonhydro_stencil_12(
                z_theta_v_pr_ic,
                d2dexdz2_fac1_mc,
                d2dexdz2_fac2_mc,
                z_rth_pr_2,
                z_dexner_dz_c_2,
                offset_provider={"Koff": KDim},
            )

        mo_solve_nonhydro_stencil_13(
            rho,
            rho_ref_mc,
            theta_v,
            theta_ref_mc,
            z_rth_pr_1,
            z_rth_pr_2,
            offset_provider={},
        )

        if (iadv_rhotheta == 1):
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl()
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl()
        elif (iadv_rhotheta == 2):
            mo_math_gradients_grad_green_gauss_cell_dsl()
        elif (iadv_rhotheta == 3):
            upwind_hflux_miura3

        if (iadv_rhotheta <= 2):
            if (idiv_method == 1):

            else:

        mo_math_gradients_grad_green_gauss_cell_dsl(
            p_grad_1_u,
            p_grad_1_v,
            p_grad_2_u,
            p_grad_2_v,
            p_ccpr1,
            p_ccpr2,
            geofac_grg_x,
            geofac_grg_y,
            offset_provider={
                "C2E2CO": mesh.get_c2e2cO_offset_provider(),
                "C2E2CODim": C2E2CODim,
            },
        )

        if (l_limited_area):
            _mo_solve_nonhydro_stencil_15()

        if (iadv_rhotheta == 2):
        else:
            mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1()

        _mo_solve_nonhydro_stencil_18()

        if (igradp_method <= 3):
            _mo_solve_nonhydro_stencil_19()
            _mo_solve_nonhydro_stencil_20()
        else if (igradp == 4 or igradp_method == 5):

        if (igradp_method == 3):
            _mo_solve_nonhydro_stencil_21()
        else if (igradp_method == 5):

        if (igradp_method == 3 or igradp_method == 5):
            _mo_solve_nonhydro_stencil_22()

        _mo_solve_nonhydro_stencil_24()

        if (is_iau_active):
            _mo_solve_nonhydro_stencil_28()

        if (l_limited_area):
            _mo_solve_nonhydro_stencil_29()

        ##### COMMUNICATION PHASE

        _mo_solve_nonhydro_stencil_30()

        #####  Not sure about  _mo_solve_nonhydro_stencil_31()

        if (idiv_method == 1):
            _mo_solve_nonhydro_stencil_32()

        _mo_solve_nonhydro_stencil_35()

        _mo_solve_nonhydro_stencil_36()

        if (not l_vert_nested):
            _mo_solve_nonhydro_stencil_37()
            _mo_solve_nonhydro_stencil_38()

        _mo_solve_nonhydro_stencil_39()
        _mo_solve_nonhydro_stencil_40()

        if (idiv_method == 2):
            if (l_limited_area):
                init_zero_contiguous_dp()

            ##stencil not translated

        if (idiv_method == 2):
            div_avg()

        if (idiv_method == 1):
            _mo_solve_nonhydro_stencil_41()

        nhsolve_prog.nhsolve_predictor_tendencies_43_44_45_45b
        #_mo_solve_nonhydro_stencil_43()
        #_mo_solve_nonhydro_stencil_44()
        #_mo_solve_nonhydro_stencil_45()
        #_mo_solve_nonhydro_stencil_45_b()

        if (not l_open_ubc and not l_vert_nested):
            _mo_solve_nonhydro_stencil_46()

        nhsolve_prog.nhsolve_predictor_tendencies_47_48_49()
        #_mo_solve_nonhydro_stencil_47()
        #_mo_solve_nonhydro_stencil_48()
        #_mo_solve_nonhydro_stencil_49()

        if (is_iau_active):
            _mo_solve_nonhydro_stencil_50()

        nhsolve_prog.nhsolve_predictor_tendencies_52_53()
        #_mo_solve_nonhydro_stencil_52()
        #_mo_solve_nonhydro_stencil_53()

        if (rayleigh_type == RAYLEIGH_KLEMP):
            ## ACC w_1 -> p_nh%w
            _mo_solve_nonhydro_stencil_54()

        _mo_solve_nonhydro_stencil_55()

        if (lhdiff_rcf and divdamp_type >= 3):
            _mo_solve_nonhydro_stencil_56_63()

        if (idyn_timestep == 1):
            nhsolve_prog.nhsolve_predictor_tendencies_59_60()
            #_mo_solve_nonhydro_stencil_59()
            #_mo_solve_nonhydro_stencil_60()

        if (l_limited_area):  # for MPI-parallelized case
            nhsolve_prog.nhsolve_predictor_tendencies_61_62()
            #_mo_solve_nonhydro_stencil_61()
            #_mo_solve_nonhydro_stencil_62()

        if (lhdiff_rcf and divdamp_type >= 3):
            _mo_solve_nonhydro_stencil_56_63()

    ##### COMMUNICATION PHASE

    def run_corrector_step(
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

        _mo_solve_nonhydro_stencil_10()

        if (l_open_ubc and not l_vert_nested):

        _mo_solve_nonhydro_stencil_17()

        if (itime_scheme >= 4):
            _mo_solve_nonhydro_stencil_23()

        if (lhdiff_rcf and (divdamp_order == 4.OR.divdamp_order == 24)):
            _mo_solve_nonhydro_stencil_25()

        if (lhdiff_rcf):
            if (divdamp_order == 2 or (divdamp_order == 24 and scal_divdamp_o2 > 1e-6)):
                _mo_solve_nonhydro_stencil_26()
            if (divdamp_order == 4 or (divdamp_order == 24 and ivdamp_fac_o2 <= 4 * divdamp_fac)):
                if (l_limited_area):
                    _mo_solve_nonhydro_stencil_27()
                else:
                    _mo_solve_nonhydro_4th_order_divdamp()

        if (is_iau_active):
            _mo_solve_nonhydro_stencil_28()

        ##### COMMUNICATION PHASE

        if (idiv_method == 1):
            _mo_solve_nonhydro_stencil_32()

            if (lpred_adv):
                if (lclean_mflx):
                    _mo_solve_nonhydro_stencil_33()
                _mo_solve_nonhydro_stencil_34()

        if (itime_scheme >= 5):
            _mo_solve_nonhydro_stencil_35()
            _mo_solve_nonhydro_stencil_39()
            _mo_solve_nonhydro_stencil_40()

        if (idiv_method == 2):
            if (l_limited_area):
                init_zero_contiguous_dp()

            ##stencil not translated

        if (idiv_method == 2):
            div_avg()

        if (idiv_method == 1):
            _mo_solve_nonhydro_stencil_41()

        if (itime_scheme >= 4):
            _mo_solve_nonhydro_stencil_42()
        else:
            _mo_solve_nonhydro_stencil_43()

        _mo_solve_nonhydro_stencil_44()

        _mo_solve_nonhydro_stencil_45()
        _mo_solve_nonhydro_stencil_45_b()

        if (not l_open_ubc and not l_vert_nested):
            _mo_solve_nonhydro_stencil_46()

        _mo_solve_nonhydro_stencil_47()
        _mo_solve_nonhydro_stencil_48()
        _mo_solve_nonhydro_stencil_49()

        if (is_iau_active):
            _mo_solve_nonhydro_stencil_50()

        _mo_solve_nonhydro_stencil_52()
        _mo_solve_nonhydro_stencil_53()

        if (rayleigh_type == RAYLEIGH_KLEMP):
            ## ACC w_1 -> p_nh%w
            _mo_solve_nonhydro_stencil_54()

        _mo_solve_nonhydro_stencil_55()

        if (lpred_adv):
            if (lclean_mflx):
                _mo_solve_nonhydro_stencil_57()

        _mo_solve_nonhydro_stencil_58()

        if (lpred_adv):
            if (lclean_mflx):
                _mo_solve_nonhydro_stencil_64()
            _mo_solve_nonhydro_stencil_65()

        ##### COMMUNICATION PHASE

    def _do_nonhydrosolve_step(
        self,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        dtime: float,
        tangent_orientation: Field[[EdgeDim], float],
        inverse_primal_edge_lengths: Field[[EdgeDim], float],
        inverse_dual_edge_length: Field[[EdgeDim], float],
        inverse_vertex_vertex_lengths: Field[[EdgeDim], float],
        primal_normal_vert: Tuple[Field[[ECVDim], float], Field[[ECVDim], float]],
        dual_normal_vert: Tuple[Field[[ECVDim], float], Field[[ECVDim], float]],
        edge_areas: Field[[EdgeDim], float],
        cell_areas: Field[[CellDim], float],
        diff_multfac_vn: Field[[KDim], float],
        smag_limit: Field[[KDim], float],
        smag_offset: float,
    ):
        """
        Run a diffusion step.

        Args:
            diagnostic_state: output argument, data class that contains diagnostic variables
            prognostic_state: output argument, data class that contains prognostic variables
            dtime: the time step,
            tangent_orientation:
            inverse_primal_edge_lengths:
            inverse_dual_edge_length:
            inverse_vertex_vertex_lengths:
            primal_normal_vert:
            dual_normal_vert:
            edge_areas:
            cell_areas:
            diff_multfac_vn:
            smag_limit:
            smag_offset:

        """
        klevels = self.grid.n_lev()
        k_start_end_minus2 = klevels - 2

        cell_start_nudging_minus1, cell_end_local_plus1 = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim) - 1,
            HorizontalMarkerIndex.local(CellDim) - 1,
        )

        cell_start_interior, cell_end_local = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.interior(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        cell_start_nudging, _ = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        edge_start_nudging_plus_one, edge_end_local = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim) + 1,
            HorizontalMarkerIndex.local(EdgeDim),
        )

        edge_start_lb_plus4, _ = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.local_boundary(EdgeDim) + 4,
            HorizontalMarkerIndex.local_boundary(EdgeDim) + 4,
        )

        (
            edge_start_nudging_minus1,
            edge_end_local_minus2,
        ) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim) - 1,
            HorizontalMarkerIndex.local(EdgeDim) - 2,
        )

        (
            vertex_start_local_boundary_plus3,
            vertex_end_local,
        ) = self.grid.get_indices_from_to(
            VertexDim,
            HorizontalMarkerIndex.local_boundary(VertexDim) + 3,
            HorizontalMarkerIndex.local(VertexDim),
        )
        (
            vertex_start_local_boundary_plus1,
            vertex_end_local_minus1,
        ) = self.grid.get_indices_from_to(
            VertexDim,
            HorizontalMarkerIndex.local_boundary(VertexDim) + 1,
            HorizontalMarkerIndex.local(VertexDim) - 1,
        )



