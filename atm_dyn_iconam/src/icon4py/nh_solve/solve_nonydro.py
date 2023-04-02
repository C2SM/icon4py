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

        if self.l_open_ubc:
            raise NotImplementedError(
                "Open upper boundary conditions not supported"
                "(to allow vertical motions related to diabatic heating to extend beyond the model top)"
            )

        if self.lvert_nest or self.l_vert_nested:
            raise NotImplementedError(
                "Vertical nesting support not implemented"
            )

        if self.igradp_method == 4 or self.igradp_method == 5:
            raise NotImplementedError(
                "igradp_method 4 and 5 not implemented"
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

        #velocity_advection is referenced inside

        self.run_predictor_step()

        self.run_corrector_step()

        if l_limited_area or jg > 1:
            nhsolve_prog.stencils_66_67()
            #_mo_solve_nonhydro_stencil_66()
            #_mo_solve_nonhydro_stencil_67()

        mo_solve_nonhydro_stencil_68()


    def run_predictor_step(
        self,
        vn_only: bool,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        z_fields: ZFields,
        dtime: float,
    ):
        if itime_scheme >=6 or l_init or l_recompute:
            if itime_scheme<6 and not l_init:
                lvn_only=true
            else:
                lvn_only=false
            velocity_advection.run_predictor_step()
        nvar = nnow

        if l_limited_area:
            set_zero_c_k(self.z_rth_pr_1, offset_provider={})
            set_zero_c_k(self.z_rth_pr_2, offset_provider={})
            #_mo_solve_nonhydro_stencil_01()

        nhsolve_prog.predictor_stencils_2_3(
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

        if igradp_method <= 3:
            nhsolve_prog.predictor_stencils_4_5_6()
            #_mo_solve_nonhydro_stencil_04()
            #_mo_solve_nonhydro_stencil_05()
            #_mo_solve_nonhydro_stencil_06()

            if nflatlev == 1:
                raise NotImplementedError(
                    "nflatlev=1 not implemented"
                )

        nhsolve_prog.predictor_stencils_7_8_9()
        #_mo_solve_nonhydro_stencil_07()
        #_mo_solve_nonhydro_stencil_08()
        #_mo_solve_nonhydro_stencil_09()

        if l_open_ubc and not l_vert_nested:
            raise NotImplementedError(
                "Nesting support not implemented. "
                "l_open_ubc not implemented"
            )

        nhsolve_prog.predictor_stencils_11_lower_upper(
        )
        #_mo_solve_nonhydro_stencil_11_lower()
        #_mo_solve_nonhydro_stencil_11_upper()

        if igradp_method <= 3:
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

        if iadv_rhotheta == 1:
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl()
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl()
        elif iadv_rhotheta == 2:
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
        elif iadv_rhotheta == 3:
            #First call: compute backward trajectory with wind at time level nnow
            lcompute=true
            lcleanup=false
            upwind_hflux_miura3()
            # Second call: compute only reconstructed value for flux divergence
            lcompute = false
            lcleanup = true
            upwind_hflux_miura3()

        # Please see test.f90 for this section. Above call to 'wrap_run_mo_solve_nonhydro_stencil_14'
        if iadv_rhotheta <= 2:
            if idiv_method == 1:
                pass
            else:
                pass
        mo_solve_nonhydro_stencil_14()

        if jg > 1 or l_limited_area:
            mo_solve_nonhydro_stencil_15()

        if iadv_rhotheta == 2:
            #Operations from upwind_hflux_miura are inlined in order to process both fields in one step
            pass
        else:
            mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1()

        mo_solve_nonhydro_stencil_18()

        if igradp_method <= 3: #stencil_20 is tricky, there's no regular field operator
            mo_solve_nonhydro_stencil_19()
            mo_solve_nonhydro_stencil_20()
        elif igradp_method == 4 or igradp_method == 5:
            raise NotImplementedError(
                "igradp_method 4 and 5 not implemented"
            )

        if igradp_method == 3:
            mo_solve_nonhydro_stencil_21()
        elif igradp_method == 5:
            raise NotImplementedError(
                "igradp_method 4 and 5 not implemented"
            )

        if igradp_method == 3 or igradp_method == 5:
            mo_solve_nonhydro_stencil_22()

        mo_solve_nonhydro_stencil_24()

        if is_iau_active:
            mo_solve_nonhydro_stencil_28()

        if l_limited_area:
            mo_solve_nonhydro_stencil_29()

        ##### COMMUNICATION PHASE

        mo_solve_nonhydro_stencil_30()

        #####  Not sure about  _mo_solve_nonhydro_stencil_31()

        if idiv_method == 1:
            mo_solve_nonhydro_stencil_32()

        nhsolve_prog.predictor_stencils_35_36()
        #_mo_solve_nonhydro_stencil_35()
        #_mo_solve_nonhydro_stencil_36()

        if not l_vert_nested:
            nhsolve_prog.predictor_stencils_37_38()
            #_mo_solve_nonhydro_stencil_37()
            #_mo_solve_nonhydro_stencil_38()

        nhsolve_prog.predictor_stencils_39_40()
        #_mo_solve_nonhydro_stencil_39()
        #_mo_solve_nonhydro_stencil_40()

        if idiv_method == 2:
            if l_limited_area:
                init_zero_contiguous_dp()

            ##stencil not translated

        if idiv_method == 2:
            div_avg()

        if idiv_method == 1:
            mo_solve_nonhydro_stencil_41()

        nhsolve_prog.stencils_43_44_45_45b()
        #_mo_solve_nonhydro_stencil_43()
        #_mo_solve_nonhydro_stencil_44()
        #_mo_solve_nonhydro_stencil_45()
        #_mo_solve_nonhydro_stencil_45_b()

        if not (l_open_ubc and l_vert_nested):
            _mo_solve_nonhydro_stencil_46()

        nhsolve_prog.stencils_47_48_49()
        #_mo_solve_nonhydro_stencil_47()
        #_mo_solve_nonhydro_stencil_48()
        #_mo_solve_nonhydro_stencil_49()

        if (is_iau_active):
            mo_solve_nonhydro_stencil_50()

        nhsolve_prog.stencils_52_53()
        #_mo_solve_nonhydro_stencil_52()
        #_mo_solve_nonhydro_stencil_53()

        if rayleigh_type == RAYLEIGH_KLEMP:
            ## ACC w_1 -> p_nh%w
            _mo_solve_nonhydro_stencil_54()

        mo_solve_nonhydro_stencil_55()

        if lhdiff_rcf and divdamp_type >= 3:
            mo_solve_nonhydro_stencil_56_63()

        if idyn_timestep == 1:
            nhsolve_prog.predictor_stencils_59_60()
            #_mo_solve_nonhydro_stencil_59()
            #_mo_solve_nonhydro_stencil_60()

        if l_limited_area:  # for MPI-parallelized case
            nhsolve_prog.predictor_stencils_61_62()
            #_mo_solve_nonhydro_stencil_61()
            #_mo_solve_nonhydro_stencil_62()

        if lhdiff_rcf and divdamp_type >= 3:
            mo_solve_nonhydro_stencil_56_63()

        ##### COMMUNICATION PHASE



    def run_corrector_step(
        self,
        vn_only: bool,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
    ):

        lvn_only = false
        velocity_advection.run_corrector_step()
        nvar=nnew

        mo_solve_nonhydro_stencil_10()

        if l_open_ubc and not l_vert_nested:
            raise NotImplementedError(
                "l_open_ubc not implemented"
            )

        mo_solve_nonhydro_stencil_17()

        if itime_scheme >= 4:
            mo_solve_nonhydro_stencil_23()

        if lhdiff_rcf and (divdamp_order == 4 or divdamp_order == 24):
            mo_solve_nonhydro_stencil_25()

        if lhdiff_rcf:
            if divdamp_order == 2 or (divdamp_order == 24 and scal_divdamp_o2 > 1e-6):
                mo_solve_nonhydro_stencil_26()
            if divdamp_order == 4 or (divdamp_order == 24 and ivdamp_fac_o2 <= 4 * divdamp_fac):
                if l_limited_area:
                    mo_solve_nonhydro_stencil_27()
                else:
                    mo_solve_nonhydro_4th_order_divdamp()

        if is_iau_active:
            mo_solve_nonhydro_stencil_28()

        ##### COMMUNICATION PHASE

        if idiv_method == 1:
            mo_solve_nonhydro_stencil_32()

            if lpred_adv:
                if lclean_mflx:
                    mo_solve_nonhydro_stencil_33()
                mo_solve_nonhydro_stencil_34()

        if itime_scheme >= 5:
            nhsolve_prog.corrector_stencils_35_39_40()
            #_mo_solve_nonhydro_stencil_35()
            #_mo_solve_nonhydro_stencil_39()
            #_mo_solve_nonhydro_stencil_40()

        if idiv_method == 2:
            if l_limited_area:
                init_zero_contiguous_dp()

            ##stencil not translated

        if idiv_method == 2:
            div_avg()

        if idiv_method == 1:
            mo_solve_nonhydro_stencil_41()

        if itime_scheme >= 4:
            mo_solve_nonhydro_stencil_42()
        else:
            mo_solve_nonhydro_stencil_43()

        mo_solve_nonhydro_stencil_44()

        mo_solve_nonhydro_stencil_45()
        mo_solve_nonhydro_stencil_45_b()

        if not l_open_ubc and not l_vert_nested:
            mo_solve_nonhydro_stencil_46()

        nhsolve_prog.stencils_47_48_49()
        #_mo_solve_nonhydro_stencil_47()
        #_mo_solve_nonhydro_stencil_48()
        #_mo_solve_nonhydro_stencil_49()

        if is_iau_active:
            mo_solve_nonhydro_stencil_50()

        nhsolve_prog.stencils_52_53()
        #_mo_solve_nonhydro_stencil_52()
        #_mo_solve_nonhydro_stencil_53()

        if rayleigh_type == RAYLEIGH_KLEMP:
            ## ACC w_1 -> p_nh%w
            mo_solve_nonhydro_stencil_54()

        mo_solve_nonhydro_stencil_55()

        if lpred_adv:
            if lclean_mflx:
                mo_solve_nonhydro_stencil_57()

        mo_solve_nonhydro_stencil_58()

        if lpred_adv:
            if lclean_mflx:
                mo_solve_nonhydro_stencil_64()
            mo_solve_nonhydro_stencil_65()

        ##### COMMUNICATION PHASE

