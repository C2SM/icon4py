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
from typing import Final

from gt4py.next.common import Field

import icon4py.common.constants as constants
import icon4py.nh_solve.solve_nonhydro_program as nhsolve_prog
from icon4py.atm_dyn_iconam.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.atm_dyn_iconam.mo_math_gradients_grad_green_gauss_cell_dsl import (
    mo_math_gradients_grad_green_gauss_cell_dsl,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_4th_order_divdamp import (
    mo_solve_nonhydro_4th_order_divdamp,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_01 import (
    mo_solve_nonhydro_stencil_01,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_10 import (
    mo_solve_nonhydro_stencil_10,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_12 import (
    mo_solve_nonhydro_stencil_12,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_13 import (
    mo_solve_nonhydro_stencil_13,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_14 import (
    mo_solve_nonhydro_stencil_14,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_15 import (
    mo_solve_nonhydro_stencil_15,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_17 import (
    mo_solve_nonhydro_stencil_17,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_18 import (
    mo_solve_nonhydro_stencil_18,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_21 import (
    mo_solve_nonhydro_stencil_21,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_22 import (
    mo_solve_nonhydro_stencil_22,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_23 import (
    mo_solve_nonhydro_stencil_23,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_24 import (
    mo_solve_nonhydro_stencil_24,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_25 import (
    mo_solve_nonhydro_stencil_25,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_26 import (
    mo_solve_nonhydro_stencil_26,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_27 import (
    mo_solve_nonhydro_stencil_27,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_28 import (
    mo_solve_nonhydro_stencil_28,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_29 import (
    mo_solve_nonhydro_stencil_29,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_30 import (
    mo_solve_nonhydro_stencil_30,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_32 import (
    mo_solve_nonhydro_stencil_32,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_33 import (
    mo_solve_nonhydro_stencil_33,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_34 import (
    mo_solve_nonhydro_stencil_34,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_41 import (
    mo_solve_nonhydro_stencil_41,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_46 import (
    mo_solve_nonhydro_stencil_46,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_50 import (
    mo_solve_nonhydro_stencil_50,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_54 import (
    mo_solve_nonhydro_stencil_54,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_55 import (
    mo_solve_nonhydro_stencil_55,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_58 import (
    mo_solve_nonhydro_stencil_58,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_65 import (
    mo_solve_nonhydro_stencil_65,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_68 import (
    mo_solve_nonhydro_stencil_68,
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
    V2CDim,
    VertexDim,
)
from icon4py.state_utils.diagnostic_state import (
    DiagnosticState,
    DiagnosticStateNonHydro,
)
from icon4py.state_utils.horizontal import HorizontalMarkerIndex
from icon4py.state_utils.icon_grid import IconGrid, VerticalModelParams
from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.state_utils.metric_state import MetricState, MetricStateNonHydro
from icon4py.state_utils.prep_adv_state import PrepAdvection
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.state_utils.utils import _allocate, set_zero_c_k
from icon4py.state_utils.z_fields import ZFields
from icon4py.velocity import velocity_advection


class NonHydrostaticConfig:
    """
    Contains necessary parameter to configure a nonhydro run.

    Encapsulates namelist parameters and derived parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    TODO: @abishekg7 to be read from config
    TODO: @abishekg7 handle dependencies on other namelists (see below...)
    """

    def __init__(
        self,
        itime_scheme: int = 5,
        iadv_rhotheta: int = 5,
        igradp_method: int = 5,
        ndyn_substeps_var: float = 3.0,
        rayleigh_type: int = 2,
        divdamp_order: int = 24,
        idiv_method: int = 4,  # from mo_dynamics_config in FORTRAN
        is_iau_active: bool = False,
        iau_wgt_dyn: float = 0.0,
        divdamp_type: int = 3,
        lhdiff_rcf: bool = True,
        l_vert_nested: bool = False,
        l_open_ubc: bool = True,
        rhotheta_offctr: float = -0.1,
        veladv_offctr: float = 2.0,
        divdamp_fac_o2: float = 2.0,
    ):

        # parameters from namelist diffusion_nml
        self.itime_scheme: int = itime_scheme
        self.iadv_rhotheta: int = iadv_rhotheta

        self._validate()
        self.l_open_ubc: bool = l_open_ubc
        self.igradp_method: int = igradp_method
        self.ndyn_substeps_var = ndyn_substeps_var
        self.idiv_method: int = idiv_method
        self.is_iau_active: bool = is_iau_active
        self.iau_wgt_dyn: float = iau_wgt_dyn
        self.divdamp_type: int = divdamp_type
        self.lhdiff_rcf: bool = lhdiff_rcf
        self.rayleigh_type: int = rayleigh_type
        self.divdamp_order: int = divdamp_order
        self.l_vert_nested: bool = l_vert_nested
        self.rhotheta_offctr: float = rhotheta_offctr
        self.veladv_offctr: float = veladv_offctr
        self.divdamp_fac_o2: float = divdamp_fac_o2

    def _validate(self):
        """Apply consistency checks and validation on configuration parameters."""
        if self.l_open_ubc:
            raise NotImplementedError(
                "Open upper boundary conditions not supported"
                "(to allow vertical motions related to diabatic heating to extend beyond the model top)"
            )

        if self.lvert_nest or self.l_vert_nested:
            raise NotImplementedError("Vertical nesting support not implemented")

        if self.igradp_method == 4 or self.igradp_method == 5:
            raise NotImplementedError("igradp_method 4 and 5 not implemented")


class NonHydrostaticParams:
    """Calculates derived quantities depending on the NonHydrostaticConfig."""

    def __init__(self, config: NonHydrostaticConfig):

        self.rd_o_cvd: Final[float] = constants.RD / constants.CPD
        self.rd_o_p0ref: Final[float] = constants.RD / constants.P0REF
        self.grav_o_cpd: Final[float] = constants.GRAV / constants.CPD

        # start level for 3D divergence damping terms
        self.kstart_dd3d: int = (
            # TODO: @abishekg7 See mo_vertical_grid.f90
        )
        # start level for moist physics processes (specified by htop_moist_proc)
        self.kstart_moist: int  # see mo_nonhydrostatic_config.f90

        self.alin = (config.divdamp_fac2 - config.divdamp_fac) / (
            config.divdamp_z2 - config.divdamp_z
        )

        self.df32 = config.divdamp_fac3 - config.divdamp_fac2
        self.dz32 = config.divdamp_z3 - config.divdamp_z2
        self.df42 = config.divdamp_fac4 - config.divdamp_fac2
        self.dz42 = config.divdamp_z4 - config.divdamp_z2

        self.bqdr = (self.df42 * self.dz32 - self.df32 * self.dz42) / (
            self.dz32 * self.dz42 * (self.dz42 - self.dz32)
        )
        self.aqdr = self.df32 / self.dz32 - self.bqdr * self.dz32


class SolveNonhydro:
    # def __init__(self, run_program=True):

    def init(
        self,
        grid: IconGrid,
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        metric_state: MetricState,
        metric_state_nonhydro: MetricStateNonHydro,
        interpolation_state: InterpolationState,
        vertical_params: VerticalModelParams,
    ):
        """
        Initialize NonHydrostatic granule with configuration.

        calculates all local fields that are used in nh_solve within the time loop
        """
        self.config: NonHydrostaticConfig = config
        self.params: NonHydrostaticParams = params
        self.grid = grid
        self.vertical_params = vertical_params
        self.metric_state: MetricState = metric_state
        self.metric_state_nonhydro: MetricStateNonHydro = metric_state_nonhydro
        self.interpolation_state: InterpolationState = interpolation_state

        self._allocate_local_fields()
        self._initialized = True

        if self.grid.lvert_nest():
            self.l_vert_nested = True

        self.enh_divdamp_fac = _en_smag_fac_for_zero_nshift(
            a_vec, *fac, *z, out=enh_smag_fac, offset_provider={"Koff": KDim}
        )

        # TODO: @abishekg7 geometry_info
        self.scal_divdamp = (
            -self.enh_divdamp_fac * p_patch % geometry_info % mean_cell_area**2
        )

    @property
    def initialized(self):
        return self._initialized

    def _allocate_local_fields(self):
        self.z_exner_ex_pr = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_exner_ic = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_dexner_dz_c_1 = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_theta_v_pr_ic = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_th_ddz_exner_c = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_rth_pr = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_rth_pr_1 = self.z_rth_pr[:, 0]
        self.z_rth_pr_2 = self.z_rth_pr[:, 1]
        self.z_grad_rth = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_grad_rth_1 = self.z_rth_pr[:, 0]
        self.z_grad_rth_2 = self.z_rth_pr[:, 1]
        self.z_grad_rth_3 = self.z_rth_pr[:, 2]
        self.z_grad_rth_4 = self.z_rth_pr[:, 3]
        self.z_dexner_dz_c_2 = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_gradh_exner = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_alpha = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_beta = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_w_expl = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_exner_expl = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_q = _allocate(CellDim, KDim, mesh=self.grid)
        self.exner_dyn_incr = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_contr_w_fl_l = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_rho_e = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_theta_v_e = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_hydro_corr = _allocate(EdgeDim, mesh=self.grid)
        self.z_vn_avg = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_graddiv_vn = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_theta_v_fl_e = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_flxdiv_mass = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_flxdiv_theta = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_rho_expl = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_dwdz_dd = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_rho_v = _allocate(VertexDim, KDim, mesh=self.grid)
        self.z_theta_v_v = _allocate(VertexDim, KDim, mesh=self.grid)
        self.p_grad = _allocate(CellDim, KDim, mesh=self.grid)
        self.p_ccpr = _allocate(CellDim, KDim, mesh=self.grid)
        self.ikoffset = _allocate(EdgeDim, E2CDim, KDim, mesh=self.grid)
        self.z_graddiv2_vn = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.mass_flx_ic = _allocate(CellDim, KDim, mesh=self.grid)

    # def initial_step(self):

    def time_step(
        self,
        diagnostic_state: DiagnosticState,
        diagnostic_state_nonhydro: DiagnosticStateNonHydro,
        prognostic_state: list[PrognosticState],
        prep_adv: PrepAdvection,
        config: NonHydrostaticConfig,
        z_fields: ZFields,
        inv_dual_edge_length: Field[[EdgeDim], float],
        primal_normal_cell_1: Field[[ECDim], float],
        dual_normal_cell_1: Field[[ECDim], float],
        primal_normal_cell_2: Field[[ECDim], float],
        dual_normal_cell_2: Field[[ECDim], float],
        inv_primal_edge_length: Field[[EdgeDim], float],
        tangent_orientation: Field[[EdgeDim], float],
        cfl_w_limit: float,
        scalfac_exdiff: float,
        cell_areas: Field[[CellDim], float],
        owner_mask: Field[[CellDim], bool],
        f_e: Field[[EdgeDim], float],
        area_edge: Field[[EdgeDim], float],
        dtime: float,
        idyn_timestep: float,
        l_recompute: bool,
        l_init: bool,
        nnow: int,
        nnew: int,
        lprep_adv: bool,
        lclean_mflx: bool,
    ):
        """
        Do one diffusion step within regular time loop.
        runs a diffusion step for the parameter linit=False, within regular time loop.
        """

        (
            edge_startindex_nudging,
            edge_endindex_nudging,
            edge_startindex_interior,
            edge_endindex_interior,
            edge_startindex_local,
            edge_endindex_local,
            cell_startindex_nudging,
            cell_endindex_nudging,
            cell_startindex_interior,
            cell_endindex_interior,
            cell_startindex_local,
            cell_endindex_local,
            vertex_startindex_interior,
            vertex_endindex_interior,
        ) = self.init_dimensions_boundaries()
        # velocity_advection is referenced inside

        # Inverse value of ndyn_substeps for tracer advection precomputations
        r_nsubsteps = 1.0 / config.ndyn_substeps_var

        #  Precompute Rayleigh damping factor
        z_raylfac = 1.0 / (
            1.0 + dtime * self.metric_state.rayleigh_w
        )  # TODO: @nfarabullini make this a program

        # Coefficient for reduced fourth-order divergence damping along nest boundaries
        bdy_divdamp = 0.75 / (nudge_max_coeff + dbl_eps) * abs(self.scal_divdamp)

        # scaling factor for second-order divergence damping: divdamp_fac_o2*delta_x**2
        # delta_x**2 is approximated by the mean cell area
        scal_divdamp_o2 = (
            config.divdamp_fac_o2 * p_patch % geometry_info % mean_cell_area
        )

        if self.p_test_run:
            nhsolve_prog.init_test_fields(
                self.z_rho_e,
                self.z_theta_v_e,
                self.z_dwdz_dd,
                self.z_graddiv_vn,
                edge_endindex_local,
                cell_endindex_local,
                self.grid.n_lev(),
                offset_provider={},
            )

        #  Set time levels of ddt_adv fields for call to velocity_tendencies
        if self.itime_scheme >= 4:
            self.ntl1 = nnow
            self.ntl2 = nnew
        else:
            self.ntl1 = 1
            self.ntl2 = 1

        self.wgt_nnow_vel = 0.5 - config.veladv_offctr  # TODO: add to config
        self.wgt_nnew_vel = 0.5 + config.veladv_offctr

        self.wgt_nnew_rth = 0.5 + config.rhotheta_offctr  # TODO: add to config
        self.wgt_nnow_rth = 1.0 - self.wgt_nnew_rth

        self.run_predictor_step(
            diagnostic_state,
            diagnostic_state_nonhydro,
            prognostic_state,
            config,
            z_fields,
            inv_dual_edge_length,
            primal_normal_cell_1,
            dual_normal_cell_1,
            primal_normal_cell_2,
            dual_normal_cell_2,
            inv_primal_edge_length,
            tangent_orientation,
            cfl_w_limit,
            scalfac_exdiff,
            cell_areas,
            owner_mask,
            f_e,
            area_edge,
            dtime,
            idyn_timestep,
            l_recompute,
            l_init,
            nnow,
            nnew,
        )

        self.run_corrector_step(
            diagnostic_state,
            diagnostic_state_nonhydro,
            prognostic_state,
            config,
            z_fields,
            inv_dual_edge_length,
            inv_primal_edge_length,
            tangent_orientation,
            prep_adv,
            dtime,
            nnew,
            nnow,
            cfl_w_limit,
            scalfac_exdiff,
            cell_areas,
            owner_mask,
            f_e,
            area_edge,
            lprep_adv,
            lclean_mflx,
        )

        if self.grid.limited_area():
            nhsolve_prog.stencils_66_67(
                self.metric_state_nonhydro.bdy_halo_c,  # TODO: @abishekg7 check if this should be mask_prog_halo_c_dsl_low_refin
                prognostic_state[nnew].rho,
                prognostic_state[nnew].theta_v,
                prognostic_state[nnew].exner,
                self.rd_o_cvd,
                self.rd_o_p0ref,
                cell_startindex_interior - 1,
                cell_endindex_local,
                cell_endindex_nudging,
                self.grid.n_lev(),
                offset_provider={},
            )

        mo_solve_nonhydro_stencil_68(
            self.metric_state_nonhydro.mask_prog_halo_c,
            prognostic_state[nnow].rho,
            prognostic_state[nnow].theta_v,
            prognostic_state[nnew].exner,
            prognostic_state[nnow].exner,
            prognostic_state[nnew].rho,
            prognostic_state[nnew].theta_v,
            constants.CVD_O_RD,
            horizontal_start=cell_startindex_interior - 1,
            horizontal_end=cell_endindex_local,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )

    def run_predictor_step(
        self,
        diagnostic_state: DiagnosticState,
        diagnostic_state_nonhydro: DiagnosticStateNonHydro,
        prognostic_state: list[PrognosticState],
        config: NonHydrostaticConfig,
        z_fields: ZFields,
        inv_dual_edge_length: Field[[EdgeDim], float],
        primal_normal_cell_1: Field[[ECDim], float],
        dual_normal_cell_1: Field[[ECDim], float],
        primal_normal_cell_2: Field[[ECDim], float],
        dual_normal_cell_2: Field[[ECDim], float],
        inv_primal_edge_length: Field[[EdgeDim], float],
        tangent_orientation: Field[[EdgeDim], float],
        cfl_w_limit: float,
        scalfac_exdiff: float,
        cell_areas: Field[[CellDim], float],
        owner_mask: Field[[CellDim], bool],
        f_e: Field[[EdgeDim], float],
        area_edge: Field[[EdgeDim], float],
        dtime: float,
        idyn_timestep: float,
        l_recompute: bool,
        l_init: bool,
        nnow: int,
        nnew: int,
    ):
        if config.itime_scheme >= 6 or l_init or l_recompute:
            if config.itime_scheme < 6 and not l_init:
                lvn_only = True  # Recompute only vn tendency
            else:
                lvn_only = False
            velocity_advection.VelocityAdvection.run_predictor_step(
                lvn_only,
                DiagnosticState,
                PrognosticState,
                ZFields,
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

        p_dthalf = 0.5 * dtime

        (
            edge_startindex_nudging,
            edge_endindex_nudging,
            edge_startindex_interior,
            edge_endindex_interior,
            edge_startindex_local,
            edge_endindex_local,
            cell_startindex_nudging,
            cell_endindex_nudging,
            cell_startindex_interior,
            cell_endindex_interior,
            cell_startindex_local,
            cell_endindex_local,
            vertex_startindex_interior,
            vertex_endindex_interior,
        ) = self.init_dimensions_boundaries()

        # initialize nest boundary points of z_rth_pr with zero
        if self.grid.limited_area():
            mo_solve_nonhydro_stencil_01(
                self.z_rth_pr_1,
                self.z_rth_pr_2,
                horizontal_start=0,
                horizontal_end=cell_endindex_local,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        nhsolve_prog.predictor_stencils_2_3(
            self.metric_state_nonhydro.exner_exfac,
            prognostic_state[nnow].exner,
            self.metric_state_nonhydro.exner_ref_mc,
            diagnostic_state_nonhydro.exner_pr,
            self.z_exner_ex_pr,
            cell_endindex_interior - 1,
            self.grid.n_lev(),
            offset_provider={},
        )

        if config.igradp_method <= 3:
            nhsolve_prog.predictor_stencils_4_5_6(
                self.metric_state_nonhydro.wgtfacq_c,
                self.z_exner_ex_pr,
                self.z_exner_ic,
                self.metric_state.wgtfac_c,
                self.metric_state_nonhydro.inv_ddqz_z_full,
                self.z_dexner_dz_c_1,
                cell_endindex_interior - 1,
                self.vertical_params.nflatlev,
                self.grid.n_lev(),
                self.grid.n_lev() + 1,
                offset_provider={"Koff": KDim},
            )

            if self.vertical_params.nflatlev == 1:
                # Perturbation Exner pressure on top half level
                raise NotImplementedError("nflatlev=1 not implemented")

        nhsolve_prog.predictor_stencils_7_8_9(
            prognostic_state[nnow].rho,
            self.metric_state_nonhydro.rho_ref_mc,
            prognostic_state[nnow].theta_v,
            self.metric_state.theta_ref_mc,
            diagnostic_state_nonhydro.rho_ic,
            self.z_rth_pr_1,
            self.z_rth_pr_2,
            self.metric_state.wgtfac_c,
            self.metric_state_nonhydro.vwind_expl_wgt,
            diagnostic_state_nonhydro.exner_pr,
            self.metric_state_nonhydro.d_exner_dz_ref_ic,
            self.metric_state.ddqz_z_half,
            self.z_theta_v_pr_ic,
            diagnostic_state_nonhydro.theta_v_ic,
            self.z_th_ddz_exner_c,
            cell_endindex_interior - 1,
            self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        if config.l_open_ubc and not self.l_vert_nested:
            raise NotImplementedError(
                "Nesting support not implemented. " "l_open_ubc not implemented"
            )

        # Perturbation theta at top and surface levels
        nhsolve_prog.predictor_stencils_11_lower_upper(
            self.metric_state_nonhydro.wgtfacq_c,
            self.z_rth_pr,
            self.metric_state_nonhydro.theta_ref_ic,
            self.z_theta_v_pr_ic,
            diagnostic_state_nonhydro.theta_v_ic,
            cell_endindex_interior - 1,
            self.grid.n_lev() + 1,
            offset_provider={"Koff": KDim},
        )

        if config.igradp_method <= 3:
            # Second vertical derivative of perturbation Exner pressure (hydrostatic approximation)
            mo_solve_nonhydro_stencil_12(
                self.z_theta_v_pr_ic,
                self.metric_state_nonhydro.d2dexdz2_fac1_mc,
                self.metric_state_nonhydro.d2dexdz2_fac2_mc,
                self.z_rth_pr_2,
                self.z_dexner_dz_c_2,
                horizontal_start=2,
                horizontal_end=cell_endindex_interior - 1,
                vertical_start=self.grid.nflat_gradp(),
                vertical_end=self.grid.n_lev(),
                offset_provider={"Koff": KDim},
            )

        # Add computation of z_grad_rth (perturbation density and virtual potential temperature at main levels)
        # at outer halo points: needed for correct calculation of the upwind gradients for Miura scheme
        mo_solve_nonhydro_stencil_13(
            prognostic_state[nnow].rho,
            self.metric_state_nonhydro.rho_ref_mc,
            prognostic_state[nnow].theta_v,
            self.metric_state.theta_ref_mc,
            self.z_rth_pr_1,
            self.z_rth_pr_2,
            horizontal_start=cell_startindex_interior - 2,
            horizontal_end=cell_endindex_interior - 2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )

        # Compute rho and theta at edges for horizontal flux divergence term
        if config.iadv_rhotheta == 1:
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                prognostic_state[nnow].rho,
                self.interpolation_state.c_intp,
                self.z_rho_v,
                vertex_endindex_interior - 1,
                self.grid.n_lev(),
                horizontal_start=1,
                horizontal_end=vertex_endindex_interior - 1,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),  # UBOUND(p_cell_in,2)
                offset_provider={
                    "V2C": self.grid.get_v2c_connectivity(),
                    "V2CDim": V2CDim,
                },
            )
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                prognostic_state[nnow].theta_v,
                self.interpolation_state.c_intp,
                self.z_theta_v_v,
                vertex_endindex_interior - 1,
                self.grid.n_lev(),
                horizontal_start=1,
                horizontal_end=vertex_endindex_interior - 1,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),  # UBOUND(p_cell_in,2)
                offset_provider={
                    "V2C": self.grid.get_v2c_connectivity(),
                    "V2CDim": V2CDim,
                },
            )
        elif config.iadv_rhotheta == 2:
            # Compute Green-Gauss gradients for rho and theta
            mo_math_gradients_grad_green_gauss_cell_dsl(
                self.p_grad[:, 1],
                self.p_grad[:, 2],
                self.p_grad[:, 3],
                self.p_grad[:, 4],
                self.p_ccpr[:, 1],
                self.p_ccpr[:, 2],
                self.interpolation_state.geofac_grg_x,
                self.interpolation_state.geofac_grg_y,
                cell_endindex_local,
                self.grid.n_lev(),
                horizontal_start=2,
                horizontal_end=cell_endindex_interior - 1,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),  # UBOUND(p_ccpr,2)
                offset_provider={
                    "C2E2CO": self.grid.get_c2e2co_connectivity(),
                    "C2E2CODim": C2E2CODim,
                },
            )
        elif config.iadv_rhotheta == 3:
            # First call: compute backward trajectory with wind at time level nnow
            lcompute = True
            lcleanup = False
            # upwind_hflux_miura3()
            # Second call: compute only reconstructed value for flux divergence
            lcompute = False
            lcleanup = True
            # upwind_hflux_miura3()

        # TODO: @abishekg7 Please see test.f90 for this section. Above call to 'wrap_run_mo_solve_nonhydro_stencil_14'
        if config.iadv_rhotheta <= 2:
            if config.idiv_method == 1:
                pass
            else:
                pass
        # TODO: @abishekg7 this is just zero stencils two fields
        mo_solve_nonhydro_stencil_14(
            self.z_rho_e,
            self.z_theta_v_e,
            horizontal_start=edge_startindex_interior - 2,
            horizontal_end=edge_endindex_interior
            - 2,  # TODO: @abishekg7 conditional on idiv_method
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )
        # initialize also nest boundary points with zero
        # TODO: @abishekg7 this is just zero stencils two fields
        if self.grid.limited_area():
            mo_solve_nonhydro_stencil_15(
                self.z_rho_e,
                self.z_theta_v_e,
                horizontal_start=0,
                horizontal_end=edge_endindex_interior - 1,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        if config.iadv_rhotheta == 2:
            # Operations from upwind_hflux_miura are inlined in order to process both fields in one step
            pass
        else:
            # Compute upwind-biased values for rho and theta starting from centered differences
            # Note: the length of the backward trajectory should be 0.5*dtime*(vn,vt) in order to arrive
            # at a second-order accurate FV discretization, but twice the length is needed for numerical stability
            nhsolve_prog.mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1(
                prognostic_state[nnow].vn,
                diagnostic_state.vt,
                self.interpolation_state.pos_on_tplane_e_1,
                self.interpolation_state.pos_on_tplane_e_2,
                primal_normal_cell_1,
                dual_normal_cell_1,
                primal_normal_cell_2,
                dual_normal_cell_2,
                p_dthalf,
                self.metric_state_nonhydro.rho_ref_me,
                self.metric_state_nonhydro.theta_ref_me,
                self.z_grad_rth_1,
                self.z_grad_rth_2,
                self.z_grad_rth_3,
                self.z_grad_rth_4,
                self.z_rth_pr_1,
                self.z_rth_pr_2,
                edge_endindex_interior - 1,
                self.grid.n_lev(),
                offset_provider={
                    "E2C": self.grid.get_e2c_connectivity(),
                    "E2CDim": E2CDim,
                },
            )

        # Remaining computations at edge points
        mo_solve_nonhydro_stencil_18(
            inv_dual_edge_length,
            self.z_exner_ex_pr,
            self.z_gradh_exner,
            horizontal_start=edge_startindex_nudging + 1,
            horizontal_end=edge_endindex_interior,
            vertical_start=0,
            vertical_end=self.vertical_params.nflatlev - 1,
            offset_provider={"E2C": self.grid.get_e2c_connectivity(), "E2CDim": E2CDim},
        )

        if (
            config.igradp_method <= 3
        ):  # stencil_20 is tricky, there's no regular field operator
            # horizontal gradient of Exner pressure, including metric correction
            # horizontal gradient of Exner pressure, Taylor-expansion-based reconstruction
            nhsolve_prog.nhsolve_predictor_tendencies_19_20(
                inv_dual_edge_length,
                self.z_exner_ex_pr,
                self.metric_state.ddxn_z_full,
                self.interpolation_state.c_lin_e,
                self.z_dexner_dz_c_1,
                self.z_gradh_exner,
                edge_startindex_nudging + 1,
                edge_endindex_interior,
                self.vertical_params.nflatlev,
                self.grid.nflat_gradp(),
                offset_provider={},
            )

        elif config.igradp_method == 4 or config.igradp_method == 5:
            # horizontal gradient of Exner pressure, cubic/quadratic interpolation
            raise NotImplementedError("igradp_method 4 and 5 not implemented")

        # compute hydrostatically approximated correction term that replaces downward extrapolation
        if config.igradp_method == 3:
            mo_solve_nonhydro_stencil_21(
                prognostic_state[nnow].theta_v,
                self.ikoffset,
                self.metric_state_nonhydro.zdiff_gradp,
                diagnostic_state_nonhydro.theta_v_ic,
                self.metric_state_nonhydro.inv_ddqz_z_full,
                inv_dual_edge_length,
                self.grav_o_cpd,
                self.z_hydro_corr,
                horizontal_start=edge_startindex_nudging + 1,
                horizontal_end=edge_endindex_interior,
                vertical_start=self.grid.n_lev(),
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "E2C": self.grid.get_e2c_connectivity(),
                    "E2CDim": E2CDim,
                    "Koff": KDim,
                },
            )
        elif config.igradp_method == 5:
            raise NotImplementedError("igradp_method 4 and 5 not implemented")

        if config.igradp_method == 3 or config.igradp_method == 5:
            mo_solve_nonhydro_stencil_22(
                self.metric_state_nonhydro.ipeidx_dsl,
                self.metric_state_nonhydro.pg_exdist,
                self.z_hydro_corr,
                self.z_gradh_exner,
                horizontal_start=edge_startindex_nudging + 1,
                horizontal_end=edge_endindex_local,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        mo_solve_nonhydro_stencil_24(
            prognostic_state[nnow].vn,
            diagnostic_state_nonhydro.ddt_vn_adv_ntl1,  # TODO: @nfarabullini: is the self.ntl1 correct or this one?
            diagnostic_state_nonhydro.ddt_vn_phy,
            self.z_theta_v_e,
            self.z_gradh_exner,
            prognostic_state[nnew].vn,
            dtime,
            constants.CPD,
            horizontal_start=edge_startindex_nudging + 1,
            horizontal_end=edge_endindex_interior,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )

        if config.is_iau_active:
            mo_solve_nonhydro_stencil_28(
                diagnostic_state_nonhydro.vn_incr,
                prognostic_state[nnew].vn,
                config.iau_wgt_dyn,
                horizontal_start=edge_startindex_nudging + 1,
                horizontal_end=edge_endindex_interior,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        if self.grid.limited_area():
            mo_solve_nonhydro_stencil_29(
                diagnostic_state_nonhydro.grf_tend_vn,
                prognostic_state[nnow].vn,
                prognostic_state[nnew].vn,
                dtime,
                horizontal_start=0,
                horizontal_end=edge_endindex_nudging,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        ##### COMMUNICATION PHASE

        mo_solve_nonhydro_stencil_30(
            self.interpolation_state.e_flx_avg,
            prognostic_state[nnew].vn,
            self.interpolation_state.geofac_grdiv,
            self.interpolation_state.rbf_vec_coeff_e,
            self.z_vn_avg,
            self.z_graddiv_vn,
            diagnostic_state.vt,
            horizontal_start=4,
            horizontal_end=edge_endindex_interior - 2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "E2C2EO": self.grid.get_e2c2eo_connectivity(),
                "E2C2EODim": E2C2EODim,
                "E2C2E": self.grid.get_e2c2e_connectivity(),
                "E2C2EDim": E2C2EDim,
            },
        )

        #####  Not sure about  _mo_solve_nonhydro_stencil_31()

        if config.idiv_method == 1:
            mo_solve_nonhydro_stencil_32(
                self.z_rho_e,
                self.z_vn_avg,
                self.metric_state.ddqz_z_full_e,
                self.z_theta_v_e,
                diagnostic_state_nonhydro.mass_fl_e,
                self.z_theta_v_fl_e,
                horizontal_start=4,
                horizontal_end=edge_endindex_interior - 2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        nhsolve_prog.predictor_stencils_35_36(
            prognostic_state[nnew].vn,
            self.metric_state.ddxn_z_full,
            self.metric_state.ddxt_z_full,
            diagnostic_state.vt,
            z_fields.z_w_concorr_me,
            self.metric_state.wgtfac_e,
            diagnostic_state.vn_ie,
            z_fields.z_vt_ie,
            z_fields.z_kin_hor_e,
            edge_endindex_interior - 2,
            self.vertical_params.nflatlev,
            self.grid.n_lev(),
            offset_provider={},
        )

        if not self.l_vert_nested:
            nhsolve_prog.predictor_stencils_37_38(
                prognostic_state[nnew].vn,
                diagnostic_state.vt,
                diagnostic_state.vn_ie,
                z_fields.z_vt_ie,
                z_fields.z_kin_hor_e,
                self.metric_state.wgtfacq_e,
                edge_endindex_local - 2,
                self.grid.n_lev() + 1,
                offset_provider={"Koff": KDim},
            )

        nhsolve_prog.predictor_stencils_39_40(
            self.interpolation_state.e_bln_c_s,
            z_fields.z_w_concorr_me,
            self.metric_state.wgtfac_c,
            self.metric_state_nonhydro.wgtfacq_c,
            diagnostic_state.w_concorr_c,
            cell_endindex_local - 1,
            self.vertical_params.nflatlev + 1,
            self.grid.n_lev(),
            self.grid.n_lev() + 1,
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "C2EDim": C2EDim,
                "Koff": KDim,
            },
        )

        if config.idiv_method == 2:
            pass
            # if self.grid.limited_area():
            #     init_zero_contiguous_dp()

        if config.idiv_method == 2:
            pass
            # div_avg()

        if config.idiv_method == 1:
            mo_solve_nonhydro_stencil_41(
                self.interpolation_state.geofac_div,
                diagnostic_state_nonhydro.mass_fl_e,
                self.z_theta_v_fl_e,
                self.z_flxdiv_mass,
                self.z_flxdiv_theta,
                horizontal_start=cell_startindex_nudging + 1,
                horizontal_end=cell_endindex_interior,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "C2E": self.grid.get_c2e_connectivity(),
                    "C2EDim": C2EDim,
                },
            )

        # check for p_nh%prog(nnow)% fields and new
        nhsolve_prog.stencils_43_44_45_45b(
            self.z_w_expl,
            prognostic_state[nnow].w,
            diagnostic_state.ddt_w_adv_pc[self.ntl1],  # =ddt_w_adv_ntl1
            self.z_th_ddz_exner_c,
            self.z_contr_w_fl_l,
            diagnostic_state_nonhydro.rho_ic,
            diagnostic_state.w_concorr_c,
            self.metric_state_nonhydro.vwind_expl_wgt,
            self.z_beta,
            prognostic_state[nnow].exner,
            prognostic_state[nnow].rho,
            prognostic_state[nnow].theta_v,
            self.metric_state_nonhydro.inv_ddqz_z_full,
            self.z_alpha,
            self.metric_state_nonhydro.vwind_impl_wgt,
            diagnostic_state_nonhydro.theta_v_ic,
            self.z_q,
            constants.RD,
            constants.CVD,
            dtime,
            constants.CPD,
            cell_startindex_nudging + 1,
            cell_endindex_local,
            self.grid.n_lev(),
            self.grid.n_lev() + 1,
            offset_provider={},
        )

        if not (config.l_open_ubc and self.l_vert_nested):
            mo_solve_nonhydro_stencil_46(
                prognostic_state[nnew].w,
                self.z_contr_w_fl_l,
                cell_startindex_nudging + 1,
                cell_endindex_local,
                horizontal_start=cell_startindex_nudging + 1,
                horizontal_end=cell_endindex_interior,
                vertical_start=0,
                vertical_end=0,
                offset_provider={},
            )

        nhsolve_prog.stencils_47_48_49(
            prognostic_state[nnew].w,
            self.z_contr_w_fl_l,
            diagnostic_state.w_concorr_c,
            self.z_rho_expl,
            self.z_exner_expl,
            prognostic_state[nnow].rho,
            self.metric_state_nonhydro.inv_ddqz_z_full,
            self.z_flxdiv_mass,
            diagnostic_state_nonhydro.exner_pr,
            self.z_beta,
            self.z_flxdiv_theta,
            diagnostic_state_nonhydro.theta_v_ic,
            diagnostic_state_nonhydro.ddt_exner_phy,
            dtime,
            cell_startindex_nudging + 1,
            cell_endindex_interior,
            self.grid.n_lev(),
            self.grid.n_lev() + 1,
            offset_provider={},
        )

        if config.is_iau_active:
            mo_solve_nonhydro_stencil_50(
                self.z_rho_expl,
                self.z_exner_expl,
                diagnostic_state_nonhydro.rho_incr,
                diagnostic_state_nonhydro.exner_incr,
                config.iau_wgt_dyn,
                horizontal_start=cell_startindex_nudging + 1,
                horizontal_end=cell_endindex_interior,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        nhsolve_prog.stencils_52_53(
            self.metric_state_nonhydro.vwind_impl_wgt,
            diagnostic_state_nonhydro.theta_v_ic,
            self.metric_state.ddqz_z_half,
            self.z_alpha,
            self.z_beta,
            self.z_w_expl,
            self.z_exner_expl,
            self.z_q,
            prognostic_state[nnew].w,
            dtime,
            constants.CPD,
            cell_startindex_nudging + 1,
            cell_endindex_local,
            self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        if config.rayleigh_type == constants.RAYLEIGH_KLEMP:
            ## ACC w_1 -> p_nh%w
            mo_solve_nonhydro_stencil_54(
                z_raylfac,
                prognostic_state[nnew].w_1,
                prognostic_state[nnew].w,
                horizontal_start=cell_startindex_nudging + 1,
                horizontal_end=cell_endindex_interior,
                vertical_start=1,
                vertical_end=self.vertical_params.index_of_damping_layer,  # nrdmax
                offset_provider={},
            )

        mo_solve_nonhydro_stencil_55(
            self.z_rho_expl,
            self.metric_state_nonhydro.vwind_impl_wgt,
            self.metric_state_nonhydro.inv_ddqz_z_full,
            diagnostic_state_nonhydro.rho_ic,
            prognostic_state[nnew].w,
            self.z_exner_expl,
            self.metric_state_nonhydro.exner_ref_mc,
            self.z_alpha,
            self.z_beta,
            prognostic_state[nnow].rho,
            prognostic_state[nnow].theta_v,
            prognostic_state[nnow].exner,
            prognostic_state[nnew].rho,
            prognostic_state[nnew].exner,
            prognostic_state[nnew].theta_v,
            dtime,
            constants.CVD_O_RD,
            horizontal_start=cell_startindex_nudging + 1,
            horizontal_end=cell_endindex_local,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        # compute dw/dz for divergence damping term
        if config.lhdiff_rcf and config.divdamp_type >= 3:
            nhsolve_prog.mo_solve_nonhydro_stencil_56_63(
                self.metric_state_nonhydro.inv_ddqz_z_full,
                prognostic_state[nnew].w,
                diagnostic_state.w_concorr_c,
                self.z_dwdz_dd,
                cell_startindex_nudging + 1,
                cell_endindex_local,
                config.kstart_dd3d,  # TODO: @abishekg7
                self.grid.n_lev(),
                offset_provider={"Koff": KDim},
            )

        if idyn_timestep == 1:
            nhsolve_prog.predictor_stencils_59_60(
                prognostic_state[nnow].exner,
                prognostic_state[nnew].exner,
                self.exner_dyn_incr,
                diagnostic_state_nonhydro.ddt_exner_phy,
                config.ndyn_substeps_var,
                dtime,
                cell_startindex_nudging + 1,
                cell_endindex_interior,
                config.kstart_moist,  # TODO: @abishekg7
                self.grid.n_lev(),
                offset_provider={},
            )

        if self.grid.limited_area():  # for MPI-parallelized case
            nhsolve_prog.predictor_stencils_61_62(
                prognostic_state[nnow].rho,
                diagnostic_state_nonhydro.grf_tend_rho,
                prognostic_state[nnow].theta_v,
                diagnostic_state_nonhydro.grf_tend_thv,
                prognostic_state[nnow].w,
                diagnostic_state_nonhydro.grf_tend_w,
                prognostic_state[nnew].rho,
                prognostic_state[nnew].exner,
                prognostic_state[nnew].w,
                dtime,
                cell_endindex_nudging,
                self.grid.n_lev(),
                self.grid.n_lev() + 1,
                offset_provider={},
            )

        if config.lhdiff_rcf and config.divdamp_type >= 3:
            nhsolve_prog.mo_solve_nonhydro_stencil_56_63(
                self.metric_state_nonhydro.inv_ddqz_z_full,
                prognostic_state[nnew].w,
                diagnostic_state.w_concorr_c,
                self.z_dwdz_dd,
                cell_endindex_nudging,  # TODO: @abishekg7 double check domains
                config.kstart_dd3d,  # TODO: @abishekg7
                self.grid.n_lev(),
                offset_provider={"Koff": KDim},
            )

        ##### COMMUNICATION PHASE

    def run_corrector_step(
        self,
        diagnostic_state: DiagnosticState,
        diagnostic_state_nonhydro: DiagnosticStateNonHydro,
        prognostic_state: list[PrognosticState],
        config: NonHydrostaticConfig,
        z_fields: ZFields,
        inv_dual_edge_length: Field[[EdgeDim], float],
        inv_primal_edge_length: Field[[EdgeDim], float],
        tangent_orientation: Field[[EdgeDim], float],
        prep_adv: PrepAdvection,
        dtime: float,
        nnew: int,
        nnow: int,
        cfl_w_limit: float,
        scalfac_exdiff: float,
        cell_areas: Field[[CellDim], float],
        owner_mask: Field[[CellDim], bool],
        f_e: Field[[EdgeDim], float],
        area_edge: Field[[EdgeDim], float],
        lprep_adv: bool,
        lclean_mflx: bool,
    ):

        (
            edge_startindex_nudging,
            edge_endindex_nudging,
            edge_startindex_interior,
            edge_endindex_interior,
            edge_startindex_local,
            edge_endindex_local,
            cell_startindex_nudging,
            cell_endindex_nudging,
            cell_startindex_interior,
            cell_endindex_interior,
            cell_startindex_local,
            cell_endindex_local,
            vertex_startindex_interior,
            vertex_endindex_interior,
        ) = self.init_dimensions_boundaries()

        lvn_only = False
        velocity_advection.VelocityAdvection.run_corrector_step(
            lvn_only,
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
        nvar = nnew

        mo_solve_nonhydro_stencil_10(
            prognostic_state.w,
            diagnostic_state.w_concorr_c,
            self.metric_state.ddqz_z_half,
            prognostic_state[nnow].rho,
            prognostic_state[nvar].rho,
            prognostic_state[nnow].theta_v,
            prognostic_state[nvar].theta_v,
            self.metric_state.wgtfac_c,
            self.metric_state.theta_ref_mc,
            self.metric_state_nonhydro.vwind_expl_wgt,
            diagnostic_state_nonhydro.exner_pr,
            self.metric_state_nonhydro.d_exner_dz_ref_ic,
            diagnostic_state_nonhydro.rho_ic,
            self.z_theta_v_pr_ic,
            diagnostic_state_nonhydro.theta_v_ic,
            self.z_th_ddz_exner_c,
            dtime,
            wgt_nnow_rth,
            wgt_nnew_rth,
            horizontal_start=2,
            horizontal_end=cell_endindex_interior,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        if config.l_open_ubc and not self.l_vert_nested:
            raise NotImplementedError("l_open_ubc not implemented")

        mo_solve_nonhydro_stencil_17(
            self.metric_state_nonhydro.hmask_dd3d,
            self.metric_state_nonhydro.scalfac_dd3d,
            inv_dual_edge_length,
            z_dwdz_dd,
            self.z_graddiv_vn,
            horizontal_start=6,
            horizontal_end=edge_endindex_interior - 2,
            vertical_start=config.kstart_dd3d,  # TODO: @abishekg7 resolve
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
            },
        )

        if config.itime_scheme >= 4:
            mo_solve_nonhydro_stencil_23(
                prognostic_state[nnow].vn,
                diagnostic_state_nonhydro.ddt_vn_adv_ntl1,
                diagnostic_state_nonhydro.ddt_vn_adv_ntl2,
                diagnostic_state_nonhydro.ddt_vn_phy,
                self.z_theta_v_e,
                self.z_gradh_exner,
                prognostic_state[nnew].vn,
                dtime,
                self.wgt_nnow_vel,
                self.wgt_nnew_vel,
                constants.CPD,
                horizontal_start=edge_startindex_nudging + 1,
                horizontal_end=edge_endindex_interior,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        if config.lhdiff_rcf and (
            config.divdamp_order == 4 or config.divdamp_order == 24
        ):
            mo_solve_nonhydro_stencil_25(
                self.interpolation_state.geofac_grdiv,
                self.z_graddiv_vn,
                self.z_graddiv2_vn,
                horizontal_start=edge_startindex_nudging + 1,
                horizontal_end=edge_endindex_interior,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "E2C2EO": self.grid.get_e2c2eo_connectivity(),
                    "E2C2EODim": E2C2EODim,
                },
            )

        if config.lhdiff_rcf:
            if config.divdamp_order == 2 or (
                config.divdamp_order == 24 and scal_divdamp_o2 > 1.0e-6
            ):
                mo_solve_nonhydro_stencil_26(
                    self.z_graddiv_vn,
                    prognostic_state[nnew].vn,
                    scal_divdamp_o2,
                    horizontal_start=edge_startindex_nudging + 1,
                    horizontal_end=edge_endindex_interior,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev(),
                    offset_provider={},
                )
            if config.divdamp_order == 4 or (
                config.divdamp_order == 24 and config.divdamp_fac_o2 <= 4 * divdamp_fac
            ):
                if self.grid.limited_area():
                    mo_solve_nonhydro_stencil_27(
                        scal_divdamp,
                        bdy_divdamp,
                        self.interpolation_state.nudgecoeff_e,
                        self.z_graddiv2_vn,
                        prognostic_state[nnew].vn,
                        horizontal_start=edge_startindex_nudging + 1,
                        horizontal_end=edge_endindex_interior,
                        vertical_start=0,
                        vertical_end=self.grid.n_lev(),
                        offset_provider={},
                    )
                else:
                    mo_solve_nonhydro_4th_order_divdamp(
                        scal_divdamp,
                        self.z_graddiv2_vn,
                        prognostic_state[nnew].vn,
                        edge_startindex_nudging + 1,
                        edge_endindex_interior,
                        self.grid.n_lev(),
                        horizontal_start=edge_startindex_nudging + 1,
                        horizontal_end=edge_endindex_interior,
                        vertical_start=0,
                        vertical_end=self.grid.n_lev(),
                        offset_provider={},
                    )

        if config.is_iau_active:
            mo_solve_nonhydro_stencil_28(
                diagnostic_state_nonhydro.vn_incr,
                prognostic_state[nnew].vn,
                config.iau_wgt_dyn,
                horizontal_start=edge_startindex_nudging + 1,
                horizontal_end=edge_endindex_interior,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        ##### COMMUNICATION PHASE

        if config.idiv_method == 1:
            mo_solve_nonhydro_stencil_32(
                self.z_rho_e,
                self.z_vn_avg,
                self.metric_state.ddqz_z_full_e,
                self.z_theta_v_e,
                diagnostic_state_nonhydro.mass_fl_e,
                self.z_theta_v_fl_e,
                horizontal_start=4,
                horizontal_end=edge_endindex_local - 2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

            # TODO: @abishekg7 resolve prep_adv
            if lprep_adv:  # Preparations for tracer advection
                if lclean_mflx:
                    mo_solve_nonhydro_stencil_33(
                        prep_adv.vn_traj,
                        prep_adv.mass_flx_me,
                        horizontal_start=0,
                        horizontal_end=edge_endindex_local,
                        vertical_start=0,
                        vertical_end=self.grid.n_lev(),
                        offset_provider={},
                    )
                mo_solve_nonhydro_stencil_34(
                    self.z_vn_avg,
                    diagnostic_state_nonhydro.mass_fl_e,
                    prep_adv.vn_traj,
                    prep_adv.mass_flx_me,
                    r_nsubsteps,
                    horizontal_start=4,
                    horizontal_end=cell_endindex_interior - 2,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev(),
                    offset_provider={},
                )

        if config.itime_scheme >= 5:
            nhsolve_prog.corrector_stencils_35_39_40(  # TODO: @abishekg7 bounds are complicated
                prognostic_state[nnew].vn,
                self.metric_state.ddxn_z_full,
                self.metric_state.ddxt_z_full,
                diagnostic_state.vt,
                z_fields.z_w_concorr_me,
                self.interpolation_state.e_bln_c_s,
                self.metric_state.wgtfac_c,
                self.metric_state_nonhydro.wgtfacq_c,
                diagnostic_state.w_concorr_c,
                edge_endindex_local - 2,
                cell_endindex_local - 1,
                self.vertical_params.nflatlev + 1,
                self.grid.n_lev(),
                self.grid.n_lev() + 1,
                self.vertical_params.nflatlev,
                offset_provider={
                    "C2E": self.grid.get_c2e_connectivity(),
                    "C2EDim": C2EDim,
                    "Koff": KDim,
                },
            )

        if config.idiv_method == 2:
            pass
            # if self.grid.limited_area():
            #     init_zero_contiguous_dp()

        if config.idiv_method == 2:
            pass
            # div_avg()

        if config.idiv_method == 1:
            mo_solve_nonhydro_stencil_41(
                self.interpolation_state.geofac_div,
                diagnostic_state_nonhydro.mass_fl_e,
                self.z_theta_v_fl_e,
                self.z_flxdiv_mass,
                self.z_flxdiv_theta,
                horizontal_start=cell_startindex_nudging + 1,
                horizontal_end=cell_endindex_interior,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "C2E": self.grid.get_c2e_connectivity(),
                    "C2EDim": C2EDim,
                },
            )

        # TODO: @abishekg7 45_b is just a zero stencil
        if config.itime_scheme >= 4:
            nhsolve_prog.stencils_42_44_45_45b(
                self.z_w_expl,
                prognostic_state[nnow].w,
                diagnostic_state.ddt_w_adv_pc[self.ntl1],  # =ddt_w_adv_ntl1
                diagnostic_state.ddt_w_adv_pc[self.ntl2],  # =ddt_w_adv_ntl2
                self.z_th_ddz_exner_c,
                self.z_contr_w_fl_l,
                diagnostic_state_nonhydro.rho_ic,
                diagnostic_state.w_concorr_c,
                self.metric_state_nonhydro.vwind_expl_wgt,
                self.z_beta,
                prognostic_state[nnow].exner,
                prognostic_state[nnow].rho,
                prognostic_state[nnow].theta_v,
                self.metric_state_nonhydro.inv_ddqz_z_full,
                self.z_alpha,
                self.metric_state_nonhydro.vwind_impl_wgt,
                diagnostic_state_nonhydro.theta_v_ic,
                self.z_q,
                constants.RD,
                constants.CVD,
                dtime,
                constants.CPD,
                self.wgt_nnow_vel,
                self.wgt_nnew_vel,
                cell_startindex_nudging + 1,
                cell_endindex_interior,
                self.grid.n_lev(),
                self.grid.n_lev() + 1,
                offset_provider={},
            )
        else:
            nhsolve_prog.stencils_43_44_45_45b(
                self.z_w_expl,
                prognostic_state[nnow].w,
                diagnostic_state.ddt_w_adv_pc[self.ntl1],  # =ddt_w_adv_ntl1
                self.z_th_ddz_exner_c,
                self.z_contr_w_fl_l,
                diagnostic_state_nonhydro.rho_ic,
                diagnostic_state.w_concorr_c,
                self.metric_state_nonhydro.vwind_expl_wgt,
                self.z_beta,
                prognostic_state[nnow].exner,
                prognostic_state[nnow].rho,
                prognostic_state[nnow].theta_v,
                self.metric_state_nonhydro.inv_ddqz_z_full,
                self.z_alpha,
                self.metric_state_nonhydro.vwind_impl_wgt,
                diagnostic_state_nonhydro.theta_v_ic,
                self.z_q,
                constants.RD,
                constants.CVD,
                dtime,
                constants.CPD,
                cell_startindex_nudging + 1,
                cell_endindex_interior,
                self.grid.n_lev(),
                self.grid.n_lev() + 1,
                offset_provider={},
            )

        if not config.l_open_ubc and not self.l_vert_nested:
            mo_solve_nonhydro_stencil_46(
                prognostic_state[nnew].w,
                self.z_contr_w_fl_l,
                horizontal_start=cell_startindex_nudging + 1,
                horizontal_end=cell_endindex_interior,
                vertical_start=0,
                vertical_end=0,
                offset_provider={},
            )

        nhsolve_prog.stencils_47_48_49(
            prognostic_state[nnew].w,
            self.z_contr_w_fl_l,
            diagnostic_state.w_concorr_c,
            self.z_rho_expl,
            self.z_exner_expl,
            prognostic_state[nnow].rho,
            self.metric_state_nonhydro.inv_ddqz_z_full,
            self.z_flxdiv_mass,
            diagnostic_state_nonhydro.exner_pr,
            self.z_beta,
            self.z_flxdiv_theta,
            diagnostic_state_nonhydro.theta_v_ic,
            diagnostic_state_nonhydro.ddt_exner_phy,
            dtime,
            cell_startindex_nudging + 1,
            cell_endindex_interior,
            self.grid.n_lev(),
            self.grid.n_lev() + 1,
            offset_provider={"Koff": KDim},
        )

        if config.is_iau_active:
            mo_solve_nonhydro_stencil_50(
                self.z_rho_expl,
                self.z_exner_expl,
                diagnostic_state_nonhydro.rho_incr,
                diagnostic_state_nonhydro.exner_incr,
                config.iau_wgt_dyn,
                horizontal_start=cell_startindex_nudging + 1,
                horizontal_end=cell_endindex_interior,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        nhsolve_prog.stencils_52_53(
            self.metric_state_nonhydro.vwind_impl_wgt,
            diagnostic_state_nonhydro.theta_v_ic,
            self.metric_state.ddqz_z_half,
            self.z_alpha,
            self.z_beta,
            self.z_w_expl,
            self.z_exner_expl,
            self.z_q,
            prognostic_state[nnew].w,
            dtime,
            constants.CPD,
            cell_startindex_nudging,
            cell_endindex_interior,
            self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        if config.rayleigh_type == constants.RAYLEIGH_KLEMP:
            ## ACC w_1 -> p_nh%w
            mo_solve_nonhydro_stencil_54(
                z_raylfac,
                prognostic_state[nnew].w_1,
                prognostic_state[nnew].w,
                cell_startindex_nudging + 1,
                cell_endindex_interior,
                horizontal_start=cell_startindex_nudging + 1,
                horizontal_end=cell_endindex_interior,
                vertical_start=1,
                vertical_end=self.vertical_params.index_of_damping_layer,  # nrdmax
                offset_provider={},
            )

        mo_solve_nonhydro_stencil_55(
            self.z_rho_expl,
            self.metric_state_nonhydro.vwind_impl_wgt,
            self.metric_state_nonhydro.inv_ddqz_z_full,
            diagnostic_state_nonhydro.rho_ic,
            prognostic_state[nnew].w,
            self.z_exner_expl,
            self.metric_state_nonhydro.exner_ref_mc,
            self.z_alpha,
            self.z_beta,
            prognostic_state[nnow].rho,
            prognostic_state[nnow].theta_v,
            prognostic_state[nnow].exner,
            prognostic_state[nnew].rho,
            prognostic_state[nnew].exner,
            prognostic_state[nnew].theta_v,
            dtime,
            constants.CVD_O_RD,
            horizontal_start=cell_startindex_nudging + 1,
            horizontal_end=cell_endindex_interior,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        if lprep_adv:
            if lclean_mflx:
                # TODO: @abishekg7 domains for this
                set_zero_c_k(
                    field=self.mass_flx_ic,
                    horizontal_start=cell_startindex_nudging + 1,
                    horizontal_end=cell_endindex_interior,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev(),
                    offset_provider={},
                )

        mo_solve_nonhydro_stencil_58(
            self.z_contr_w_fl_l,
            diagnostic_state_nonhydro.rho_ic,
            self.metric_state_nonhydro.vwind_impl_wgt,
            prognostic_state[nnew].w,
            self.mass_flx_ic,
            r_nsubsteps,
            horizontal_start=cell_startindex_nudging + 1,
            horizontal_end=cell_endindex_interior,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )

        if lprep_adv:
            if lclean_mflx:
                set_zero_c_k(
                    field=self.mass_flx_ic,
                    horizontal_start=0,
                    horizontal_end=cell_endindex_nudging,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev() + 1,
                    offset_provider={},
                )

            mo_solve_nonhydro_stencil_65(
                diagnostic_state_nonhydro.rho_ic,
                self.metric_state_nonhydro.vwind_expl_wgt,
                self.metric_state_nonhydro.vwind_impl_wgt,
                prognostic_state[nnow].w,
                prognostic_state[nnew].w,
                diagnostic_state_nonhydro.w_concorr_c,
                self.mass_flx_ic,
                r_nsubsteps,
                horizontal_start=0,
                horizontal_end=cell_endindex_nudging,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        ##### COMMUNICATION PHASE

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

        (
            cell_startindex_interior,
            cell_endindex_interior,
        ) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.interior(CellDim),
            HorizontalMarkerIndex.interior(CellDim),
        )

        (cell_startindex_local, cell_endindex_local,) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.local(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        (
            vertex_startindex_interior,
            vertex_endindex_interior,
        ) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.interior(VertexDim),
            HorizontalMarkerIndex.interior(VertexDim),
        )

        return (
            edge_startindex_nudging,
            edge_endindex_nudging,
            edge_startindex_interior,
            edge_endindex_interior,
            edge_startindex_local,
            edge_endindex_local,
            cell_startindex_nudging,
            cell_endindex_nudging,
            cell_startindex_interior,
            cell_endindex_interior,
            cell_startindex_local,
            cell_endindex_local,
            vertex_startindex_interior,
            vertex_endindex_interior,
        )
