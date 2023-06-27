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
from typing import Final, Optional

import numpy as np
from gt4py.next.common import Field
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.program_processors.runners.gtfn_cpu import run_gtfn

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
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_17 import (
    mo_solve_nonhydro_stencil_17,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_18 import (
    mo_solve_nonhydro_stencil_18,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_19 import (
    mo_solve_nonhydro_stencil_19,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_20 import (
    mo_solve_nonhydro_stencil_20,
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
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_31 import (
    mo_solve_nonhydro_stencil_31,
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
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_52 import (
    mo_solve_nonhydro_stencil_52,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_53 import (
    mo_solve_nonhydro_stencil_53,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_54 import (
    mo_solve_nonhydro_stencil_54,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_55 import (
    mo_solve_nonhydro_stencil_55,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_56_63 import (
    mo_solve_nonhydro_stencil_56_63,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_58 import (
    mo_solve_nonhydro_stencil_58,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_65 import (
    mo_solve_nonhydro_stencil_65,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_66 import (
    mo_solve_nonhydro_stencil_66,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_67 import (
    mo_solve_nonhydro_stencil_67,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_68 import (
    mo_solve_nonhydro_stencil_68,
)
from icon4py.common.dimension import CellDim, ECDim, EdgeDim, KDim, VertexDim
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
from icon4py.state_utils.utils import (
    _allocate,
    _allocate_indices,
    _calculate_bdy_divdamp,
    _en_smag_fac_for_zero_nshift,
    compute_z_raylfac,
    scal_divdamp_calcs,
    set_zero_c_k,
    set_zero_e_k,
)
from icon4py.velocity import velocity_advection


class NonHydrostaticConfig:
    """
    Contains necessary parameter to configure a nonhydro run.

    Encapsulates namelist parameters and derived parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    """

    def __init__(
        self,
        itime_scheme: int = 4,
        iadv_rhotheta: int = 2,
        igradp_method: int = 3,
        ndyn_substeps_var: float = 5.0,
        rayleigh_type: int = 2,
        divdamp_order: int = 24,
        idiv_method: int = 1,
        is_iau_active: bool = False,
        iau_wgt_dyn: float = 1.0,
        divdamp_type: int = 3,
        lhdiff_rcf: bool = True,
        l_vert_nested: bool = False,
        l_open_ubc: bool = False,
        rhotheta_offctr: float = -0.1,
        veladv_offctr: float = 0.25,
        divdamp_fac: float = 0.0025,
        divdamp_fac_o2: float = 2.0,
        max_nudging_coeff: float = 0.075,
        divdamp_fac2: float = 0.004,
        divdamp_fac3: float = 0.004,
        divdamp_fac4: float = 0.004,
        divdamp_z: float = 32500,
        divdamp_z2: float = 40000,
        divdamp_z3: float = 60000,
        divdamp_z4: float = 80000,
    ):

        # parameters from namelist diffusion_nml
        self.itime_scheme: int = itime_scheme
        self.iadv_rhotheta: int = iadv_rhotheta

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
        self.divdamp_fac: float = divdamp_fac
        self.divdamp_fac_o2: float = divdamp_fac_o2
        self.nudge_max_coeff: float = max_nudging_coeff

        self.divdamp_fac2: float = divdamp_fac2
        self.divdamp_fac3: float = divdamp_fac3
        self.divdamp_fac4: float = divdamp_fac4
        self.divdamp_z: float = divdamp_z
        self.divdamp_z2: float = divdamp_z2
        self.divdamp_z3: float = divdamp_z3
        self.divdamp_z4: float = divdamp_z4

        self._validate()

    def _validate(self):
        """Apply consistency checks and validation on configuration parameters."""
        if self.l_open_ubc:
            raise NotImplementedError(
                "Open upper boundary conditions not supported"
                "(to allow vertical motions related to diabatic heating to extend beyond the model top)"
            )

        if self.l_vert_nested:
            raise NotImplementedError("Vertical nesting support not implemented")

        if self.igradp_method != 3:
            raise NotImplementedError("igradp_method can only be 3")

        if self.itime_scheme != 4:
            raise NotImplementedError("itime_scheme can only be 4")

        if self.idiv_method != 1:
            raise NotImplementedError("idiv_method can only be 1")

        if self.divdamp_order != 24:
            raise NotImplementedError("divdamp_order can only be 24")

        if self.divdamp_type == 32:
            raise NotImplementedError("divdamp_type with value 32 not yet implemented")


class NonHydrostaticParams:
    """Calculates derived quantities depending on the NonHydrostaticConfig."""

    def __init__(self, config: NonHydrostaticConfig):
        self.rd_o_cvd: Final[float] = constants.RD / constants.CPD
        self.rd_o_p0ref: Final[float] = constants.RD / constants.P0REF
        self.grav_o_cpd: Final[float] = constants.GRAV / constants.CPD

        # start level for 3D divergence damping terms
        self.kstart_dd3d: int = 1

        # start level for moist physics processes (specified by htop_moist_proc)
        self.kstart_moist: int = 1
        if self.kstart_moist != 1:
            raise NotImplementedError("kstart_moist can only be 1")

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
    def __init__(self):
        self._initialized = False
        self.grid: Optional[IconGrid] = None
        self.interpolation_state = None
        self.metric_state = None
        self.vertical_params: Optional[VerticalModelParams] = None

        self.config: Optional[NonHydrostaticConfig] = None
        self.params: Optional[NonHydrostaticParams] = None
        self.metric_state_nonhydro = None

        self.l_vert_nested = None
        self.enh_divdamp_fac = None
        self.scal_divdamp = None
        self.p_test_run = None

    def init(
        self,
        grid: IconGrid,
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        metric_state: MetricState,
        metric_state_nonhydro: MetricStateNonHydro,
        interpolation_state: InterpolationState,
        vertical_params: VerticalModelParams,
        a_vec: Field[[KDim], float],
        enh_smag_fac: Field[[KDim], float],
        cell_areas: Field[[CellDim], float],
        fac: tuple,
        z: tuple,
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

        out = enh_smag_fac
        _en_smag_fac_for_zero_nshift(
            a_vec, *fac, *z, out=enh_smag_fac, offset_provider={"Koff": KDim}
        )
        self.enh_divdamp_fac = enh_smag_fac

        cell_areas_avg = np.sum(cell_areas) / float(self.grid.num_cells())

        scal_divdamp_calcs(
            enh_smag_fac,
            cell_areas_avg,
            out=out,
            offset_provider={},
        )
        self.scal_divdamp = out
        self.p_test_run = True

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
        self.z_rth_pr_1 = self.z_rth_pr
        self.z_rth_pr_2 = self.z_rth_pr
        self.z_grad_rth = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_grad_rth_1 = self.z_grad_rth
        self.z_grad_rth_2 = self.z_grad_rth
        self.z_grad_rth_3 = self.z_grad_rth
        self.z_grad_rth_4 = self.z_grad_rth
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
        self.z_hydro_corr = _allocate(EdgeDim, KDim, mesh=self.grid)
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
        self.p_grad_1 = self.p_grad
        self.p_grad_2 = self.p_grad
        self.p_grad_3 = self.p_grad
        self.p_grad_4 = self.p_grad
        self.p_ccpr_1 = self.p_ccpr
        self.p_ccpr_2 = self.p_ccpr
        self.ikoffset = np_as_located_field(ECDim, KDim)(
            np.zeros((100000, 65), dtype=np.int32)
        )  # TODO: this should not be np_as_located_field, rather a separate function
        self.z_graddiv2_vn = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.mass_flx_ic = _allocate(CellDim, KDim, mesh=self.grid)
        self.k_field = _allocate_indices(KDim, mesh=self.grid)
        self.z_w_concorr_me = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_kin_hor_e = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_vt_ie = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_raylfac = _allocate(KDim, mesh=self.grid)

    def time_step(
        self,
        diagnostic_state: DiagnosticState,
        diagnostic_state_nonhydro: DiagnosticStateNonHydro,
        prognostic_state: list[PrognosticState],
        prep_adv: PrepAdvection,
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        inv_dual_edge_length: Field[[EdgeDim], float],
        primal_normal_cell: tuple[Field[[ECDim], float], Field[[ECDim], float]],
        dual_normal_cell: tuple[Field[[ECDim], float], Field[[ECDim], float]],
        inv_primal_edge_length: Field[[EdgeDim], float],
        tangent_orientation: Field[[EdgeDim], float],
        cfl_w_limit: float,
        scalfac_exdiff: float,
        cell_areas: Field[[CellDim], float],
        owner_mask: Field[[CellDim], bool],
        f_e: Field[[EdgeDim], float],
        area_edge: Field[[EdgeDim], float],
        bdy_divdamp: Field[[KDim], float],
        dtime: float,
        idyn_timestep: float,
        l_recompute: bool,
        l_init: bool,
        nnow: int,
        nnew: int,
        lclean_mflx: bool,
        lprep_adv: bool,
    ):
        primal_normal_cell_1 = primal_normal_cell[0]
        primal_normal_cell_2 = primal_normal_cell[1]
        dual_normal_cell_1 = dual_normal_cell[0]
        dual_normal_cell_2 = dual_normal_cell[1]
        # Inverse value of ndyn_substeps for tracer advection precomputations
        r_nsubsteps = 1.0 / config.ndyn_substeps_var

        # Coefficient for reduced fourth-order divergence damping along nest boundaries
        # TODO: this should be a program maybe(?)
        _calculate_bdy_divdamp(
            self.scal_divdamp,
            config.nudge_max_coeff,
            constants.dbl_eps,
            out=bdy_divdamp,
            offset_provider={},
        )

        # scaling factor for second-order divergence damping: divdamp_fac_o2*delta_x**2
        # delta_x**2 is approximated by the mean cell area
        cell_areas_avg = np.sum(cell_areas) / float(self.grid.num_cells())
        scal_divdamp_o2 = config.divdamp_fac_o2 * (cell_areas_avg)

        (indices_cells_1, indices_cells_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.local_boundary(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        (indices_edges_1, indices_edges_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.local_boundary(EdgeDim),
            HorizontalMarkerIndex.local(EdgeDim),
        )

        if self.p_test_run:
            nhsolve_prog.init_test_fields.with_backend(run_gtfn)(
                self.z_rho_e,
                self.z_theta_v_e,
                self.z_dwdz_dd,
                self.z_graddiv_vn,
                indices_edges_1,
                indices_edges_2,
                indices_cells_1,
                indices_cells_2,
                self.grid.n_lev(),
                offset_provider={},
            )

        self.wgt_nnow_vel = 0.5 - config.veladv_offctr
        self.wgt_nnew_vel = 0.5 + config.veladv_offctr

        self.wgt_nnew_rth = 0.5 + config.rhotheta_offctr
        self.wgt_nnow_rth = 1.0 - self.wgt_nnew_rth

        self.run_predictor_step(
            diagnostic_state,
            diagnostic_state_nonhydro,
            prognostic_state,
            config,
            params,
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
            params,
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
            lclean_mflx,
            scal_divdamp_o2,
            bdy_divdamp,
            lprep_adv,
        )

        (indices_0_1, indices_0_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.local(CellDim) - 1,
            HorizontalMarkerIndex.local(CellDim),
        )

        (indices_1_1, indices_1_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.local_boundary(CellDim),
            HorizontalMarkerIndex.nudging(CellDim) - 1,
        )

        if self.grid.limited_area():
            mo_solve_nonhydro_stencil_66.with_backend(run_gtfn)(
                self.metric_state_nonhydro.bdy_halo_c,
                prognostic_state[nnew].rho,
                prognostic_state[nnew].theta_v,
                prognostic_state[nnew].exner,
                params.rd_o_cvd,
                params.rd_o_p0ref,
                horizontal_start=indices_0_1,
                horizontal_end=indices_0_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

            mo_solve_nonhydro_stencil_67.with_backend(run_gtfn)(
                prognostic_state[nnew].rho,
                prognostic_state[nnew].theta_v,
                prognostic_state[nnew].exner,
                params.rd_o_cvd,
                params.rd_o_p0ref,
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        mo_solve_nonhydro_stencil_68.with_backend(run_gtfn)(
            self.metric_state_nonhydro.mask_prog_halo_c,
            prognostic_state[nnow].rho,
            prognostic_state[nnow].theta_v,
            prognostic_state[nnew].exner,
            prognostic_state[nnow].exner,
            prognostic_state[nnew].rho,
            prognostic_state[nnew].theta_v,
            constants.CVD_O_RD,
            horizontal_start=indices_0_1,
            horizontal_end=indices_0_2,
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
        params: NonHydrostaticParams,
        inv_dual_edge_length: Field[[EdgeDim], float],
        primal_normal_cell: tuple[Field[[ECDim], float], Field[[ECDim], float]],
        dual_normal_cell: tuple[Field[[ECDim], float], Field[[ECDim], float]],
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
        if l_init or l_recompute:
            if config.itime_scheme == 4 and not l_init:
                lvn_only = True  # Recompute only vn tendency
            else:
                lvn_only = False

        #  Set time levels of ddt_adv fields for call to velocity_tendencies
        if config.itime_scheme == 4:
            ntl1 = nnow
            ntl2 = nnew
        else:
            ntl1 = 1
            ntl2 = 1

        primal_normal_cell_1 = primal_normal_cell[0]
        dual_normal_cell_1 = dual_normal_cell[0]
        primal_normal_cell_2 = primal_normal_cell[1]
        dual_normal_cell_2 = dual_normal_cell[1]

        velocity_advection.VelocityAdvection(
            self.grid, self.metric_state, self.interpolation_state, self.vertical_params
        ).run_predictor_step(
            lvn_only,
            diagnostic_state,
            prognostic_state[nnow],
            self.z_w_concorr_me,
            self.z_kin_hor_e,
            self.z_vt_ie,
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

        #  Precompute Rayleigh damping factor
        compute_z_raylfac.with_backend(run_gtfn)(
            self.metric_state_nonhydro.rayleigh_w,
            dtime,
            self.z_raylfac,
            offset_provider={},
        )
        z_raylfac = self.z_raylfac

        (indices_0_1, indices_0_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.local_boundary(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        # initialize nest boundary points of z_rth_pr with zero
        if self.grid.limited_area():
            mo_solve_nonhydro_stencil_01.with_backend(run_gtfn)(
                self.z_rth_pr_1,
                self.z_rth_pr_2,
                horizontal_start=indices_0_1,
                horizontal_end=indices_0_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        (indices_1_1, indices_1_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.local_boundary(CellDim) + 2,
            HorizontalMarkerIndex.local(CellDim) - 1,
        )

        nhsolve_prog.predictor_stencils_2_3.with_backend(run_gtfn)(
            self.metric_state_nonhydro.exner_exfac,
            prognostic_state[nnow].exner,
            self.metric_state_nonhydro.exner_ref_mc,
            diagnostic_state_nonhydro.exner_pr,
            self.z_exner_ex_pr,
            horizontal_start=indices_1_1,
            horizontal_end=indices_1_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )
        # TODO: @abishekg7 check size and ordering of wgtfac_c
        if config.igradp_method == 3:
            nhsolve_prog.predictor_stencils_4_5_6.with_backend(run_gtfn)(
                self.metric_state_nonhydro.wgtfacq_c_dsl,
                self.z_exner_ex_pr,
                self.z_exner_ic,
                self.metric_state.wgtfac_c,
                self.metric_state_nonhydro.inv_ddqz_z_full,
                self.z_dexner_dz_c_1,
                self.k_field,
                self.grid.n_lev(),
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=max(1, self.vertical_params.nflatlev),
                vertical_end=self.grid.n_lev() + 1,
                offset_provider={"Koff": KDim},
            )

            if self.vertical_params.nflatlev == 1:
                # Perturbation Exner pressure on top half level
                raise NotImplementedError("nflatlev=1 not implemented")

        nhsolve_prog.predictor_stencils_7_8_9.with_backend(run_gtfn)(
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
            self.k_field,
            self.grid.n_lev(),
            horizontal_start=indices_1_1,
            horizontal_end=indices_1_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        # Perturbation theta at top and surface levels
        nhsolve_prog.predictor_stencils_11_lower_upper.with_backend(run_gtfn)(
            self.metric_state_nonhydro.wgtfacq_c_dsl,
            self.z_rth_pr_2,
            self.metric_state_nonhydro.theta_ref_ic,
            self.z_theta_v_pr_ic,
            diagnostic_state_nonhydro.theta_v_ic,
            self.k_field,
            self.grid.n_lev(),
            horizontal_start=indices_1_1,
            horizontal_end=indices_1_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev() + 1,
            offset_provider={"Koff": KDim},
        )

        if config.igradp_method == 3:
            # Second vertical derivative of perturbation Exner pressure (hydrostatic approximation)
            mo_solve_nonhydro_stencil_12.with_backend(run_gtfn)(
                self.z_theta_v_pr_ic,
                self.metric_state_nonhydro.d2dexdz2_fac1_mc,
                self.metric_state_nonhydro.d2dexdz2_fac2_mc,
                self.z_rth_pr_2,
                self.z_dexner_dz_c_2,
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=self.grid.nflat_gradp(),
                vertical_end=self.grid.n_lev(),
                offset_provider={"Koff": KDim},
            )

        # Add computation of z_grad_rth (perturbation density and virtual potential temperature at main levels)
        # at outer halo points: needed for correct calculation of the upwind gradients for Miura scheme
        (indices_2_1, indices_2_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.local(CellDim) - 2,
            HorizontalMarkerIndex.local(CellDim) - 2,
        )

        mo_solve_nonhydro_stencil_13.with_backend(run_gtfn)(
            prognostic_state[nnow].rho,
            self.metric_state_nonhydro.rho_ref_mc,
            prognostic_state[nnow].theta_v,
            self.metric_state.theta_ref_mc,
            self.z_rth_pr_1,
            self.z_rth_pr_2,
            horizontal_start=indices_2_1,
            horizontal_end=indices_2_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )

        (indices_3_1, indices_3_2) = self.grid.get_indices_from_to(
            VertexDim,
            HorizontalMarkerIndex.local_boundary(VertexDim) + 1,  # TODO: check
            HorizontalMarkerIndex.local(VertexDim) - 1,
        )

        # Compute rho and theta at edges for horizontal flux divergence term
        if config.iadv_rhotheta == 1:
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl.with_backend(
                run_gtfn
            )(
                prognostic_state[nnow].rho,
                self.interpolation_state.c_intp,
                self.z_rho_v,
                horizontal_start=indices_3_1,
                horizontal_end=indices_3_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),  # UBOUND(p_cell_in,2)
                offset_provider={
                    "V2C": self.grid.get_v2c_connectivity(),
                },
            )
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl.with_backend(
                run_gtfn
            )(
                prognostic_state[nnow].theta_v,
                self.interpolation_state.c_intp,
                self.z_theta_v_v,
                horizontal_start=indices_3_1,
                horizontal_end=indices_3_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),  # UBOUND(p_cell_in,2)
                offset_provider={
                    "V2C": self.grid.get_v2c_connectivity(),
                },
            )
        elif config.iadv_rhotheta == 2:
            # Compute Green-Gauss gradients for rho and theta
            mo_math_gradients_grad_green_gauss_cell_dsl.with_backend(run_gtfn)(
                self.p_grad_1,
                self.p_grad_2,
                self.p_grad_3,
                self.p_grad_4,
                self.p_ccpr_1,
                self.p_ccpr_2,
                self.interpolation_state.geofac_grg[0],
                self.interpolation_state.geofac_grg[1],
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),  # UBOUND(p_ccpr,2)
                offset_provider={
                    "C2E2CO": self.grid.get_c2e2co_connectivity(),
                },
            )

        if config.iadv_rhotheta <= 2:
            (tmp_0_0, tmp_0_1) = self.grid.get_indices_from_to(
                EdgeDim,
                HorizontalMarkerIndex.local(EdgeDim) - 2,
                HorizontalMarkerIndex.local(EdgeDim) - 3,
            )
            if config.idiv_method == 1:
                (tmp_0_0, tmp_0_1) = self.grid.get_indices_from_to(
                    EdgeDim,
                    HorizontalMarkerIndex.local(EdgeDim) - 2,
                    HorizontalMarkerIndex.local(EdgeDim) - 2,
                )
            # TODO: this makes the code crash but it should not, it is most likely a bounds problem
            set_zero_e_k.with_backend(run_gtfn)(
                field=self.z_rho_e,
                horizontal_start=tmp_0_0,
                horizontal_end=tmp_0_1,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

            set_zero_e_k.with_backend(run_gtfn)(
                field=self.z_theta_v_e,
                horizontal_start=tmp_0_0,
                horizontal_end=tmp_0_1,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

            (indices_4_1, indices_4_2) = self.grid.get_indices_from_to(
                EdgeDim,
                HorizontalMarkerIndex.local_boundary(EdgeDim),
                HorizontalMarkerIndex.local(EdgeDim) - 1,
            )

            # initialize also nest boundary points with zero
            if self.grid.limited_area():
                set_zero_e_k.with_backend(run_gtfn)(
                    field=self.z_rho_e,
                    horizontal_start=indices_4_1,
                    horizontal_end=indices_4_2,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev(),
                    offset_provider={},
                )

                set_zero_e_k.with_backend(run_gtfn)(
                    field=self.z_theta_v_e,
                    horizontal_start=indices_4_1,
                    horizontal_end=indices_4_2,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev(),
                    offset_provider={},
                )

            (indices_12_1, indices_12_2) = self.grid.get_indices_from_to(
                EdgeDim,
                HorizontalMarkerIndex.local_boundary(EdgeDim) + 6,
                HorizontalMarkerIndex.interior(EdgeDim) - 1,
            )

            if config.iadv_rhotheta == 2:
                # Compute upwind-biased values for rho and theta starting from centered differences
                # Note: the length of the backward trajectory should be 0.5*dtime*(vn,vt) in order to arrive
                # at a second-order accurate FV discretization, but twice the length is needed for numerical stability

                nhsolve_prog.mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1.with_backend(
                    run_gtfn
                )(
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
                    self.z_rho_e,
                    self.z_theta_v_e,
                    horizontal_start=indices_12_1,
                    horizontal_end=indices_12_2,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev(),
                    offset_provider={
                        "E2C": self.grid.get_e2c_connectivity(),
                        "E2EC": self.grid.get_e2ec_connectivity(),
                    },
                )
        (indices_5_1, indices_5_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim) + 1,
            HorizontalMarkerIndex.local(EdgeDim),
        )

        # Remaining computations at edge points
        mo_solve_nonhydro_stencil_18.with_backend(run_gtfn)(
            inv_dual_edge_length,
            self.z_exner_ex_pr,
            self.z_gradh_exner,
            horizontal_start=indices_5_1,
            horizontal_end=indices_5_2,
            vertical_start=0,
            vertical_end=self.vertical_params.nflatlev - 1,
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
            },
        )

        if config.igradp_method == 3:
            # horizontal gradient of Exner pressure, including metric correction
            # horizontal gradient of Exner pressure, Taylor-expansion-based reconstruction

            mo_solve_nonhydro_stencil_19.with_backend(run_gtfn)(
                inv_dual_edge_length,
                self.z_exner_ex_pr,
                self.metric_state.ddxn_z_full,
                self.interpolation_state.c_lin_e,
                self.z_dexner_dz_c_1,
                self.z_gradh_exner,
                horizontal_start=indices_5_1,
                horizontal_end=indices_5_2,
                vertical_start=self.vertical_params.nflatlev,
                vertical_end=self.grid.nflat_gradp(),
                offset_provider={
                    "E2C": self.grid.get_e2c_connectivity(),
                },
            )

            mo_solve_nonhydro_stencil_20.with_backend(run_gtfn)(
                inv_dual_edge_length,
                self.z_exner_ex_pr,
                self.metric_state_nonhydro.zdiff_gradp,
                self.ikoffset,
                self.z_dexner_dz_c_1,
                self.z_dexner_dz_c_2,
                self.z_gradh_exner,
                horizontal_start=indices_5_1,
                horizontal_end=indices_5_2,
                vertical_start=self.vertical_params.nflatlev + 1,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "E2C": self.grid.get_e2c_connectivity(),
                    "E2EC": self.grid.get_e2ec_connectivity(),
                    "Koff": KDim,
                },
            )
        # compute hydrostatically approximated correction term that replaces downward extrapolation
        if config.igradp_method == 3:
            mo_solve_nonhydro_stencil_21.with_backend(run_gtfn)(
                prognostic_state[nnow].theta_v,
                self.ikoffset,
                self.metric_state_nonhydro.zdiff_gradp,
                diagnostic_state_nonhydro.theta_v_ic,
                self.metric_state_nonhydro.inv_ddqz_z_full,
                inv_dual_edge_length,
                params.grav_o_cpd,
                self.z_hydro_corr,
                horizontal_start=indices_5_1,
                horizontal_end=indices_5_2,
                vertical_start=self.grid.n_lev(),
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "E2C": self.grid.get_e2c_connectivity(),
                    "E2EC": self.grid.get_e2ec_connectivity(),
                    "Koff": KDim,
                },
            )

        if config.igradp_method == 3:

            (indices_22_1, indices_22_2) = self.grid.get_indices_from_to(
                EdgeDim,
                HorizontalMarkerIndex.nudging(EdgeDim) + 1,
                HorizontalMarkerIndex.local(EdgeDim),
            )

            mo_solve_nonhydro_stencil_22.with_backend(run_gtfn)(
                self.metric_state_nonhydro.ipeidx_dsl,
                self.metric_state_nonhydro.pg_exdist,
                self.z_hydro_corr,
                self.z_gradh_exner,
                horizontal_start=indices_22_1,
                horizontal_end=indices_22_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        mo_solve_nonhydro_stencil_24.with_backend(run_gtfn)(
            prognostic_state[nnow].vn,
            diagnostic_state_nonhydro.ddt_vn_adv_ntl[ntl1],
            diagnostic_state_nonhydro.ddt_vn_phy,
            self.z_theta_v_e,
            self.z_gradh_exner,
            prognostic_state[nnew].vn,
            dtime,
            constants.CPD,
            horizontal_start=indices_5_1,
            horizontal_end=indices_5_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )

        # keep commented out for now
        # if config.is_iau_active:
        #     mo_solve_nonhydro_stencil_28(
        #         diagnostic_state_nonhydro.vn_incr,
        #         prognostic_state[nnew].vn,
        #         config.iau_wgt_dyn,
        #         horizontal_start=edge_startindex_nudging + 1,
        #         horizontal_end=edge_endindex_interior,
        #         vertical_start=0,
        #         vertical_end=self.grid.n_lev(),
        #         offset_provider={},
        #     )

        (indices_6_1, indices_6_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.local_boundary(EdgeDim),
            HorizontalMarkerIndex.nudging(EdgeDim),
        )

        if self.grid.limited_area():
            mo_solve_nonhydro_stencil_29.with_backend(run_gtfn)(
                diagnostic_state_nonhydro.grf_tend_vn,
                prognostic_state[nnow].vn,
                prognostic_state[nnew].vn,
                dtime,
                horizontal_start=indices_6_1,
                horizontal_end=indices_6_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        ##### COMMUNICATION PHASE

        (indices_7_1, indices_7_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.local_boundary(EdgeDim) + 4,
            HorizontalMarkerIndex.local(EdgeDim) - 2,
        )

        mo_solve_nonhydro_stencil_30.with_backend(run_gtfn)(
            self.interpolation_state.e_flx_avg,
            prognostic_state[nnew].vn,
            self.interpolation_state.geofac_grdiv,
            self.interpolation_state.rbf_vec_coeff_e,
            self.z_vn_avg,
            self.z_graddiv_vn,
            diagnostic_state.vt,
            horizontal_start=indices_7_1,
            horizontal_end=indices_7_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "E2C2EO": self.grid.get_e2c2eo_connectivity(),
                "E2C2E": self.grid.get_e2c2e_connectivity(),
            },
        )

        if config.idiv_method == 1:
            mo_solve_nonhydro_stencil_32.with_backend(run_gtfn)(
                self.z_rho_e,
                self.z_vn_avg,
                self.metric_state.ddqz_z_full_e,
                self.z_theta_v_e,
                diagnostic_state_nonhydro.mass_fl_e,
                self.z_theta_v_fl_e,
                horizontal_start=indices_7_1,
                horizontal_end=indices_7_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        nhsolve_prog.predictor_stencils_35_36.with_backend(run_gtfn)(
            prognostic_state[nnew].vn,
            self.metric_state.ddxn_z_full,
            self.metric_state.ddxt_z_full,
            diagnostic_state.vt,
            self.z_w_concorr_me,
            self.metric_state.wgtfac_e,
            diagnostic_state.vn_ie,
            self.z_vt_ie,
            self.z_kin_hor_e,
            self.k_field,
            self.vertical_params.nflatlev,
            self.grid.n_lev(),
            horizontal_start=indices_7_1,
            horizontal_end=indices_7_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        if not self.l_vert_nested:
            nhsolve_prog.predictor_stencils_37_38.with_backend(run_gtfn)(
                prognostic_state[nnew].vn,
                diagnostic_state.vt,
                diagnostic_state.vn_ie,
                self.z_vt_ie,
                self.z_kin_hor_e,
                self.metric_state.wgtfac_e_dsl,
                self.k_field,
                self.grid.n_lev(),
                horizontal_start=indices_7_1,
                horizontal_end=indices_7_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev() + 1,
                offset_provider={"Koff": KDim},
            )

        (indices_9_1, indices_9_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.local_boundary(CellDim) + 2,
            HorizontalMarkerIndex.local(CellDim) - 1,
        )

        nhsolve_prog.stencils_39_40.with_backend(run_gtfn)(
            self.interpolation_state.e_bln_c_s,
            self.z_w_concorr_me,
            self.metric_state.wgtfac_c,
            self.metric_state_nonhydro.wgtfacq_c_dsl,
            diagnostic_state.w_concorr_c,
            self.k_field,
            self.vertical_params.nflatlev + 1,
            self.grid.n_lev(),
            horizontal_start=indices_9_1,
            horizontal_end=indices_9_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev() + 1,
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "Koff": KDim,
            },
        )

        (indices_10_1, indices_10_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        if config.idiv_method == 1:
            mo_solve_nonhydro_stencil_41.with_backend(run_gtfn)(
                self.interpolation_state.geofac_div,
                diagnostic_state_nonhydro.mass_fl_e,
                self.z_theta_v_fl_e,
                self.z_flxdiv_mass,
                self.z_flxdiv_theta,
                horizontal_start=indices_10_1,
                horizontal_end=indices_10_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "C2E": self.grid.get_c2e_connectivity(),
                },
            )

        # check for p_nh%prog(nnow)% fields and new
        nhsolve_prog.stencils_43_44_45_45b.with_backend(run_gtfn)(
            self.z_w_expl,
            prognostic_state[nnow].w,
            diagnostic_state_nonhydro.ddt_w_adv_ntl[ntl1],
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
            self.k_field,
            constants.RD,
            constants.CVD,
            dtime,
            constants.CPD,
            indices_10_1,
            indices_10_2,
            self.grid.n_lev(),
            self.grid.n_lev() + 1,
            offset_provider={},
        )

        if not (config.l_open_ubc and self.l_vert_nested):
            mo_solve_nonhydro_stencil_46.with_backend(run_gtfn)(
                prognostic_state[nnew].w,
                self.z_contr_w_fl_l,
                horizontal_start=indices_10_1,
                horizontal_end=indices_10_2,
                vertical_start=0,
                vertical_end=0,
                offset_provider={},
            )
        nhsolve_prog.stencils_47_48_49.with_backend(run_gtfn)(
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
            self.k_field,
            dtime,
            indices_10_1,
            indices_10_2,
            self.grid.n_lev(),
            self.grid.n_lev() + 1,
            offset_provider={
                "Koff": KDim,
            },
        )

        # keep comment for now
        # if config.is_iau_active:
        #     mo_solve_nonhydro_stencil_50.with_backend(run_gtfn)(
        #         self.z_rho_expl,
        #         self.z_exner_expl,
        #         diagnostic_state_nonhydro.rho_incr,
        #         diagnostic_state_nonhydro.exner_incr,
        #         config.iau_wgt_dyn,
        #         horizontal_start=indices_10_1,
        #         horizontal_end=indices_10_2,
        #         vertical_start=0,
        #         vertical_end=self.grid.n_lev(),
        #         offset_provider={},
        #     )

        mo_solve_nonhydro_stencil_52.with_backend(run_gtfn)(
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
            horizontal_start=indices_10_1,
            horizontal_end=indices_10_2,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        mo_solve_nonhydro_stencil_53.with_backend(run_gtfn)(
            self.z_q,
            prognostic_state[nnew].w,
            horizontal_start=indices_10_1,
            horizontal_end=indices_10_2,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )

        if config.rayleigh_type == constants.RAYLEIGH_KLEMP:
            ## ACC w_1 -> p_nh%w
            mo_solve_nonhydro_stencil_54.with_backend(run_gtfn)(
                z_raylfac,
                prognostic_state[nnew].w_1,
                prognostic_state[nnew].w,
                horizontal_start=indices_10_1,
                horizontal_end=indices_10_2,
                vertical_start=1,
                vertical_end=self.vertical_params.index_of_damping_layer,  # nrdmax
                offset_provider={},
            )

        mo_solve_nonhydro_stencil_55.with_backend(run_gtfn)(
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
            horizontal_start=indices_10_1,
            horizontal_end=indices_10_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        # compute dw/dz for divergence damping term
        if config.lhdiff_rcf and config.divdamp_type >= 3:
            mo_solve_nonhydro_stencil_56_63.with_backend(run_gtfn)(
                self.metric_state_nonhydro.inv_ddqz_z_full,
                prognostic_state[nnew].w,
                diagnostic_state.w_concorr_c,
                self.z_dwdz_dd,
                horizontal_start=indices_10_1,
                horizontal_end=indices_10_2,
                vertical_start=params.kstart_dd3d,
                vertical_end=self.grid.n_lev(),
                offset_provider={"Koff": KDim},
            )

        if idyn_timestep == 1:
            nhsolve_prog.predictor_stencils_59_60.with_backend(run_gtfn)(
                prognostic_state[nnow].exner,
                prognostic_state[nnew].exner,
                self.exner_dyn_incr,
                diagnostic_state_nonhydro.ddt_exner_phy,
                config.ndyn_substeps_var,
                dtime,
                indices_10_1,
                indices_10_2,
                params.kstart_moist,
                self.grid.n_lev(),
                offset_provider={},
            )

        (indices_11_1, indices_11_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.local_boundary(EdgeDim),
            HorizontalMarkerIndex.nudging(CellDim) - 1,
        )

        if self.grid.limited_area():  # for MPI-parallelized case
            nhsolve_prog.stencils_61_62.with_backend(run_gtfn)(
                prognostic_state[nnow].rho,
                diagnostic_state_nonhydro.grf_tend_rho,
                prognostic_state[nnow].theta_v,
                diagnostic_state_nonhydro.grf_tend_thv,
                prognostic_state[nnow].w,
                diagnostic_state_nonhydro.grf_tend_w,
                prognostic_state[nnew].rho,
                prognostic_state[nnew].exner,
                prognostic_state[nnew].w,
                self.k_field,
                dtime,
                self.grid.n_lev(),
                horizontal_start=indices_11_1,
                horizontal_end=indices_11_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev() + 1,
                offset_provider={},
            )

        if config.lhdiff_rcf and config.divdamp_type >= 3:
            mo_solve_nonhydro_stencil_56_63.with_backend(run_gtfn)(
                self.metric_state_nonhydro.inv_ddqz_z_full,
                prognostic_state[nnew].w,
                diagnostic_state.w_concorr_c,
                self.z_dwdz_dd,
                horizontal_start=indices_11_1,
                horizontal_end=indices_11_2,
                vertical_start=params.kstart_dd3d,
                vertical_end=self.grid.n_lev(),
                offset_provider={"Koff": KDim},
            )

        ##### COMMUNICATION PHASE

    def run_corrector_step(
        self,
        diagnostic_state: DiagnosticState,
        diagnostic_state_nonhydro: DiagnosticStateNonHydro,
        prognostic_state: list[PrognosticState],
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
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
        lclean_mflx: bool,
        scal_divdamp_o2: float,
        bdy_divdamp: Field[[KDim], float],
        lprep_adv: bool,
    ):

        #  Set time levels of ddt_adv fields for call to velocity_tendencies
        if config.itime_scheme == 4:
            ntl1 = nnow
            ntl2 = nnew
        else:
            ntl1 = 1
            ntl2 = 1

        self.wgt_nnow_vel = 0.5 - config.veladv_offctr
        self.wgt_nnew_vel = 0.5 + config.veladv_offctr

        self.wgt_nnew_rth = 0.5 + config.rhotheta_offctr
        self.wgt_nnow_rth = 1.0 - self.wgt_nnew_rth

        # Inverse value of ndyn_substeps for tracer advection precomputations
        r_nsubsteps = 1.0 / config.ndyn_substeps_var

        lvn_only = False
        velocity_advection.VelocityAdvection(
            self.grid, self.metric_state, self.interpolation_state, self.vertical_params
        ).run_corrector_step(
            lvn_only,
            diagnostic_state,
            prognostic_state[nnew],
            self.z_kin_hor_e,
            self.z_vt_ie,
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

        #  Precompute Rayleigh damping factor
        compute_z_raylfac.with_backend(run_gtfn)(
            self.metric_state_nonhydro.rayleigh_w,
            dtime,
            self.z_raylfac,
            offset_provider={},
        )
        z_raylfac = self.z_raylfac

        (indices_0_1, indices_0_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.local_boundary(CellDim) + 2,
            HorizontalMarkerIndex.local(CellDim),
        )

        mo_solve_nonhydro_stencil_10.with_backend(run_gtfn)(
            prognostic_state[nnew].w,
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
            self.wgt_nnow_rth,
            self.wgt_nnew_rth,
            horizontal_start=indices_0_1,
            horizontal_end=indices_0_2,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        if config.l_open_ubc and not self.l_vert_nested:
            raise NotImplementedError("l_open_ubc not implemented")

        (indices_0_2, indices_0_3) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.local_boundary(EdgeDim) + 6,
            HorizontalMarkerIndex.local(EdgeDim) - 2,
        )

        mo_solve_nonhydro_stencil_17.with_backend(run_gtfn)(
            self.metric_state_nonhydro.hmask_dd3d,
            self.metric_state_nonhydro.scalfac_dd3d,
            inv_dual_edge_length,
            self.z_dwdz_dd,
            self.z_graddiv_vn,
            horizontal_start=indices_0_2,
            horizontal_end=indices_0_3,
            vertical_start=params.kstart_dd3d,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
            },
        )

        (indices_1_1, indices_1_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim) + 1,
            HorizontalMarkerIndex.local(EdgeDim),
        )

        if config.itime_scheme == 4:
            mo_solve_nonhydro_stencil_23.with_backend(run_gtfn)(
                prognostic_state[nnow].vn,
                diagnostic_state_nonhydro.ddt_vn_adv_ntl[ntl1],
                diagnostic_state_nonhydro.ddt_vn_adv_ntl[ntl2],
                diagnostic_state_nonhydro.ddt_vn_phy,
                self.z_theta_v_e,
                self.z_gradh_exner,
                prognostic_state[nnew].vn,
                dtime,
                self.wgt_nnow_vel,
                self.wgt_nnew_vel,
                constants.CPD,
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        if config.lhdiff_rcf and config.divdamp_order == 24:
            mo_solve_nonhydro_stencil_25.with_backend(run_gtfn)(
                self.interpolation_state.geofac_grdiv,
                self.z_graddiv_vn,
                self.z_graddiv2_vn,
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "E2C2EO": self.grid.get_e2c2eo_connectivity(),
                },
            )

        if config.lhdiff_rcf:
            if config.divdamp_order == 24 and scal_divdamp_o2 > 1.0e-6:
                mo_solve_nonhydro_stencil_26.with_backend(run_gtfn)(
                    self.z_graddiv_vn,
                    prognostic_state[nnew].vn,
                    scal_divdamp_o2,
                    horizontal_start=indices_1_1,
                    horizontal_end=indices_1_2,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev(),
                    offset_provider={},
                )
            if (
                config.divdamp_order == 24
                and config.divdamp_fac_o2 <= 4 * config.divdamp_fac
            ):
                if self.grid.limited_area():
                    mo_solve_nonhydro_stencil_27.with_backend(run_gtfn)(
                        self.scal_divdamp,
                        bdy_divdamp,
                        self.interpolation_state.nudgecoeff_e,
                        self.z_graddiv2_vn,
                        prognostic_state[nnew].vn,
                        horizontal_start=indices_1_1,
                        horizontal_end=indices_1_2,
                        vertical_start=0,
                        vertical_end=self.grid.n_lev(),
                        offset_provider={},
                    )
                else:
                    mo_solve_nonhydro_4th_order_divdamp.with_backend(run_gtfn)(
                        self.scal_divdamp,
                        self.z_graddiv2_vn,
                        prognostic_state[nnew].vn,
                        horizontal_start=indices_1_1,
                        horizontal_end=indices_1_2,
                        vertical_start=0,
                        vertical_end=self.grid.n_lev(),
                        offset_provider={},
                    )

        # keep commented out for now
        # if config.is_iau_active:
        #     mo_solve_nonhydro_stencil_28(
        #         diagnostic_state_nonhydro.vn_incr,
        #         prognostic_state[nnew].vn,
        #         config.iau_wgt_dyn,
        #         horizontal_start=edge_startindex_nudging + 1,
        #         horizontal_end=edge_endindex_interior,
        #         vertical_start=0,
        #         vertical_end=self.grid.n_lev(),
        #         offset_provider={},
        #     )

        (indices_2_1, indices_2_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.local_boundary(EdgeDim) + 4,
            HorizontalMarkerIndex.local(EdgeDim) - 2,
        )

        mo_solve_nonhydro_stencil_31.with_backend(run_gtfn)(
            self.interpolation_state.e_flx_avg,
            prognostic_state[nnew].vn,
            self.z_vn_avg,
            horizontal_start=indices_2_1,
            horizontal_end=indices_2_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "E2C2EO": self.grid.get_e2c2eo_connectivity(),
            },
        )

        ##### COMMUNICATION PHASE

        if config.idiv_method == 1:
            mo_solve_nonhydro_stencil_32.with_backend(run_gtfn)(
                self.z_rho_e,
                self.z_vn_avg,
                self.metric_state.ddqz_z_full_e,
                self.z_theta_v_e,
                diagnostic_state_nonhydro.mass_fl_e,
                self.z_theta_v_fl_e,
                horizontal_start=indices_2_1,
                horizontal_end=indices_2_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

            (indices_3_1, indices_3_2) = self.grid.get_indices_from_to(
                EdgeDim,
                HorizontalMarkerIndex.local_boundary(EdgeDim),
                HorizontalMarkerIndex.local(EdgeDim),
            )

            if lprep_adv:  # Preparations for tracer advection
                if lclean_mflx:
                    mo_solve_nonhydro_stencil_33.with_backend(run_gtfn)(
                        prep_adv.vn_traj,
                        prep_adv.mass_flx_me,
                        horizontal_start=indices_3_1,
                        horizontal_end=indices_3_2,
                        vertical_start=0,
                        vertical_end=self.grid.n_lev(),
                        offset_provider={},
                    )
                mo_solve_nonhydro_stencil_34.with_backend(run_gtfn)(
                    self.z_vn_avg,
                    diagnostic_state_nonhydro.mass_fl_e,
                    prep_adv.vn_traj,
                    prep_adv.mass_flx_me,
                    r_nsubsteps,
                    horizontal_start=indices_2_1,
                    horizontal_end=indices_2_2,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev(),
                    offset_provider={},
                )

        if config.idiv_method == 1:
            mo_solve_nonhydro_stencil_41.with_backend(run_gtfn)(
                self.interpolation_state.geofac_div,
                diagnostic_state_nonhydro.mass_fl_e,
                self.z_theta_v_fl_e,
                self.z_flxdiv_mass,
                self.z_flxdiv_theta,
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "C2E": self.grid.get_c2e_connectivity(),
                },
            )

        (indices_5_1, indices_5_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        if config.itime_scheme == 4:
            nhsolve_prog.stencils_42_44_45_45b.with_backend(run_gtfn)(
                self.z_w_expl,
                prognostic_state[nnow].w,
                diagnostic_state_nonhydro.ddt_w_adv_ntl[ntl1],
                diagnostic_state_nonhydro.ddt_w_adv_ntl[ntl2],
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
                self.k_field,
                constants.RD,
                constants.CVD,
                dtime,
                constants.CPD,
                self.wgt_nnow_vel,
                self.wgt_nnew_vel,
                indices_1_1,
                indices_1_2,
                self.grid.n_lev(),
                self.grid.n_lev() + 1,
                offset_provider={},
            )
        else:
            nhsolve_prog.stencils_43_44_45_45b.with_backend(run_gtfn)(
                self.z_w_expl,
                prognostic_state[nnow].w,
                diagnostic_state_nonhydro.ddt_w_adv_ntl[ntl1],
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
                self.k_field,
                constants.RD,
                constants.CVD,
                dtime,
                constants.CPD,
                indices_5_1,
                indices_5_2,
                self.grid.n_lev(),
                self.grid.n_lev() + 1,
                offset_provider={},
            )

        if not config.l_open_ubc and not self.l_vert_nested:
            mo_solve_nonhydro_stencil_46.with_backend(run_gtfn)(
                prognostic_state[nnew].w,
                self.z_contr_w_fl_l,
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=0,
                vertical_end=0,
                offset_provider={},
            )

        nhsolve_prog.stencils_47_48_49.with_backend(run_gtfn)(
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
            self.k_field,
            dtime,
            indices_1_1,
            indices_1_2,
            self.grid.n_lev(),
            self.grid.n_lev() + 1,
            offset_provider={
                "Koff": KDim,
            },
        )

        # keep commented out for now
        # if config.is_iau_active:
        #     mo_solve_nonhydro_stencil_50(
        #         self.z_rho_expl,
        #         self.z_exner_expl,
        #         diagnostic_state_nonhydro.rho_incr,
        #         diagnostic_state_nonhydro.exner_incr,
        #         config.iau_wgt_dyn,
        #         horizontal_start=cell_startindex_nudging + 1,
        #         horizontal_end=cell_endindex_interior,
        #         vertical_start=0,
        #         vertical_end=self.grid.n_lev(),
        #         offset_provider={},
        #     )

        mo_solve_nonhydro_stencil_52.with_backend(run_gtfn)(
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
            horizontal_start=indices_1_1,
            horizontal_end=indices_1_2,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        mo_solve_nonhydro_stencil_53.with_backend(run_gtfn)(
            self.z_q,
            prognostic_state[nnew].w,
            horizontal_start=indices_1_1,
            horizontal_end=indices_1_2,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )

        if config.rayleigh_type == constants.RAYLEIGH_KLEMP:
            ## ACC w_1 -> p_nh%w
            mo_solve_nonhydro_stencil_54.with_backend(run_gtfn)(
                z_raylfac,
                prognostic_state[nnew].w_1,
                prognostic_state[nnew].w,
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=1,
                vertical_end=self.vertical_params.index_of_damping_layer,  # nrdmax
                offset_provider={},
            )

        mo_solve_nonhydro_stencil_55.with_backend(run_gtfn)(
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
            horizontal_start=indices_1_1,
            horizontal_end=indices_1_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        if lprep_adv:
            if lclean_mflx:
                set_zero_c_k.with_backend(run_gtfn)(
                    field=self.mass_flx_ic,
                    horizontal_start=indices_1_1,
                    horizontal_end=indices_1_2,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev(),
                    offset_provider={},
                )

        mo_solve_nonhydro_stencil_58.with_backend(run_gtfn)(
            self.z_contr_w_fl_l,
            diagnostic_state_nonhydro.rho_ic,
            self.metric_state_nonhydro.vwind_impl_wgt,
            prognostic_state[nnew].w,
            self.mass_flx_ic,
            r_nsubsteps,
            horizontal_start=indices_1_1,
            horizontal_end=indices_1_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )

        (indices_4_1, indices_4_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.local_boundary(CellDim),
            HorizontalMarkerIndex.nudging(CellDim) - 1,
        )

        if lprep_adv:
            if lclean_mflx:
                set_zero_c_k.with_backend(run_gtfn)(
                    field=self.mass_flx_ic,
                    horizontal_start=indices_4_1,
                    horizontal_end=indices_4_2,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev() + 1,
                    offset_provider={},
                )

            mo_solve_nonhydro_stencil_65.with_backend(run_gtfn)(
                diagnostic_state_nonhydro.rho_ic,
                self.metric_state_nonhydro.vwind_expl_wgt,
                self.metric_state_nonhydro.vwind_impl_wgt,
                prognostic_state[nnow].w,
                prognostic_state[nnew].w,
                diagnostic_state.w_concorr_c,
                self.mass_flx_ic,
                r_nsubsteps,
                horizontal_start=indices_4_1,
                horizontal_end=indices_4_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        ##### COMMUNICATION PHASE
