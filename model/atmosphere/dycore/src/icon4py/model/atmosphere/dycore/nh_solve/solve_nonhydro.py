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
from dataclasses import dataclass
from typing import Final, Optional

from gt4py.next import as_field
from gt4py.next.common import Field
from gt4py.next.ffront.fbuiltins import int32

import icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro_program as nhsolve_prog
import icon4py.model.common.constants as constants
from icon4py.model.atmosphere.dycore.init_cell_kdim_field_with_zero_wp import (
    init_cell_kdim_field_with_zero_wp,
)

from icon4py.model.atmosphere.dycore.accumulate_prep_adv_fields import (
    accumulate_prep_adv_fields,
)
from icon4py.model.atmosphere.dycore.add_analysis_increments_from_data_assimilation import (
    add_analysis_increments_from_data_assimilation,
)
from icon4py.model.atmosphere.dycore.add_analysis_increments_to_vn import (
    add_analysis_increments_to_vn,
)
from icon4py.model.atmosphere.dycore.add_temporal_tendencies_to_vn import (
    add_temporal_tendencies_to_vn,
)
from icon4py.model.atmosphere.dycore.add_temporal_tendencies_to_vn_by_interpolating_between_time_levels import (
    add_temporal_tendencies_to_vn_by_interpolating_between_time_levels,
)
from icon4py.model.atmosphere.dycore.add_vertical_wind_derivative_to_divergence_damping import (
    add_vertical_wind_derivative_to_divergence_damping,
)
from icon4py.model.atmosphere.dycore.apply_2nd_order_divergence_damping import (
    apply_2nd_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.apply_4th_order_divergence_damping import (
    apply_4th_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure import (
    apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure,
)
from icon4py.model.atmosphere.dycore.apply_rayleigh_damping_mechanism import (
    apply_rayleigh_damping_mechanism,
)
from icon4py.model.atmosphere.dycore.apply_weighted_2nd_and_4th_order_divergence_damping import (
    apply_weighted_2nd_and_4th_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.compute_approx_of_2nd_vertical_derivative_of_exner import (
    compute_approx_of_2nd_vertical_derivative_of_exner,
)
from icon4py.model.atmosphere.dycore.compute_avg_vn import compute_avg_vn
from icon4py.model.atmosphere.dycore.compute_avg_vn_and_graddiv_vn_and_vt import (
    compute_avg_vn_and_graddiv_vn_and_vt,
)
from icon4py.model.atmosphere.dycore.compute_divergence_of_fluxes_of_rho_and_theta import (
    compute_divergence_of_fluxes_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.compute_dwdz_for_divergence_damping import (
    compute_dwdz_for_divergence_damping,
)
from icon4py.model.atmosphere.dycore.compute_exner_from_rhotheta import (
    compute_exner_from_rhotheta,
)
from icon4py.model.atmosphere.dycore.compute_graddiv2_of_vn import (
    compute_graddiv2_of_vn,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates import (
    compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates import (
    compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_gradient_of_exner_pressure_for_multiple_levels import (
    compute_horizontal_gradient_of_exner_pressure_for_multiple_levels,
)
from icon4py.model.atmosphere.dycore.compute_hydrostatic_correction_term import (
    compute_hydrostatic_correction_term,
)
from icon4py.model.atmosphere.dycore.compute_mass_flux import compute_mass_flux
from icon4py.model.atmosphere.dycore.compute_perturbation_of_rho_and_theta import (
    compute_perturbation_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.compute_results_for_thermodynamic_variables import (
    compute_results_for_thermodynamic_variables,
)
from icon4py.model.atmosphere.dycore.compute_rho_virtual_potential_temperatures_and_pressure_gradient import (
    compute_rho_virtual_potential_temperatures_and_pressure_gradient,
)
from icon4py.model.atmosphere.dycore.compute_theta_and_exner import (
    compute_theta_and_exner,
)
from icon4py.model.atmosphere.dycore.compute_vn_on_lateral_boundary import (
    compute_vn_on_lateral_boundary,
)
from icon4py.model.atmosphere.dycore.copy_cell_kdim_field_to_vp import (
    copy_cell_kdim_field_to_vp,
)
from icon4py.model.atmosphere.dycore.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.atmosphere.dycore.mo_math_gradients_grad_green_gauss_cell_dsl import (
    mo_math_gradients_grad_green_gauss_cell_dsl,
)
from icon4py.model.atmosphere.dycore.init_two_cell_kdim_fields_with_zero_vp import (
    init_two_cell_kdim_fields_with_zero_vp,
)
from icon4py.model.atmosphere.dycore.init_two_cell_kdim_fields_with_zero_wp import (
    init_two_cell_kdim_fields_with_zero_wp,
)
from icon4py.model.atmosphere.dycore.init_two_edge_kdim_fields_with_zero_wp import (
    init_two_edge_kdim_fields_with_zero_wp,
)
from icon4py.model.atmosphere.dycore.solve_tridiagonal_matrix_for_w_back_substitution import (
    solve_tridiagonal_matrix_for_w_back_substitution,
)
from icon4py.model.atmosphere.dycore.solve_tridiagonal_matrix_for_w_forward_sweep import (
    solve_tridiagonal_matrix_for_w_forward_sweep,
)
from icon4py.model.atmosphere.dycore.state_utils.states import (
    DiagnosticStateNonHydro,
    InterpolationState,
    MetricStateNonHydro,
    PrepAdvection,
)
from icon4py.model.atmosphere.dycore.state_utils.utils import (
    _allocate,
    _allocate_indices,
    _calculate_divdamp_fields,
    compute_z_raylfac,
)
from icon4py.model.atmosphere.dycore.update_dynamical_exner_time_increment import (
    update_dynamical_exner_time_increment,
)
from icon4py.model.atmosphere.dycore.update_mass_volume_flux import (
    update_mass_volume_flux,
)
from icon4py.model.atmosphere.dycore.update_mass_flux_weighted import (
    update_mass_flux_weighted,
)
from icon4py.model.atmosphere.dycore.update_theta_v import update_theta_v
from icon4py.model.atmosphere.dycore.velocity.velocity_advection import (
    VelocityAdvection,
)
from icon4py.model.common.decomposition.definitions import (
    ExchangeRuntime,
    SingleNodeExchange,
)
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.grid.base import BaseGrid
from icon4py.model.common.grid.horizontal import (
    CellParams,
    EdgeParams,
    HorizontalMarkerIndex,
)
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.math.smagorinsky import en_smag_fac_for_zero_nshift
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.settings import backend

from model.common.tests import field_type_aliases as fa

# flake8: noqa
log = logging.getLogger(__name__)


@dataclass
class IntermediateFields:
    """
    Encapsulate internal fields of SolveNonHydro that contain shared state over predictor and corrector step.

    Encapsulates internal fields used in SolveNonHydro. Fields (and the class!)
    follow the naming convention of ICON to prepend local fields of a module with z_. Contrary to
    other such z_ fields inside SolveNonHydro the fields in this dataclass
    contain state that is built up over the predictor and corrector part in a timestep.
    """

    z_gradh_exner: Field[[EdgeDim, KDim], float]
    z_alpha: Field[
        [EdgeDim, KDim], float
    ]  # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    z_beta: fa.CKfloatField
    z_w_expl: Field[
        [EdgeDim, KDim], float
    ]  # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    z_exner_expl: fa.CKfloatField
    z_q: fa.CKfloatField
    z_contr_w_fl_l: Field[
        [EdgeDim, KDim], float
    ]  # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    z_rho_e: Field[[EdgeDim, KDim], float]
    z_theta_v_e: Field[[EdgeDim, KDim], float]
    z_kin_hor_e: Field[[EdgeDim, KDim], float]
    z_vt_ie: Field[[EdgeDim, KDim], float]
    z_graddiv_vn: Field[[EdgeDim, KDim], float]
    z_rho_expl: fa.CKfloatField
    z_dwdz_dd: fa.CKfloatField

    @classmethod
    def allocate(cls, grid: BaseGrid):
        return IntermediateFields(
            z_gradh_exner=_allocate(EdgeDim, KDim, grid=grid),
            z_alpha=_allocate(CellDim, KDim, is_halfdim=True, grid=grid),
            z_beta=_allocate(CellDim, KDim, grid=grid),
            z_w_expl=_allocate(CellDim, KDim, is_halfdim=True, grid=grid),
            z_exner_expl=_allocate(CellDim, KDim, grid=grid),
            z_q=_allocate(CellDim, KDim, grid=grid),
            z_contr_w_fl_l=_allocate(CellDim, KDim, is_halfdim=True, grid=grid),
            z_rho_e=_allocate(EdgeDim, KDim, grid=grid),
            z_theta_v_e=_allocate(EdgeDim, KDim, grid=grid),
            z_graddiv_vn=_allocate(EdgeDim, KDim, grid=grid),
            z_rho_expl=_allocate(CellDim, KDim, grid=grid),
            z_dwdz_dd=_allocate(CellDim, KDim, grid=grid),
            z_kin_hor_e=_allocate(EdgeDim, KDim, grid=grid),
            z_vt_ie=_allocate(EdgeDim, KDim, grid=grid),
        )


class NonHydrostaticConfig:
    """
    Contains necessary parameter to configure a nonhydro run.

    Encapsulates namelist parameters and derived parameters.
    TODO: (magdalena) values should be read from a configuration file.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    """

    def __init__(
        self,
        itime_scheme: int = 4,
        iadv_rhotheta: int = 2,
        igradp_method: int = 3,
        ndyn_substeps_var: float = 5.0,
        rayleigh_type: int = 2,
        rayleigh_coeff: float = 0.05,
        divdamp_order: int = 24,  # the ICON default is 4,
        is_iau_active: bool = False,
        iau_wgt_dyn: float = 0.0,
        divdamp_type: int = 3,
        divdamp_trans_start: float = 12500.0,
        divdamp_trans_end: float = 17500.0,
        l_vert_nested: bool = False,
        rhotheta_offctr: float = -0.1,
        veladv_offctr: float = 0.25,
        max_nudging_coeff: float = 0.02,
        divdamp_fac: float = 0.0025,
        divdamp_fac2: float = 0.004,
        divdamp_fac3: float = 0.004,
        divdamp_fac4: float = 0.004,
        divdamp_z: float = 32500.0,
        divdamp_z2: float = 40000.0,
        divdamp_z3: float = 60000.0,
        divdamp_z4: float = 80000.0,
        htop_moist_proc: float = 22500.0,
    ):
        # parameters from namelist diffusion_nml
        self.itime_scheme: int = itime_scheme

        #: Miura scheme for advection of rho and theta
        self.iadv_rhotheta: int = iadv_rhotheta
        #: Use truly horizontal pressure-gradient computation to ensure numerical
        #: stability without heavy orography smoothing
        self.igradp_method: int = igradp_method

        #: number of dynamics substeps per fast-physics timestep
        self.ndyn_substeps_var = ndyn_substeps_var

        #: type of Rayleigh damping
        self.rayleigh_type: int = rayleigh_type
        # used for calculation of rayleigh_w, rayleigh_vn in mo_vertical_grid.f90
        self.rayleigh_coeff: float = rayleigh_coeff

        #: order of divergence damping
        self.divdamp_order: int = divdamp_order

        #: type of divergence damping
        self.divdamp_type: int = divdamp_type
        #: Lower and upper bound of transition zone between 2D and 3D divergence damping in case of divdamp_type = 32 [m]
        self.divdamp_trans_start: float = divdamp_trans_start
        self.divdamp_trans_end: float = divdamp_trans_end

        #: off-centering for density and potential temperature at interface levels.
        #: Specifying a negative value here reduces the amount of vertical
        #: wind off-centering needed for stability of sound waves.
        self.rhotheta_offctr: float = rhotheta_offctr

        #: off-centering of velocity advection in corrector step
        self.veladv_offctr: float = veladv_offctr

        #: scaling factor for divergence damping
        self.divdamp_fac: float = divdamp_fac
        self.divdamp_fac2: float = divdamp_fac2
        self.divdamp_fac3: float = divdamp_fac3
        self.divdamp_fac4: float = divdamp_fac4
        self.divdamp_z: float = divdamp_z
        self.divdamp_z2: float = divdamp_z2
        self.divdamp_z3: float = divdamp_z3
        self.divdamp_z4: float = divdamp_z4

        #: height [m] where moist physics is turned off
        self.htop_moist_proc: float = htop_moist_proc

        #: parameters from other namelists:

        #: from mo_interpol_nml.f90
        self.nudge_max_coeff: float = max_nudging_coeff

        #: from mo_run_nml.f90
        #: use vertical nesting
        self.l_vert_nested: bool = l_vert_nested

        #: from mo_initicon_nml.f90/ mo_initicon_config.f90
        #: whether IAU is active at current time
        self.is_iau_active: bool = is_iau_active
        #: IAU weight for dynamics fields
        self.iau_wgt_dyn: float = iau_wgt_dyn

        self._validate()

    def _validate(self):
        """Apply consistency checks and validation on configuration parameters."""

        if self.l_vert_nested:
            raise NotImplementedError("Vertical nesting support not implemented")

        if self.igradp_method != 3:
            raise NotImplementedError("igradp_method can only be 3")

        if self.itime_scheme != 4:
            raise NotImplementedError("itime_scheme can only be 4")

        if self.divdamp_order != 24:
            raise NotImplementedError("divdamp_order can only be 24")

        if self.divdamp_type == 32:
            raise NotImplementedError("divdamp_type with value 32 not yet implemented")


class NonHydrostaticParams:
    """Calculates derived quantities depending on the NonHydrostaticConfig."""

    def __init__(self, config: NonHydrostaticConfig):
        self.rd_o_cvd: Final[float] = constants.RD / constants.CVD
        self.cvd_o_rd: Final[float] = constants.CVD / constants.RD
        self.rd_o_p0ref: Final[float] = constants.RD / constants.P0REF
        self.grav_o_cpd: Final[float] = constants.GRAV / constants.CPD

        #:  start level for 3D divergence damping terms
        #: this is only different from 0 if divdamp_type == 32: calculation done in mo_vertical_grid.f90
        self.kstart_dd3d: Final[int] = 0

        #: Weighting coefficients for velocity advection if tendency averaging is used
        #: The off-centering specified here turned out to be beneficial to numerical
        #: stability in extreme situations
        self.wgt_nnow_vel: Final[float] = 0.5 - config.veladv_offctr
        self.wgt_nnew_vel: Final[float] = 0.5 + config.veladv_offctr

        #: Weighting coefficients for rho and theta at interface levels in the corrector step
        #: This empirically determined weighting minimizes the vertical wind off-centering
        #: needed for numerical stability of vertical sound wave propagation
        self.wgt_nnew_rth: Final[float] = 0.5 + config.rhotheta_offctr
        self.wgt_nnow_rth: Final[float] = 1.0 - self.wgt_nnew_rth


class SolveNonhydro:
    def __init__(self, exchange: ExchangeRuntime = SingleNodeExchange()):
        self._exchange = exchange
        self._initialized = False
        self.grid: Optional[IconGrid] = None
        self.config: Optional[NonHydrostaticConfig] = None
        self.params: Optional[NonHydrostaticParams] = None
        self.metric_state_nonhydro: Optional[MetricStateNonHydro] = None
        self.interpolation_state: Optional[InterpolationState] = None
        self.vertical_params: Optional[VerticalModelParams] = None
        self.edge_geometry: Optional[EdgeParams] = None
        self.cell_params: Optional[CellParams] = None
        self.velocity_advection: Optional[VelocityAdvection] = None
        self.l_vert_nested: bool = False
        self.enh_divdamp_fac: Optional[Field[[KDim], float]] = None
        self.scal_divdamp: Optional[Field[[KDim], float]] = None
        self._bdy_divdamp: Optional[Field[[KDim], float]] = None
        self.p_test_run = True
        self.jk_start = 0  # used in stencil_55
        self.ntl1 = 0
        self.ntl2 = 0

    def init(
        self,
        grid: IconGrid,
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        metric_state_nonhydro: MetricStateNonHydro,
        interpolation_state: InterpolationState,
        vertical_params: VerticalModelParams,
        edge_geometry: EdgeParams,
        cell_geometry: CellParams,
        owner_mask: fa.CboolField,
    ):
        """
        Initialize NonHydrostatic granule with configuration.

        calculates all local fields that are used in nh_solve within the time loop
        """
        self.grid = grid
        self.config: NonHydrostaticConfig = config
        self.params: NonHydrostaticParams = params
        self.metric_state_nonhydro: MetricStateNonHydro = metric_state_nonhydro
        self.interpolation_state: InterpolationState = interpolation_state
        self.vertical_params = vertical_params
        self.edge_geometry = edge_geometry
        self.cell_params = cell_geometry
        self.velocity_advection = VelocityAdvection(
            grid,
            metric_state_nonhydro,
            interpolation_state,
            vertical_params,
            edge_geometry,
            owner_mask,
        )
        self._allocate_local_fields()

        # TODO (magdalena) vertical nesting is only relevant in the context of
        #      horizontal nesting, since we don't support this we should remove this option
        if grid.lvert_nest:
            self.l_vert_nested = True
            self.jk_start = 1
        else:
            self.jk_start = 0

        en_smag_fac_for_zero_nshift(
            self.vertical_params.vct_a,
            self.config.divdamp_fac,
            self.config.divdamp_fac2,
            self.config.divdamp_fac3,
            self.config.divdamp_fac4,
            self.config.divdamp_z,
            self.config.divdamp_z2,
            self.config.divdamp_z3,
            self.config.divdamp_z4,
            self.enh_divdamp_fac,
            offset_provider={"Koff": KDim},
        )

        self.p_test_run = True
        self._initialized = True

    @property
    def initialized(self):
        return self._initialized

    def _allocate_local_fields(self):
        self.z_exner_ex_pr = _allocate(CellDim, KDim, is_halfdim=True, grid=self.grid)
        self.z_exner_ic = _allocate(CellDim, KDim, is_halfdim=True, grid=self.grid)
        self.z_dexner_dz_c_1 = _allocate(CellDim, KDim, grid=self.grid)
        self.z_theta_v_pr_ic = _allocate(CellDim, KDim, is_halfdim=True, grid=self.grid)
        self.z_th_ddz_exner_c = _allocate(CellDim, KDim, grid=self.grid)
        self.z_rth_pr_1 = _allocate(CellDim, KDim, grid=self.grid)
        self.z_rth_pr_2 = _allocate(CellDim, KDim, grid=self.grid)
        self.z_grad_rth_1 = _allocate(CellDim, KDim, grid=self.grid)
        self.z_grad_rth_2 = _allocate(CellDim, KDim, grid=self.grid)
        self.z_grad_rth_3 = _allocate(CellDim, KDim, grid=self.grid)
        self.z_grad_rth_4 = _allocate(CellDim, KDim, grid=self.grid)
        self.z_dexner_dz_c_2 = _allocate(CellDim, KDim, grid=self.grid)
        self.z_hydro_corr = _allocate(EdgeDim, KDim, grid=self.grid)
        self.z_vn_avg = _allocate(EdgeDim, KDim, grid=self.grid)
        self.z_theta_v_fl_e = _allocate(EdgeDim, KDim, grid=self.grid)
        self.z_flxdiv_mass = _allocate(CellDim, KDim, grid=self.grid)
        self.z_flxdiv_theta = _allocate(CellDim, KDim, grid=self.grid)
        self.z_rho_v = _allocate(VertexDim, KDim, grid=self.grid)
        self.z_theta_v_v = _allocate(VertexDim, KDim, grid=self.grid)
        self.z_graddiv2_vn = _allocate(EdgeDim, KDim, grid=self.grid)
        self.k_field = _allocate_indices(KDim, grid=self.grid, is_halfdim=True)
        self.z_w_concorr_me = _allocate(EdgeDim, KDim, grid=self.grid)
        self.z_hydro_corr_horizontal = _allocate(EdgeDim, grid=self.grid)
        self.z_raylfac = _allocate(KDim, grid=self.grid)
        self.enh_divdamp_fac = _allocate(KDim, grid=self.grid)
        self._bdy_divdamp = _allocate(KDim, grid=self.grid)
        self.scal_divdamp = _allocate(KDim, grid=self.grid)
        self.intermediate_fields = IntermediateFields.allocate(self.grid)

    def set_timelevels(self, nnow, nnew):
        #  Set time levels of ddt_adv fields for call to velocity_tendencies
        if self.config.itime_scheme == 4:
            self.ntl1 = nnow
            self.ntl2 = nnew
        else:
            self.ntl1 = 0
            self.ntl2 = 0

    def time_step(
        self,
        diagnostic_state_nh: DiagnosticStateNonHydro,
        prognostic_state_ls: list[PrognosticState],
        prep_adv: PrepAdvection,
        divdamp_fac_o2: float,
        dtime: float,
        l_recompute: bool,
        l_init: bool,
        nnow: int,
        nnew: int,
        lclean_mflx: bool,
        lprep_adv: bool,
        at_first_substep: bool,
        at_last_substep: bool,
    ):
        log.info(
            f"running timestep: dtime = {dtime}, init = {l_init}, recompute = {l_recompute}, prep_adv = {lprep_adv}  clean_mflx={lclean_mflx} "
        )
        start_cell_lb = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim)
        )
        end_cell_end = self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim))
        start_edge_lb = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim)
        )
        end_edge_local = self.grid.get_end_index(EdgeDim, HorizontalMarkerIndex.local(EdgeDim))
        # # TODO: abishekg7 move this to tests
        if self.p_test_run:
            nhsolve_prog.init_test_fields(
                self.intermediate_fields.z_rho_e,
                self.intermediate_fields.z_theta_v_e,
                self.intermediate_fields.z_dwdz_dd,
                self.intermediate_fields.z_graddiv_vn,
                start_edge_lb,
                end_edge_local,
                start_cell_lb,
                end_cell_end,
                self.grid.num_levels,
                offset_provider={},
            )

        self.set_timelevels(nnow, nnew)

        self.run_predictor_step(
            diagnostic_state_nh=diagnostic_state_nh,
            prognostic_state=prognostic_state_ls,
            z_fields=self.intermediate_fields,
            dtime=dtime,
            l_recompute=l_recompute,
            l_init=l_init,
            at_first_substep=at_first_substep,
            nnow=nnow,
            nnew=nnew,
        )

        self.run_corrector_step(
            diagnostic_state_nh=diagnostic_state_nh,
            prognostic_state=prognostic_state_ls,
            z_fields=self.intermediate_fields,
            prep_adv=prep_adv,
            divdamp_fac_o2=divdamp_fac_o2,
            dtime=dtime,
            nnew=nnew,
            nnow=nnow,
            lclean_mflx=lclean_mflx,
            lprep_adv=lprep_adv,
            at_last_substep=at_last_substep,
        )

        start_cell_lb = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim)
        )
        end_cell_nudging_minus1 = self.grid.get_end_index(
            CellDim, HorizontalMarkerIndex.nudging(CellDim) - 1
        )
        start_cell_halo = self.grid.get_start_index(CellDim, HorizontalMarkerIndex.halo(CellDim))
        end_cell_end = self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim))
        if self.grid.limited_area:
            compute_theta_and_exner(
                bdy_halo_c=self.metric_state_nonhydro.bdy_halo_c,
                rho=prognostic_state_ls[nnew].rho,
                theta_v=prognostic_state_ls[nnew].theta_v,
                exner=prognostic_state_ls[nnew].exner,
                rd_o_cvd=self.params.rd_o_cvd,
                rd_o_p0ref=self.params.rd_o_p0ref,
                horizontal_start=0,
                horizontal_end=end_cell_end,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

            compute_exner_from_rhotheta(
                rho=prognostic_state_ls[nnew].rho,
                theta_v=prognostic_state_ls[nnew].theta_v,
                exner=prognostic_state_ls[nnew].exner,
                rd_o_cvd=self.params.rd_o_cvd,
                rd_o_p0ref=self.params.rd_o_p0ref,
                horizontal_start=start_cell_lb,
                horizontal_end=end_cell_nudging_minus1,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        update_theta_v(
            mask_prog_halo_c=self.metric_state_nonhydro.mask_prog_halo_c,
            rho_now=prognostic_state_ls[nnow].rho,
            theta_v_now=prognostic_state_ls[nnow].theta_v,
            exner_new=prognostic_state_ls[nnew].exner,
            exner_now=prognostic_state_ls[nnow].exner,
            rho_new=prognostic_state_ls[nnew].rho,
            theta_v_new=prognostic_state_ls[nnew].theta_v,
            cvd_o_rd=self.params.cvd_o_rd,
            horizontal_start=start_cell_halo,
            horizontal_end=end_cell_end,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

    # flake8: noqa: C901
    def run_predictor_step(
        self,
        diagnostic_state_nh: DiagnosticStateNonHydro,
        prognostic_state: list[PrognosticState],
        z_fields: IntermediateFields,
        dtime: float,
        l_recompute: bool,
        l_init: bool,
        at_first_substep: bool,
        nnow: int,
        nnew: int,
    ):
        log.info(
            f"running predictor step: dtime = {dtime}, init = {l_init}, recompute = {l_recompute} "
        )
        if l_init or l_recompute:
            if self.config.itime_scheme == 4 and not l_init:
                lvn_only = True  # Recompute only vn tendency
            else:
                lvn_only = False

            self.velocity_advection.run_predictor_step(
                vn_only=lvn_only,
                diagnostic_state=diagnostic_state_nh,
                prognostic_state=prognostic_state[nnow],
                z_w_concorr_me=self.z_w_concorr_me,
                z_kin_hor_e=z_fields.z_kin_hor_e,
                z_vt_ie=z_fields.z_vt_ie,
                dtime=dtime,
                ntnd=self.ntl1,
                cell_areas=self.cell_params.area,
            )

        p_dthalf = 0.5 * dtime

        end_cell_end = self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim))

        start_cell_local_minus2 = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.local(CellDim) - 2
        )
        end_cell_local_minus2 = self.grid.get_end_index(
            CellDim, HorizontalMarkerIndex.local(CellDim) - 2
        )

        start_vertex_lb_plus1 = self.grid.get_start_index(
            VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1
        )  # TODO: check
        end_vertex_local_minus1 = self.grid.get_end_index(
            VertexDim, HorizontalMarkerIndex.local(VertexDim) - 1
        )

        start_cell_lb = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim)
        )
        end_cell_nudging_minus1 = self.grid.get_end_index(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim) - 1,
        )

        start_edge_lb_plus6 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6
        )
        end_edge_local_minus1 = self.grid.get_end_index(
            EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 1
        )
        end_edge_local = self.grid.get_end_index(EdgeDim, HorizontalMarkerIndex.local(EdgeDim))

        start_edge_nudging_plus1 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim) + 1
        )
        end_edge_end = self.grid.get_end_index(EdgeDim, HorizontalMarkerIndex.end(EdgeDim))

        start_edge_lb = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim)
        )
        end_edge_nudging = self.grid.get_end_index(EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim))

        start_edge_lb_plus4 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4
        )
        start_edge_local_minus2 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 2
        )
        end_edge_local_minus2 = self.grid.get_end_index(
            EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 2
        )

        start_cell_lb_plus2 = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 2
        )

        end_cell_halo = self.grid.get_end_index(CellDim, HorizontalMarkerIndex.halo(CellDim))
        start_cell_nudging = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.nudging(CellDim)
        )
        end_cell_local = self.grid.get_end_index(CellDim, HorizontalMarkerIndex.local(CellDim))

        #  Precompute Rayleigh damping factor
        compute_z_raylfac(
            rayleigh_w=self.metric_state_nonhydro.rayleigh_w,
            dtime=dtime,
            z_raylfac=self.z_raylfac,
            offset_provider={},
        )

        # initialize nest boundary points of z_rth_pr with zero
        if self.grid.limited_area:
            init_two_cell_kdim_fields_with_zero_vp(
                cell_kdim_field_with_zero_vp_1=self.z_rth_pr_1,
                cell_kdim_field_with_zero_vp_2=self.z_rth_pr_2,
                horizontal_start=start_cell_lb,
                horizontal_end=end_cell_end,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        nhsolve_prog.predictor_stencils_2_3(
            exner_exfac=self.metric_state_nonhydro.exner_exfac,
            exner=prognostic_state[nnow].exner,
            exner_ref_mc=self.metric_state_nonhydro.exner_ref_mc,
            exner_pr=diagnostic_state_nh.exner_pr,
            z_exner_ex_pr=self.z_exner_ex_pr,
            horizontal_start=start_cell_lb_plus2,
            horizontal_end=end_cell_halo,
            k_field=self.k_field,
            nlev=self.grid.num_levels,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider={},
        )

        if self.config.igradp_method == 3:
            nhsolve_prog.predictor_stencils_4_5_6(
                wgtfacq_c_dsl=self.metric_state_nonhydro.wgtfacq_c,
                z_exner_ex_pr=self.z_exner_ex_pr,
                z_exner_ic=self.z_exner_ic,
                wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                z_dexner_dz_c_1=self.z_dexner_dz_c_1,
                k_field=self.k_field,
                nlev=self.grid.num_levels,
                horizontal_start=start_cell_lb_plus2,
                horizontal_end=end_cell_halo,
                vertical_start=max(1, self.vertical_params.nflatlev),
                vertical_end=self.grid.num_levels + 1,
                offset_provider=self.grid.offset_providers,
            )

            if self.vertical_params.nflatlev == 1:
                # Perturbation Exner pressure on top half level
                raise NotImplementedError("nflatlev=1 not implemented")

        nhsolve_prog.predictor_stencils_7_8_9(
            rho=prognostic_state[nnow].rho,
            rho_ref_mc=self.metric_state_nonhydro.rho_ref_mc,
            theta_v=prognostic_state[nnow].theta_v,
            theta_ref_mc=self.metric_state_nonhydro.theta_ref_mc,
            rho_ic=diagnostic_state_nh.rho_ic,
            z_rth_pr_1=self.z_rth_pr_1,
            z_rth_pr_2=self.z_rth_pr_2,
            wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
            vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
            exner_pr=diagnostic_state_nh.exner_pr,
            d_exner_dz_ref_ic=self.metric_state_nonhydro.d_exner_dz_ref_ic,
            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
            z_theta_v_pr_ic=self.z_theta_v_pr_ic,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            z_th_ddz_exner_c=self.z_th_ddz_exner_c,
            k_field=self.k_field,
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_lb_plus2,
            horizontal_end=end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        # Perturbation theta at top and surface levels
        nhsolve_prog.predictor_stencils_11_lower_upper(
            wgtfacq_c_dsl=self.metric_state_nonhydro.wgtfacq_c,
            z_rth_pr=self.z_rth_pr_2,
            theta_ref_ic=self.metric_state_nonhydro.theta_ref_ic,
            z_theta_v_pr_ic=self.z_theta_v_pr_ic,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            k_field=self.k_field,
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_lb_plus2,
            horizontal_end=end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        if self.config.igradp_method == 3:
            # Second vertical derivative of perturbation Exner pressure (hydrostatic approximation)
            compute_approx_of_2nd_vertical_derivative_of_exner(
                z_theta_v_pr_ic=self.z_theta_v_pr_ic,
                d2dexdz2_fac1_mc=self.metric_state_nonhydro.d2dexdz2_fac1_mc,
                d2dexdz2_fac2_mc=self.metric_state_nonhydro.d2dexdz2_fac2_mc,
                z_rth_pr_2=self.z_rth_pr_2,
                z_dexner_dz_c_2=self.z_dexner_dz_c_2,
                horizontal_start=start_cell_lb_plus2,
                horizontal_end=end_cell_halo,
                vertical_start=self.vertical_params.nflat_gradp,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        # Add computation of z_grad_rth (perturbation density and virtual potential temperature at main levels)
        # at outer halo points: needed for correct calculation of the upwind gradients for Miura scheme

        compute_perturbation_of_rho_and_theta(
            rho=prognostic_state[nnow].rho,
            rho_ref_mc=self.metric_state_nonhydro.rho_ref_mc,
            theta_v=prognostic_state[nnow].theta_v,
            theta_ref_mc=self.metric_state_nonhydro.theta_ref_mc,
            z_rth_pr_1=self.z_rth_pr_1,
            z_rth_pr_2=self.z_rth_pr_2,
            horizontal_start=start_cell_local_minus2,
            horizontal_end=end_cell_local_minus2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        # Compute rho and theta at edges for horizontal flux divergence term
        if self.config.iadv_rhotheta == 1:
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                p_cell_in=prognostic_state[nnow].rho,
                c_intp=self.interpolation_state.c_intp,
                p_vert_out=self.z_rho_v,
                horizontal_start=start_vertex_lb_plus1,
                horizontal_end=end_vertex_local_minus1,
                vertical_start=0,
                vertical_end=self.grid.num_levels,  # UBOUND(p_cell_in,2)
                offset_provider=self.grid.offset_providers,
            )
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                p_cell_in=prognostic_state[nnow].theta_v,
                c_intp=self.interpolation_state.c_intp,
                p_vert_out=self.z_theta_v_v,
                horizontal_start=start_vertex_lb_plus1,
                horizontal_end=end_vertex_local_minus1,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
        elif self.config.iadv_rhotheta == 2:
            # Compute Green-Gauss gradients for rho and theta
            mo_math_gradients_grad_green_gauss_cell_dsl(
                p_grad_1_u=self.z_grad_rth_1,
                p_grad_1_v=self.z_grad_rth_2,
                p_grad_2_u=self.z_grad_rth_3,
                p_grad_2_v=self.z_grad_rth_4,
                p_ccpr1=self.z_rth_pr_1,
                p_ccpr2=self.z_rth_pr_2,
                geofac_grg_x=self.interpolation_state.geofac_grg_x,
                geofac_grg_y=self.interpolation_state.geofac_grg_y,
                horizontal_start=start_cell_lb_plus2,
                horizontal_end=end_cell_halo,
                vertical_start=0,
                vertical_end=self.grid.num_levels,  # UBOUND(p_ccpr,2)
                offset_provider=self.grid.offset_providers,
            )
        if self.config.iadv_rhotheta <= 2:
            init_two_edge_kdim_fields_with_zero_wp(
                edge_kdim_field_with_zero_wp_1=z_fields.z_rho_e,
                edge_kdim_field_with_zero_wp_2=z_fields.z_theta_v_e,
                horizontal_start=start_edge_local_minus2,
                horizontal_end=end_edge_local_minus2,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
            # initialize also nest boundary points with zero
            if self.grid.limited_area:
                init_two_edge_kdim_fields_with_zero_wp(
                    edge_kdim_field_with_zero_wp_1=z_fields.z_rho_e,
                    edge_kdim_field_with_zero_wp_2=z_fields.z_theta_v_e,
                    horizontal_start=start_edge_lb,
                    horizontal_end=end_edge_local_minus1,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )
            if self.config.iadv_rhotheta == 2:
                # Compute upwind-biased values for rho and theta starting from centered differences
                # Note: the length of the backward trajectory should be 0.5*dtime*(vn,vt) in order to arrive
                # at a second-order accurate FV discretization, but twice the length is needed for numerical stability

                nhsolve_prog.compute_horizontal_advection_of_rho_and_theta(
                    p_vn=prognostic_state[nnow].vn,
                    p_vt=diagnostic_state_nh.vt,
                    pos_on_tplane_e_1=self.interpolation_state.pos_on_tplane_e_1,
                    pos_on_tplane_e_2=self.interpolation_state.pos_on_tplane_e_2,
                    primal_normal_cell_1=self.edge_geometry.primal_normal_cell[0],
                    dual_normal_cell_1=self.edge_geometry.dual_normal_cell[0],
                    primal_normal_cell_2=self.edge_geometry.primal_normal_cell[1],
                    dual_normal_cell_2=self.edge_geometry.dual_normal_cell[1],
                    p_dthalf=p_dthalf,
                    rho_ref_me=self.metric_state_nonhydro.rho_ref_me,
                    theta_ref_me=self.metric_state_nonhydro.theta_ref_me,
                    z_grad_rth_1=self.z_grad_rth_1,
                    z_grad_rth_2=self.z_grad_rth_2,
                    z_grad_rth_3=self.z_grad_rth_3,
                    z_grad_rth_4=self.z_grad_rth_4,
                    z_rth_pr_1=self.z_rth_pr_1,
                    z_rth_pr_2=self.z_rth_pr_2,
                    z_rho_e=z_fields.z_rho_e,
                    z_theta_v_e=z_fields.z_theta_v_e,
                    horizontal_start=start_edge_lb_plus6,
                    horizontal_end=end_edge_local_minus1,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider=self.grid.offset_providers,
                )

        # Remaining computations at edge points
        compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates(
            inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
            z_exner_ex_pr=self.z_exner_ex_pr,
            z_gradh_exner=z_fields.z_gradh_exner,
            horizontal_start=start_edge_nudging_plus1,
            horizontal_end=end_edge_local,
            vertical_start=0,
            vertical_end=self.vertical_params.nflatlev,
            offset_provider=self.grid.offset_providers,
        )

        if self.config.igradp_method == 3:
            # horizontal gradient of Exner pressure, including metric correction
            # horizontal gradient of Exner pressure, Taylor-expansion-based reconstruction

            compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates(
                inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                z_exner_ex_pr=self.z_exner_ex_pr,
                ddxn_z_full=self.metric_state_nonhydro.ddxn_z_full,
                c_lin_e=self.interpolation_state.c_lin_e,
                z_dexner_dz_c_1=self.z_dexner_dz_c_1,
                z_gradh_exner=z_fields.z_gradh_exner,
                horizontal_start=start_edge_nudging_plus1,
                horizontal_end=end_edge_local,
                vertical_start=self.vertical_params.nflatlev,
                vertical_end=int32(self.vertical_params.nflat_gradp + 1),
                offset_provider=self.grid.offset_providers,
            )

            compute_horizontal_gradient_of_exner_pressure_for_multiple_levels(
                inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                z_exner_ex_pr=self.z_exner_ex_pr,
                zdiff_gradp=self.metric_state_nonhydro.zdiff_gradp,
                ikoffset=self.metric_state_nonhydro.vertoffset_gradp,
                z_dexner_dz_c_1=self.z_dexner_dz_c_1,
                z_dexner_dz_c_2=self.z_dexner_dz_c_2,
                z_gradh_exner=z_fields.z_gradh_exner,
                horizontal_start=start_edge_nudging_plus1,
                horizontal_end=end_edge_local,
                vertical_start=int32(self.vertical_params.nflat_gradp + 1),
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
        # compute hydrostatically approximated correction term that replaces downward extrapolation
        if self.config.igradp_method == 3:
            compute_hydrostatic_correction_term(
                theta_v=prognostic_state[nnow].theta_v,
                ikoffset=self.metric_state_nonhydro.vertoffset_gradp,
                zdiff_gradp=self.metric_state_nonhydro.zdiff_gradp,
                theta_v_ic=diagnostic_state_nh.theta_v_ic,
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                z_hydro_corr=self.z_hydro_corr,
                grav_o_cpd=self.params.grav_o_cpd,
                horizontal_start=start_edge_nudging_plus1,
                horizontal_end=end_edge_local,
                vertical_start=self.grid.num_levels - 1,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
        # TODO (Nikki) check when merging fused stencil
        lowest_level = self.grid.num_levels - 1
        hydro_corr_horizontal = as_field((EdgeDim,), self.z_hydro_corr.asnumpy()[:, lowest_level])

        if self.config.igradp_method == 3:
            apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure(
                ipeidx_dsl=self.metric_state_nonhydro.ipeidx_dsl,
                pg_exdist=self.metric_state_nonhydro.pg_exdist,
                z_hydro_corr=hydro_corr_horizontal,
                z_gradh_exner=z_fields.z_gradh_exner,
                horizontal_start=start_edge_nudging_plus1,
                horizontal_end=end_edge_end,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        add_temporal_tendencies_to_vn(
            vn_nnow=prognostic_state[nnow].vn,
            ddt_vn_apc_ntl1=diagnostic_state_nh.ddt_vn_apc_pc[self.ntl1],
            ddt_vn_phy=diagnostic_state_nh.ddt_vn_phy,
            z_theta_v_e=z_fields.z_theta_v_e,
            z_gradh_exner=z_fields.z_gradh_exner,
            vn_nnew=prognostic_state[nnew].vn,
            dtime=dtime,
            cpd=constants.CPD,
            horizontal_start=start_edge_nudging_plus1,
            horizontal_end=end_edge_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        if self.config.is_iau_active:
            add_analysis_increments_to_vn(
                vn_incr=diagnostic_state_nh.vn_incr,
                vn=prognostic_state[nnew].vn,
                iau_wgt_dyn=self.config.iau_wgt_dyn,
                horizontal_start=start_edge_nudging_plus1,
                horizontal_end=end_edge_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        if self.grid.limited_area:
            compute_vn_on_lateral_boundary(
                grf_tend_vn=diagnostic_state_nh.grf_tend_vn,
                vn_now=prognostic_state[nnow].vn,
                vn_new=prognostic_state[nnew].vn,
                dtime=dtime,
                horizontal_start=start_edge_lb,
                horizontal_end=end_edge_nudging,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
        log.debug("exchanging prognostic field 'vn' and local field 'z_rho_e'")
        self._exchange.exchange_and_wait(EdgeDim, prognostic_state[nnew].vn, z_fields.z_rho_e)

        compute_avg_vn_and_graddiv_vn_and_vt(
            e_flx_avg=self.interpolation_state.e_flx_avg,
            vn=prognostic_state[nnew].vn,
            geofac_grdiv=self.interpolation_state.geofac_grdiv,
            rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
            z_vn_avg=self.z_vn_avg,
            z_graddiv_vn=z_fields.z_graddiv_vn,
            vt=diagnostic_state_nh.vt,
            horizontal_start=start_edge_lb_plus4,
            horizontal_end=end_edge_local_minus2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        compute_mass_flux(
            z_rho_e=z_fields.z_rho_e,
            z_vn_avg=self.z_vn_avg,
            ddqz_z_full_e=self.metric_state_nonhydro.ddqz_z_full_e,
            z_theta_v_e=z_fields.z_theta_v_e,
            mass_fl_e=diagnostic_state_nh.mass_fl_e,
            z_theta_v_fl_e=self.z_theta_v_fl_e,
            horizontal_start=start_edge_lb_plus4,
            horizontal_end=end_edge_local_minus2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        nhsolve_prog.predictor_stencils_35_36(
            vn=prognostic_state[nnew].vn,
            ddxn_z_full=self.metric_state_nonhydro.ddxn_z_full,
            ddxt_z_full=self.metric_state_nonhydro.ddxt_z_full,
            vt=diagnostic_state_nh.vt,
            z_w_concorr_me=self.z_w_concorr_me,
            wgtfac_e=self.metric_state_nonhydro.wgtfac_e,
            vn_ie=diagnostic_state_nh.vn_ie,
            z_vt_ie=z_fields.z_vt_ie,
            z_kin_hor_e=z_fields.z_kin_hor_e,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            horizontal_start=start_edge_lb_plus4,
            horizontal_end=end_edge_local_minus2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        if not self.l_vert_nested:
            nhsolve_prog.predictor_stencils_37_38(
                vn=prognostic_state[nnew].vn,
                vt=diagnostic_state_nh.vt,
                vn_ie=diagnostic_state_nh.vn_ie,
                z_vt_ie=z_fields.z_vt_ie,
                z_kin_hor_e=z_fields.z_kin_hor_e,
                wgtfacq_e_dsl=self.metric_state_nonhydro.wgtfacq_e,
                horizontal_start=start_edge_lb_plus4,
                horizontal_end=end_edge_local_minus2,
                vertical_start=0,
                vertical_end=self.grid.num_levels + 1,
                offset_provider=self.grid.offset_providers,
            )

        nhsolve_prog.stencils_39_40(
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            z_w_concorr_me=self.z_w_concorr_me,
            wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
            wgtfacq_c_dsl=self.metric_state_nonhydro.wgtfacq_c,
            w_concorr_c=diagnostic_state_nh.w_concorr_c,
            k_field=self.k_field,
            nflatlev_startindex_plus1=int32(self.vertical_params.nflatlev + 1),
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_lb_plus2,
            horizontal_end=end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        compute_divergence_of_fluxes_of_rho_and_theta(
            geofac_div=self.interpolation_state.geofac_div,
            mass_fl_e=diagnostic_state_nh.mass_fl_e,
            z_theta_v_fl_e=self.z_theta_v_fl_e,
            z_flxdiv_mass=self.z_flxdiv_mass,
            z_flxdiv_theta=self.z_flxdiv_theta,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        nhsolve_prog.stencils_43_44_45_45b(
            z_w_expl=z_fields.z_w_expl,
            w_nnow=prognostic_state[nnow].w,
            ddt_w_adv_ntl1=diagnostic_state_nh.ddt_w_adv_pc[self.ntl1],
            z_th_ddz_exner_c=self.z_th_ddz_exner_c,
            z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
            rho_ic=diagnostic_state_nh.rho_ic,
            w_concorr_c=diagnostic_state_nh.w_concorr_c,
            vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
            z_beta=z_fields.z_beta,
            exner_nnow=prognostic_state[nnow].exner,
            rho_nnow=prognostic_state[nnow].rho,
            theta_v_nnow=prognostic_state[nnow].theta_v,
            inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
            z_alpha=z_fields.z_alpha,
            vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            z_q=z_fields.z_q,
            k_field=self.k_field,
            rd=constants.RD,
            cvd=constants.CVD,
            dtime=dtime,
            cpd=constants.CPD,
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider={},
        )

        if not self.l_vert_nested:
            init_two_cell_kdim_fields_with_zero_wp(
                cell_kdim_field_with_zero_wp_1=prognostic_state[nnew].w,
                cell_kdim_field_with_zero_wp_2=z_fields.z_contr_w_fl_l,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=0,
                vertical_end=1,
                offset_provider={},
            )
        nhsolve_prog.stencils_47_48_49(
            w_nnew=prognostic_state[nnew].w,
            z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
            w_concorr_c=diagnostic_state_nh.w_concorr_c,
            z_rho_expl=z_fields.z_rho_expl,
            z_exner_expl=z_fields.z_exner_expl,
            rho_nnow=prognostic_state[nnow].rho,
            inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
            z_flxdiv_mass=self.z_flxdiv_mass,
            exner_pr=diagnostic_state_nh.exner_pr,
            z_beta=z_fields.z_beta,
            z_flxdiv_theta=self.z_flxdiv_theta,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            ddt_exner_phy=diagnostic_state_nh.ddt_exner_phy,
            k_field=self.k_field,
            dtime=dtime,
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        if self.config.is_iau_active:
            add_analysis_increments_from_data_assimilation(
                z_fields.z_rho_expl,
                z_fields.z_exner_expl,
                diagnostic_state_nh.rho_incr,
                diagnostic_state_nh.exner_incr,
                self.config.iau_wgt_dyn,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        solve_tridiagonal_matrix_for_w_forward_sweep(
            vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
            z_alpha=z_fields.z_alpha,
            z_beta=z_fields.z_beta,
            z_w_expl=z_fields.z_w_expl,
            z_exner_expl=z_fields.z_exner_expl,
            z_q=z_fields.z_q,
            w=prognostic_state[nnew].w,
            dtime=dtime,
            cpd=constants.CPD,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        solve_tridiagonal_matrix_for_w_back_substitution(
            z_q=z_fields.z_q,
            w=prognostic_state[nnew].w,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        if self.config.rayleigh_type == constants.RayleighType.RAYLEIGH_KLEMP:
            apply_rayleigh_damping_mechanism(
                z_raylfac=self.z_raylfac,
                w_1=prognostic_state[nnew].w_1,
                w=prognostic_state[nnew].w,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=1,
                vertical_end=int32(
                    self.vertical_params.index_of_damping_layer + 1
                ),  # +1 since Fortran includes boundaries
                offset_provider={},
            )

        compute_results_for_thermodynamic_variables(
            z_rho_expl=z_fields.z_rho_expl,
            vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
            inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
            rho_ic=diagnostic_state_nh.rho_ic,
            w=prognostic_state[nnew].w,
            z_exner_expl=z_fields.z_exner_expl,
            exner_ref_mc=self.metric_state_nonhydro.exner_ref_mc,
            z_alpha=z_fields.z_alpha,
            z_beta=z_fields.z_beta,
            rho_now=prognostic_state[nnow].rho,
            theta_v_now=prognostic_state[nnow].theta_v,
            exner_now=prognostic_state[nnow].exner,
            rho_new=prognostic_state[nnew].rho,
            exner_new=prognostic_state[nnew].exner,
            theta_v_new=prognostic_state[nnew].theta_v,
            dtime=dtime,
            cvd_o_rd=constants.CVD_O_RD,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=int32(self.jk_start),
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        # compute dw/dz for divergence damping term
        if self.config.divdamp_type >= 3:
            compute_dwdz_for_divergence_damping(
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                w=prognostic_state[nnew].w,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                z_dwdz_dd=z_fields.z_dwdz_dd,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=self.params.kstart_dd3d,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        if at_first_substep:
            copy_cell_kdim_field_to_vp(
                field=prognostic_state[nnow].exner,
                field_copy=diagnostic_state_nh.exner_dyn_incr,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=self.vertical_params.kstart_moist,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        if self.grid.limited_area:
            nhsolve_prog.stencils_61_62(
                rho_now=prognostic_state[nnow].rho,
                grf_tend_rho=diagnostic_state_nh.grf_tend_rho,
                theta_v_now=prognostic_state[nnow].theta_v,
                grf_tend_thv=diagnostic_state_nh.grf_tend_thv,
                w_now=prognostic_state[nnow].w,
                grf_tend_w=diagnostic_state_nh.grf_tend_w,
                rho_new=prognostic_state[nnew].rho,
                exner_new=prognostic_state[nnew].exner,
                w_new=prognostic_state[nnew].w,
                k_field=self.k_field,
                dtime=dtime,
                nlev=self.grid.num_levels,
                horizontal_start=start_cell_lb,
                horizontal_end=end_cell_nudging_minus1,
                vertical_start=0,
                vertical_end=int32(self.grid.num_levels + 1),
                offset_provider={},
            )

        if self.config.divdamp_type >= 3:
            compute_dwdz_for_divergence_damping(
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                w=prognostic_state[nnew].w,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                z_dwdz_dd=z_fields.z_dwdz_dd,
                horizontal_start=start_cell_lb,
                horizontal_end=end_cell_nudging_minus1,
                vertical_start=self.params.kstart_dd3d,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug("exchanging prognostic field 'w' and local field 'z_dwdz_dd'")
            self._exchange.exchange_and_wait(CellDim, prognostic_state[nnew].w, z_fields.z_dwdz_dd)
        else:
            log.debug("exchanging prognostic field 'w'")
            self._exchange.exchange_and_wait(CellDim, prognostic_state[nnew].w)

    def run_corrector_step(
        self,
        diagnostic_state_nh: DiagnosticStateNonHydro,
        prognostic_state: list[PrognosticState],
        z_fields: IntermediateFields,
        divdamp_fac_o2: float,
        prep_adv: PrepAdvection,
        dtime: float,
        nnew: int,
        nnow: int,
        lclean_mflx: bool,
        lprep_adv: bool,
        at_last_substep: bool,
    ):
        log.info(
            f"running corrector step: dtime = {dtime}, prep_adv = {lprep_adv},  divdamp_fac_o2 = {divdamp_fac_o2} clean_mfxl= {lclean_mflx}  "
        )

        # TODO (magdalena) is it correct to to use a config parameter here? the actual number of substeps can vary dynmically...
        #                  should this config parameter exist at all in SolveNonHydro?
        # Inverse value of ndyn_substeps for tracer advection precomputations
        r_nsubsteps = 1.0 / self.config.ndyn_substeps_var

        # scaling factor for second-order divergence damping: divdamp_fac_o2*delta_x**2
        # delta_x**2 is approximated by the mean cell area
        # Coefficient for reduced fourth-order divergence d
        scal_divdamp_o2 = divdamp_fac_o2 * self.cell_params.mean_cell_area

        _calculate_divdamp_fields(
            self.enh_divdamp_fac,
            int32(self.config.divdamp_order),
            self.cell_params.mean_cell_area,
            divdamp_fac_o2,
            self.config.nudge_max_coeff,
            constants.dbl_eps,
            out=(self.scal_divdamp, self._bdy_divdamp),
            offset_provider={},
        )

        start_cell_lb_plus2 = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 2
        )

        start_edge_lb_plus6 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6
        )

        start_edge_nudging_plus1 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim) + 1
        )
        end_edge_local = self.grid.get_end_index(EdgeDim, HorizontalMarkerIndex.local(EdgeDim))

        start_edge_lb_plus4 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4
        )
        end_edge_local_minus2 = self.grid.get_end_index(
            EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 2
        )

        start_edge_lb = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim)
        )
        end_edge_end = self.grid.get_end_index(EdgeDim, HorizontalMarkerIndex.end(EdgeDim))

        start_cell_lb = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim)
        )
        end_cell_nudging = self.grid.get_end_index(CellDim, HorizontalMarkerIndex.nudging(CellDim))

        start_cell_nudging = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.nudging(CellDim)
        )
        end_cell_local = self.grid.get_end_index(CellDim, HorizontalMarkerIndex.local(CellDim))

        lvn_only = False
        log.debug(f"corrector run velocity advection")
        self.velocity_advection.run_corrector_step(
            vn_only=lvn_only,
            diagnostic_state=diagnostic_state_nh,
            prognostic_state=prognostic_state[nnew],
            z_kin_hor_e=z_fields.z_kin_hor_e,
            z_vt_ie=z_fields.z_vt_ie,
            dtime=dtime,
            ntnd=self.ntl2,
            cell_areas=self.cell_params.area,
        )

        nvar = nnew

        #  Precompute Rayleigh damping factor
        compute_z_raylfac(
            self.metric_state_nonhydro.rayleigh_w,
            dtime,
            self.z_raylfac,
            offset_provider={},
        )
        log.debug(f"corrector: start stencil 10")
        compute_rho_virtual_potential_temperatures_and_pressure_gradient(
            w=prognostic_state[nnew].w,
            w_concorr_c=diagnostic_state_nh.w_concorr_c,
            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
            rho_now=prognostic_state[nnow].rho,
            rho_var=prognostic_state[nvar].rho,
            theta_now=prognostic_state[nnow].theta_v,
            theta_var=prognostic_state[nvar].theta_v,
            wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
            theta_ref_mc=self.metric_state_nonhydro.theta_ref_mc,
            vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
            exner_pr=diagnostic_state_nh.exner_pr,
            d_exner_dz_ref_ic=self.metric_state_nonhydro.d_exner_dz_ref_ic,
            rho_ic=diagnostic_state_nh.rho_ic,
            z_theta_v_pr_ic=self.z_theta_v_pr_ic,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            z_th_ddz_exner_c=self.z_th_ddz_exner_c,
            dtime=dtime,
            wgt_nnow_rth=self.params.wgt_nnow_rth,
            wgt_nnew_rth=self.params.wgt_nnew_rth,
            horizontal_start=start_cell_lb_plus2,
            horizontal_end=end_cell_local,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        log.debug(f"corrector: start stencil 17")
        add_vertical_wind_derivative_to_divergence_damping(
            hmask_dd3d=self.metric_state_nonhydro.hmask_dd3d,
            scalfac_dd3d=self.metric_state_nonhydro.scalfac_dd3d,
            inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
            z_dwdz_dd=z_fields.z_dwdz_dd,
            z_graddiv_vn=z_fields.z_graddiv_vn,
            horizontal_start=start_edge_lb_plus6,
            horizontal_end=end_edge_local_minus2,
            vertical_start=self.params.kstart_dd3d,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        if self.config.itime_scheme == 4:
            log.debug(f"corrector: start stencil 23")
            add_temporal_tendencies_to_vn_by_interpolating_between_time_levels(
                vn_nnow=prognostic_state[nnow].vn,
                ddt_vn_apc_ntl1=diagnostic_state_nh.ddt_vn_apc_pc[self.ntl1],
                ddt_vn_apc_ntl2=diagnostic_state_nh.ddt_vn_apc_pc[self.ntl2],
                ddt_vn_phy=diagnostic_state_nh.ddt_vn_phy,
                z_theta_v_e=z_fields.z_theta_v_e,
                z_gradh_exner=z_fields.z_gradh_exner,
                vn_nnew=prognostic_state[nnew].vn,
                dtime=dtime,
                wgt_nnow_vel=self.params.wgt_nnow_vel,
                wgt_nnew_vel=self.params.wgt_nnew_vel,
                cpd=constants.CPD,
                horizontal_start=start_edge_nudging_plus1,
                horizontal_end=end_edge_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        if self.config.divdamp_order == 24 or self.config.divdamp_order == 4:
            # verified for e-10
            log.debug(f"corrector start stencil 25")
            compute_graddiv2_of_vn(
                geofac_grdiv=self.interpolation_state.geofac_grdiv,
                z_graddiv_vn=z_fields.z_graddiv_vn,
                z_graddiv2_vn=self.z_graddiv2_vn,
                horizontal_start=start_edge_nudging_plus1,
                horizontal_end=end_edge_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        if self.config.divdamp_order == 24 and scal_divdamp_o2 > 1.0e-6:
            log.debug(f"corrector: start stencil 26")
            apply_2nd_order_divergence_damping(
                z_graddiv_vn=z_fields.z_graddiv_vn,
                vn=prognostic_state[nnew].vn,
                scal_divdamp_o2=scal_divdamp_o2,
                horizontal_start=start_edge_nudging_plus1,
                horizontal_end=end_edge_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        # TODO: this does not get accessed in FORTRAN
        if self.config.divdamp_order == 24 and divdamp_fac_o2 <= 4 * self.config.divdamp_fac:
            if self.grid.limited_area:
                log.debug("corrector: start stencil 27")
                apply_weighted_2nd_and_4th_order_divergence_damping(
                    scal_divdamp=self.scal_divdamp,
                    bdy_divdamp=self._bdy_divdamp,
                    nudgecoeff_e=self.interpolation_state.nudgecoeff_e,
                    z_graddiv2_vn=self.z_graddiv2_vn,
                    vn=prognostic_state[nnew].vn,
                    horizontal_start=start_edge_nudging_plus1,
                    horizontal_end=end_edge_local,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )
            else:
                log.debug("corrector start stencil 4th order divdamp")
                apply_4th_order_divergence_damping(
                    scal_divdamp=self.scal_divdamp,
                    z_graddiv2_vn=self.z_graddiv2_vn,
                    vn=prognostic_state[nnew].vn,
                    horizontal_start=start_edge_nudging_plus1,
                    horizontal_end=end_edge_local,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )

        # TODO: this does not get accessed in FORTRAN
        if self.config.is_iau_active:
            log.debug("corrector start stencil 28")
            add_analysis_increments_to_vn(
                diagnostic_state_nh.vn_incr,
                prognostic_state[nnew].vn,
                self.config.iau_wgt_dyn,
                horizontal_start=start_edge_nudging_plus1,
                horizontal_end=end_edge_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
        log.debug("exchanging prognostic field 'vn'")
        self._exchange.exchange_and_wait(EdgeDim, (prognostic_state[nnew].vn))
        log.debug("corrector: start stencil 31")
        compute_avg_vn(
            e_flx_avg=self.interpolation_state.e_flx_avg,
            vn=prognostic_state[nnew].vn,
            z_vn_avg=self.z_vn_avg,
            horizontal_start=start_edge_lb_plus4,
            horizontal_end=end_edge_local_minus2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        log.debug("corrector: start stencil 32")
        compute_mass_flux(
            z_rho_e=z_fields.z_rho_e,
            z_vn_avg=self.z_vn_avg,
            ddqz_z_full_e=self.metric_state_nonhydro.ddqz_z_full_e,
            z_theta_v_e=z_fields.z_theta_v_e,
            mass_fl_e=diagnostic_state_nh.mass_fl_e,
            z_theta_v_fl_e=self.z_theta_v_fl_e,
            horizontal_start=start_edge_lb_plus4,
            horizontal_end=end_edge_local_minus2,  # TODO: (halungge) this is actually the second halo line
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        if lprep_adv:  # Preparations for tracer advection
            log.debug("corrector: doing prep advection")
            if lclean_mflx:
                log.debug("corrector: start stencil 33")
                init_two_edge_kdim_fields_with_zero_wp(
                    edge_kdim_field_with_zero_wp_1=prep_adv.vn_traj,
                    edge_kdim_field_with_zero_wp_2=prep_adv.mass_flx_me,
                    horizontal_start=start_edge_lb,
                    horizontal_end=end_edge_end,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )
            log.debug(f"corrector: start stencil 34")
            accumulate_prep_adv_fields(
                z_vn_avg=self.z_vn_avg,
                mass_fl_e=diagnostic_state_nh.mass_fl_e,
                vn_traj=prep_adv.vn_traj,
                mass_flx_me=prep_adv.mass_flx_me,
                r_nsubsteps=r_nsubsteps,
                horizontal_start=start_edge_lb_plus4,
                horizontal_end=end_edge_local_minus2,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        # verified for e-9
        log.debug(f"corrector: start stencil 41")
        compute_divergence_of_fluxes_of_rho_and_theta(
            geofac_div=self.interpolation_state.geofac_div,
            mass_fl_e=diagnostic_state_nh.mass_fl_e,
            z_theta_v_fl_e=self.z_theta_v_fl_e,
            z_flxdiv_mass=self.z_flxdiv_mass,
            z_flxdiv_theta=self.z_flxdiv_theta,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        if self.config.itime_scheme == 4:
            log.debug(f"corrector start stencil 42 44 45 45b")
            nhsolve_prog.stencils_42_44_45_45b(
                z_w_expl=z_fields.z_w_expl,
                w_nnow=prognostic_state[nnow].w,
                ddt_w_adv_ntl1=diagnostic_state_nh.ddt_w_adv_pc[self.ntl1],
                ddt_w_adv_ntl2=diagnostic_state_nh.ddt_w_adv_pc[self.ntl2],
                z_th_ddz_exner_c=self.z_th_ddz_exner_c,
                z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
                rho_ic=diagnostic_state_nh.rho_ic,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
                z_beta=z_fields.z_beta,
                exner_nnow=prognostic_state[nnow].exner,
                rho_nnow=prognostic_state[nnow].rho,
                theta_v_nnow=prognostic_state[nnow].theta_v,
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                z_alpha=z_fields.z_alpha,
                vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
                theta_v_ic=diagnostic_state_nh.theta_v_ic,
                z_q=z_fields.z_q,
                k_field=self.k_field,
                rd=constants.RD,
                cvd=constants.CVD,
                dtime=dtime,
                cpd=constants.CPD,
                wgt_nnow_vel=self.params.wgt_nnow_vel,
                wgt_nnew_vel=self.params.wgt_nnew_vel,
                nlev=self.grid.num_levels,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels + 1,
                offset_provider={},
            )
        else:
            log.debug(f"corrector start stencil 43 44 45 45b")
            nhsolve_prog.stencils_43_44_45_45b(
                z_w_expl=z_fields.z_w_expl,
                w_nnow=prognostic_state[nnow].w,
                ddt_w_adv_ntl1=diagnostic_state_nh.ddt_w_adv_pc[self.ntl1],
                z_th_ddz_exner_c=self.z_th_ddz_exner_c,
                z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
                rho_ic=diagnostic_state_nh.rho_ic,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
                z_beta=z_fields.z_beta,
                exner_nnow=prognostic_state[nnow].exner,
                rho_nnow=prognostic_state[nnow].rho,
                theta_v_nnow=prognostic_state[nnow].theta_v,
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                z_alpha=z_fields.z_alpha,
                vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
                theta_v_ic=diagnostic_state_nh.theta_v_ic,
                z_q=z_fields.z_q,
                k_field=self.k_field,
                rd=constants.RD,
                cvd=constants.CVD,
                dtime=dtime,
                cpd=constants.CPD,
                nlev=self.grid.num_levels,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels + 1,
                offset_provider={},
            )
        if not self.l_vert_nested:
            init_two_cell_kdim_fields_with_zero_wp(
                cell_kdim_field_with_zero_wp_1=prognostic_state[nnew].w,
                cell_kdim_field_with_zero_wp_2=z_fields.z_contr_w_fl_l,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=0,
                vertical_end=0,
                offset_provider={},
            )

        log.debug(f"corrector start stencil 47 48 49")
        nhsolve_prog.stencils_47_48_49(
            w_nnew=prognostic_state[nnew].w,
            z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
            w_concorr_c=diagnostic_state_nh.w_concorr_c,
            z_rho_expl=z_fields.z_rho_expl,
            z_exner_expl=z_fields.z_exner_expl,
            rho_nnow=prognostic_state[nnow].rho,
            inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
            z_flxdiv_mass=self.z_flxdiv_mass,
            exner_pr=diagnostic_state_nh.exner_pr,
            z_beta=z_fields.z_beta,
            z_flxdiv_theta=self.z_flxdiv_theta,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            ddt_exner_phy=diagnostic_state_nh.ddt_exner_phy,
            k_field=self.k_field,
            dtime=dtime,
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        # TODO: this is not tested in green line so far
        if self.config.is_iau_active:
            log.debug(f"corrector start stencil 50")
            add_analysis_increments_from_data_assimilation(
                z_rho_expl=z_fields.z_rho_expl,
                z_exner_expl=z_fields.z_exner_expl,
                rho_incr=diagnostic_state_nh.rho_incr,
                exner_incr=diagnostic_state_nh.exner_incr,
                iau_wgt_dyn=self.config.iau_wgt_dyn,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
        log.debug(f"corrector start stencil 52")
        solve_tridiagonal_matrix_for_w_forward_sweep(
            vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
            z_alpha=z_fields.z_alpha,
            z_beta=z_fields.z_beta,
            z_w_expl=z_fields.z_w_expl,
            z_exner_expl=z_fields.z_exner_expl,
            z_q=z_fields.z_q,
            w=prognostic_state[nnew].w,
            dtime=dtime,
            cpd=constants.CPD,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )
        log.debug(f"corrector start stencil 53")
        solve_tridiagonal_matrix_for_w_back_substitution(
            z_q=z_fields.z_q,
            w=prognostic_state[nnew].w,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        if self.config.rayleigh_type == constants.RayleighType.RAYLEIGH_KLEMP:
            log.debug(f"corrector start stencil 54")
            apply_rayleigh_damping_mechanism(
                z_raylfac=self.z_raylfac,
                w_1=prognostic_state[nnew].w_1,
                w=prognostic_state[nnew].w,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=1,
                vertical_end=int32(
                    self.vertical_params.index_of_damping_layer + 1
                ),  # +1 since Fortran includes boundaries
                offset_provider={},
            )
        log.debug(f"corrector start stencil 55")
        compute_results_for_thermodynamic_variables(
            z_rho_expl=z_fields.z_rho_expl,
            vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
            inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
            rho_ic=diagnostic_state_nh.rho_ic,
            w=prognostic_state[nnew].w,
            z_exner_expl=z_fields.z_exner_expl,
            exner_ref_mc=self.metric_state_nonhydro.exner_ref_mc,
            z_alpha=z_fields.z_alpha,
            z_beta=z_fields.z_beta,
            rho_now=prognostic_state[nnow].rho,
            theta_v_now=prognostic_state[nnow].theta_v,
            exner_now=prognostic_state[nnow].exner,
            rho_new=prognostic_state[nnew].rho,
            exner_new=prognostic_state[nnew].exner,
            theta_v_new=prognostic_state[nnew].theta_v,
            dtime=dtime,
            cvd_o_rd=constants.CVD_O_RD,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=int32(self.jk_start),
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        if lprep_adv:
            if lclean_mflx:
                log.debug(f"corrector set prep_adv.mass_flx_ic to zero")
                init_two_cell_kdim_fields_with_zero_wp(
                    prep_adv.mass_flx_ic,
                    prep_adv.vol_flx_ic,
                    horizontal_start=start_cell_nudging,
                    horizontal_end=end_cell_local,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )
        log.debug(f"corrector start stencil 58")
        update_mass_volume_flux(
            z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
            rho_ic=diagnostic_state_nh.rho_ic,
            vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
            w=prognostic_state[nnew].w,
            mass_flx_ic=prep_adv.mass_flx_ic,
            vol_flx_ic=prep_adv.vol_flx_ic,
            r_nsubsteps=r_nsubsteps,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )
        if at_last_substep:
            update_dynamical_exner_time_increment(
                exner=prognostic_state[nnew].exner,
                ddt_exner_phy=diagnostic_state_nh.ddt_exner_phy,
                exner_dyn_incr=diagnostic_state_nh.exner_dyn_incr,
                ndyn_substeps_var=float(self.config.ndyn_substeps_var),
                dtime=dtime,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=self.vertical_params.kstart_moist,
                vertical_end=int32(self.grid.num_levels),
                offset_provider={},
            )

        if lprep_adv:
            if lclean_mflx:
                log.debug(f"corrector set prep_adv.mass_flx_ic to zero")
                init_cell_kdim_field_with_zero_wp(
                    field_with_zero_wp=prep_adv.mass_flx_ic,
                    horizontal_start=start_cell_lb,
                    horizontal_end=end_cell_nudging,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels + 1,
                    offset_provider={},
                )
            log.debug(f" corrector: start stencil 65")
            update_mass_flux_weighted(
                rho_ic=diagnostic_state_nh.rho_ic,
                vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
                vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
                w_now=prognostic_state[nnow].w,
                w_new=prognostic_state[nnew].w,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                mass_flx_ic=prep_adv.mass_flx_ic,
                r_nsubsteps=r_nsubsteps,
                horizontal_start=start_cell_lb,
                horizontal_end=end_cell_nudging,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
            log.debug("exchange prognostic fields 'rho' , 'exner', 'w'")
            self._exchange.exchange_and_wait(
                CellDim,
                prognostic_state[nnew].rho,
                prognostic_state[nnew].exner,
                prognostic_state[nnew].w,
            )
