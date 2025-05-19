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

import icon4py.model.common.constants as constants
from gt4py.next import as_field
from gt4py.next.common import Field
from gt4py.next.ffront.fbuiltins import int32
from icon4py.model.atmosphere.dycore.nh_solve.helpers import (
    accumulate_prep_adv_fields,
    add_analysis_increments_from_data_assimilation,
    add_analysis_increments_to_vn,
    add_temporal_tendencies_to_vn,
    add_temporal_tendencies_to_vn_by_interpolating_between_time_levels,
    add_vertical_wind_derivative_to_divergence_damping,
    apply_2nd_order_divergence_damping,
    apply_4th_order_divergence_damping,
    apply_4th_order_divergence_damping_nonmeancell,
    apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure,
    apply_rayleigh_damping_mechanism,
    apply_weighted_2nd_and_4th_order_divergence_damping,
    compute_approx_of_2nd_vertical_derivative_of_exner,
    compute_avg_vn,
    compute_avg_vn_and_graddiv_vn_and_vt,
    compute_divergence_of_fluxes_of_rho_and_theta,
    # New divergence stencils
    compute_divergence_of_flux_of_normal_wind,
    interpolate_2nd_order_divergence_of_flux_from_cell_to_vertex,
    add_dwdz_to_divergence_of_flux_of_normal_wind,
    compute_full3d_graddiv_normal,
    compute_full3d_graddiv_vertical,
    compute_dgraddiv_dz_for_full3d_divergence_damping,
    compute_divergence_of_flux_of_full3d_graddiv,
    add_dgraddiv_dz_to_full3d_divergence_flux_of_graddiv,
    compute_full3d_graddiv2_normal,
    compute_full3d_graddiv2_vertical,
    apply_4th_order_3d_divergence_damping_to_vn,
    apply_4th_order_3d_divergence_damping_to_w,
    compute_2nd_order_divergence_of_flux_of_normal_wind,
    interpolate_2nd_order_divergence_of_flux_of_normal_wind_to_cell,
    compute_2nd_order_divergence_of_flux_of_full3d_graddiv,
    interpolate_2nd_order_divergence_of_flux_of_full3d_graddiv_to_cell,
    compute_divergence_of_flux,
    compute_2nd_order_divergence_of_flux,
    compute_pure_2nd_order_divergence_of_flux,
    compute_graddiv,
    apply_3d_divergence_damping,
    apply_3d_divergence_damping_only_to_w,
    compute_tangential_wind_and_contravariant,
    # end of new divergence stencils
    compute_dwdz_for_divergence_damping,
    compute_exner_from_rhotheta,
    compute_graddiv_of_vn,
    compute_graddiv2_of_vn,
    compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates,
    compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates,
    compute_horizontal_gradient_of_exner_pressure_for_multiple_levels,
    compute_hydrostatic_correction_term,
    compute_mass_flux,
    compute_perturbation_of_rho_and_theta,
    compute_results_for_thermodynamic_variables,
    compute_rho_virtual_potential_temperatures_and_pressure_gradient,
    compute_theta_and_exner,
    compute_vn_on_lateral_boundary,
    copy_cell_kdim_field_to_vp,
    copy_edge_kdim_field_to_vp,
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
    mo_math_gradients_grad_green_gauss_cell_dsl,
    solve_tridiagonal_matrix_for_w_back_substitution,
    solve_tridiagonal_matrix_for_w_forward_sweep,
    update_dynamical_exner_time_increment,
    update_mass_volume_flux,
    update_mass_flux_weighted,
    update_theta_v,
    en_smag_fac_for_zero_nshift,
    init_cell_kdim_field_with_zero_wp,
    init_two_cell_kdim_fields_with_zero_vp,
    init_two_cell_kdim_fields_with_zero_wp,
    init_two_edge_kdim_fields_with_zero_wp,
    init_test_fields,
    predictor_stencils_2_3,
    predictor_stencils_4_5_6,
    predictor_stencils_7_8_9,
    predictor_stencils_7_8_9_firststep,
    predictor_stencils_7_8_9_secondstep,
    compute_perturbed_rho_and_potential_temperatures_at_half_and_full_levels,
    compute_pressure_gradient,
    predictor_stencils_11_lower_upper,
    compute_horizontal_advection_of_rho_and_theta,
    predictor_stencils_35_36,
    predictor_stencils_37_38,
    stencils_39_40,
    stencils_43_44_45_45b,
    stencils_47_48_49,
    stencils_61_62,
    stencils_42_44_45_45b,
    compute_z_raylfac,
    calculate_divdamp_fields,
    calculate_scal_divdamp_half,
    init_cell_kdim_field_with_zero_vp,
    compute_contravariant_correction,
)
from icon4py.model.atmosphere.dycore.velocity.helpers import (
    compute_tangential_wind,
    fused_stencils_9_10,
)
from icon4py.model.atmosphere.dycore.state_utils.states import (
    DiagnosticStateNonHydro,
    InterpolationState,
    MetricStateNonHydro,
    PrepAdvection,
    OutputIntermediateFields,
)
from icon4py.model.atmosphere.dycore.state_utils.utils import (
    _allocate,
    _allocate_indices,
    _calculate_divdamp_fields,
)

from icon4py.model.atmosphere.dycore.velocity.velocity_advection import (
    VelocityAdvection,
)
from icon4py.model.common.decomposition.definitions import (
    ExchangeRuntime,
    SingleNodeExchange,
)
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim, C2EDim
from icon4py.model.common.grid.base import BaseGrid
from icon4py.model.common.grid.horizontal import (
    CellParams,
    EdgeParams,
    HorizontalMarkerIndex,
)
from icon4py.model.common.settings import xp, device
from icon4py.model.common.config import Device
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.dimension import V2C2EDim
import os

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
    z_beta: Field[[CellDim, KDim], float]
    z_w_divdamp: Field[[EdgeDim, KDim], float]
    z_w_expl: Field[
        [EdgeDim, KDim], float
    ]  # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    z_exner_expl: Field[[CellDim, KDim], float]
    z_q: Field[[CellDim, KDim], float]
    z_contr_w_fl_l: Field[
        [EdgeDim, KDim], float
    ]  # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    z_rho_e: Field[[EdgeDim, KDim], float]
    z_theta_v_e: Field[[EdgeDim, KDim], float]
    z_kin_hor_e: Field[[EdgeDim, KDim], float]
    z_vt_ie: Field[[EdgeDim, KDim], float]
    z_graddiv_vn: Field[[EdgeDim, KDim], float]
    z_rho_expl: Field[[CellDim, KDim], float]
    z_dwdz_dd: Field[[CellDim, KDim], float]
    # New variables for new divergence damping
    z_flxdiv_vn_and_w: Field[[CellDim, KDim], float]
    z_graddiv_normal: Field[[EdgeDim, KDim], float]
    z_graddiv_vertical: Field[[CellDim, KDim], float]
    z_dgraddiv_dz: Field[[CellDim, KDim], float]
    z_flxdiv_graddiv_vn_and_w: Field[[CellDim, KDim], float]
    z_flxdiv_vn_and_w_residual: Field[[CellDim, KDim], float]
    z_graddiv_normal_residual: Field[[EdgeDim, KDim], float]
    z_graddiv_vertical_residual: Field[[CellDim, KDim], float]
    z_dgraddiv_dz_residual: Field[[CellDim, KDim], float]
    z_flxdiv_graddiv_vn_and_w_residual: Field[[CellDim, KDim], float]

    # z_flxdiv2order_vn_vertex: Field[[VertexDim, KDim], float]
    # z_flxdiv2order_graddiv_vn_vertex: Field[[VertexDim, KDim], float]

    vt: Field[[EdgeDim, KDim], float]
    z_w_concorr_me: Field[[EdgeDim, KDim], float]
    z_w_concorr_mc: Field[[CellDim, KDim], float]

    z_graddiv_vt: Field[[EdgeDim, KDim], float]
    z_graddiv_vt_residual: Field[[EdgeDim, KDim], float]
    z_graddiv_w_concorr_me: Field[[EdgeDim, KDim], float]
    z_graddiv_w_concorr_me_residual: Field[[EdgeDim, KDim], float]

    @classmethod
    def allocate(cls, grid: BaseGrid):
        return IntermediateFields(
            z_gradh_exner=_allocate(EdgeDim, KDim, grid=grid),
            z_alpha=_allocate(CellDim, KDim, is_halfdim=True, grid=grid),
            z_beta=_allocate(CellDim, KDim, grid=grid),
            z_w_divdamp=_allocate(CellDim, KDim, is_halfdim=True, grid=grid),
            z_w_expl=_allocate(CellDim, KDim, is_halfdim=True, grid=grid),
            z_exner_expl=_allocate(CellDim, KDim, grid=grid),
            z_q=_allocate(CellDim, KDim, grid=grid),
            z_contr_w_fl_l=_allocate(CellDim, KDim, is_halfdim=True, grid=grid),
            z_rho_e=_allocate(EdgeDim, KDim, grid=grid),
            z_theta_v_e=_allocate(EdgeDim, KDim, grid=grid),
            z_graddiv_vn=_allocate(EdgeDim, KDim, grid=grid),
            z_rho_expl=_allocate(CellDim, KDim, grid=grid),
            z_dwdz_dd=_allocate(CellDim, KDim, grid=grid),
            z_flxdiv_vn_and_w=_allocate(CellDim, KDim, grid=grid),
            z_flxdiv_vn_and_w_residual=_allocate(CellDim, KDim, grid=grid),
            z_graddiv_normal=_allocate(EdgeDim, KDim, grid=grid),
            z_graddiv_vertical=_allocate(CellDim, KDim, is_halfdim=True, grid=grid),
            z_graddiv_normal_residual=_allocate(EdgeDim, KDim, grid=grid),
            z_graddiv_vertical_residual=_allocate(CellDim, KDim, is_halfdim=True, grid=grid),
            z_dgraddiv_dz=_allocate(CellDim, KDim, grid=grid),
            z_dgraddiv_dz_residual=_allocate(CellDim, KDim, grid=grid),
            z_flxdiv_graddiv_vn_and_w=_allocate(CellDim, KDim, grid=grid),
            z_flxdiv_graddiv_vn_and_w_residual=_allocate(CellDim, KDim, grid=grid),
            # z_flxdiv2order_vn_vertex=_allocate(VertexDim, KDim, grid=grid),
            # z_flxdiv2order_graddiv_vn_vertex=_allocate(VertexDim, KDim, grid=grid),
            z_kin_hor_e=_allocate(EdgeDim, KDim, grid=grid),
            z_vt_ie=_allocate(EdgeDim, KDim, grid=grid),
            vt=_allocate(EdgeDim, KDim, grid=grid),
            z_w_concorr_me=_allocate(EdgeDim, KDim, grid=grid),
            z_w_concorr_mc=_allocate(CellDim, KDim, grid=grid),
            z_graddiv_vt=_allocate(EdgeDim, KDim, grid=grid),
            z_graddiv_vt_residual=_allocate(EdgeDim, KDim, grid=grid),
            z_graddiv_w_concorr_me=_allocate(EdgeDim, KDim, grid=grid),
            z_graddiv_w_concorr_me_residual=_allocate(EdgeDim, KDim, grid=grid),
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
        divdamp_fac_w: float = 1.0,
        htop_moist_proc: float = 22500.0,
        scal_divsign: float = 1.0,
        first_order_div_threshold: float = 1.0,
        do_o2_divdamp: bool = False,
        do_multiple_divdamp: bool = False,
        do_3d_divergence_damping: bool = False,
        do_proper_diagnostics_divdamp: bool = False,
        divergence_order: int = 1,
        number_of_divdamp_step: int = 1,
        do_only_divdamp: bool = False,
        do_proper_contravariant_divdamp: bool = False,
        suppress_vertical_in_3d_divdamp: bool = False,
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

        self.divdamp_fac_w: float = divdamp_fac_w

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

        self.scal_divsign: float = scal_divsign
        self.first_order_div_threshold: float = first_order_div_threshold

        self.do_o2_divdamp = do_o2_divdamp
        self.do_multiple_divdamp = do_multiple_divdamp
        self.do_3d_divergence_damping = do_3d_divergence_damping
        self.do_proper_diagnostics_divdamp = do_proper_diagnostics_divdamp
        self.divergence_order = divergence_order
        self.number_of_divdamp_step = number_of_divdamp_step

        self.do_only_divdamp = do_only_divdamp

        self.do_proper_contravariant_divdamp = do_proper_contravariant_divdamp
        self.suppress_vertical_in_3d_divdamp = suppress_vertical_in_3d_divdamp

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

        if self.divergence_order == 3 and self.do_o2_divdamp:
            raise NotImplementedError(
                "Divergence order 3 and doing o2 divdamp cannot be set simultaneously."
            )

        if self.divergence_order >= 2:
            if self.do_3d_divergence_damping is False:
                raise NotImplementedError(
                    "If you use 2d divergence, please set divergence order to 1 because only order 1 is supported."
                )


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
        log.info(f"wgt_nnow_vel and wgt_nnew_vel: {self.wgt_nnow_vel} {self.wgt_nnew_vel}")

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
        self.scal_divdamp_half: Optional[Field[[KDim], float]] = None
        self.scal_divdamp_o2: Optional[Field[[KDim], float]] = None
        self.scal_divdamp_o2_half: Optional[Field[[KDim], float]] = None
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
        owner_mask: Field[[CellDim], bool],
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

        for k in range(self.grid.num_levels):
            log.critical(f"enh_divdamp_fac {k} {self.enh_divdamp_fac.ndarray[k]}")

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
        self.redundant_z_vn_avg = _allocate(EdgeDim, KDim, grid=self.grid)
        self.redundant_vt = _allocate(EdgeDim, KDim, grid=self.grid)
        self.z_theta_v_fl_e = _allocate(EdgeDim, KDim, grid=self.grid)
        self.z_flxdiv_mass = _allocate(CellDim, KDim, grid=self.grid)
        self.z_flxdiv_theta = _allocate(CellDim, KDim, grid=self.grid)
        self.z_rho_v = _allocate(VertexDim, KDim, grid=self.grid)
        self.z_theta_v_v = _allocate(VertexDim, KDim, grid=self.grid)
        self.z_graddiv2_w = _allocate(EdgeDim, KDim, grid=self.grid, is_halfdim=True)
        self.z_graddiv2_vn = _allocate(EdgeDim, KDim, grid=self.grid)
        self.k_field = _allocate_indices(KDim, grid=self.grid, is_halfdim=True)
        self.z_w_concorr_me = _allocate(EdgeDim, KDim, grid=self.grid)
        self.z_hydro_corr_horizontal = _allocate(EdgeDim, grid=self.grid)
        self.z_raylfac = _allocate(KDim, grid=self.grid)
        self.enh_divdamp_fac = _allocate(KDim, grid=self.grid)
        self._bdy_divdamp = _allocate(KDim, grid=self.grid)
        self.scal_divdamp = _allocate(KDim, grid=self.grid)
        self.scal_divdamp_half = _allocate(
            KDim, grid=self.grid
        )  # this is on half level, however, it only has nlev levels without the ground.
        self.scal_divdamp_o2 = _allocate(KDim, grid=self.grid)
        self.scal_divdamp_o2_half = _allocate(
            KDim, grid=self.grid
        )  # this is on half level, however, it only has nlev levels without the ground.
        self.z_graddiv2_normal = _allocate(EdgeDim, KDim, grid=self.grid)
        self.z_graddiv2_vertical = _allocate(CellDim, KDim, grid=self.grid, is_halfdim=True)
        self.z_graddiv2_normal_residual = _allocate(EdgeDim, KDim, grid=self.grid)
        self.z_graddiv2_vertical_residual = _allocate(
            CellDim, KDim, grid=self.grid, is_halfdim=True
        )
        self.intermediate_fields = IntermediateFields.allocate(self.grid)
        self.output_intermediate_fields = OutputIntermediateFields.allocate(self.grid)

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
        do_output: bool = False,
        do_output_step: int = 0,
        do_output_substep: int = 0,
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
            init_test_fields(
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

        if self.config.do_only_divdamp:
            self.run_divdamp(
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
                do_output=do_output,
                do_output_step=do_output_step,
                do_output_substep=do_output_substep,
            )
        else:
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
                do_output=do_output,
                do_output_step=do_output_step,
                do_output_substep=do_output_substep,
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
            """
            theta_v (0:nlev-1):
                Update virtual temperature at full levels (cell center) at only halo cells in the boundary interpolation zone by equating it to exner.
            exner (0:nlev-1):
                Update exner function at full levels (cell center) at only halo cells in the boundary interpolation zone using the equation of state (see eq.3.9 in ICON tutorial 2023).
                exner = (rd * rho * exner / p0ref) ^ (rd / cvd)
            """
            compute_theta_and_exner(
                bdy_halo_c=self.metric_state_nonhydro.bdy_halo_c,
                rho=prognostic_state_ls[nnew].rho,
                theta_v=prognostic_state_ls[nnew].theta_v,
                exner=prognostic_state_ls[nnew].exner,
                rd_o_cvd=self.params.rd_o_cvd,
                rd_o_p0ref=self.params.rd_o_p0ref,
                horizontal_start=0,
                horizontal_end=end_cell_end,
                vertical_start=int32(0),
                vertical_end=int32(self.grid.num_levels),
                offset_provider={},
            )

            """
            theta_v (0:nlev-1):
                Update virtual temperature at full levels (cell center) at only halo cells in the boundary interpolation zone by equating it to exner.
            exner (0:nlev-1):
                Update exner function at full levels (cell center) at only halo cells in the boundary interpolation zone using the equation of state (see eq.3.9 in ICON tutorial 2023).
                exner = (rd * rho * exner / p0ref) ^ (rd / cvd)
            """
            compute_exner_from_rhotheta(
                rho=prognostic_state_ls[nnew].rho,
                theta_v=prognostic_state_ls[nnew].theta_v,
                exner=prognostic_state_ls[nnew].exner,
                rd_o_cvd=self.params.rd_o_cvd,
                rd_o_p0ref=self.params.rd_o_p0ref,
                horizontal_start=start_cell_lb,
                horizontal_end=end_cell_nudging_minus1,
                vertical_start=int32(0),
                vertical_end=int32(self.grid.num_levels),
                offset_provider={},
            )

        """
        theta_v (0:nlev-1):
            Update virtual temperature at full levels (cell center) at only halo cells in the boundary interpolation zone from the equation of state (see eqs. 3.22 and 3.23 in ICON tutorial 2023).
            rho^{n+1} theta_v^{n+1} = rho^{n} theta_v^{n} + ( cvd * rho^{n} * theta_v^{n} ) / ( rd * pi^{n} ) ( pi^{n+1} - pi^{n} )
        """
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
            vertical_start=int32(0),
            vertical_end=int32(self.grid.num_levels),
            offset_provider={},
        )

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
                output_intermediate_fields=self.output_intermediate_fields,
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
                vertical_start=int32(0),
                vertical_end=int32(self.grid.num_levels),
                offset_provider={},
            )

        """
        z_exner_ex_pr (0:nlev):
            Compute the temporal extrapolation of perturbed exner function at full levels (cell center) using the time backward scheme (page 74 in icon tutorial 2023) for horizontal momentum equations.
            Note that it has nlev+1 levels. This last level is underground and set to zero.
        exner_pr (0:nlev-1):
            Store perturbed exner function at full levels of current time step.
        """
        predictor_stencils_2_3(
            exner_exfac=self.metric_state_nonhydro.exner_exfac,
            exner=prognostic_state[nnow].exner,
            exner_ref_mc=self.metric_state_nonhydro.exner_ref_mc,
            exner_pr=diagnostic_state_nh.exner_pr,
            z_exner_ex_pr=self.z_exner_ex_pr,
            k_field=self.k_field,
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_lb_plus2,
            horizontal_end=end_cell_halo,
            vertical_start=int32(0),
            vertical_end=int32(self.grid.num_levels + 1),
            offset_provider={},
        )

        """
        z_exner_ic (1 or flat_lev:nlev):
            Linearly interpolate the temporal extrapolation of perturbed exner function computed in previous stencil to half levels.
            The ground level is based on quadratic interpolation (with hydrostatic assumption?).
            WARNING: Its value at the model top level is not updated and assumed to be zero. It should be treated in the same way as the ground level.
        z_dexner_dz_c_1 (1 or flat_lev:nlev-1):
            Vertical derivative of the temporal extrapolation of exner function at full levels is also computed (first order scheme).
        flat_lev is the height (inclusive) above which the grid is not affected by terrain following.
        """
        if self.config.igradp_method == 3:
            predictor_stencils_4_5_6(
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
                vertical_start=max(int32(1), self.vertical_params.nflatlev),
                vertical_end=int32(self.grid.num_levels + 1),
                offset_provider=self.grid.offset_providers,
            )

            if self.vertical_params.nflatlev == 1:
                # Perturbation Exner pressure on top half level
                raise NotImplementedError("nflatlev=1 not implemented")

        """
        rho_ic & theta_v_ic (1:nlev-1):
            Compute rho and virtual temperature at half levels. rho and virtual temperature at model top boundary and ground are not updated.
        z_rth_pr_1 (0:nlev-1):
            Compute perturbed rho at full levels (cell center).
        z_rth_pr_2 (0:nlev-1):
            Compute perturbed virtual temperature at full levels (cell center).
        z_theta_v_pr_ic (1:nlev-1):
            Compute the perturbed virtual temperature from z_rth_pr_2 at half levels.
        z_th_ddz_exner_c (1:nlev-1):
            theta_v' dpi_0/dz + eta_expl theta_v dpi'/dz (see eq. 3.19 in icon tutorial 2023) at half levels (cell center) is also computed. Its value at the model top is not updated. No ground value.
            dpi_0/dz is d_exner_dz_ref_ic.
            eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
            eta_expl = 1.0 - eta_impl = vwind_expl_wgt
        """

        predictor_stencils_7_8_9(
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
            vertical_start=int32(0),
            vertical_end=int32(self.grid.num_levels),
            offset_provider=self.grid.offset_providers,
        )
        """
        compute_perturbed_rho_and_potential_temperatures_at_half_and_full_levels(
            rho=prognostic_state[nnow].rho,
            z_rth_pr_1=self.z_rth_pr_1,
            z_rth_pr_2=self.z_rth_pr_2,
            rho_ref_mc=self.metric_state_nonhydro.rho_ref_mc,
            theta_v=prognostic_state[nnow].theta_v,
            theta_ref_mc=self.metric_state_nonhydro.theta_ref_mc,
            rho_ic=diagnostic_state_nh.rho_ic,
            wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
            z_theta_v_pr_ic=self.z_theta_v_pr_ic,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            k_field=self.k_field,
            horizontal_start=start_cell_lb_plus2,
            horizontal_end=end_cell_halo,
            vertical_start=int32(0),
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )
        compute_pressure_gradient(
            vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            z_theta_v_pr_ic=self.z_theta_v_pr_ic,
            exner_pr=diagnostic_state_nh.exner_pr,
            d_exner_dz_ref_ic=self.metric_state_nonhydro.d_exner_dz_ref_ic,
            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
            z_th_ddz_exner_c=self.z_th_ddz_exner_c,
            horizontal_start=start_cell_lb_plus2,
            horizontal_end=end_cell_halo,
            vertical_start=int32(1),
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )
        """
        """
        predictor_stencils_7_8_9_firststep(
            rho=prognostic_state[nnow].rho,
            rho_ref_mc=self.metric_state_nonhydro.rho_ref_mc,
            theta_v=prognostic_state[nnow].theta_v,
            theta_ref_mc=self.metric_state_nonhydro.theta_ref_mc,
            rho_ic=diagnostic_state_nh.rho_ic,
            z_rth_pr_1=self.z_rth_pr_1,
            z_rth_pr_2=self.z_rth_pr_2,
            wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
            z_theta_v_pr_ic=self.z_theta_v_pr_ic,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            k_field=self.k_field,
            horizontal_start=start_cell_lb_plus2,
            horizontal_end=end_cell_halo,
            vertical_start=int32(0),
            vertical_end=int32(self.grid.num_levels),
            offset_provider=self.grid.offset_providers,
        )

        predictor_stencils_7_8_9_secondstep(
            vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            z_theta_v_pr_ic=self.z_theta_v_pr_ic,
            exner_pr=diagnostic_state_nh.exner_pr,
            d_exner_dz_ref_ic=self.metric_state_nonhydro.d_exner_dz_ref_ic,
            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
            z_th_ddz_exner_c=self.z_th_ddz_exner_c,
            k_field=self.k_field,
            horizontal_start=start_cell_lb_plus2,
            horizontal_end=end_cell_halo,
            vertical_start=int32(0),
            vertical_end=int32(self.grid.num_levels),
            offset_provider=self.grid.offset_providers,
        )
        """

        """
        z_theta_v_pr_ic (0, nlev):
            Perturbed theta_v at half level at the model top is set to zero.
            Perturbed theta_v at half level at the ground level is computed by quadratic interpolation (hydrostatic assumption?) in the same way as z_exner_ic.
        theta_v_ic (nlev):
            virtual temperature at half level at the ground level is computed by adding theta_ref_ic to z_theta_v_pr_ic.
        """
        # Perturbation theta at top and surface levels
        predictor_stencils_11_lower_upper(
            wgtfacq_c_dsl=self.metric_state_nonhydro.wgtfacq_c,
            z_rth_pr=self.z_rth_pr_2,
            theta_ref_ic=self.metric_state_nonhydro.theta_ref_ic,
            z_theta_v_pr_ic=self.z_theta_v_pr_ic,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            k_field=self.k_field,
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_lb_plus2,
            horizontal_end=end_cell_halo,
            vertical_start=int32(0),
            vertical_end=int32(self.grid.num_levels + 1),
            offset_provider=self.grid.offset_providers,
        )

        """
        z_dexner_dz_c_2 (flat_gradp:nlev-1):
            Compute second vertical derivative of perturbed exner function at full levels (cell centers) from flat_gradp to the bottom-most level.
            This second vertical derivative is approximated by hydrostatic approximation (see eqs. 13 and 7 in Günther et al. 2012).
            d2pi'/dz2 = - dtheta_v'/dz /theta_v_0 dpi_0/dz - theta_v' d/dz( 1/theta_v_0 dpi_0/dz ), dpi_0/dz = -g /cpd / theta_v_0
            z_dexner_dz_c_2 = 1/2 d2pi'/dz2
            1/theta_v_0 dpi_0/dz is precomputed as d2dexdz2_fac1_mc.
            d/dz( 1/theta_v_0 dpi_0/dz ) is precomputed as d2dexdz2_fac2_mc. It makes use of eq. 15 in Günther et al. 2012 for refernce state of temperature when computing dtheta_v_0/dz
        flat_gradp is the maximum height index at which the height of the center of an edge lies within two neighboring cells.
        """
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
                vertical_end=int32(self.grid.num_levels),
                offset_provider=self.grid.offset_providers,
            )

        """
        z_rth_pr_1 (0:nlev-1):
            Compute perturbed rho at full levels (cell center), which is equal to rho - rho_ref_mc.
        z_rth_pr_2 (0:nlev-1):
            Compute perturbed virtual temperature at full levels (cell center), which is equal to theta_v - theta_ref_mc.
        """
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
            """
            z_rho_v (0:nlev-1):
                Compute the density at cell vertices at full levels by simple area-weighted interpolation.
            """
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
            """
            z_theta_v_v (0:nlev-1):
                Compute the virtual temperature at cell vertices at full levels by simple area-weighted interpolation.
            """
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
            """
            z_grad_rth_1 (0:nlev-1):
                Compute x derivative of perturbed rho at full levels (cell center) using the Green theorem. See https://www.cfd-online.com/Wiki/Gradient_computation.
            z_grad_rth_2 (0:nlev-1):
                Compute y derivative of perturbed rho at full levels (cell center) using the Green theorem. See https://www.cfd-online.com/Wiki/Gradient_computation.
            z_grad_rth_3 (0:nlev-1):
                Compute x derivative of perturbed virtual temperature at full levels (cell center) using the Green theorem. See https://www.cfd-online.com/Wiki/Gradient_computation.
            z_grad_rth_4 (0:nlev-1):
                Compute y derivative of perturbed virtual temperature at full levels (cell center) using the Green theorem. See https://www.cfd-online.com/Wiki/Gradient_computation.
            """
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
                """
                This long stencil computes rho (density) and theta_v (virtual temperature) on edges.
                Miura (2007) scheme is adopted. pos_on_tplane_e is the location of neighboring cell centers on (vn, vt) coordinates (normal points inwards and tangent points right-handed).
                primal_normal_cell and dual_normal_cell are the components of the (vn, vt) vector at location of neighboring cells in (lat, lon) coordinates. vn = vn_lat <lat> + vn_lon <lon>, vt = vt_lat <lat> + vt_lon <lon>
                The distance between back-trajectory point and the nearest cell center is computed first (d_n, d_t) = -(cell_center_n + vn dt/2, cell_center_t + vt dt/2) = d_n vn + d_t vt.
                It is then transformed to (lat, lon) coordinates = (d_lat, d_lon) = (d_y, d_x) by d_n*vn_lat + d_t*vt_lat, d_n*vn_lon + d_t*vt_lon.
                Then, the value at edges is simply p_at_edge = p_at_cell_center + dp/dx d_x + dp/dy d_y
                z_rho_e (0:nlev-1):
                    rho at edges.
                z_theta_v_e (0:nlev-1):
                    theta_v at edges.
                """
                # Compute upwind-biased values for rho and theta starting from centered differences
                # Note: the length of the backward trajectory should be 0.5*dtime*(vn,vt) in order to arrive
                # at a second-order accurate FV discretization, but twice the length is needed for numerical stability

                compute_horizontal_advection_of_rho_and_theta(
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

        """
        z_gradh_exner (0:flat_lev-1):
            Compute the horizontal gradient of temporal extrapolation of perturbed exner function at full levels (edge center) by simple first order scheme at altitudes witout terrain following effect.
        """
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
            """
            z_gradh_exner (flat_lev:flat_gradp):
                Compute the horizontal gradient (at constant height) of temporal extrapolation of perturbed exner function at full levels (edge center) by simple first order scheme.
                By coordinate transformation (x, y, z) <-> (x, y, eta), dpi/dn |z = dpi/dn |s + dh/dn |s dpi/dz
                dpi/dz is previously computed z_dexner_dz_c_1.
                dh/dn | s is ddxn_z_full, it is the horizontal gradient across neighboring cells at constant eta at full levels.
            """
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

            """
            z_gradh_exner (flat_gradp+1:nlev-1):
                Compute the horizontal gradient (at constant height) of temporal extrapolation of perturbed exner function at full levels (edge center) when the height of neighboring cells is at another k level.
                See eq. 8 in Günther et al. 2012.
                dpi/dn |z = (pi_1 - pi_0 + dpi_1/dz_1 dz_1 - dpi_0/dz_0 dz_0 + d^2pi_1/dz_1^2 dz_1^2/2 - d^2pi_0/dz_0^2 dz_0^2/2) / length
                dpi_0/dz_0 or dpi_1/dz_1 is z_dexner_dz_c_1 computed previously.
                d^2pi_0/dz_0^2 / 2 or d^2pi_1/dz_1^2 / 2  is z_dexner_dz_c_2 computed previously.
                dz is zdiff_gradp.
                neighboring cell k index is vertoffset_gradp.
                Note that the vertoffset_gradp and zdiff_gradp are recomputed for edges which have an neighboring underground cell center in mo_vertical_grid.f90.
                It is explained more in next stencil for computation of hydrostatic correction.
            """
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
        """
        z_hydro_corr (nlev-1):
            z_hydro_corr = g/(cpd theta_v^2) dtheta_v/dn (h_k - h_k*)
            Compute the hydrostatic correction term (see the last term in eq. 10 or 9 in Günther et al. 2012) at full levels (edge center).
            This is only computed for the last or bottom most level because all edge centers which have a neighboring cell center inside terrain
            beyond a certain limit (see last paragraph for discussion on page 3724) use the same correction term at k* level in eq. 10 in Günther
            et al. 2012.
            Note that the vertoffset_gradp and zdiff_gradp are recomputed for those special edges in mo_vertical_grid.f90.
        """
        # compute hydrostatically approximated correction term that replaces downward extrapolation
        if self.config.igradp_method == 3:
            compute_hydrostatic_correction_term(
                theta_v=prognostic_state[nnow].theta_v,
                ikoffset=self.metric_state_nonhydro.vertoffset_gradp,
                zdiff_gradp=self.metric_state_nonhydro.zdiff_gradp,
                theta_v_ic=diagnostic_state_nh.theta_v_ic,
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                grav_o_cpd=self.params.grav_o_cpd,
                z_hydro_corr=self.z_hydro_corr,
                horizontal_start=start_edge_nudging_plus1,
                horizontal_end=end_edge_local,
                vertical_start=int32(self.grid.num_levels) - int32(1),
                vertical_end=int32(self.grid.num_levels),
                offset_provider=self.grid.offset_providers,
            )
        # TODO (Nikki) check when merging fused stencil
        lowest_level = self.grid.num_levels - 1
        hydro_corr_horizontal = as_field((EdgeDim,), self.z_hydro_corr.ndarray[:, lowest_level])

        if self.config.igradp_method == 3:
            """
            z_gradh_exner (0:nlev-1):
                Apply the dydrostatic correction term to horizontal gradient (at constant height) of temporal extrapolation of perturbed exner function at full levels (edge center)
                when neighboring cells are underground beyond a certain limit.
            """
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

        """
        vn (0:nlev-1):
            Add the advection and pressure gradient terms to update the normal velocity.
            vn = vn + dt * (advection - cpd * theta_v * dpi/dz)
            advection is computed in velocity_advection.
        """
        copy_edge_kdim_field_to_vp(
            field=z_fields.z_gradh_exner,
            field_copy=self.output_intermediate_fields.output_predictor_gradh_exner,
            horizontal_start=int32(0),
            horizontal_end=end_edge_local,
            vertical_start=int32(0),
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )
        copy_edge_kdim_field_to_vp(
            field=z_fields.z_theta_v_e,
            field_copy=self.output_intermediate_fields.output_predictor_theta_v_e,
            horizontal_start=int32(0),
            horizontal_end=end_edge_local,
            vertical_start=int32(0),
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )
        copy_edge_kdim_field_to_vp(
            field=diagnostic_state_nh.ddt_vn_apc_pc[self.ntl1],
            field_copy=self.output_intermediate_fields.output_predictor_ddt_vn_apc_ntl1,
            horizontal_start=int32(0),
            horizontal_end=end_edge_local,
            vertical_start=int32(0),
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )
        # main update
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
            """
            vn (0:nlev-1):
                Add boundary velocity tendendy to the normal velocity.
                vn = vn + dt * grf_tend_vn
            """
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

        """
        z_vn_avg (0:nlev-1):
            Compute the averaged normal velocity at full levels (edge center).
            TODO (Chia Rui): Fill in details about how the coefficients are computed.
        z_graddiv_vn (0:nlev-1):
            Compute normal gradient of divergence at full levels (edge center).
            z_graddiv_vn = Del(normal_direction) divergence
        vt (0:nlev-1):
            Compute tangential velocity by rbf interpolation at full levels (edge center).
        """
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

        """
        z_flxdiv_mass (0:nlev-1):
            Compute the mass flux at full levels (edge center) by multiplying density with averaged normal velocity (z_vn_avg) computed above.
        z_flxdiv_theta (0:nlev-1):
            Compute the energy (theta_v * mass) flux by multiplying density with averaged normal velocity (z_vn_avg) computed above.
        """
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

        """
        z_w_concorr_me (flat_lev:nlev-1):
            Compute contravariant correction (due to terrain-following coordinates) to vertical wind at
            full levels (edge center). The correction is equal to vn dz/dn + vt dz/dt, where t is tangent.
        vn_ie (1:nlev-1):
            Compute normal velocity at half levels (edge center) simply by interpolating two neighboring
            normal velocity at full levels.
        z_kin_hor_e (1:nlev-1):
            Compute the horizontal kinetic energy (vn^2 + vt^2)/2 at full levels (edge center).
        z_vt_ie (1:nlev-1):
            Compute tangential velocity at half levels (edge center) simply by interpolating two neighboring
            tangential velocity at full levels.
        """
        predictor_stencils_35_36(
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
            """
            vn_ie (0):
                Compute normal wind at model top (edge center). It is simply set equal to normal wind.
            z_vt_ie (0):
                Compute tangential wind at model top (edge center). It is simply set equal to tangential wind.
            z_kin_hor_e (0):
                Compute the horizonal kinetic energy (vn^2 + vt^2)/2 at first full level (edge center).
            vn_ie (nlev):
                Compute normal wind at ground level (edge center) by quadratic extrapolation.
                ---------------  z4
                       z3'
                ---------------  z3
                       z2'
                ---------------  z2
                       z1'
                ---------------  z1 (surface)
                ///////////////
                The three reference points for extrapolation are at z2, z2', and z3'. Value at z1 is
                then obtained by quadratic interpolation polynomial based on these three points.
            """
            predictor_stencils_37_38(
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

        """
        w_concorr_c (flat_lev+1:nlev-1):
            Interpolate contravariant correction at edge center from full levels, which is
            z_w_concorr_me computed above, to half levels using simple linear interpolation.
        w_concorr_c (nlev):
            Compute contravariant correction at ground level (cell center) by quadratic extrapolation. z_w_concorr_me needs to be first
            linearly interpolated to cell center.
            ---------------  z4
                   z3'
            ---------------  z3
                   z2'
            ---------------  z2
                   z1'
            ---------------  z1 (surface)
            ///////////////
            The three reference points for extrapolation are at z2, z2', and z3'. Value at z1 is
            then obtained by quadratic interpolation polynomial based on these three points.
        """
        stencils_39_40(
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

        """
        z_flxdiv_mass (0:nlev-1):
            Compute the divergence of mass flux at full levels (cell center) by Gauss theorem.
        z_flxdiv_theta (0:nlev-1):
            Compute the divergence of energy (theta_v * mass) flux at full levels (cell center) by Gauss theorem.
        """
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

        init_cell_kdim_field_with_zero_vp(
            field_with_zero_vp=z_fields.z_w_divdamp,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=int32(self.grid.num_levels + 1),
            offset_provider={},
        )
        """
        z_w_expl (1:nlev-1):
            Compute the explicit term in vertical momentum equation at half levels (cell center). See the first equation below eq. 3.25 in ICON tutorial 2023.
            z_w_expl = advection of w + cpd theta' dpi0/dz + cpd theta (1 - eta_impl) dpi'/dz + F_divdamp @ k+1/2 level
            advection of w = ddt_w_adv_pc
            cpd theta' dpi0/dz + cpd theta (1 - eta_impl) dpi'/dz = cpd z_th_ddz_exner_c
            eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
            eta_expl = 1.0 - eta_impl = vwind_expl_wgt
        z_contr_w_fl_l (1:nlev-1):
            Compute the vertical mass flux at half levels (cell center). See second term on RHS of mass conservation in eq. 3.21 in ICON tutorial 2023.
            z_contr_w_fl_l = rho * (-contravariant_correction + vwind_expl_wgt * w) # TODO (Chia Rui: Check why minus sign)
            eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
            eta_expl = 1.0 - eta_impl = vwind_expl_wgt
            rho = rho_ic
        z_beta (0:nlev-1):
            Compute component of the coefficients in the tridiagonal matrix of w equation at full levels (cell center).
            See the middle term in each square bracket of eq. 3.27 and unnumbered equation below in ICON tutorial 2023.
            a b 0 0 0
            c a b 0 0
            0 c a b 0
            0 0 c a b
            0 0 0 c a
            z_beta_{k} = dt * rd * pi_{k} / (cvd * rho_{k} * theta_v_{k}) / dz_{k}
        z_alpha (0:nlev-1):
            Compute component of the coefficients in the tridiagonal matrix of w equation at half levels (cell center).
            See the last term in each square bracket of eq. 3.27 and unnumbered equation below in ICON tutorial 2023.
            z_alpha_{k-1/2} = vwind_impl_wgt rho_{k-1/2} theta_v_{k-1/2}
            eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
            eta_expl = 1.0 - eta_impl = vwind_expl_wgt
            rho_{k-1/2} is precomputed as rho_ic.
            theta_v_{k-1/2} is precomputed as theta_v_ic.
        z_alpha (nlev):
            Compute component of the coefficients in the tridiagonal matrix of w equation at half levels (cell center).
            z_alpha_{k-1/2} = 0
        z_q (0):
            Set the intermediate result for w in tridiagonal solver during forward seep at half levels (cell center) at model top to zero.
            Note that it also only has nlev levels because the model top w is not updated, although it is a half-level variable.
            z_q_{k-1/2} = 0
        """
        stencils_43_44_45_45b(
            z_w_expl=z_fields.z_w_expl,
            w_nnow=prognostic_state[nnow].w,
            z_w_divdamp=z_fields.z_w_divdamp,
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
            # vertical_end=self.grid.num_levels + int32(1),
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )
        init_cell_kdim_field_with_zero_vp(
            field_with_zero_vp=z_fields.z_alpha,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=self.grid.num_levels,
            vertical_end=self.grid.num_levels + 1,
            offset_provider={},
        )

        if not self.l_vert_nested:
            """
            w (0):
                Set w at half levels (cell center) at model top to zero.
            z_contr_w_fl_l (0):
                Set the vertical mass flux at half levels (cell center) at model top to zero. See second term on RHS of mass conservation in eq. 3.21 in ICON tutorial 2023.
                z_contr_w_fl_l = rho * (-contravariant_correction + vwind_expl_wgt * w) # TODO (Chia Rui: Check why minus sign)
                eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
                eta_expl = 1.0 - eta_impl = vwind_expl_wgt
                rho = rho_ic
            """
            init_two_cell_kdim_fields_with_zero_wp(
                cell_kdim_field_with_zero_wp_1=prognostic_state[nnew].w,
                cell_kdim_field_with_zero_wp_2=z_fields.z_contr_w_fl_l,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=0,
                vertical_end=1,
                offset_provider={},
            )
        """
        w^{n+1*} (nlev):
            Set updated w at half levels (cell center) at ground level to the contravariant correction (since we are using terrain following coordinates).
            w_ground = contravariant_correction
        z_contr_w_fl_l (nlev):
            Set the vertical mass flux at half levels (cell center) at ground level to zero. See second term on RHS of mass conservation in eq. 3.21 in ICON tutorial 2023.
            z_contr_w_fl_l = rho * (-contravariant_correction + vwind_expl_wgt * w) # TODO (Chia Rui: Check why minus sign)
            eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
            eta_expl = 1.0 - eta_impl = vwind_expl_wgt
            rho = rho_ic
        z_rho_expl (0:nlev-1):
            Compute the explicit term in vertical momentum equation at full levels (cell center). See RHS of mass conservation in eq. 3.21 in ICON tutorial 2023.
            z_rho_expl = rho^{n} - dt ( divergence(v^{n+1*} rho^{n}) + vwind_expl_wgt ( rho^{n}_{k-1/2} w^{n}_{k-1/2} - rho^{n}_{k+1/2} w^{n}_{k+1} ) / dz_{k} )
            eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
            eta_expl = 1.0 - eta_impl = vwind_expl_wgt
            The divergence term on RHS of the equation for z_rho_expl is precomputed as z_flxdiv_mass.
            The mass flux in second term on RHS of the equation for z_rho_expl is precomputed as z_contr_w_fl_l. i.e. z_contr_w_fl_l = vwind_expl_wgt rho^{n}_{k-1/2} w^{n}_{k-1/2}
            TODO (Chia Rui): Why /dz_{k} factor is included in divergence term?
        z_exner_expl (0:nlev-1):
            Compute the explicit term in pressure equation at full levels (cell center). See RHS of thermodynamics equation in eq. 3.21 and the second unnumbered equation below eq. 3.25 in ICON tutorial 2023.
            z_exner_expl = pi'^{n} - dt * rd * pi_{k} / (cvd * rho_{k} * theta_v_{k}) ( divergence(v^{n+1*} rho^{n} theta_v^{n}) + vwind_expl_wgt ( rho^{n}_{k-1/2} w^{n}_{k-1/2} theta_v^{n}_{k-1/2} - rho^{n}_{k+1/2} w^{n}_{k+1} theta_v^{n}_{k+1/2} ) / dz_{k} ) + dt * physics_tendency
            eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
            eta_expl = 1.0 - eta_impl = vwind_expl_wgt
            pi'^{n} is precomputed as exner_pr.
            dt * rd * pi_{k} / (cvd * rho_{k} * theta_v_{k}) / dz_{k} is precomputed as z_beta.
            The divergence term on RHS of the equation for z_exber_expl is precomputed as z_flxdiv_theta.
            The mass flux in second term on RHS of the equation for z_exner_expl is precomputed as z_contr_w_fl_l, and it is multiplied by theta_v at half levels (which is theta_v_ic) and become energy flux. i.e. z_contr_w_fl_l = vwind_expl_wgt rho^{n}_{k-1/2} w^{n}_{k-1/2}
            physics_tendency is represented by ddt_exner_phy.
            TODO (Chia Rui): Why /dz_{k} factor is included in divergence term?
        """
        stencils_47_48_49(
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
            nlev=int32(self.grid.num_levels),
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=int32(0),
            vertical_end=int32(self.grid.num_levels) + int32(1),
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

        """
        w (1:nlev-1):
            Update intermediate vertical velocity by forward sweep (RHS of the equation).
        z_q (1:nlev-1):
            Update intermediate upper element of tridiagonal matrix by forward sweep.
            During the forward seep, the middle element is normalized to 1.
        """
        # main update
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

        """
        w (1:nlev-1):
            Compute the vertical velocity by backward sweep. Model top and ground level are not updated.
            w_{k-1/2} = w_{k-1/2} + w_{k+1/2} * z_q_{k-1/2}
        """
        # main update
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
            """
            w (1:damp_nlev):
                Compute the rayleigh damping of vertical velocity at half levels (cell center).
                w_{k-1/2} = Rayleigh_damping_coeff w_{k-1/2} + (1 - Rayleigh_damping_coeff) w_{-1/2}, where w_{-1/2} is model top vertical velocity. It is zero.
                Rayleigh_damping_coeff is represented by z_raylfac.
            """
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

        """
        rho (0:nlev-1):
            Update the density at full levels (cell center) from the mass conservation equation (see eq. 3.21 in ICON tutorial 2023).
            rho^{n+1} = rho^{n} - dt ( divergence(v^{n+1*} rho^{n}) + vwind_expl_wgt ( rho^{n}_{k-1/2} w^{n}_{k-1/2} - rho^{n}_{k+1/2} w^{n}_{k+1} ) / dz_{k} ) - dt * vwind_impl_wgt ( rho^{n}_{k-1/2} w^{n+1}_{k-1/2} - rho^{n}_{k+1/2} w^{n+1}_{k+1} ) / dz_{k} )
            rho^{n+1} = z_rho_expl - dt * vwind_impl_wgt ( rho^{n}_{k-1/2} w^{n+1}_{k-1/2} - rho^{n}_{k+1/2} w^{n+1}_{k+1} ) / dz_{k} )
            eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
            eta_expl = 1.0 - eta_impl = vwind_expl_wgt
            Note that rho^{n} is used for the implicit mass flux term.
        exner (0:nlev-1):
            Update exner function at full levels (cell center) from the energy equation (see eq. 3.21 or 3.25 in ICON tutorial 2023).
            z_exner_expl = pi'^{n} - dt * rd * pi_{k} / (cvd * rho_{k} * theta_v_{k}) ( divergence(v^{n+1*} rho^{n} theta_v^{n}) + vwind_expl_wgt ( rho^{n}_{k-1/2} w^{n}_{k-1/2} theta_v^{n}_{k-1/2} - rho^{n}_{k+1/2} w^{n}_{k+1} theta_v^{n}_{k+1/2} ) / dz_{k} ) + dt * physics_tendency
            pi^{n+1} = pi_reference + z_exner_expl - dt * vwind_impl_wgt ( rd * pi^{n} ) / ( cvd * rho^{n} * theta_v^{n} ) ( rho^{n}_{k-1/2} w^{n+1}_{k-1/2} theta_v^{n}_{k-1/2} - rho^{n}_{k+1/2} w^{n+1}_{k+1} theta_v^{n}_{k+1/2} ) / dz_{k} )
            eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
            eta_expl = 1.0 - eta_impl = vwind_expl_wgt
            Note that rho^{n} and theta_v^{n} are used for the implicit flux term.
            rho^{n}_{k-1/2} theta_v^{n}_{k-1/2} is represented by z_alpha.
            dt * vwind_impl_wgt (rd * pi^{n}) / (cvd * rho^{n} * theta_v^{n} ) / dz_{k} is represented by z_beta.
        theta_v (0:nlev-1):
            Update virtual potential temperature at full levels (cell center) from the equation of state (see eqs. 3.22 and 3.23 in ICON tutorial 2023).
            rho^{n+1} theta_v^{n+1} = rho^{n} theta_v^{n} + ( cvd * rho^{n} * theta_v^{n} ) / ( rd * pi^{n} ) ( pi^{n+1} - pi^{n} )
        """
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
            """
            z_dwdz_dd (dd3d_lev:nlev-1):
                Compute vertical derivative of vertical velocity at full levels (cell center).
                z_dwdz_dḍ_{k} = ( w_{k-1/2} - w_{k+1/2} ) / dz_{k} - ( contravariant_correction_{k-1/2} - contravariant_correction_{k+1/2} ) / dz_{k}
                contravariant_correction is precomputed by w_concorr_c at half levels.
            """
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
            """
            rho (0:nlev-1):
                Add the boundary tendency to the density at full levels (cell center) for limited area simulations.
            exner (0:nlev-1):
                Add the boundary tendency to the exner function at full levels (cell center) for limited area simulations.
            w (0:nlev):
                Add the boundary tendency to the vertical velocity at full levels (cell center) for limited area simulations.
            """
            stencils_61_62(
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
            """
            z_dwdz_dd (dd3d_lev:nlev-1):
                Compute vertical derivative of vertical velocity at full levels (cell center).
                z_dwdz_dḍ_{k} = ( w_{k-1/2} - w_{k+1/2} ) / dz_{k} - ( contravariant_correction_{k-1/2} - contravariant_correction_{k+1/2} ) / dz_{k}
                contravariant_correction is precomputed by w_concorr_c at half levels.
            """
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

    def run_divdamp(
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
        do_output: bool,
        do_output_step: int,
        do_output_substep: int,
    ):
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
        start_vertex_lb_plus1 = self.grid.get_start_index(
            VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1
        )
        end_vertex_local_minus1 = self.grid.get_end_index(
            VertexDim, HorizontalMarkerIndex.local(VertexDim) - 1
        )

        start_cell_lb_plus3 = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 3
        )
        end_cell_local_minus1 = self.grid.get_end_index(
            CellDim, HorizontalMarkerIndex.local(CellDim) - 1
        )

        def aux_func_divergence(
            input_prognostic_state: PrognosticState,
            input_z_fields: IntermediateFields,
            order: int,
            step: int,
        ):
            if step == 1:
                if order == 1:
                    """
                    z_flxdiv_vn_and_w (0:nlev-1):
                        Compute the divergence of normal wind at full levels (cell center) by Gauss theorem.
                        Add vertical derivative of vertical wind to the divergence of normal wind at full levels (cell center).
                    """
                    compute_divergence_of_flux(
                        geofac_div=self.interpolation_state.geofac_div,
                        vn=input_prognostic_state.vn,
                        dwdz=input_z_fields.z_dwdz_dd,
                        divergence=input_z_fields.z_flxdiv_vn_and_w,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                elif order == 2:
                    """
                    z_flxdiv_vn_and_w (0:nlev-1):
                        Compute the 2nd order divergence of normal wind at full levels (cell center) by Gauss theorem.
                        Add vertical derivative of vertical wind to the divergence of normal wind at full levels (cell center).
                    """
                    compute_pure_2nd_order_divergence_of_flux(
                        geofac_div=self.interpolation_state.geofac_div,
                        vn=input_prognostic_state.vn,
                        dwdz=input_z_fields.z_dwdz_dd,
                        area=self.cell_params.area,
                        pentagon_mask=self.metric_state_nonhydro.pentagon_mask,
                        v2c_area_mask=self.metric_state_nonhydro.v2c_area_mask,
                        divergence=input_z_fields.z_flxdiv_vn_and_w,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )

                elif order == 3:
                    """
                    z_flxdiv_vn_and_w and z_flxdiv_vn_and_w_residual (0:nlev-1):
                        Compute the 2nd order divergence of normal wind at full levels (cell center) by Gauss theorem.
                        Add vertical derivative of vertical wind to the divergence of normal wind at full levels (cell center).
                    """
                    compute_2nd_order_divergence_of_flux(
                        geofac_div=self.interpolation_state.geofac_div,
                        vn=input_prognostic_state.vn,
                        dwdz=input_z_fields.z_dwdz_dd,
                        area=self.cell_params.area,
                        pentagon_mask=self.metric_state_nonhydro.pentagon_mask,
                        v2c_area_mask=self.metric_state_nonhydro.v2c_area_mask,
                        scale=self.config.first_order_div_threshold,
                        divergence=input_z_fields.z_flxdiv_vn_and_w,
                        divergence_residual=input_z_fields.z_flxdiv_vn_and_w_residual,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                else:
                    raise NotImplementedError("Order = {order} must be 1 or 2 or 3")
            elif step == 2:
                if order == 1:
                    """
                    z_flxdiv_graddiv_vn_and_w (0:nlev-1):
                        Compute the divergence of normal wind at full levels (cell center) by Gauss theorem.
                        Add vertical derivative of vertical wind to the divergence of normal wind at full levels (cell center).
                    """
                    compute_divergence_of_flux(
                        geofac_div=self.interpolation_state.geofac_div,
                        vn=input_z_fields.z_graddiv_normal,
                        dwdz=input_z_fields.z_dgraddiv_dz,
                        divergence=input_z_fields.z_flxdiv_graddiv_vn_and_w,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                    if self.config.suppress_vertical_in_3d_divdamp:
                        if device == Device.GPU:
                            assert xp.abs(input_z_fields.z_dgraddiv_dz.ndarray).max() == 0.0
                        else:
                            assert xp.abs(input_z_fields.z_dgraddiv_dz.asnumpy()).max() == 0.0
                elif order == 2:
                    """
                    z_flxdiv_graddiv_vn_and_w (0:nlev-1):
                        Compute the 2nd order divergence of normal wind at full levels (cell center) by Gauss theorem.
                        Add vertical derivative of vertical wind to the divergence of normal wind at full levels (cell center).
                    """
                    compute_pure_2nd_order_divergence_of_flux(
                        geofac_div=self.interpolation_state.geofac_div,
                        vn=input_z_fields.z_graddiv_normal,
                        dwdz=input_z_fields.z_dgraddiv_dz,
                        area=self.cell_params.area,
                        pentagon_mask=self.metric_state_nonhydro.pentagon_mask,
                        v2c_area_mask=self.metric_state_nonhydro.v2c_area_mask,
                        divergence=input_z_fields.z_flxdiv_graddiv_vn_and_w,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                    if self.config.suppress_vertical_in_3d_divdamp:
                        if device == Device.GPU:
                            assert xp.abs(input_z_fields.z_dgraddiv_dz.ndarray).max() == 0.0
                        else:
                            assert xp.abs(input_z_fields.z_dgraddiv_dz.asnumpy()).max() == 0.0
                elif order == 3:
                    """
                    z_flxdiv_graddiv_vn_and_w and z_flxdiv_graddiv_vn_and_w_residual (0:nlev-1):
                        Compute the 2nd order divergence of normal wind at full levels (cell center) by Gauss theorem.
                        Add vertical derivative of vertical wind to the divergence of normal wind at full levels (cell center).
                    """
                    compute_divergence_of_flux(
                        geofac_div=self.interpolation_state.geofac_div,
                        vn=input_z_fields.z_graddiv_normal_residual,
                        dwdz=input_z_fields.z_dgraddiv_dz_residual,
                        divergence=input_z_fields.z_flxdiv_graddiv_vn_and_w_residual,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                    compute_pure_2nd_order_divergence_of_flux(
                        geofac_div=self.interpolation_state.geofac_div,
                        vn=input_z_fields.z_graddiv_normal,
                        dwdz=input_z_fields.z_dgraddiv_dz,
                        area=self.cell_params.area,
                        pentagon_mask=self.metric_state_nonhydro.pentagon_mask,
                        v2c_area_mask=self.metric_state_nonhydro.v2c_area_mask,
                        divergence=input_z_fields.z_flxdiv_graddiv_vn_and_w,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                    if self.config.suppress_vertical_in_3d_divdamp:
                        if device == Device.GPU:
                            assert xp.abs(input_z_fields.z_dgraddiv_dz.ndarray).max() == 0.0
                        else:
                            assert xp.abs(input_z_fields.z_dgraddiv_dz.asnumpy()).max() == 0.0
                else:
                    raise NotImplementedError("Order = {order} must be 1 or 2 or 3")
            else:
                raise NotImplementedError("Step = {step} must be 1 or 2")

        def aux_func_divergence_for_output(
            input_prognostic_state: PrognosticState,
            input_z_fields: IntermediateFields,
            output_flxdiv1_vn: Field[[CellDim, KDim], float],
            output_flxdiv2_vn: Field[[CellDim, KDim], float],
            step: int,
        ):
            if step == 1:
                """
                z_flxdiv_vn_and_w (0:nlev-1):
                    Compute the divergence of normal wind at full levels (cell center) by Gauss theorem.
                    Add vertical derivative of vertical wind to the divergence of normal wind at full levels (cell center).
                """
                compute_divergence_of_flux(
                    geofac_div=self.interpolation_state.geofac_div,
                    vn=input_prognostic_state.vn,
                    dwdz=input_z_fields.z_dwdz_dd,
                    divergence=output_flxdiv1_vn,
                    horizontal_start=start_cell_nudging,
                    horizontal_end=end_cell_local,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider=self.grid.offset_providers,
                )
                """
                z_flxdiv_vn_and_w (0:nlev-1):
                    Compute the 2nd order divergence of normal wind at full levels (cell center) by Gauss theorem.
                    Add vertical derivative of vertical wind to the divergence of normal wind at full levels (cell center).
                """
                compute_pure_2nd_order_divergence_of_flux(
                    geofac_div=self.interpolation_state.geofac_div,
                    vn=input_prognostic_state.vn,
                    dwdz=input_z_fields.z_dwdz_dd,
                    area=self.cell_params.area,
                    pentagon_mask=self.metric_state_nonhydro.pentagon_mask,
                    v2c_area_mask=self.metric_state_nonhydro.v2c_area_mask,
                    divergence=output_flxdiv2_vn,
                    horizontal_start=start_cell_nudging,
                    horizontal_end=end_cell_local,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider=self.grid.offset_providers,
                )

            else:
                raise NotImplementedError("Step = {step} must be 1")

        def aux_func_tangential_wind_and_contravariant_correction_and_dwdz(
            input_prognostic_state: PrognosticState,
            input_diagnostic_state_nh: DiagnosticStateNonHydro,
            input_z_fields: IntermediateFields,
            order: int,
            step: int,
        ):
            if step == 1:
                """
                vt (0:nlev-1):
                    Compute tangential velocity at full levels (edge center) by RBF interpolation from four neighboring
                    edges (diamond shape) and projected to tangential direction.
                z_w_concorr_me (nflatlev:nlev-1):
                    Contravariant correction at full levels at edge center. The correction is equal to vn dz/dn + vt dz/dt, where t is tangent.
                w_concorr_c (nflatlev+1:nlev-1):
                    Contravariant correction at half levels at cell center.
                """
                compute_tangential_wind_and_contravariant(
                    rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
                    ddxn_z_full=self.metric_state_nonhydro.ddxn_z_full,
                    ddxt_z_full=self.metric_state_nonhydro.ddxt_z_full,
                    vn=input_prognostic_state.vn,
                    e_bln_c_s=self.interpolation_state.e_bln_c_s,
                    wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
                    vt=input_z_fields.vt,
                    z_w_concorr_me=input_z_fields.z_w_concorr_me,
                    w_concorr_c=input_diagnostic_state_nh.w_concorr_c,
                    k_field=self.k_field,
                    nflatlev_startindex=self.vertical_params.nflatlev,
                    nlev=self.grid.num_levels,
                    edge_horizontal_start=start_edge_lb_plus4,
                    edge_horizontal_end=end_edge_local_minus2,
                    cell_horizontal_start=start_cell_lb_plus3,
                    cell_horizontal_end=end_cell_local_minus1,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider=self.grid.offset_providers,
                )
                end_cell_nudging_minus1 = self.grid.get_end_index(
                    CellDim,
                    HorizontalMarkerIndex.nudging(CellDim) - 1,
                )
                compute_dwdz_for_divergence_damping(
                    inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                    w=input_prognostic_state.w,
                    w_concorr_c=input_diagnostic_state_nh.w_concorr_c,
                    z_dwdz_dd=input_z_fields.z_dwdz_dd,
                    horizontal_start=start_cell_nudging,
                    horizontal_end=end_cell_local,
                    vertical_start=self.params.kstart_dd3d,
                    vertical_end=self.grid.num_levels,
                    offset_provider=self.grid.offset_providers,
                )
            elif step == 2:
                if self.config.do_proper_contravariant_divdamp:
                    compute_tangential_wind_and_contravariant(
                        rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
                        ddxn_z_full=self.metric_state_nonhydro.ddxn_z_full,
                        ddxt_z_full=self.metric_state_nonhydro.ddxt_z_full,
                        vn=input_z_fields.z_graddiv_normal,
                        e_bln_c_s=self.interpolation_state.e_bln_c_s,
                        wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
                        vt=input_z_fields.z_graddiv_vt,
                        z_w_concorr_me=input_z_fields.z_graddiv_w_concorr_me,
                        w_concorr_c=input_diagnostic_state_nh.graddiv_w_concorr_c,
                        k_field=self.k_field,
                        nflatlev_startindex=self.vertical_params.nflatlev,
                        nlev=self.grid.num_levels,
                        edge_horizontal_start=start_edge_lb_plus4,
                        edge_horizontal_end=end_edge_local_minus2,
                        cell_horizontal_start=start_cell_lb_plus3,
                        cell_horizontal_end=end_cell_local_minus1,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                    end_cell_nudging_minus1 = self.grid.get_end_index(
                        CellDim,
                        HorizontalMarkerIndex.nudging(CellDim) - 1,
                    )
                    compute_dwdz_for_divergence_damping(
                        inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                        w=input_z_fields.z_graddiv_vertical,
                        w_concorr_c=input_diagnostic_state_nh.graddiv_w_concorr_c,
                        z_dwdz_dd=input_z_fields.z_dgraddiv_dz,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=self.params.kstart_dd3d,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )

                    if order == 3:
                        compute_tangential_wind_and_contravariant(
                            rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
                            ddxn_z_full=self.metric_state_nonhydro.ddxn_z_full,
                            ddxt_z_full=self.metric_state_nonhydro.ddxt_z_full,
                            vn=input_z_fields.z_graddiv_normal_residual,
                            e_bln_c_s=self.interpolation_state.e_bln_c_s,
                            wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
                            vt=input_z_fields.z_graddiv_vt_residual,
                            z_w_concorr_me=input_z_fields.z_graddiv_w_concorr_me_residual,
                            w_concorr_c=input_diagnostic_state_nh.graddiv_w_concorr_c_residual,
                            k_field=self.k_field,
                            nflatlev_startindex=self.vertical_params.nflatlev,
                            nlev=self.grid.num_levels,
                            edge_horizontal_start=start_edge_lb_plus4,
                            edge_horizontal_end=end_edge_local_minus2,
                            cell_horizontal_start=start_cell_lb_plus3,
                            cell_horizontal_end=end_cell_local_minus1,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )
                        compute_dwdz_for_divergence_damping(
                            inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                            w=input_z_fields.z_graddiv_vertical_residual,
                            w_concorr_c=input_diagnostic_state_nh.graddiv_w_concorr_c_residual,
                            z_dwdz_dd=input_z_fields.z_dgraddiv_dz_residual,
                            horizontal_start=start_cell_nudging,
                            horizontal_end=end_cell_local,
                            vertical_start=self.params.kstart_dd3d,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )
                else:
                    end_cell_nudging_minus1 = self.grid.get_end_index(
                        CellDim,
                        HorizontalMarkerIndex.nudging(CellDim) - 1,
                    )
                    compute_dgraddiv_dz_for_full3d_divergence_damping(
                        inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                        z_dgraddiv_vertical=input_z_fields.z_graddiv_vertical,
                        z_dgraddiv_dz=input_z_fields.z_dgraddiv_dz,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=self.params.kstart_dd3d,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                    if order == 3:
                        compute_dgraddiv_dz_for_full3d_divergence_damping(
                            inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                            z_dgraddiv_vertical=input_z_fields.z_graddiv_vertical_residual,
                            z_dgraddiv_dz=input_z_fields.z_dgraddiv_dz_residual,
                            horizontal_start=start_cell_nudging,
                            horizontal_end=end_cell_local,
                            vertical_start=self.params.kstart_dd3d,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )

        def aux_func_graddiv(
            input_prognostic_state: PrognosticState,
            input_z_fields: IntermediateFields,
            do_3d_divergence: bool,
            order: int,
            step: int,
        ):
            if do_3d_divergence:
                if step == 1:
                    if order == 3:
                        """
                        z_graddiv_normal and z_graddiv_normal_residual (0:nlev-1):
                            Compute the horizontal gradient of the 3d divergence of normal wind at full levels (edge center).
                        z_graddiv_vertical and z_graddiv_vertical_residual (1:nlev-1):
                            Compute the vertical gradient of the 3d divergence of normal wind at half levels (cell center).
                        """
                        compute_graddiv(
                            inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
                            divergence=input_z_fields.z_flxdiv_vn_and_w,
                            graddiv_normal=input_z_fields.z_graddiv_normal,
                            graddiv_vertical=input_z_fields.z_graddiv_vertical,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=start_cell_nudging,
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )
                        compute_graddiv(
                            inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
                            divergence=input_z_fields.z_flxdiv_vn_and_w_residual,
                            graddiv_normal=input_z_fields.z_graddiv_normal_residual,
                            graddiv_vertical=input_z_fields.z_graddiv_vertical_residual,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=start_cell_nudging,
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )
                        if self.config.suppress_vertical_in_3d_divdamp:
                            init_cell_kdim_field_with_zero_vp(
                                field_with_zero_vp=input_z_fields.z_graddiv_vertical_residual,
                                horizontal_start=start_cell_nudging,
                                horizontal_end=end_cell_local,
                                vertical_start=0,
                                vertical_end=int32(self.grid.num_levels + 1),
                                offset_provider={},
                            )
                            if device == Device.GPU:
                                assert (
                                    xp.abs(input_z_fields.z_graddiv_vertical_residual.ndarray).max()
                                    == 0.0
                                ), f"the max of z_graddiv_vertical_residual and z_graddiv_normal_residual are {xp.abs(input_z_fields.z_graddiv_vertical_residual.ndarray).max()} {xp.abs(input_z_fields.z_graddiv_normal_residual.ndarray).max()}"
                            else:
                                assert (
                                    xp.abs(
                                        input_z_fields.z_graddiv_vertical_residual.asnumpy()
                                    ).max()
                                    == 0.0
                                ), f"the max of z_graddiv_vertical_residual and z_graddiv_normal_residual are {xp.abs(input_z_fields.z_graddiv_vertical_residual.asnumpy()).max()} {xp.abs(input_z_fields.z_graddiv_normal_residual.asnumpy()).max()}"
                    else:
                        """
                        z_graddiv_normal (0:nlev-1):
                            Compute the horizontal gradient of the 3d divergence of normal wind at full levels (edge center).
                        z_graddiv_vertical (1:nlev-1):
                            Compute the vertical gradient of the 3d divergence of normal wind at half levels (cell center).
                        """
                        compute_graddiv(
                            inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
                            divergence=input_z_fields.z_flxdiv_vn_and_w,
                            graddiv_normal=input_z_fields.z_graddiv_normal,
                            graddiv_vertical=input_z_fields.z_graddiv_vertical,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=start_cell_nudging,
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )
                    if self.config.suppress_vertical_in_3d_divdamp:
                        init_cell_kdim_field_with_zero_vp(
                            field_with_zero_vp=input_z_fields.z_graddiv_vertical,
                            horizontal_start=start_cell_nudging,
                            horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=int32(self.grid.num_levels + 1),
                            offset_provider={},
                        )
                        if device == Device.GPU:
                            assert (
                                xp.abs(input_z_fields.z_graddiv_vertical.ndarray).max() == 0.0
                            ), f"the max of z_graddiv_vertical and z_graddiv_normal are {xp.abs(input_z_fields.z_graddiv_vertical.ndarray).max()} {xp.abs(input_z_fields.z_graddiv_normal.ndarray).max()}"
                        else:
                            assert (
                                xp.abs(input_z_fields.z_graddiv_vertical.asnumpy()).max() == 0.0
                            ), f"the max of z_graddiv_vertical and z_graddiv_normal are {xp.abs(input_z_fields.z_graddiv_vertical.asnumpy()).max()} {xp.abs(input_z_fields.z_graddiv_normal.asnumpy()).max()}"
                elif step == 2:
                    if order == 3:
                        """
                        z_graddiv2_normal and z_graddiv2_normal_residual (0:nlev-1):
                            Compute the horizontal gradient of the 3d divergence of normal wind at full levels (edge center).
                        z_graddiv2_vertical and z_graddiv2_vertical_residual (1:nlev-1):
                            Compute the vertical gradient of the 3d divergence of normal wind at half levels (cell center).
                        """
                        compute_graddiv(
                            inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
                            divergence=input_z_fields.z_flxdiv_graddiv_vn_and_w,
                            graddiv_normal=self.z_graddiv2_normal,
                            graddiv_vertical=self.z_graddiv2_vertical,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=start_cell_nudging,
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )
                        compute_graddiv(
                            inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
                            divergence=input_z_fields.z_flxdiv_graddiv_vn_and_w_residual,
                            graddiv_normal=self.z_graddiv2_normal_residual,
                            graddiv_vertical=self.z_graddiv2_vertical_residual,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=start_cell_nudging,
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )
                        if self.config.suppress_vertical_in_3d_divdamp:
                            init_cell_kdim_field_with_zero_vp(
                                field_with_zero_vp=self.z_graddiv2_vertical_residual,
                                horizontal_start=start_cell_nudging,
                                horizontal_end=end_cell_local,
                                vertical_start=0,
                                vertical_end=int32(self.grid.num_levels + 1),
                                offset_provider={},
                            )
                            if device == Device.GPU:
                                assert (
                                    xp.abs(self.z_graddiv2_vertical_residual.ndarray).max() == 0.0
                                ), f"the max of z_graddiv2_vertical_residual and z_graddiv2_normal_residual are {xp.abs(self.z_graddiv2_vertical_residual.ndarray).max()} {xp.abs(self.z_graddiv2_normal_residual.ndarray).max()}"
                            else:
                                assert (
                                    xp.abs(self.z_graddiv2_vertical_residual.asnumpy()).max() == 0.0
                                ), f"the max of z_graddiv2_vertical_residual and z_graddiv2_normal_residual are {xp.abs(self.z_graddiv2_vertical_residual.asnumpy()).max()} {xp.abs(self.z_graddiv2_normal_residual.asnumpy()).max()}"
                    else:
                        """
                        z_graddiv2_normal (0:nlev-1):
                            Compute the horizontal gradient of the 3d divergence of normal wind at full levels (edge center).
                        z_graddiv2_vertical (1:nlev-1):
                            Compute the vertical gradient of the 3d divergence of normal wind at half levels (cell center).
                        """
                        compute_graddiv(
                            inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
                            divergence=input_z_fields.z_flxdiv_graddiv_vn_and_w,
                            graddiv_normal=self.z_graddiv2_normal,
                            graddiv_vertical=self.z_graddiv2_vertical,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=start_cell_nudging,
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )
                    if self.config.suppress_vertical_in_3d_divdamp:
                        init_cell_kdim_field_with_zero_vp(
                            field_with_zero_vp=self.z_graddiv2_vertical,
                            horizontal_start=start_cell_nudging,
                            horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=int32(self.grid.num_levels + 1),
                            offset_provider={},
                        )
                        if device == Device.GPU:
                            assert (
                                xp.abs(self.z_graddiv2_vertical.ndarray).max() == 0.0
                            ), f"the max of z_graddiv2_vertical and z_graddiv2_normal are {xp.abs(self.z_graddiv2_vertical.ndarray).max()} {xp.abs(self.z_graddiv2_normal.ndarray).max()}"
                        else:
                            assert (
                                xp.abs(self.z_graddiv2_vertical.asnumpy()).max() == 0.0
                            ), f"the max of z_graddiv2_vertical and z_graddiv2_normal are {xp.abs(self.z_graddiv2_vertical.asnumpy()).max()} {xp.abs(self.z_graddiv2_normal.asnumpy()).max()}"
            else:
                if step == 1:
                    """
                    z_graddiv_vn (0:nlev-1):
                        Compute the laplacian of vn at full levels (edge center).
                    """
                    compute_graddiv_of_vn(
                        geofac_grdiv=self.interpolation_state.geofac_grdiv,
                        vn=input_prognostic_state.vn,
                        z_graddiv_vn=input_z_fields.z_graddiv_vn,
                        horizontal_start=start_edge_nudging_plus1,
                        horizontal_end=end_edge_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                    """
                    z_graddiv_vn (dd3d_lev:nlev-1):
                        Add vertical wind derivative to the normal gradient of divergence at full levels (edge center).
                        z_graddiv_vn_{k} = z_graddiv_vn_{k} + scalfac_dd3d_{k} d2w_{k}/dzdn
                    """
                    add_vertical_wind_derivative_to_divergence_damping(
                        hmask_dd3d=self.metric_state_nonhydro.hmask_dd3d,
                        scalfac_dd3d=self.metric_state_nonhydro.scalfac_dd3d,
                        inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                        z_dwdz_dd=input_z_fields.z_dwdz_dd,
                        z_graddiv_vn=input_z_fields.z_graddiv_vn,
                        horizontal_start=start_edge_lb_plus6,
                        horizontal_end=end_edge_local_minus2,
                        vertical_start=self.params.kstart_dd3d,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                elif step == 2:
                    """
                    z_graddiv2_vn (0:nlev-1):
                        Compute the double laplacian of vn at full levels (edge center).
                    """
                    compute_graddiv2_of_vn(
                        geofac_grdiv=self.interpolation_state.geofac_grdiv,
                        z_graddiv_vn=input_z_fields.z_graddiv_vn,
                        z_graddiv2_vn=self.z_graddiv2_vn,
                        horizontal_start=start_edge_nudging_plus1,
                        horizontal_end=end_edge_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )

                else:
                    raise NotImplementedError("Step = {step} must be 1 or 2")

        def copy_data_to_output(
            input_prognostic_state: PrognosticState,
            input_z_fields: IntermediateFields,
            output_group: str,
            do_o2: bool,
            do_3d_divergence_damping: bool,
        ):
            if output_group == "before":
                aux_func_divergence_for_output(
                    input_prognostic_state,
                    input_z_fields,
                    self.output_intermediate_fields.output_before_flxdiv1_vn,
                    self.output_intermediate_fields.output_before_flxdiv2_vn,
                    step=1,
                )
                copy_edge_kdim_field_to_vp(
                    field=input_prognostic_state.vn,
                    field_copy=self.output_intermediate_fields.output_before_vn,
                    horizontal_start=int32(0),
                    horizontal_end=end_edge_local,
                    vertical_start=int32(0),
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )
                copy_cell_kdim_field_to_vp(
                    field=input_prognostic_state.w,
                    field_copy=self.output_intermediate_fields.output_before_w,
                    horizontal_start=int32(0),
                    horizontal_end=end_cell_local,
                    vertical_start=int32(0),
                    vertical_end=self.grid.num_levels + int32(1),
                    offset_provider={},
                )
            elif output_group == "mid":
                if do_o2:
                    if do_3d_divergence_damping:
                        copy_edge_kdim_field_to_vp(
                            field=input_z_fields.z_graddiv_normal,
                            field_copy=self.output_intermediate_fields.output_graddiv_normal,
                            horizontal_start=int32(0),
                            horizontal_end=end_edge_local,
                            vertical_start=int32(0),
                            vertical_end=self.grid.num_levels,
                            offset_provider={},
                        )
                        copy_cell_kdim_field_to_vp(
                            field=input_z_fields.z_graddiv_vertical,
                            field_copy=self.output_intermediate_fields.output_graddiv_vertical,
                            horizontal_start=int32(0),
                            horizontal_end=end_cell_local,
                            vertical_start=int32(0),
                            vertical_end=self.grid.num_levels + int32(1),
                            offset_provider={},
                        )
                    else:
                        copy_edge_kdim_field_to_vp(
                            field=input_z_fields.z_graddiv_vn,
                            field_copy=self.output_intermediate_fields.output_graddiv_vn,
                            horizontal_start=int32(0),
                            horizontal_end=end_edge_local,
                            vertical_start=int32(0),
                            vertical_end=self.grid.num_levels,
                            offset_provider={},
                        )
                else:
                    if do_3d_divergence_damping:
                        copy_edge_kdim_field_to_vp(
                            field=self.z_graddiv2_normal,
                            field_copy=self.output_intermediate_fields.output_graddiv_normal,
                            horizontal_start=int32(0),
                            horizontal_end=end_edge_local,
                            vertical_start=int32(0),
                            vertical_end=self.grid.num_levels,
                            offset_provider={},
                        )
                        copy_cell_kdim_field_to_vp(
                            field=self.z_graddiv2_vertical,
                            field_copy=self.output_intermediate_fields.output_graddiv_vertical,
                            horizontal_start=int32(0),
                            horizontal_end=end_cell_local,
                            vertical_start=int32(0),
                            vertical_end=self.grid.num_levels + int32(1),
                            offset_provider={},
                        )
                    else:
                        copy_edge_kdim_field_to_vp(
                            field=self.z_graddiv2_vn,
                            field_copy=self.output_intermediate_fields.output_graddiv_vn,
                            horizontal_start=int32(0),
                            horizontal_end=end_edge_local,
                            vertical_start=int32(0),
                            vertical_end=self.grid.num_levels,
                            offset_provider={},
                        )

            elif output_group == "after":
                aux_func_divergence_for_output(
                    input_prognostic_state,
                    input_z_fields,
                    self.output_intermediate_fields.output_after_flxdiv1_vn,
                    self.output_intermediate_fields.output_after_flxdiv2_vn,
                    step=1,
                )
                copy_edge_kdim_field_to_vp(
                    field=input_prognostic_state.vn,
                    field_copy=self.output_intermediate_fields.output_after_vn,
                    horizontal_start=int32(0),
                    horizontal_end=end_edge_local,
                    vertical_start=int32(0),
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )
                copy_cell_kdim_field_to_vp(
                    field=input_prognostic_state.w,
                    field_copy=self.output_intermediate_fields.output_after_w,
                    horizontal_start=int32(0),
                    horizontal_end=end_cell_local,
                    vertical_start=int32(0),
                    vertical_end=self.grid.num_levels + int32(1),
                    offset_provider={},
                )

        def aux_func_compute_divergence_damping(
            input_prognostic_state: PrognosticState,
            input_diagnostic_state_nh: DiagnosticStateNonHydro,
            input_z_fields: IntermediateFields,
            order: int,
            do_o2: bool,
            do_compute_diagnostics: bool,
            do_3d_divergence_damping: bool,
        ):
            if do_o2:
                if do_compute_diagnostics:
                    aux_func_tangential_wind_and_contravariant_correction_and_dwdz(
                        input_prognostic_state,
                        input_diagnostic_state_nh,
                        input_z_fields,
                        order,
                        step=1,
                    )

                if do_3d_divergence_damping:
                    aux_func_divergence(input_prognostic_state, input_z_fields, order, step=1)
                aux_func_graddiv(
                    input_prognostic_state, input_z_fields, do_3d_divergence_damping, order, step=1
                )
            else:
                if do_compute_diagnostics:
                    aux_func_tangential_wind_and_contravariant_correction_and_dwdz(
                        input_prognostic_state,
                        input_diagnostic_state_nh,
                        input_z_fields,
                        order=order,
                        step=1,
                    )

                if do_3d_divergence_damping:
                    aux_func_divergence(input_prognostic_state, input_z_fields, order, step=1)
                aux_func_graddiv(
                    input_prognostic_state, input_z_fields, do_3d_divergence_damping, order, step=1
                )

                if do_3d_divergence_damping:
                    aux_func_tangential_wind_and_contravariant_correction_and_dwdz(
                        input_prognostic_state,
                        input_diagnostic_state_nh,
                        input_z_fields,
                        order,
                        step=2,
                    )
                    aux_func_divergence(input_prognostic_state, input_z_fields, order, step=2)
                aux_func_graddiv(
                    input_prognostic_state, input_z_fields, do_3d_divergence_damping, order, step=2
                )

        def aux_func_apply_divergence_damping(
            input_prognostic_state: PrognosticState,
            input_z_fields: IntermediateFields,
            do_o2: bool,
            do_3d_divergence_damping: bool,
            order: int,
        ):
            if do_3d_divergence_damping:
                if do_o2:
                    """
                    vn (0:nlev-1):
                        Apply the higher order divergence damping to vn at full levels (edge center).
                        vn = vn + scal_divdamp * Del(normal_direction) Div(V)
                    w (1:nlev-1):
                        Apply the higher order divergence damping to w at half levels (cell center).
                        w = w + scal_divdamp_half * Del(vertical_direction) Div(V)
                    """
                    apply_3d_divergence_damping(
                        scal_divdamp=self.scal_divdamp_o2,
                        scal_divdamp_half=self.scal_divdamp_o2_half,
                        graddiv_normal=input_z_fields.z_graddiv_normal,
                        graddiv_vertical=input_z_fields.z_graddiv_vertical,
                        vn=input_prognostic_state.vn,
                        # w=input_prognostic_state.w,
                        z_w_divdamp=input_z_fields.z_w_divdamp,
                        edge_horizontal_start=start_edge_nudging_plus1,
                        edge_horizontal_end=end_edge_local,
                        cell_horizontal_start=int32(0),
                        cell_horizontal_end=end_cell_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider={},
                    )
                else:
                    if order == 3:
                        apply_3d_divergence_damping(
                            scal_divdamp=self.scal_divdamp,
                            scal_divdamp_half=self.scal_divdamp_half,
                            graddiv_normal=self.z_graddiv2_normal,
                            graddiv_vertical=self.z_graddiv2_vertical,
                            vn=input_prognostic_state.vn,
                            # w=input_prognostic_state.w,
                            z_w_divdamp=input_z_fields.z_w_divdamp,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=int32(0),
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider={},
                        )
                        apply_3d_divergence_damping(
                            scal_divdamp=self.scal_divdamp,
                            scal_divdamp_half=self.scal_divdamp_half,
                            graddiv_normal=self.z_graddiv2_normal_residual,
                            graddiv_vertical=self.z_graddiv2_vertical_residual,
                            vn=input_prognostic_state.vn,
                            # w=input_prognostic_state.w,
                            z_w_divdamp=input_z_fields.z_w_divdamp,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=int32(0),
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider={},
                        )
                    else:
                        apply_3d_divergence_damping(
                            scal_divdamp=self.scal_divdamp,
                            scal_divdamp_half=self.scal_divdamp_half,
                            graddiv_normal=self.z_graddiv2_normal,
                            graddiv_vertical=self.z_graddiv2_vertical,
                            vn=input_prognostic_state.vn,
                            # w=input_prognostic_state.w,
                            z_w_divdamp=input_z_fields.z_w_divdamp,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=int32(0),
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider={},
                        )
            else:
                if do_o2:
                    apply_4th_order_divergence_damping(
                        scal_divdamp=self.scal_divdamp_o2,
                        z_graddiv2_vn=input_z_fields.z_graddiv_vn,
                        vn=input_prognostic_state.vn,
                        horizontal_start=start_edge_nudging_plus1,
                        horizontal_end=end_edge_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider={},
                    )
                else:
                    apply_4th_order_divergence_damping(
                        scal_divdamp=self.scal_divdamp,
                        z_graddiv2_vn=self.z_graddiv2_vn,
                        vn=input_prognostic_state.vn,
                        horizontal_start=start_edge_nudging_plus1,
                        horizontal_end=end_edge_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider={},
                    )

        scal_divdamp_o2 = divdamp_fac_o2 * self.cell_params.mean_cell_area

        init_cell_kdim_field_with_zero_vp(
            field_with_zero_vp=z_fields.z_w_divdamp,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=0,
            vertical_end=int32(self.grid.num_levels + 1),
            offset_provider={},
        )

        copy_cell_kdim_field_to_vp(
            field=prognostic_state[nnow].w,
            field_copy=prognostic_state[nnew].w,
            horizontal_start=int32(0),
            horizontal_end=end_cell_local,
            vertical_start=int32(0),
            vertical_end=int32(self.grid.num_levels + 1),
            offset_provider={},
        )
        copy_edge_kdim_field_to_vp(
            field=prognostic_state[nnow].vn,
            field_copy=prognostic_state[nnew].vn,
            horizontal_start=int32(0),
            horizontal_end=end_edge_local,
            vertical_start=int32(0),
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )
        # TODO: to cached program
        # _calculate_divdamp_fields(
        #     self.enh_divdamp_fac,
        #     int32(self.config.divdamp_order),
        #     self.cell_params.mean_cell_area,
        #     divdamp_fac_o2,
        #     self.config.nudge_max_coeff,
        #     constants.dbl_eps,
        #     out=(self.scal_divdamp, self.scal_divdamp_o2, self._bdy_divdamp),
        #     offset_provider={},
        # )
        calculate_divdamp_fields(
            self.enh_divdamp_fac,
            self.scal_divdamp,
            self.scal_divdamp_o2,
            self._bdy_divdamp,
            int32(self.config.divdamp_order),
            self.cell_params.mean_cell_area,
            float(0.0),
            self.config.nudge_max_coeff,
            constants.dbl_eps,
            self.config.scal_divsign,
            offset_provider={},
        )
        calculate_scal_divdamp_half(
            scal_divdamp=self.scal_divdamp,
            scal_divdamp_o2=self.scal_divdamp_o2,
            vct_a=self.vertical_params.vct_a,
            divdamp_fac_w=self.config.divdamp_fac_w,
            scal_divdamp_half=self.scal_divdamp_half,
            scal_divdamp_o2_half=self.scal_divdamp_o2_half,
            # k_field=self.k_field,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider={"Koff": KDim},
        )

        aux_func_compute_divergence_damping(
            prognostic_state[nnew],
            diagnostic_state_nh,
            z_fields,
            order=self.config.divergence_order,
            do_o2=self.config.do_o2_divdamp,
            do_compute_diagnostics=True,
            do_3d_divergence_damping=self.config.do_3d_divergence_damping,
        )
        copy_data_to_output(
            prognostic_state[nnew],
            z_fields,
            "before",
            self.config.do_o2_divdamp,
            self.config.do_3d_divergence_damping,
        )
        aux_func_apply_divergence_damping(
            prognostic_state[nnew],
            z_fields,
            do_o2=self.config.do_o2_divdamp,
            do_3d_divergence_damping=self.config.do_3d_divergence_damping,
            order=self.config.divergence_order,
        )
        apply_3d_divergence_damping_only_to_w(
            prognostic_state[nnew].w,
            z_fields.z_w_divdamp,
            horizontal_start=int32(0),
            horizontal_end=end_cell_local,
            vertical_start=int32(1),
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )
        aux_func_compute_divergence_damping(
            prognostic_state[nnew],
            diagnostic_state_nh,
            z_fields,
            order=self.config.divergence_order,
            do_o2=self.config.do_o2_divdamp,
            do_compute_diagnostics=True,
            do_3d_divergence_damping=self.config.do_3d_divergence_damping,
        )
        copy_data_to_output(
            prognostic_state[nnew],
            z_fields,
            "after",
            self.config.do_o2_divdamp,
            self.config.do_3d_divergence_damping,
        )

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
        do_output: bool,
        do_output_step: int,
        do_output_substep: int,
    ):
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
        start_vertex_lb_plus1 = self.grid.get_start_index(
            VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1
        )
        end_vertex_local_minus1 = self.grid.get_end_index(
            VertexDim, HorizontalMarkerIndex.local(VertexDim) - 1
        )

        start_cell_lb_plus3 = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 3
        )
        end_cell_local_minus1 = self.grid.get_end_index(
            CellDim, HorizontalMarkerIndex.local(CellDim) - 1
        )

        def aux_func_divergence(
            input_prognostic_state: PrognosticState,
            input_z_fields: IntermediateFields,
            order: int,
            step: int,
        ):
            if step == 1:
                if order == 1:
                    """
                    z_flxdiv_vn_and_w (0:nlev-1):
                        Compute the divergence of normal wind at full levels (cell center) by Gauss theorem.
                        Add vertical derivative of vertical wind to the divergence of normal wind at full levels (cell center).
                    """
                    compute_divergence_of_flux(
                        geofac_div=self.interpolation_state.geofac_div,
                        vn=input_prognostic_state.vn,
                        dwdz=input_z_fields.z_dwdz_dd,
                        divergence=input_z_fields.z_flxdiv_vn_and_w,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                elif order == 2:
                    """
                    z_flxdiv_vn_and_w (0:nlev-1):
                        Compute the 2nd order divergence of normal wind at full levels (cell center) by Gauss theorem.
                        Add vertical derivative of vertical wind to the divergence of normal wind at full levels (cell center).
                    """
                    compute_pure_2nd_order_divergence_of_flux(
                        geofac_div=self.interpolation_state.geofac_div,
                        vn=input_prognostic_state.vn,
                        dwdz=input_z_fields.z_dwdz_dd,
                        area=self.cell_params.area,
                        pentagon_mask=self.metric_state_nonhydro.pentagon_mask,
                        v2c_area_mask=self.metric_state_nonhydro.v2c_area_mask,
                        divergence=input_z_fields.z_flxdiv_vn_and_w,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                elif order == 3:
                    """
                    z_flxdiv_vn_and_w and z_flxdiv_vn_and_w_residual (0:nlev-1):
                        Compute the 2nd order divergence of normal wind at full levels (cell center) by Gauss theorem.
                        Add vertical derivative of vertical wind to the divergence of normal wind at full levels (cell center).
                    """
                    compute_2nd_order_divergence_of_flux(
                        geofac_div=self.interpolation_state.geofac_div,
                        vn=input_prognostic_state.vn,
                        dwdz=input_z_fields.z_dwdz_dd,
                        area=self.cell_params.area,
                        pentagon_mask=self.metric_state_nonhydro.pentagon_mask,
                        v2c_area_mask=self.metric_state_nonhydro.v2c_area_mask,
                        scale=self.config.first_order_div_threshold,
                        divergence=input_z_fields.z_flxdiv_vn_and_w,
                        divergence_residual=input_z_fields.z_flxdiv_vn_and_w_residual,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                else:
                    raise NotImplementedError("Order = {order} must be 1 or 2 or 3")
            elif step == 2:
                if order == 1:
                    """
                    z_flxdiv_graddiv_vn_and_w (0:nlev-1):
                        Compute the divergence of normal wind at full levels (cell center) by Gauss theorem.
                        Add vertical derivative of vertical wind to the divergence of normal wind at full levels (cell center).
                    """
                    compute_divergence_of_flux(
                        geofac_div=self.interpolation_state.geofac_div,
                        vn=input_z_fields.z_graddiv_normal,
                        dwdz=input_z_fields.z_dgraddiv_dz,
                        divergence=input_z_fields.z_flxdiv_graddiv_vn_and_w,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                    if self.config.suppress_vertical_in_3d_divdamp:
                        if device == Device.GPU:
                            assert xp.abs(input_z_fields.z_dgraddiv_dz.ndarray).max() == 0.0
                        else:
                            assert xp.abs(input_z_fields.z_dgraddiv_dz.asnumpy()).max() == 0.0
                elif order == 2:
                    """
                    z_flxdiv_graddiv_vn_and_w (0:nlev-1):
                        Compute the 2nd order divergence of normal wind at full levels (cell center) by Gauss theorem.
                        Add vertical derivative of vertical wind to the divergence of normal wind at full levels (cell center).
                    """
                    compute_pure_2nd_order_divergence_of_flux(
                        geofac_div=self.interpolation_state.geofac_div,
                        vn=input_z_fields.z_graddiv_normal,
                        dwdz=input_z_fields.z_dgraddiv_dz,
                        area=self.cell_params.area,
                        pentagon_mask=self.metric_state_nonhydro.pentagon_mask,
                        v2c_area_mask=self.metric_state_nonhydro.v2c_area_mask,
                        divergence=input_z_fields.z_flxdiv_graddiv_vn_and_w,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                    if self.config.suppress_vertical_in_3d_divdamp:
                        if device == Device.GPU:
                            assert xp.abs(input_z_fields.z_dgraddiv_dz.ndarray).max() == 0.0
                        else:
                            assert xp.abs(input_z_fields.z_dgraddiv_dz.asnumpy()).max() == 0.0
                elif order == 3:
                    """
                    z_flxdiv_graddiv_vn_and_w and z_flxdiv_graddiv_vn_and_w_residual (0:nlev-1):
                        Compute the 2nd order divergence of normal wind at full levels (cell center) by Gauss theorem.
                        Add vertical derivative of vertical wind to the divergence of normal wind at full levels (cell center).
                    """
                    compute_divergence_of_flux(
                        geofac_div=self.interpolation_state.geofac_div,
                        vn=input_z_fields.z_graddiv_normal_residual,
                        dwdz=input_z_fields.z_dgraddiv_dz_residual,
                        divergence=input_z_fields.z_flxdiv_graddiv_vn_and_w_residual,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                    compute_pure_2nd_order_divergence_of_flux(
                        geofac_div=self.interpolation_state.geofac_div,
                        vn=input_z_fields.z_graddiv_normal,
                        dwdz=input_z_fields.z_dgraddiv_dz,
                        area=self.cell_params.area,
                        pentagon_mask=self.metric_state_nonhydro.pentagon_mask,
                        v2c_area_mask=self.metric_state_nonhydro.v2c_area_mask,
                        divergence=input_z_fields.z_flxdiv_graddiv_vn_and_w,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                    if self.config.suppress_vertical_in_3d_divdamp:
                        if device == Device.GPU:
                            assert xp.abs(input_z_fields.z_dgraddiv_dz.ndarray).max() == 0.0
                            assert (
                                xp.abs(input_z_fields.z_dgraddiv_dz_residual.ndarray).max() == 0.0
                            )
                        else:
                            assert xp.abs(input_z_fields.z_dgraddiv_dz.asnumpy()).max() == 0.0
                            assert (
                                xp.abs(input_z_fields.z_dgraddiv_dz_residual.asnumpy()).max() == 0.0
                            )
                else:
                    raise NotImplementedError("Order = {order} must be 1 or 2 or 3")
            else:
                raise NotImplementedError("Step = {step} must be 1 or 2")

        def aux_func_divergence_for_output(
            input_prognostic_state: PrognosticState,
            input_z_fields: IntermediateFields,
            output_flxdiv1_vn: Field[[CellDim, KDim], float],
            output_flxdiv2_vn: Field[[CellDim, KDim], float],
            step: int,
        ):
            if step == 1:
                """
                z_flxdiv_vn_and_w (0:nlev-1):
                    Compute the divergence of normal wind at full levels (cell center) by Gauss theorem.
                    Add vertical derivative of vertical wind to the divergence of normal wind at full levels (cell center).
                """
                compute_divergence_of_flux(
                    geofac_div=self.interpolation_state.geofac_div,
                    vn=input_prognostic_state.vn,
                    dwdz=input_z_fields.z_dwdz_dd,
                    divergence=output_flxdiv1_vn,
                    horizontal_start=start_cell_nudging,
                    horizontal_end=end_cell_local,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider=self.grid.offset_providers,
                )
                """
                z_flxdiv_vn_and_w (0:nlev-1):
                    Compute the 2nd order divergence of normal wind at full levels (cell center) by Gauss theorem.
                    Add vertical derivative of vertical wind to the divergence of normal wind at full levels (cell center).
                """
                compute_pure_2nd_order_divergence_of_flux(
                    geofac_div=self.interpolation_state.geofac_div,
                    vn=input_prognostic_state.vn,
                    dwdz=input_z_fields.z_dwdz_dd,
                    area=self.cell_params.area,
                    pentagon_mask=self.metric_state_nonhydro.pentagon_mask,
                    v2c_area_mask=self.metric_state_nonhydro.v2c_area_mask,
                    divergence=output_flxdiv2_vn,
                    horizontal_start=start_cell_nudging,
                    horizontal_end=end_cell_local,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider=self.grid.offset_providers,
                )

            else:
                raise NotImplementedError("Step = {step} must be 1")

        def aux_func_tangential_wind_and_contravariant_correction_and_dwdz(
            input_prognostic_state: PrognosticState,
            input_diagnostic_state_nh: DiagnosticStateNonHydro,
            input_z_fields: IntermediateFields,
            order: int,
            step: int,
        ):
            if step == 1:
                """
                vt (0:nlev-1):
                    Compute tangential velocity at full levels (edge center) by RBF interpolation from four neighboring
                    edges (diamond shape) and projected to tangential direction.
                z_w_concorr_me (nflatlev:nlev-1):
                    Contravariant correction at full levels at edge center. The correction is equal to vn dz/dn + vt dz/dt, where t is tangent.
                w_concorr_c (nflatlev+1:nlev-1):
                    Contravariant correction at half levels at cell center.
                """
                compute_tangential_wind_and_contravariant(
                    rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
                    ddxn_z_full=self.metric_state_nonhydro.ddxn_z_full,
                    ddxt_z_full=self.metric_state_nonhydro.ddxt_z_full,
                    vn=input_prognostic_state.vn,
                    e_bln_c_s=self.interpolation_state.e_bln_c_s,
                    wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
                    vt=input_z_fields.vt,
                    z_w_concorr_me=input_z_fields.z_w_concorr_me,
                    w_concorr_c=input_diagnostic_state_nh.w_concorr_c,
                    k_field=self.k_field,
                    nflatlev_startindex=self.vertical_params.nflatlev,
                    nlev=self.grid.num_levels,
                    edge_horizontal_start=start_edge_lb_plus4,
                    edge_horizontal_end=end_edge_local_minus2,
                    cell_horizontal_start=start_cell_lb_plus3,
                    cell_horizontal_end=end_cell_local_minus1,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider=self.grid.offset_providers,
                )
                end_cell_nudging_minus1 = self.grid.get_end_index(
                    CellDim,
                    HorizontalMarkerIndex.nudging(CellDim) - 1,
                )
                compute_dwdz_for_divergence_damping(
                    inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                    w=input_prognostic_state.w,
                    w_concorr_c=input_diagnostic_state_nh.w_concorr_c,
                    z_dwdz_dd=input_z_fields.z_dwdz_dd,
                    horizontal_start=start_cell_nudging,
                    horizontal_end=end_cell_local,
                    vertical_start=self.params.kstart_dd3d,
                    vertical_end=self.grid.num_levels,
                    offset_provider=self.grid.offset_providers,
                )
            elif step == 2:
                if self.config.do_proper_contravariant_divdamp:
                    compute_tangential_wind_and_contravariant(
                        rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
                        ddxn_z_full=self.metric_state_nonhydro.ddxn_z_full,
                        ddxt_z_full=self.metric_state_nonhydro.ddxt_z_full,
                        vn=input_z_fields.z_graddiv_normal,
                        e_bln_c_s=self.interpolation_state.e_bln_c_s,
                        wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
                        vt=input_z_fields.z_graddiv_vt,
                        z_w_concorr_me=input_z_fields.z_graddiv_w_concorr_me,
                        w_concorr_c=input_diagnostic_state_nh.graddiv_w_concorr_c,
                        k_field=self.k_field,
                        nflatlev_startindex=self.vertical_params.nflatlev,
                        nlev=self.grid.num_levels,
                        edge_horizontal_start=start_edge_lb_plus4,
                        edge_horizontal_end=end_edge_local_minus2,
                        cell_horizontal_start=start_cell_lb_plus3,
                        cell_horizontal_end=end_cell_local_minus1,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                    end_cell_nudging_minus1 = self.grid.get_end_index(
                        CellDim,
                        HorizontalMarkerIndex.nudging(CellDim) - 1,
                    )
                    compute_dwdz_for_divergence_damping(
                        inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                        w=input_z_fields.z_graddiv_vertical,
                        w_concorr_c=input_diagnostic_state_nh.graddiv_w_concorr_c,
                        z_dwdz_dd=input_z_fields.z_dgraddiv_dz,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=self.params.kstart_dd3d,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )

                    if order == 3:
                        compute_tangential_wind_and_contravariant(
                            rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
                            ddxn_z_full=self.metric_state_nonhydro.ddxn_z_full,
                            ddxt_z_full=self.metric_state_nonhydro.ddxt_z_full,
                            vn=input_z_fields.z_graddiv_normal_residual,
                            e_bln_c_s=self.interpolation_state.e_bln_c_s,
                            wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
                            vt=input_z_fields.z_graddiv_vt_residual,
                            z_w_concorr_me=input_z_fields.z_graddiv_w_concorr_me_residual,
                            w_concorr_c=input_diagnostic_state_nh.graddiv_w_concorr_c_residual,
                            k_field=self.k_field,
                            nflatlev_startindex=self.vertical_params.nflatlev,
                            nlev=self.grid.num_levels,
                            edge_horizontal_start=start_edge_lb_plus4,
                            edge_horizontal_end=end_edge_local_minus2,
                            cell_horizontal_start=start_cell_lb_plus3,
                            cell_horizontal_end=end_cell_local_minus1,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )
                        compute_dwdz_for_divergence_damping(
                            inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                            w=input_z_fields.z_graddiv_vertical_residual,
                            w_concorr_c=input_diagnostic_state_nh.graddiv_w_concorr_c_residual,
                            z_dwdz_dd=input_z_fields.z_dgraddiv_dz_residual,
                            horizontal_start=start_cell_nudging,
                            horizontal_end=end_cell_local,
                            vertical_start=self.params.kstart_dd3d,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )
                else:
                    end_cell_nudging_minus1 = self.grid.get_end_index(
                        CellDim,
                        HorizontalMarkerIndex.nudging(CellDim) - 1,
                    )
                    compute_dgraddiv_dz_for_full3d_divergence_damping(
                        inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                        z_dgraddiv_vertical=input_z_fields.z_graddiv_vertical,
                        z_dgraddiv_dz=input_z_fields.z_dgraddiv_dz,
                        horizontal_start=start_cell_nudging,
                        horizontal_end=end_cell_local,
                        vertical_start=self.params.kstart_dd3d,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                    if order == 3:
                        compute_dgraddiv_dz_for_full3d_divergence_damping(
                            inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                            z_dgraddiv_vertical=input_z_fields.z_graddiv_vertical_residual,
                            z_dgraddiv_dz=input_z_fields.z_dgraddiv_dz_residual,
                            horizontal_start=start_cell_nudging,
                            horizontal_end=end_cell_local,
                            vertical_start=self.params.kstart_dd3d,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )

        def aux_func_graddiv(
            input_prognostic_state: PrognosticState,
            input_z_fields: IntermediateFields,
            do_3d_divergence: bool,
            order: int,
            step: int,
        ):
            if do_3d_divergence:
                if step == 1:
                    if order == 3:
                        """
                        z_graddiv_normal and z_graddiv_normal_residual (0:nlev-1):
                            Compute the horizontal gradient of the 3d divergence of normal wind at full levels (edge center).
                        z_graddiv_vertical and z_graddiv_vertical_residual (1:nlev-1):
                            Compute the vertical gradient of the 3d divergence of normal wind at half levels (cell center).
                        """
                        compute_graddiv(
                            inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
                            divergence=input_z_fields.z_flxdiv_vn_and_w,
                            graddiv_normal=input_z_fields.z_graddiv_normal,
                            graddiv_vertical=input_z_fields.z_graddiv_vertical,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=start_cell_nudging,
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )
                        compute_graddiv(
                            inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
                            divergence=input_z_fields.z_flxdiv_vn_and_w_residual,
                            graddiv_normal=input_z_fields.z_graddiv_normal_residual,
                            graddiv_vertical=input_z_fields.z_graddiv_vertical_residual,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=start_cell_nudging,
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )
                        if self.config.suppress_vertical_in_3d_divdamp:
                            init_cell_kdim_field_with_zero_vp(
                                field_with_zero_vp=input_z_fields.z_graddiv_vertical_residual,
                                horizontal_start=start_cell_nudging,
                                horizontal_end=end_cell_local,
                                vertical_start=0,
                                vertical_end=int32(self.grid.num_levels + 1),
                                offset_provider={},
                            )
                            if device == Device.GPU:
                                assert (
                                    xp.abs(input_z_fields.z_graddiv_vertical_residual.ndarray).max()
                                    == 0.0
                                ), f"the max of z_graddiv_vertical_residual and z_graddiv_normal_residual are {xp.abs(input_z_fields.z_graddiv_vertical_residual.ndarray).max()} {xp.abs(input_z_fields.z_graddiv_normal_residual.ndarray).max()}"
                            else:
                                assert (
                                    xp.abs(
                                        input_z_fields.z_graddiv_vertical_residual.asnumpy()
                                    ).max()
                                    == 0.0
                                ), f"the max of z_graddiv_vertical_residual and z_graddiv_normal_residual are {xp.abs(input_z_fields.z_graddiv_vertical_residual.asnumpy()).max()} {xp.abs(input_z_fields.z_graddiv_normal_residual.asnumpy()).max()}"
                    else:
                        """
                        z_graddiv_normal (0:nlev-1):
                            Compute the horizontal gradient of the 3d divergence of normal wind at full levels (edge center).
                        z_graddiv_vertical (1:nlev-1):
                            Compute the vertical gradient of the 3d divergence of normal wind at half levels (cell center).
                        """
                        compute_graddiv(
                            inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
                            divergence=input_z_fields.z_flxdiv_vn_and_w,
                            graddiv_normal=input_z_fields.z_graddiv_normal,
                            graddiv_vertical=input_z_fields.z_graddiv_vertical,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=start_cell_nudging,
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )
                    if self.config.suppress_vertical_in_3d_divdamp:
                        init_cell_kdim_field_with_zero_vp(
                            field_with_zero_vp=input_z_fields.z_graddiv_vertical,
                            horizontal_start=start_cell_nudging,
                            horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=int32(self.grid.num_levels + 1),
                            offset_provider={},
                        )
                        if device == Device.GPU:
                            assert (
                                xp.abs(input_z_fields.z_graddiv_vertical.ndarray).max() == 0.0
                            ), f"the max of z_graddiv_vertical and z_graddiv_normal are {xp.abs(input_z_fields.z_graddiv_vertical.ndarray).max()} {xp.abs(input_z_fields.z_graddiv_normal.ndarray).max()}"
                        else:
                            assert (
                                xp.abs(input_z_fields.z_graddiv_vertical.asnumpy()).max() == 0.0
                            ), f"the max of z_graddiv_vertical and z_graddiv_normal are {xp.abs(input_z_fields.z_graddiv_vertical.asnumpy()).max()} {xp.abs(input_z_fields.z_graddiv_normal.asnumpy()).max()}"
                elif step == 2:
                    if order == 3:
                        """
                        z_graddiv2_normal and z_graddiv2_normal_residual (0:nlev-1):
                            Compute the horizontal gradient of the 3d divergence of normal wind at full levels (edge center).
                        z_graddiv2_vertical and z_graddiv2_vertical_residual (1:nlev-1):
                            Compute the vertical gradient of the 3d divergence of normal wind at half levels (cell center).
                        """
                        compute_graddiv(
                            inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
                            divergence=input_z_fields.z_flxdiv_graddiv_vn_and_w,
                            graddiv_normal=self.z_graddiv2_normal,
                            graddiv_vertical=self.z_graddiv2_vertical,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=start_cell_nudging,
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )
                        compute_graddiv(
                            inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
                            divergence=input_z_fields.z_flxdiv_graddiv_vn_and_w_residual,
                            graddiv_normal=self.z_graddiv2_normal_residual,
                            graddiv_vertical=self.z_graddiv2_vertical_residual,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=start_cell_nudging,
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )
                        if self.config.suppress_vertical_in_3d_divdamp:
                            init_cell_kdim_field_with_zero_vp(
                                field_with_zero_vp=self.z_graddiv2_vertical_residual,
                                horizontal_start=start_cell_nudging,
                                horizontal_end=end_cell_local,
                                vertical_start=0,
                                vertical_end=int32(self.grid.num_levels + 1),
                                offset_provider={},
                            )
                            if device == Device.GPU:
                                assert (
                                    xp.abs(self.z_graddiv2_vertical_residual.ndarray).max() == 0.0
                                ), f"the max of z_graddiv2_vertical_residual and z_graddiv2_normal_residual are {xp.abs(self.z_graddiv2_vertical_residual.ndarray).max()} {xp.abs(self.z_graddiv2_normal_residual.ndarray).max()}"
                            else:
                                assert (
                                    xp.abs(self.z_graddiv2_vertical_residual.asnumpy()).max() == 0.0
                                ), f"the max of z_graddiv2_vertical_residual and z_graddiv2_normal_residual are {xp.abs(self.z_graddiv2_vertical_residual.asnumpy()).max()} {xp.abs(self.z_graddiv2_normal_residual.asnumpy()).max()}"
                    else:
                        """
                        z_graddiv2_normal (0:nlev-1):
                            Compute the horizontal gradient of the 3d divergence of normal wind at full levels (edge center).
                        z_graddiv2_vertical (1:nlev-1):
                            Compute the vertical gradient of the 3d divergence of normal wind at half levels (cell center).
                        """
                        compute_graddiv(
                            inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
                            divergence=input_z_fields.z_flxdiv_graddiv_vn_and_w,
                            graddiv_normal=self.z_graddiv2_normal,
                            graddiv_vertical=self.z_graddiv2_vertical,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=start_cell_nudging,
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider=self.grid.offset_providers,
                        )
                    if self.config.suppress_vertical_in_3d_divdamp:
                        init_cell_kdim_field_with_zero_vp(
                            field_with_zero_vp=self.z_graddiv2_vertical,
                            horizontal_start=start_cell_nudging,
                            horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=int32(self.grid.num_levels + 1),
                            offset_provider={},
                        )
                        if device == Device.GPU:
                            assert (
                                xp.abs(self.z_graddiv2_vertical.ndarray).max() == 0.0
                            ), f"the max of z_graddiv2_vertical and z_graddiv2_normal are {xp.abs(self.z_graddiv2_vertical.ndarray).max()} {xp.abs(self.z_graddiv2_normal.ndarray).max()}"
                        else:
                            assert (
                                xp.abs(self.z_graddiv2_vertical.asnumpy()).max() == 0.0
                            ), f"the max of z_graddiv2_vertical and z_graddiv2_normal are {xp.abs(self.z_graddiv2_vertical.asnumpy()).max()} {xp.abs(self.z_graddiv2_normal.asnumpy()).max()}"
            else:
                if step == 1:
                    """
                    z_graddiv_vn (0:nlev-1):
                        Compute the laplacian of vn at full levels (edge center).
                    """
                    compute_graddiv_of_vn(
                        geofac_grdiv=self.interpolation_state.geofac_grdiv,
                        vn=input_prognostic_state.vn,
                        z_graddiv_vn=input_z_fields.z_graddiv_vn,
                        horizontal_start=start_edge_nudging_plus1,
                        horizontal_end=end_edge_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                    """
                    z_graddiv_vn (dd3d_lev:nlev-1):
                        Add vertical wind derivative to the normal gradient of divergence at full levels (edge center).
                        z_graddiv_vn_{k} = z_graddiv_vn_{k} + scalfac_dd3d_{k} d2w_{k}/dzdn
                    """
                    add_vertical_wind_derivative_to_divergence_damping(
                        hmask_dd3d=self.metric_state_nonhydro.hmask_dd3d,
                        scalfac_dd3d=self.metric_state_nonhydro.scalfac_dd3d,
                        inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                        z_dwdz_dd=input_z_fields.z_dwdz_dd,
                        z_graddiv_vn=input_z_fields.z_graddiv_vn,
                        horizontal_start=start_edge_lb_plus6,
                        horizontal_end=end_edge_local_minus2,
                        vertical_start=self.params.kstart_dd3d,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )
                elif step == 2:
                    """
                    z_graddiv2_vn (0:nlev-1):
                        Compute the double laplacian of vn at full levels (edge center).
                    """
                    compute_graddiv2_of_vn(
                        geofac_grdiv=self.interpolation_state.geofac_grdiv,
                        z_graddiv_vn=input_z_fields.z_graddiv_vn,
                        z_graddiv2_vn=self.z_graddiv2_vn,
                        horizontal_start=start_edge_nudging_plus1,
                        horizontal_end=end_edge_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider=self.grid.offset_providers,
                    )

                else:
                    raise NotImplementedError("Step = {step} must be 1 or 2")

        def copy_data_to_output(
            input_prognostic_state: PrognosticState,
            input_z_fields: IntermediateFields,
            output_group: str,
            do_o2: bool,
            do_3d_divergence_damping: bool,
        ):
            if output_group == "before":
                aux_func_divergence_for_output(
                    input_prognostic_state,
                    input_z_fields,
                    self.output_intermediate_fields.output_before_flxdiv1_vn,
                    self.output_intermediate_fields.output_before_flxdiv2_vn,
                    step=1,
                )
                copy_edge_kdim_field_to_vp(
                    field=input_prognostic_state.vn,
                    field_copy=self.output_intermediate_fields.output_before_vn,
                    horizontal_start=int32(0),
                    horizontal_end=end_edge_local,
                    vertical_start=int32(0),
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )
                copy_cell_kdim_field_to_vp(
                    field=input_prognostic_state.w,
                    field_copy=self.output_intermediate_fields.output_before_w,
                    horizontal_start=int32(0),
                    horizontal_end=end_cell_local,
                    vertical_start=int32(0),
                    vertical_end=self.grid.num_levels + int32(1),
                    offset_provider={},
                )
            elif output_group == "mid":
                if do_o2:
                    if do_3d_divergence_damping:
                        copy_edge_kdim_field_to_vp(
                            field=input_z_fields.z_graddiv_normal,
                            field_copy=self.output_intermediate_fields.output_graddiv_normal,
                            horizontal_start=int32(0),
                            horizontal_end=end_edge_local,
                            vertical_start=int32(0),
                            vertical_end=self.grid.num_levels,
                            offset_provider={},
                        )
                        copy_cell_kdim_field_to_vp(
                            field=input_z_fields.z_graddiv_vertical,
                            field_copy=self.output_intermediate_fields.output_graddiv_vertical,
                            horizontal_start=int32(0),
                            horizontal_end=end_cell_local,
                            vertical_start=int32(0),
                            vertical_end=self.grid.num_levels + int32(1),
                            offset_provider={},
                        )
                    else:
                        copy_edge_kdim_field_to_vp(
                            field=input_z_fields.z_graddiv_vn,
                            field_copy=self.output_intermediate_fields.output_graddiv_vn,
                            horizontal_start=int32(0),
                            horizontal_end=end_edge_local,
                            vertical_start=int32(0),
                            vertical_end=self.grid.num_levels,
                            offset_provider={},
                        )
                else:
                    if do_3d_divergence_damping:
                        copy_edge_kdim_field_to_vp(
                            field=self.z_graddiv2_normal,
                            field_copy=self.output_intermediate_fields.output_graddiv_normal,
                            horizontal_start=int32(0),
                            horizontal_end=end_edge_local,
                            vertical_start=int32(0),
                            vertical_end=self.grid.num_levels,
                            offset_provider={},
                        )
                        copy_cell_kdim_field_to_vp(
                            field=self.z_graddiv2_vertical,
                            field_copy=self.output_intermediate_fields.output_graddiv_vertical,
                            horizontal_start=int32(0),
                            horizontal_end=end_cell_local,
                            vertical_start=int32(0),
                            vertical_end=self.grid.num_levels + int32(1),
                            offset_provider={},
                        )
                    else:
                        copy_edge_kdim_field_to_vp(
                            field=self.z_graddiv2_vn,
                            field_copy=self.output_intermediate_fields.output_graddiv_vn,
                            horizontal_start=int32(0),
                            horizontal_end=end_edge_local,
                            vertical_start=int32(0),
                            vertical_end=self.grid.num_levels,
                            offset_provider={},
                        )

            elif output_group == "after":
                aux_func_divergence_for_output(
                    input_prognostic_state,
                    input_z_fields,
                    self.output_intermediate_fields.output_after_flxdiv1_vn,
                    self.output_intermediate_fields.output_after_flxdiv2_vn,
                    step=1,
                )
                copy_edge_kdim_field_to_vp(
                    field=input_prognostic_state.vn,
                    field_copy=self.output_intermediate_fields.output_after_vn,
                    horizontal_start=int32(0),
                    horizontal_end=end_edge_local,
                    vertical_start=int32(0),
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )
                copy_cell_kdim_field_to_vp(
                    field=input_prognostic_state.w,
                    field_copy=self.output_intermediate_fields.output_after_w,
                    horizontal_start=int32(0),
                    horizontal_end=end_cell_local,
                    vertical_start=int32(0),
                    vertical_end=self.grid.num_levels + int32(1),
                    offset_provider={},
                )

        def output_div_data(
            input_z_fields: IntermediateFields,
            input_do_output_step: int,
            input_do_output_substep: int,
            input_do_o2: bool,
            input_do_3d_divergence_damping: bool,
        ):
            filename = (
                "divergence_data_"
                + str(input_do_output_step)
                + "_"
                + str(input_do_output_substep)
                + ".dat"
            )
            if os.path.exists(filename):
                append_write = "a"  # append if already exists
            else:
                append_write = "w"  # make a new file if not
            expanded_dz = 1.0 / self.metric_state_nonhydro.inv_ddqz_z_full.ndarray
            temporary = self.cell_params.area.ndarray
            temporary = xp.expand_dims(temporary, axis=-1)
            expanded_weight = xp.repeat(temporary, expanded_dz.shape[1], axis=1)
            total_weight = xp.sum(xp.sum(expanded_weight * expanded_dz, axis=1), axis=0)
            if input_do_3d_divergence_damping:
                if not input_do_o2:
                    temporary1 = self.z_graddiv2_vn.ndarray**2
                else:
                    temporary1 = input_z_fields.z_graddiv_vn.ndarray**2
                temporary3 = xp.zeros_like(expanded_weight)
                c2e = self.grid.connectivities[C2EDim]
                for k in range(self.grid.num_levels):
                    temporary3[:, k] = xp.sum(temporary1[c2e, k], axis=1)
                    temporary3[:, k] = xp.sqrt(temporary3[:, k]) / 2.0
                with open(filename, append_write) as f:
                    # f.write(str(xp.sum(xp.abs(input_z_fields.z_flxdiv_vn_and_w.ndarray*self.cell_params.area.ndarray))/xp.sum(self.cell_params.area.ndarray))+'\n')
                    f.write(
                        str(
                            xp.sum(
                                xp.sum(
                                    expanded_weight
                                    * expanded_dz
                                    * xp.abs(input_z_fields.z_flxdiv_vn_and_w.ndarray),
                                    axis=1,
                                ),
                                axis=0,
                            )
                            / total_weight
                        )
                        + "   "
                        + str(
                            xp.sum(
                                xp.sum(expanded_weight * expanded_dz * xp.abs(temporary3), axis=1),
                                axis=0,
                            )
                            / total_weight
                        )
                        + "\n"
                    )
            else:
                if not input_do_o2:
                    temporary1 = self.z_graddiv2_normal.ndarray**2
                    temporary2 = self.z_graddiv2_vertical.ndarray**2
                else:
                    temporary1 = input_z_fields.z_graddiv_normal.ndarray**2
                    temporary2 = input_z_fields.z_graddiv_vertical.ndarray**2
                temporary1 = input_z_fields.z_graddiv_normal.ndarray**2
                temporary2 = input_z_fields.z_graddiv_vertical.ndarray**2
                temporary3 = xp.zeros_like(expanded_weight)
                c2e = self.grid.connectivities[C2EDim]
                for k in range(self.grid.num_levels):
                    temporary3[:, k] = (
                        xp.sum(temporary1[c2e, k], axis=1) + temporary2[:, k + 1] + temporary2[:, k]
                    )
                    temporary3[:, k] = xp.sqrt(temporary3[:, k]) / 2.0
                with open(filename, append_write) as f:
                    # f.write(str(xp.sum(xp.abs(input_z_fields.z_flxdiv_vn_and_w.ndarray*self.cell_params.area.ndarray))/xp.sum(self.cell_params.area.ndarray))+'\n')
                    f.write(
                        str(
                            xp.sum(
                                xp.sum(
                                    expanded_weight
                                    * expanded_dz
                                    * xp.abs(input_z_fields.z_flxdiv_vn_and_w.ndarray),
                                    axis=1,
                                ),
                                axis=0,
                            )
                            / total_weight
                        )
                        + "   "
                        + str(
                            xp.sum(
                                xp.sum(expanded_weight * expanded_dz * xp.abs(temporary3), axis=1),
                                axis=0,
                            )
                            / total_weight
                        )
                        + "\n"
                    )

        def aux_func_compute_divergence_damping(
            input_prognostic_state: PrognosticState,
            input_diagnostic_state_nh: DiagnosticStateNonHydro,
            input_z_fields: IntermediateFields,
            order: int,
            do_o2: bool,
            do_compute_diagnostics: bool,
            do_3d_divergence_damping: bool,
        ):
            # debug_z_graddiv2_vn = _allocate(EdgeDim, KDim, grid=self.grid)
            # debug_z_graddiv_vn = _allocate(EdgeDim, KDim, grid=self.grid)
            # copy_edge_kdim_field_to_vp(
            #     field=self.z_graddiv2_vn,
            #     field_copy=debug_z_graddiv2_vn,
            #     horizontal_start=int32(0),
            #     horizontal_end=end_edge_local,
            #     vertical_start=int32(0),
            #     vertical_end=self.grid.num_levels,
            #     offset_provider={},
            # )
            # copy_edge_kdim_field_to_vp(
            #     field=input_z_fields.z_graddiv_vn,
            #     field_copy=debug_z_graddiv_vn,
            #     horizontal_start=int32(0),
            #     horizontal_end=end_edge_local,
            #     vertical_start=int32(0),
            #     vertical_end=self.grid.num_levels,
            #     offset_provider={},
            # )

            if do_o2:
                if do_compute_diagnostics:
                    aux_func_tangential_wind_and_contravariant_correction_and_dwdz(
                        input_prognostic_state,
                        input_diagnostic_state_nh,
                        input_z_fields,
                        order,
                        step=1,
                    )

                if do_3d_divergence_damping:
                    aux_func_divergence(input_prognostic_state, input_z_fields, order, step=1)
                aux_func_graddiv(
                    input_prognostic_state, input_z_fields, do_3d_divergence_damping, order, step=1
                )
            else:
                if do_compute_diagnostics:
                    aux_func_tangential_wind_and_contravariant_correction_and_dwdz(
                        input_prognostic_state,
                        input_diagnostic_state_nh,
                        input_z_fields,
                        order,
                        step=1,
                    )

                if do_3d_divergence_damping:
                    aux_func_divergence(input_prognostic_state, input_z_fields, order, step=1)
                aux_func_graddiv(
                    input_prognostic_state, input_z_fields, do_3d_divergence_damping, order, step=1
                )

                if do_3d_divergence_damping:
                    aux_func_tangential_wind_and_contravariant_correction_and_dwdz(
                        input_prognostic_state,
                        input_diagnostic_state_nh,
                        input_z_fields,
                        order,
                        step=2,
                    )
                    aux_func_divergence(input_prognostic_state, input_z_fields, order, step=2)
                aux_func_graddiv(
                    input_prognostic_state, input_z_fields, do_3d_divergence_damping, order, step=2
                )

        def aux_func_apply_divergence_damping(
            input_prognostic_state: PrognosticState,
            input_z_fields: IntermediateFields,
            do_o2: bool,
            do_3d_divergence_damping: bool,
            order: int,
        ):
            log.critical(
                f"vertical absolute max value: {xp.abs(input_z_fields.z_graddiv_vertical.ndarray).max()} {xp.abs(input_z_fields.z_dgraddiv_dz.ndarray).max()} {xp.abs(self.z_graddiv2_vertical.ndarray).max()}"
            )
            log.critical(
                f"horizontal absolute max value: {xp.abs(self.z_graddiv2_normal.ndarray).max()}"
            )
            if do_3d_divergence_damping:
                if do_o2:
                    """
                    vn (0:nlev-1):
                        Apply the higher order divergence damping to vn at full levels (edge center).
                        vn = vn + scal_divdamp * Del(normal_direction) Div(V)
                    w (1:nlev-1):
                        Apply the higher order divergence damping to w at half levels (cell center).
                        w = w + scal_divdamp_half * Del(vertical_direction) Div(V)
                    """
                    apply_3d_divergence_damping(
                        scal_divdamp=self.scal_divdamp_o2,
                        scal_divdamp_half=self.scal_divdamp_o2_half,
                        graddiv_normal=input_z_fields.z_graddiv_normal,
                        graddiv_vertical=input_z_fields.z_graddiv_vertical,
                        vn=input_prognostic_state.vn,
                        # w=input_prognostic_state.w,
                        z_w_divdamp=input_z_fields.z_w_divdamp,
                        edge_horizontal_start=start_edge_nudging_plus1,
                        edge_horizontal_end=end_edge_local,
                        cell_horizontal_start=int32(0),
                        cell_horizontal_end=end_cell_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider={},
                    )
                else:
                    if order == 3:
                        apply_3d_divergence_damping(
                            scal_divdamp=self.scal_divdamp,
                            scal_divdamp_half=self.scal_divdamp_half,
                            graddiv_normal=self.z_graddiv2_normal,
                            graddiv_vertical=self.z_graddiv2_vertical,
                            vn=input_prognostic_state.vn,
                            # w=input_prognostic_state.w,
                            z_w_divdamp=input_z_fields.z_w_divdamp,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=int32(0),
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider={},
                        )
                        apply_3d_divergence_damping(
                            scal_divdamp=self.scal_divdamp,
                            scal_divdamp_half=self.scal_divdamp_half,
                            graddiv_normal=self.z_graddiv2_normal_residual,
                            graddiv_vertical=self.z_graddiv2_vertical_residual,
                            vn=input_prognostic_state.vn,
                            # w=input_prognostic_state.w,
                            z_w_divdamp=input_z_fields.z_w_divdamp,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=int32(0),
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider={},
                        )
                    else:
                        w_copy = xp.array(input_prognostic_state.w.ndarray, copy=True)
                        vn_copy = xp.array(input_prognostic_state.vn.ndarray, copy=True)
                        apply_3d_divergence_damping(
                            scal_divdamp=self.scal_divdamp,
                            scal_divdamp_half=self.scal_divdamp_half,
                            graddiv_normal=self.z_graddiv2_normal,
                            graddiv_vertical=self.z_graddiv2_vertical,
                            vn=input_prognostic_state.vn,
                            # w=input_prognostic_state.w,
                            z_w_divdamp=input_z_fields.z_w_divdamp,
                            edge_horizontal_start=start_edge_nudging_plus1,
                            edge_horizontal_end=end_edge_local,
                            cell_horizontal_start=int32(0),
                            cell_horizontal_end=end_cell_local,
                            vertical_start=0,
                            vertical_end=self.grid.num_levels,
                            offset_provider={},
                        )
                        log.critical(
                            f"wind max diff value: {xp.abs(input_prognostic_state.vn.ndarray - vn_copy).max()} {xp.abs(input_prognostic_state.w.ndarray - w_copy).max()}"
                        )
            else:
                if do_o2:
                    apply_4th_order_divergence_damping(
                        scal_divdamp=self.scal_divdamp_o2,
                        z_graddiv2_vn=input_z_fields.z_graddiv_vn,
                        vn=input_prognostic_state.vn,
                        horizontal_start=start_edge_nudging_plus1,
                        horizontal_end=end_edge_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider={},
                    )
                else:
                    apply_4th_order_divergence_damping(
                        scal_divdamp=self.scal_divdamp,
                        z_graddiv2_vn=self.z_graddiv2_vn,
                        vn=input_prognostic_state.vn,
                        horizontal_start=start_edge_nudging_plus1,
                        horizontal_end=end_edge_local,
                        vertical_start=0,
                        vertical_end=self.grid.num_levels,
                        offset_provider={},
                    )

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

        # TODO: to cached program
        # _calculate_divdamp_fields(
        #     self.enh_divdamp_fac,
        #     int32(self.config.divdamp_order),
        #     self.cell_params.mean_cell_area,
        #     divdamp_fac_o2,
        #     self.config.nudge_max_coeff,
        #     constants.dbl_eps,
        #     out=(self.scal_divdamp, self.scal_divdamp_o2, self._bdy_divdamp),
        #     offset_provider={},
        # )
        calculate_divdamp_fields(
            self.enh_divdamp_fac,
            self.scal_divdamp,
            self.scal_divdamp_o2,
            self._bdy_divdamp,
            int32(self.config.divdamp_order),
            self.cell_params.mean_cell_area,
            divdamp_fac_o2,
            self.config.nudge_max_coeff,
            constants.dbl_eps,
            self.config.scal_divsign,
            offset_provider={},
        )
        calculate_scal_divdamp_half(
            scal_divdamp=self.scal_divdamp,
            scal_divdamp_o2=self.scal_divdamp_o2,
            vct_a=self.vertical_params.vct_a,
            divdamp_fac_w=self.config.divdamp_fac_w,
            scal_divdamp_half=self.scal_divdamp_half,
            scal_divdamp_o2_half=self.scal_divdamp_o2_half,
            # k_field=self.k_field,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider={"Koff": KDim},
        )
        # if do_output:
        #     for k in range(self.grid.num_levels):
        #         log.critical(f"scal_divdamp {k} {self.scal_divdamp.ndarray[k]} {self.scal_divdamp_o2.ndarray[k]} {self.scal_divdamp_half.ndarray[k]} {self.scal_divdamp_o2_half.ndarray[k]}")

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
            output_intermediate_fields=self.output_intermediate_fields,
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
        """
        rho_ic (1:nlev-1):
            Compute the density at half levels (cell center) that includes the vertical mass flux term. Its value at the model top and ground level is not updated.
            wgt_nnew_rth = 0.5 + rhotheta_offctr
            wgt_nnow_rth = 1.0 - wgt_nnew_rth
            rho_avg_{k} = wgt_nnow_rth rho^{n}_{k} + wgt_nnew_rth rho^{n+1*}_{k}
            rho_{k-1/2} = vertical_weight rho_avg_{k-1} + (1 - vertical_weight) rho_avg_{k} - dt (w^{n+1*}_{k-1/2} - contravariant_correction^{n+1*}_{k-1/2} ) / dz_{k-1/2}
        z_theta_v_pr_ic (1:nlev-1):
            Compute the perturbed virtual temperature at half levels (cell center). Its value at the model top and ground level is not updated.
            wgt_nnew_rth = 0.5 + rhotheta_offctr
            wgt_nnow_rth = 1.0 - wgt_nnew_rth
            theta_v_avg_{k} = wgt_nnow_rth theta_v^{n}_{k} + wgt_nnew_rth theta_v^{n+1*}_{k}
            perturbed_theta_v_avg_{k} = theta_v_avg_{k} - theta_v_ref_{k}
            z_theta_v_pr_ic{k-1/2} = vertical_weight perturbed_theta_v_avg_{k-1} + (1 - vertical_weight) perturbed_theta_v_avg_{k}
        theta_v_ic (1:nlev-1):
            Compute the virtual temperature at half levels (cell center) that includes the vertical flux term. Its value at the model top and ground level is not updated.
            wgt_nnew_rth = 0.5 + rhotheta_offctr
            wgt_nnow_rth = 1.0 - wgt_nnew_rth
            rho_avg_{k} = wgt_nnow_rth theta_v^{n}_{k} + wgt_nnew_rth theta_v^{n+1*}_{k}
            rho_{k-1/2} = vertical_weight theta_v_avg_{k-1} + (1 - vertical_weight) theta_v_avg_{k} - dt (w^{n+1*}_{k-1/2} - contravariant_correction^{n+1*}_{k-1/2} ) / dz_{k-1/2}
        z_th_ddz_exner_c (1:nlev-1):
            theta_v' dpi_0/dz + eta_expl theta_v dpi'/dz (see the last two terms on the RHS of vertical momentum equation in eq. 3.21 in icon tutorial 2023) at half levels (cell center) is also computed. Its value at the model top is not updated. No ground value.
            z_th_ddz_exner_c_{k-1/2} = vwind_expl_wgt theta_v_ic_{k-1/2} (exner_pr_{k-1} - exner_pr_{k}) / dz_{k-1/2} + perturbed_theta_v_avg_{k} dpi0/dz_{k-1/2}
            eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
            eta_expl = 1.0 - eta_impl = vwind_expl_wgt
        """
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
        """
        z_graddiv_vn (dd3d_lev:nlev-1):
            Add vertical wind derivative to the normal gradient of divergence at full levels (edge center).
            z_graddiv_vn_{k} = z_graddiv_vn_{k} + scalfac_dd3d_{k} d2w_{k}/dzdn
        """
        # add_vertical_wind_derivative_to_divergence_damping(
        #     hmask_dd3d=self.metric_state_nonhydro.hmask_dd3d,
        #     scalfac_dd3d=self.metric_state_nonhydro.scalfac_dd3d,
        #     inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
        #     z_dwdz_dd=z_fields.z_dwdz_dd,
        #     z_graddiv_vn=z_fields.z_graddiv_vn,
        #     horizontal_start=start_edge_lb_plus6,
        #     horizontal_end=end_edge_local_minus2,
        #     vertical_start=self.params.kstart_dd3d,
        #     vertical_end=self.grid.num_levels,
        #     offset_provider=self.grid.offset_providers,
        # )

        # if self.config.divdamp_order == 24 or self.config.divdamp_order == 4:
        #     # verified for e-10
        #     log.debug(f"corrector start stencil 25")
        #     """
        #     z_graddiv2_vn (0:nlev-1):
        #         Compute the double laplacian of vn at full levels (edge center).
        #     """
        #     compute_graddiv2_of_vn(
        #         geofac_grdiv=self.interpolation_state.geofac_grdiv,
        #         z_graddiv_vn=z_fields.z_graddiv_vn,
        #         z_graddiv2_vn=self.z_graddiv2_vn,
        #         horizontal_start=start_edge_nudging_plus1,
        #         horizontal_end=end_edge_local,
        #         vertical_start=0,
        #         vertical_end=self.grid.num_levels,
        #         offset_provider=self.grid.offset_providers,
        #     )

        aux_func_compute_divergence_damping(
            prognostic_state[nnew],
            diagnostic_state_nh,
            z_fields,
            order=self.config.divergence_order,
            do_o2=self.config.do_o2_divdamp,
            do_compute_diagnostics=False,
            do_3d_divergence_damping=self.config.do_3d_divergence_damping,
        )

        if self.config.itime_scheme == 4:
            log.debug(f"corrector: start stencil 23")
            """
            vn (0:nlev-1):
                Update the normal wind at full levels (edge center) in the corrector step (see the horizontal momentum equaton in eq. 3.21 in ICON tutoral 2023 and the assumption below that simplifies the computation).
                The second term on the RHS is cpd z_theta_v_e z_gradh_exner.
                vn^{n+1}_{k} = vn^{n}_{k} - dt advection_avg_{k} - cpd theta_v dpi/dn
                advection_avg_{k} = wgt_nnow_vel advection^{n}_{k} + wgt_nnew_rth advecton^{n+1*}_{k}
                ddt_vn_apc_pc[self.ntl2] is computed in velocity advecton corrector step.
                wgt_nnew_vel = 0.5 + veladv_offctr
                wgt_nnow_vel = 1.0 - wgt_nnew_vel
            """
            copy_edge_kdim_field_to_vp(
                field=z_fields.z_gradh_exner,
                field_copy=self.output_intermediate_fields.output_corrector_gradh_exner,
                horizontal_start=int32(0),
                horizontal_end=end_edge_local,
                vertical_start=int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
            copy_edge_kdim_field_to_vp(
                field=z_fields.z_theta_v_e,
                field_copy=self.output_intermediate_fields.output_corrector_theta_v_e,
                horizontal_start=int32(0),
                horizontal_end=end_edge_local,
                vertical_start=int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
            copy_edge_kdim_field_to_vp(
                field=diagnostic_state_nh.ddt_vn_apc_pc[self.ntl2],
                field_copy=self.output_intermediate_fields.output_corrector_ddt_vn_apc_ntl2,
                horizontal_start=int32(0),
                horizontal_end=end_edge_local,
                vertical_start=int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

            # main update
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

        # """
        # vt (0:nlev-1):
        #     Compute tangential velocity at full levels (edge center) by RBF interpolation from four neighboring
        #     edges (diamond shape) and projected to tangential direction.
        # """
        # compute_tangential_wind(
        #     vn=prognostic_state[nnew].vn,
        #     rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
        #     vt=z_fields.vt,
        #     horizontal_start=start_edge_lb_plus4,
        #     horizontal_end=end_edge_local_minus2,
        #     vertical_start=0,
        #     vertical_end=self.grid.num_levels,
        #     offset_provider=self.grid.offset_providers,
        # )
        # compute_contravariant_correction(
        #     vn=prognostic_state[nnew].vn,
        #     ddxn_z_full=self.metric_state_nonhydro.ddxn_z_full,
        #     ddxt_z_full=self.metric_state_nonhydro.ddxt_z_full,
        #     vt=z_fields.vt,
        #     z_w_concorr_me=z_fields.z_w_concorr_me,
        #     horizontal_start=start_edge_lb_plus4,
        #     horizontal_end=end_edge_local_minus2,
        #     vertical_start=0,
        #     vertical_end=self.grid.num_levels,
        #     offset_provider=self.grid.offset_providers,
        # )
        # fused_stencils_9_10(
        #     z_w_concorr_me=z_fields.z_w_concorr_me,
        #     e_bln_c_s=self.interpolation_state.e_bln_c_s,
        #     local_z_w_concorr_mc=z_fields.z_w_concorr_mc,
        #     wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
        #     w_concorr_c=diagnostic_state_nh.w_concorr_c,
        #     k_field=self.k_field,
        #     nflatlev_startindex=self.vertical_params.nflatlev,
        #     nlev=self.grid.num_levels,
        #     horizontal_start=start_cell_lb_plus3,
        #     horizontal_end=end_cell_local_minus1,
        #     vertical_start=0,
        #     vertical_end=self.grid.num_levels,
        #     offset_provider=self.grid.offset_providers,
        # )
        # """
        # rho_ic (1:nlev-1):
        #     Compute the density at half levels (cell center) that includes the vertical mass flux term. Its value at the model top and ground level is not updated.
        #     wgt_nnew_rth = 0.5 + rhotheta_offctr
        #     wgt_nnow_rth = 1.0 - wgt_nnew_rth
        #     rho_avg_{k} = wgt_nnow_rth rho^{n}_{k} + wgt_nnew_rth rho^{n+1*}_{k}
        #     rho_{k-1/2} = vertical_weight rho_avg_{k-1} + (1 - vertical_weight) rho_avg_{k} - dt (w^{n+1*}_{k-1/2} - contravariant_correction^{n+1*}_{k-1/2} ) / dz_{k-1/2}
        # z_theta_v_pr_ic (1:nlev-1):
        #     Compute the perturbed virtual temperature at half levels (cell center). Its value at the model top and ground level is not updated.
        #     wgt_nnew_rth = 0.5 + rhotheta_offctr
        #     wgt_nnow_rth = 1.0 - wgt_nnew_rth
        #     theta_v_avg_{k} = wgt_nnow_rth theta_v^{n}_{k} + wgt_nnew_rth theta_v^{n+1*}_{k}
        #     perturbed_theta_v_avg_{k} = theta_v_avg_{k} - theta_v_ref_{k}
        #     z_theta_v_pr_ic{k-1/2} = vertical_weight perturbed_theta_v_avg_{k-1} + (1 - vertical_weight) perturbed_theta_v_avg_{k}
        # theta_v_ic (1:nlev-1):
        #     Compute the virtual temperature at half levels (cell center) that includes the vertical flux term. Its value at the model top and ground level is not updated.
        #     wgt_nnew_rth = 0.5 + rhotheta_offctr
        #     wgt_nnow_rth = 1.0 - wgt_nnew_rth
        #     rho_avg_{k} = wgt_nnow_rth theta_v^{n}_{k} + wgt_nnew_rth theta_v^{n+1*}_{k}
        #     rho_{k-1/2} = vertical_weight theta_v_avg_{k-1} + (1 - vertical_weight) theta_v_avg_{k} - dt (w^{n+1*}_{k-1/2} - contravariant_correction^{n+1*}_{k-1/2} ) / dz_{k-1/2}
        # z_th_ddz_exner_c (1:nlev-1):
        #     theta_v' dpi_0/dz + eta_expl theta_v dpi'/dz (see the last two terms on the RHS of vertical momentum equation in eq. 3.21 in icon tutorial 2023) at half levels (cell center) is also computed. Its value at the model top is not updated. No ground value.
        #     z_th_ddz_exner_c_{k-1/2} = vwind_expl_wgt theta_v_ic_{k-1/2} (exner_pr_{k-1} - exner_pr_{k}) / dz_{k-1/2} + perturbed_theta_v_avg_{k} dpi0/dz_{k-1/2}
        #     eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
        #     eta_expl = 1.0 - eta_impl = vwind_expl_wgt
        # """
        # compute_rho_virtual_potential_temperatures_and_pressure_gradient(
        #     w=prognostic_state[nnew].w,
        #     w_concorr_c=diagnostic_state_nh.w_concorr_c,
        #     ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
        #     rho_now=prognostic_state[nnow].rho,
        #     rho_var=prognostic_state[nvar].rho,
        #     theta_now=prognostic_state[nnow].theta_v,
        #     theta_var=prognostic_state[nvar].theta_v,
        #     wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
        #     theta_ref_mc=self.metric_state_nonhydro.theta_ref_mc,
        #     vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
        #     exner_pr=diagnostic_state_nh.exner_pr,
        #     d_exner_dz_ref_ic=self.metric_state_nonhydro.d_exner_dz_ref_ic,
        #     rho_ic=diagnostic_state_nh.rho_ic,
        #     z_theta_v_pr_ic=self.z_theta_v_pr_ic,
        #     theta_v_ic=diagnostic_state_nh.theta_v_ic,
        #     z_th_ddz_exner_c=self.z_th_ddz_exner_c,
        #     dtime=dtime,
        #     wgt_nnow_rth=self.params.wgt_nnow_rth,
        #     wgt_nnew_rth=self.params.wgt_nnew_rth,
        #     horizontal_start=start_cell_lb_plus2,
        #     horizontal_end=end_cell_local,
        #     vertical_start=1,
        #     vertical_end=self.grid.num_levels,
        #     offset_provider=self.grid.offset_providers,
        # )

        if self.config.divdamp_order == 24 and scal_divdamp_o2 > 1.0e-6:
            """
            vn (0:nlev-1):
                Apply the divergence damping to vn at full levels (edge center).
                vn = vn + scal_divdamp_o2 * Del(normal_direction) Div(vn)
            """
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
                """
                vn (0:nlev-1):
                    Apply the higher order divergence damping to vn at full levels (edge center).
                    vn = vn + (scal_divdamp + bdy_divdamp * nudgecoeff_e) * Del(normal_direction) Div( Del(normal_direction) Div(vn) )
                """
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
                if self.config.do_proper_diagnostics_divdamp:
                    aux_func_compute_divergence_damping(
                        prognostic_state[nnew],
                        diagnostic_state_nh,
                        z_fields,
                        order=self.config.divergence_order,
                        do_o2=self.config.do_o2_divdamp,
                        do_compute_diagnostics=True,
                        do_3d_divergence_damping=self.config.do_3d_divergence_damping,
                    )
                if do_output:
                    copy_data_to_output(
                        prognostic_state[nnew],
                        z_fields,
                        "before",
                        self.config.do_o2_divdamp,
                        self.config.do_3d_divergence_damping,
                    )
                    output_div_data(
                        z_fields,
                        do_output_step,
                        do_output_substep,
                        self.config.do_o2_divdamp,
                        self.config.do_3d_divergence_damping,
                    )
                aux_func_apply_divergence_damping(
                    prognostic_state[nnew],
                    z_fields,
                    do_o2=self.config.do_o2_divdamp,
                    do_3d_divergence_damping=self.config.do_3d_divergence_damping,
                    order=self.config.divergence_order,
                )

                if do_output:
                    copy_data_to_output(
                        prognostic_state[nnew],
                        z_fields,
                        "mid",
                        self.config.do_o2_divdamp,
                        self.config.do_3d_divergence_damping,
                    )
                    self.output_intermediate_fields.output_scal_divdamp = self.scal_divdamp
                    output_div_data(
                        z_fields,
                        do_output_step,
                        do_output_substep,
                        self.config.do_o2_divdamp,
                        self.config.do_3d_divergence_damping,
                    )

                if self.config.do_multiple_divdamp:
                    for _ in range(self.config.number_of_divdamp_step - 1):
                        aux_func_compute_divergence_damping(
                            prognostic_state[nnew],
                            diagnostic_state_nh,
                            z_fields,
                            order=self.config.divergence_order,
                            do_o2=self.config.do_o2_divdamp,
                            do_compute_diagnostics=True,
                            do_3d_divergence_damping=self.config.do_3d_divergence_damping,
                        )
                        aux_func_apply_divergence_damping(
                            prognostic_state[nnew],
                            z_fields,
                            do_o2=self.config.do_o2_divdamp,
                            do_3d_divergence_damping=self.config.do_3d_divergence_damping,
                            order=self.config.divergence_order,
                        )
                        if do_output:
                            output_div_data(
                                z_fields,
                                do_output_step,
                                do_output_substep,
                                self.config.do_o2_divdamp,
                                self.config.do_3d_divergence_damping,
                            )

        # log.info(
        #     f" MAXDIV VN: {prognostic_state[nnew].vn.ndarray.max():.15e} , MAXDIV W: {prognostic_state[nnew].w.ndarray.max():.15e}"
        # )
        # log.info(
        #     f" MAXDIV RHO: {prognostic_state[nnew].rho.ndarray.max():.15e} , MAXDIV THETA_V: {prognostic_state[nnew].theta_v.ndarray.max():.15e}"
        # )
        # log.info(
        #     f" AVEDIV VN: {prognostic_state[nnew].vn.ndarray.mean(axis=(0,1)):.15e} , AVEDIV W: {prognostic_state[nnew].w.ndarray.mean(axis=(0,1)):.15e}"
        # )
        # log.info(
        #     f" AVEDIV RHO: {prognostic_state[nnew].rho.ndarray.mean(axis=(0,1)):.15e} , AVEDIV THETA_V: {prognostic_state[nnew].theta_v.ndarray.mean(axis=(0,1)):.15e}"
        # )

        if do_output:
            copy_data_to_output(
                prognostic_state[nnew],
                z_fields,
                "after",
                self.config.do_o2_divdamp,
                self.config.do_3d_divergence_damping,
            )

        # """
        # rho_ic (1:nlev-1):
        #     Compute the density at half levels (cell center) that includes the vertical mass flux term. Its value at the model top and ground level is not updated.
        #     wgt_nnew_rth = 0.5 + rhotheta_offctr
        #     wgt_nnow_rth = 1.0 - wgt_nnew_rth
        #     rho_avg_{k} = wgt_nnow_rth rho^{n}_{k} + wgt_nnew_rth rho^{n+1*}_{k}
        #     rho_{k-1/2} = vertical_weight rho_avg_{k-1} + (1 - vertical_weight) rho_avg_{k} - dt (w^{n+1*}_{k-1/2} - contravariant_correction^{n+1*}_{k-1/2} ) / dz_{k-1/2}
        # z_theta_v_pr_ic (1:nlev-1):
        #     Compute the perturbed virtual temperature at half levels (cell center). Its value at the model top and ground level is not updated.
        #     wgt_nnew_rth = 0.5 + rhotheta_offctr
        #     wgt_nnow_rth = 1.0 - wgt_nnew_rth
        #     theta_v_avg_{k} = wgt_nnow_rth theta_v^{n}_{k} + wgt_nnew_rth theta_v^{n+1*}_{k}
        #     perturbed_theta_v_avg_{k} = theta_v_avg_{k} - theta_v_ref_{k}
        #     z_theta_v_pr_ic{k-1/2} = vertical_weight perturbed_theta_v_avg_{k-1} + (1 - vertical_weight) perturbed_theta_v_avg_{k}
        # theta_v_ic (1:nlev-1):
        #     Compute the virtual temperature at half levels (cell center) that includes the vertical flux term. Its value at the model top and ground level is not updated.
        #     wgt_nnew_rth = 0.5 + rhotheta_offctr
        #     wgt_nnow_rth = 1.0 - wgt_nnew_rth
        #     rho_avg_{k} = wgt_nnow_rth theta_v^{n}_{k} + wgt_nnew_rth theta_v^{n+1*}_{k}
        #     rho_{k-1/2} = vertical_weight theta_v_avg_{k-1} + (1 - vertical_weight) theta_v_avg_{k} - dt (w^{n+1*}_{k-1/2} - contravariant_correction^{n+1*}_{k-1/2} ) / dz_{k-1/2}
        # z_th_ddz_exner_c (1:nlev-1):
        #     theta_v' dpi_0/dz + eta_expl theta_v dpi'/dz (see the last two terms on the RHS of vertical momentum equation in eq. 3.21 in icon tutorial 2023) at half levels (cell center) is also computed. Its value at the model top is not updated. No ground value.
        #     z_th_ddz_exner_c_{k-1/2} = vwind_expl_wgt theta_v_ic_{k-1/2} (exner_pr_{k-1} - exner_pr_{k}) / dz_{k-1/2} + perturbed_theta_v_avg_{k} dpi0/dz_{k-1/2}
        #     eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
        #     eta_expl = 1.0 - eta_impl = vwind_expl_wgt
        # """
        # compute_rho_virtual_potential_temperatures_and_pressure_gradient(
        #     w=prognostic_state[nnew].w,
        #     w_concorr_c=diagnostic_state_nh.w_concorr_c,
        #     ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
        #     rho_now=prognostic_state[nnow].rho,
        #     rho_var=prognostic_state[nvar].rho,
        #     theta_now=prognostic_state[nnow].theta_v,
        #     theta_var=prognostic_state[nvar].theta_v,
        #     wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
        #     theta_ref_mc=self.metric_state_nonhydro.theta_ref_mc,
        #     vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
        #     exner_pr=diagnostic_state_nh.exner_pr,
        #     d_exner_dz_ref_ic=self.metric_state_nonhydro.d_exner_dz_ref_ic,
        #     rho_ic=diagnostic_state_nh.rho_ic,
        #     z_theta_v_pr_ic=self.z_theta_v_pr_ic,
        #     theta_v_ic=diagnostic_state_nh.theta_v_ic,
        #     z_th_ddz_exner_c=self.z_th_ddz_exner_c,
        #     dtime=dtime,
        #     wgt_nnow_rth=self.params.wgt_nnow_rth,
        #     wgt_nnew_rth=self.params.wgt_nnew_rth,
        #     horizontal_start=start_cell_lb_plus2,
        #     horizontal_end=end_cell_local,
        #     vertical_start=1,
        #     vertical_end=self.grid.num_levels,
        #     offset_provider=self.grid.offset_providers,
        # )

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
        """
        z_vn_avg (0:nlev-1):
            Compute the averaged normal velocity at full levels (edge center).
            TODO (Chia Rui): Fill in details about how the coefficients are computed.
        """
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
        """
        z_flxdiv_mass (0:nlev-1):
            Compute the mass flux at full levels (edge center) by multiplying density with averaged normal velocity (z_vn_avg) computed above.
        z_flxdiv_theta (0:nlev-1):
            Compute the energy (theta_v * mass) flux by multiplying density with averaged normal velocity (z_vn_avg) computed above.
        """
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
        """
        z_flxdiv_mass (0:nlev-1):
            Compute the divergence of mass flux at full levels (cell center) by Gauss theorem.
        z_flxdiv_theta (0:nlev-1):
            Compute the divergence of energy (theta_v * mass) flux at full levels (cell center) by Gauss theorem.
        """
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
            """
            z_w_expl (1:nlev-1):
                Compute the explicit term in vertical momentum equation at half levels (cell center). See the first equation below eq. 3.25 in ICON tutorial 2023.
                z_w_expl = advection of w + cpd theta' dpi0/dz + cpd theta (1 - eta_impl) dpi'/dz + F_divdamp @ k+1/2 level
                advection of w_{k} = wgt_nnow_vel advection^{n}_{k} + wgt_nnew_rth advecton^{n+1*}_{k}
                advection^{n} is ddt_vn_apc_pc[self.ntl1], which is computed in predictor step.
                advection^{n+1*} is ddt_vn_apc_pc[self.ntl2], which is computed in velocity advecton corrector step.
                wgt_nnew_vel = 0.5 + veladv_offctr
                wgt_nnow_vel = 1.0 - wgt_nnew_vel
                cpd theta' dpi0/dz + cpd theta (1 - eta_impl) dpi'/dz = cpd z_th_ddz_exner_c
                eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
                eta_expl = 1.0 - eta_impl = vwind_expl_wgt
            z_contr_w_fl_l (1:nlev-1):
                Compute the vertical mass flux at half levels (cell center). See second term on RHS of mass conservation in eq. 3.21 in ICON tutorial 2023.
                z_contr_w_fl_l = rho * (-contravariant_correction + vwind_expl_wgt * w) # TODO (Chia Rui: Check why minus sign)
                eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
                eta_expl = 1.0 - eta_impl = vwind_expl_wgt
                rho = rho_ic
            z_beta (0:nlev-1):
                Compute component of the coefficients in the tridiagonal matrix of w equation at full levels (cell center).
                See the middle term in each square bracket of eq. 3.27 and unnumbered equation below in ICON tutorial 2023.
                a b 0 0 0
                c a b 0 0
                0 c a b 0
                0 0 c a b
                0 0 0 c a
                z_beta_{k} = dt * rd * pi_{k} / (cvd * rho_{k} * theta_v_{k}) / dz_{k}
            z_alpha (0:nlev-1):
                Compute component of the coefficients in the tridiagonal matrix of w equation at half levels (cell center).
                See the last term in each square bracket of eq. 3.27 and unnumbered equation below in ICON tutorial 2023.
                z_alpha_{k-1/2} = vwind_impl_wgt rho_{k-1/2} theta_v_{k-1/2}
                eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
                eta_expl = 1.0 - eta_impl = vwind_expl_wgt
                rho_{k-1/2} is precomputed as rho_ic.
                theta_v_{k-1/2} is precomputed as theta_v_ic.
            z_alpha (nlev):
                Compute component of the coefficients in the tridiagonal matrix of w equation at half levels (cell center).
                z_alpha_{k-1/2} = 0
            z_q (0):
                Set the intermediate result for w in tridiagonal solver during forward seep at half levels (cell center) at model top to zero.
                Note that it also only has nlev levels because the model top w is not updated, although it is a half-level variable.
                z_q_{k-1/2} = 0
            """
            stencils_42_44_45_45b(
                z_w_expl=z_fields.z_w_expl,
                w_nnow=prognostic_state[nnow].w,
                z_w_divdamp=z_fields.z_w_divdamp,
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
                # vertical_end=self.grid.num_levels + 1,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
            init_cell_kdim_field_with_zero_vp(
                field_with_zero_vp=z_fields.z_alpha,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=self.grid.num_levels,
                vertical_end=self.grid.num_levels + 1,
                offset_provider={},
            )
        else:
            log.debug(f"corrector start stencil 43 44 45 45b")
            """
            z_w_expl (1:nlev-1):
                Compute the explicit term in vertical momentum equation at half levels (cell center). See the first equation below eq. 3.25 in ICON tutorial 2023.
                z_w_expl = advection of w + cpd theta' dpi0/dz + cpd theta (1 - eta_impl) dpi'/dz + F_divdamp @ k+1/2 level
                advection of w = ddt_w_adv_pc
                cpd theta' dpi0/dz + cpd theta (1 - eta_impl) dpi'/dz = cpd z_th_ddz_exner_c
                eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
                eta_expl = 1.0 - eta_impl = vwind_expl_wgt
            z_contr_w_fl_l (1:nlev-1):
                Compute the vertical mass flux at half levels (cell center). See second term on RHS of mass conservation in eq. 3.21 in ICON tutorial 2023.
                z_contr_w_fl_l = rho * (-contravariant_correction + vwind_expl_wgt * w) # TODO (Chia Rui: Check why minus sign)
                eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
                eta_expl = 1.0 - eta_impl = vwind_expl_wgt
                rho = rho_ic
            z_beta (0:nlev-1):
                Compute component of the coefficients in the tridiagonal matrix of w equation at full levels (cell center).
                See the middle term in each square bracket of eq. 3.27 and unnumbered equation below in ICON tutorial 2023.
                a b 0 0 0
                c a b 0 0
                0 c a b 0
                0 0 c a b
                0 0 0 c a
                z_beta_{k} = dt * rd * pi_{k} / (cvd * rho_{k} * theta_v_{k}) / dz_{k}
            z_alpha (0:nlev-1):
                Compute component of the coefficients in the tridiagonal matrix of w equation at half levels (cell center).
                See the last term in each square bracket of eq. 3.27 and unnumbered equation below in ICON tutorial 2023.
                z_alpha_{k-1/2} = vwind_impl_wgt rho_{k-1/2} theta_v_{k-1/2}
                eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
                eta_expl = 1.0 - eta_impl = vwind_expl_wgt
                rho_{k-1/2} is precomputed as rho_ic.
                theta_v_{k-1/2} is precomputed as theta_v_ic.
            z_alpha (nlev):
                Compute component of the coefficients in the tridiagonal matrix of w equation at half levels (cell center).
                z_alpha_{k-1/2} = 0
            z_q (0):
                Set the intermediate result for w in tridiagonal solver during forward seep at half levels (cell center) at model top to zero.
                Note that it also only has nlev levels because the model top w is not updated, although it is a half-level variable.
                z_q_{k-1/2} = 0
            """
            stencils_43_44_45_45b(
                z_w_expl=z_fields.z_w_expl,
                w_nnow=prognostic_state[nnow].w,
                z_w_divdamp=z_fields.z_w_divdamp,
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
                # vertical_end=self.grid.num_levels + 1,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
            init_cell_kdim_field_with_zero_vp(
                field_with_zero_vp=z_fields.z_alpha,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=self.grid.num_levels,
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
        """
        w^{n+1*} (nlev):
            Set updated w at half levels (cell center) at ground level to the contravariant correction (since we are using terrain following coordinates).
            w_ground = contravariant_correction
        z_contr_w_fl_l (nlev):
            Set the vertical mass flux at half levels (cell center) at ground level to zero. See second term on RHS of mass conservation in eq. 3.21 in ICON tutorial 2023.
            z_contr_w_fl_l = rho * (-contravariant_correction + vwind_expl_wgt * w) # TODO (Chia Rui: Check why minus sign)
            eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
            eta_expl = 1.0 - eta_impl = vwind_expl_wgt
            rho = rho_ic
        z_rho_expl (0:nlev-1):
            Compute the explicit term in vertical momentum equation at full levels (cell center). See RHS of mass conservation in eq. 3.21 in ICON tutorial 2023.
            z_rho_expl = rho^{n} - dt ( divergence(v^{n+1*} rho^{n}) + vwind_expl_wgt ( rho^{n}_{k-1/2} w^{n}_{k-1/2} - rho^{n}_{k+1/2} w^{n}_{k+1} ) / dz_{k} )
            eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
            eta_expl = 1.0 - eta_impl = vwind_expl_wgt
            The divergence term on RHS of the equation for z_rho_expl is precomputed as z_flxdiv_mass.
            The mass flux in second term on RHS of the equation for z_rho_expl is precomputed as z_contr_w_fl_l. i.e. z_contr_w_fl_l = vwind_expl_wgt rho^{n}_{k-1/2} w^{n}_{k-1/2}
            TODO (Chia Rui): Why /dz_{k} factor is included in divergence term?
        z_exner_expl (0:nlev-1):
            Compute the explicit term in pressure equation at full levels (cell center). See RHS of thermodynamics equation in eq. 3.21 and the second unnumbered equation below eq. 3.25 in ICON tutorial 2023.
            z_exner_expl = pi'^{n} - dt * rd * pi_{k} / (cvd * rho_{k} * theta_v_{k}) ( divergence(v^{n+1*} rho^{n} theta_v^{n}) + vwind_expl_wgt ( rho^{n}_{k-1/2} w^{n}_{k-1/2} theta_v^{n}_{k-1/2} - rho^{n}_{k+1/2} w^{n}_{k+1} theta_v^{n}_{k+1/2} ) / dz_{k} ) + dt * physics_tendency
            eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
            eta_expl = 1.0 - eta_impl = vwind_expl_wgt
            pi'^{n} is precomputed as exner_pr.
            dt * rd * pi_{k} / (cvd * rho_{k} * theta_v_{k}) / dz_{k} is precomputed as z_beta.
            The divergence term on RHS of the equation for z_exber_expl is precomputed as z_flxdiv_theta.
            The mass flux in second term on RHS of the equation for z_exner_expl is precomputed as z_contr_w_fl_l, and it is multiplied by virtual temperature at half levels (which is theta_v_ic) and become energy flux. i.e. z_contr_w_fl_l = vwind_expl_wgt rho^{n}_{k-1/2} w^{n}_{k-1/2}
            physics_tendency is represented by ddt_exner_phy.
            TODO (Chia Rui): Why /dz_{k} factor is included in divergence term?
        """
        stencils_47_48_49(
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
            vertical_start=int32(0),
            vertical_end=int32(self.grid.num_levels) + int32(1),
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
        """
        w (1:nlev-1):
            Update intermediate vertical velocity by forward sweep (RHS of the equation).
        z_q (1:nlev-1):
            Update intermediate upper element of tridiagonal matrix by forward sweep.
            During the forward seep, the middle element is normalized to 1.
        """
        # main update
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
        """
        w (1:nlev-1):
            Compute the vertical velocity by backward sweep. Model top and ground level are not updated.
            w_{k-1/2} = w_{k-1/2} + w_{k+1/2} * z_q_{k-1/2}
        """
        # main update
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
            """
            w (1:damp_nlev):
                Compute the rayleigh damping of vertical velocity at half levels (cell center).
                w_{k-1/2} = Rayleigh_damping_coeff w_{k-1/2} + (1 - Rayleigh_damping_coeff) w_{-1/2}, where w_{-1/2} is model top vertical velocity. It is zero.
                Rayleigh_damping_coeff is represented by z_raylfac.
            """
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
        """
        rho (0:nlev-1):
            Update the density at full levels (cell center) from the mass conservation equation (see eq. 3.21 in ICON tutorial 2023).
            rho^{n+1} = rho^{n} - dt ( divergence(v^{n+1*} rho^{n}) + vwind_expl_wgt ( rho^{n}_{k-1/2} w^{n}_{k-1/2} - rho^{n}_{k+1/2} w^{n}_{k+1} ) / dz_{k} ) - dt * vwind_impl_wgt ( rho^{n}_{k-1/2} w^{n+1}_{k-1/2} - rho^{n}_{k+1/2} w^{n+1}_{k+1} ) / dz_{k} )
            rho^{n+1} = z_rho_expl - dt * vwind_impl_wgt ( rho^{n}_{k-1/2} w^{n+1}_{k-1/2} - rho^{n}_{k+1/2} w^{n+1}_{k+1} ) / dz_{k} )
            eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
            eta_expl = 1.0 - eta_impl = vwind_expl_wgt
            Note that rho^{n} is used for the implicit mass flux term.
        exner (0:nlev-1):
            Update exner function at full levels (cell center) from the energy equation (see eq. 3.21 or 3.25 in ICON tutorial 2023).
            z_exner_expl = pi'^{n} - dt * rd * pi_{k} / (cvd * rho_{k} * theta_v_{k}) ( divergence(v^{n+1*} rho^{n} theta_v^{n}) + vwind_expl_wgt ( rho^{n}_{k-1/2} w^{n}_{k-1/2} theta_v^{n}_{k-1/2} - rho^{n}_{k+1/2} w^{n}_{k+1} theta_v^{n}_{k+1/2} ) / dz_{k} ) + dt * physics_tendency
            pi^{n+1} = pi_reference + z_exner_expl - dt * vwind_impl_wgt ( rd * pi^{n} ) / ( cvd * rho^{n} * theta_v^{n} ) ( rho^{n}_{k-1/2} w^{n+1}_{k-1/2} theta_v^{n}_{k-1/2} - rho^{n}_{k+1/2} w^{n+1}_{k+1} theta_v^{n}_{k+1/2} ) / dz_{k} )
            eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
            eta_expl = 1.0 - eta_impl = vwind_expl_wgt
            Note that rho^{n} and theta_v^{n} are used for the implicit flux term.
            rho^{n}_{k-1/2} theta_v^{n}_{k-1/2} is represented by z_alpha.
            dt * vwind_impl_wgt (rd * pi^{n}) / (cvd * rho^{n} * theta_v^{n} ) / dz_{k} is represented by z_beta.
        theta_v (0:nlev-1):
            Update virtual potential temperature at full levels (cell center) from the equation of state (see eqs. 3.22 and 3.23 in ICON tutorial 2023).
            rho^{n+1} theta_v^{n+1} = rho^{n} theta_v^{n} + ( cvd * rho^{n} * theta_v^{n} ) / ( rd * pi^{n} ) ( pi^{n+1} - pi^{n} )
        """
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
