# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import dataclasses
from typing import Final, Optional

import gt4py.next as gtx

import icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro_program as nhsolve_prog
import icon4py.model.common.grid.geometry as geometry
from gt4py.next import backend
from icon4py.model.common import constants
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
from icon4py.model.atmosphere.dycore.state_utils import (
    states as solve_nh_states,
    utils as solve_nh_utils,
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
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import (
    base as grid_def,
    horizontal as h_grid,
    vertical as v_grid,
    icon as icon_grid,
)
from icon4py.model.common.math import smagorinsky
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc
from icon4py.model.common import field_type_aliases as fa
import enum

# flake8: noqa
log = logging.getLogger(__name__)


class TimeSteppingScheme(enum.IntEnum):
    """Parameter called `itime_scheme` in ICON namelist."""

    #: Contravariant vertical velocity is computed in the predictor step only, velocity tendencies are computed in the corrector step only
    MOST_EFFICIENT = 4
    #: Contravariant vertical velocity is computed in both substeps (beneficial for numerical stability in very-high resolution setups with extremely steep slopes)
    STABLE = 5
    #:  As STABLE, but velocity tendencies are also computed in both substeps (no benefit, but more expensive)
    EXPENSIVE = 6


class DivergenceDampingType(enum.IntEnum):
    #: divergence damping acting on 3D divergence
    THREE_DIMENSIONAL = 3
    #: combination of 3D div.damping in the troposphere with transition to 2D div. damping in the stratosphere
    COMBINED = 32


class DivergenceDampingOrder(enum.IntEnum):
    #: 2nd order divergence damping
    SECOND_ORDER = 2
    #: 4th order divergence damping
    FOURTH_ORDER = 4
    #: combined 2nd and 4th orders divergence damping and enhanced vertical wind off - centering during initial spinup phase
    COMBINED = 24


class HorizontalPressureDiscretizationType(enum.IntEnum):
    """Parameter called igradp_method in ICON namelist."""

    #: conventional discretization with metric correction term
    CONVENTIONAL = 1
    #: Taylor-expansion-based reconstruction of pressure
    TAYLOR = 2
    #: Similar discretization as igradp_method_taylor, but uses hydrostatic approximation for downward extrapolation over steep slopes
    TAYLOR_HYDRO = 3
    #: Cubic / quadratic polynomial interpolation for pressure reconstruction
    POLYNOMIAL = 4
    #: Same as igradp_method_polynomial, but hydrostatic approximation for downward extrapolation over steep slopes
    POLYNOMIAL_HYDRO = 5


class RhoThetaAdvectionType(enum.IntEnum):
    """Parameter called iadv_rhotheta in ICON namelist."""

    #: simple 2nd order upwind-biased scheme
    SIMPLE = 1
    #: 2nd order Miura horizontal
    MIURA = 2


@dataclasses.dataclass
class IntermediateFields:
    """
    Encapsulate internal fields of SolveNonHydro that contain shared state over predictor and corrector step.

    Encapsulates internal fields used in SolveNonHydro. Fields (and the class!)
    follow the naming convention of ICON to prepend local fields of a module with z_. Contrary to
    other such z_ fields inside SolveNonHydro the fields in this dataclass
    contain state that is built up over the predictor and corrector part in a timestep.
    """

    z_gradh_exner: fa.EdgeKField[float]
    z_alpha: fa.EdgeKField[
        float
    ]  # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    z_beta: fa.CellKField[float]
    z_w_expl: fa.EdgeKField[
        float
    ]  # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    z_exner_expl: fa.CellKField[float]
    z_q: fa.CellKField[float]
    z_contr_w_fl_l: fa.EdgeKField[
        float
    ]  # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    z_rho_e: fa.EdgeKField[float]
    z_theta_v_e: fa.EdgeKField[float]
    z_kin_hor_e: fa.EdgeKField[float]
    z_vt_ie: fa.EdgeKField[float]
    z_graddiv_vn: fa.EdgeKField[float]
    z_rho_expl: fa.CellKField[float]
    z_dwdz_dd: fa.CellKField[float]

    @classmethod
    def allocate(
        cls,
        grid: grid_def.BaseGrid,
        backend: Optional[backend.Backend] = None,
    ):
        return IntermediateFields(
            z_gradh_exner=field_alloc.allocate_zero_field(
                dims.EdgeDim, dims.KDim, grid=grid, backend=backend
            ),
            z_alpha=field_alloc.allocate_zero_field(
                dims.CellDim, dims.KDim, is_halfdim=True, grid=grid, backend=backend
            ),
            z_beta=field_alloc.allocate_zero_field(
                dims.CellDim, dims.KDim, grid=grid, backend=backend
            ),
            z_w_expl=field_alloc.allocate_zero_field(
                dims.CellDim, dims.KDim, is_halfdim=True, grid=grid, backend=backend
            ),
            z_exner_expl=field_alloc.allocate_zero_field(
                dims.CellDim, dims.KDim, grid=grid, backend=backend
            ),
            z_q=field_alloc.allocate_zero_field(
                dims.CellDim, dims.KDim, grid=grid, backend=backend
            ),
            z_contr_w_fl_l=field_alloc.allocate_zero_field(
                dims.CellDim, dims.KDim, is_halfdim=True, grid=grid, backend=backend
            ),
            z_rho_e=field_alloc.allocate_zero_field(
                dims.EdgeDim, dims.KDim, grid=grid, backend=backend
            ),
            z_theta_v_e=field_alloc.allocate_zero_field(
                dims.EdgeDim, dims.KDim, grid=grid, backend=backend
            ),
            z_graddiv_vn=field_alloc.allocate_zero_field(
                dims.EdgeDim, dims.KDim, grid=grid, backend=backend
            ),
            z_rho_expl=field_alloc.allocate_zero_field(
                dims.CellDim, dims.KDim, grid=grid, backend=backend
            ),
            z_dwdz_dd=field_alloc.allocate_zero_field(
                dims.CellDim, dims.KDim, grid=grid, backend=backend
            ),
            z_kin_hor_e=field_alloc.allocate_zero_field(
                dims.EdgeDim, dims.KDim, grid=grid, backend=backend
            ),
            z_vt_ie=field_alloc.allocate_zero_field(
                dims.EdgeDim, dims.KDim, grid=grid, backend=backend
            ),
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
        itime_scheme: TimeSteppingScheme = TimeSteppingScheme.MOST_EFFICIENT,
        iadv_rhotheta: RhoThetaAdvectionType = RhoThetaAdvectionType.MIURA,
        igradp_method: HorizontalPressureDiscretizationType = HorizontalPressureDiscretizationType.TAYLOR_HYDRO,
        ndyn_substeps_var: float = 5.0,
        rayleigh_type: constants.RayleighType = constants.RayleighType.KLEMP,
        rayleigh_coeff: float = 0.05,
        divdamp_order: DivergenceDampingOrder = DivergenceDampingOrder.COMBINED,  # the ICON default is 4,
        is_iau_active: bool = False,
        iau_wgt_dyn: float = 0.0,
        divdamp_type: DivergenceDampingType = DivergenceDampingType.THREE_DIMENSIONAL,
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

        if self.igradp_method != HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
            raise NotImplementedError("igradp_method can only be 3")

        if self.itime_scheme != TimeSteppingScheme.MOST_EFFICIENT:
            raise NotImplementedError("itime_scheme can only be 4")

        if self.divdamp_order != DivergenceDampingOrder.COMBINED:
            raise NotImplementedError("divdamp_order can only be 24")

        if self.divdamp_type == DivergenceDampingType.COMBINED:
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
    def __init__(
        self,
        backend: backend.Backend,
        exchange: decomposition.ExchangeRuntime = decomposition.SingleNodeExchange(),
    ):
        self._exchange = exchange
        self._backend = backend
        self._initialized = False
        self.grid: Optional[icon_grid.IconGrid] = None
        self.config: Optional[NonHydrostaticConfig] = None
        self.params: Optional[NonHydrostaticParams] = None
        self.metric_state_nonhydro: Optional[solve_nh_states.MetricStateNonHydro] = None
        self.interpolation_state: Optional[solve_nh_states.InterpolationState] = None
        self.vertical_params: Optional[v_grid.VerticalGrid] = None
        self.edge_geometry: Optional[geometry.EdgeParams] = None
        self.cell_params: Optional[geometry.CellParams] = None
        self.velocity_advection: Optional[VelocityAdvection] = None
        self.l_vert_nested: bool = False
        self.enh_divdamp_fac: Optional[fa.KField[float]] = None
        self.scal_divdamp: Optional[fa.KField[float]] = None
        self._bdy_divdamp: Optional[fa.KField[float]] = None
        self.p_test_run = True
        self.jk_start = 0  # used in stencil_55
        self.ntl1 = 0
        self.ntl2 = 0

        self._compute_theta_and_exner = compute_theta_and_exner.with_backend(self._backend)
        self._compute_exner_from_rhotheta = compute_exner_from_rhotheta.with_backend(self._backend)
        self._update_theta_v = update_theta_v.with_backend(self._backend)
        self._init_two_cell_kdim_fields_with_zero_vp = (
            init_two_cell_kdim_fields_with_zero_vp.with_backend(self._backend)
        )
        self._compute_approx_of_2nd_vertical_derivative_of_exner = (
            compute_approx_of_2nd_vertical_derivative_of_exner.with_backend(self._backend)
        )
        self._compute_perturbation_of_rho_and_theta = (
            compute_perturbation_of_rho_and_theta.with_backend(self._backend)
        )
        self._mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl = (
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl.with_backend(self._backend)
        )
        self._mo_math_gradients_grad_green_gauss_cell_dsl = (
            mo_math_gradients_grad_green_gauss_cell_dsl.with_backend(self._backend)
        )
        self._init_two_edge_kdim_fields_with_zero_wp = (
            init_two_edge_kdim_fields_with_zero_wp.with_backend(self._backend)
        )
        self._compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates = (
            compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates.with_backend(
                self._backend
            )
        )
        self._compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates = (
            compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates.with_backend(
                self._backend
            )
        )
        self._compute_horizontal_gradient_of_exner_pressure_for_multiple_levels = (
            compute_horizontal_gradient_of_exner_pressure_for_multiple_levels.with_backend(
                self._backend
            )
        )
        self._compute_hydrostatic_correction_term = (
            compute_hydrostatic_correction_term.with_backend(self._backend)
        )
        self._apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure = (
            apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure.with_backend(
                self._backend
            )
        )
        self._add_temporal_tendencies_to_vn = add_temporal_tendencies_to_vn.with_backend(
            self._backend
        )
        self._add_analysis_increments_to_vn = add_analysis_increments_to_vn.with_backend(
            self._backend
        )
        self._compute_vn_on_lateral_boundary = compute_vn_on_lateral_boundary.with_backend(
            self._backend
        )
        self._compute_avg_vn_and_graddiv_vn_and_vt = (
            compute_avg_vn_and_graddiv_vn_and_vt.with_backend(self._backend)
        )
        self._compute_mass_flux = compute_mass_flux.with_backend(self._backend)
        self._compute_divergence_of_fluxes_of_rho_and_theta = (
            compute_divergence_of_fluxes_of_rho_and_theta.with_backend(self._backend)
        )
        self._init_two_cell_kdim_fields_with_zero_wp = (
            init_two_cell_kdim_fields_with_zero_wp.with_backend(self._backend)
        )
        self._add_analysis_increments_from_data_assimilation = (
            add_analysis_increments_from_data_assimilation.with_backend(self._backend)
        )
        self._solve_tridiagonal_matrix_for_w_forward_sweep = (
            solve_tridiagonal_matrix_for_w_forward_sweep.with_backend(self._backend)
        )
        self._solve_tridiagonal_matrix_for_w_back_substitution = (
            solve_tridiagonal_matrix_for_w_back_substitution.with_backend(self._backend)
        )
        self._apply_rayleigh_damping_mechanism = apply_rayleigh_damping_mechanism.with_backend(
            self._backend
        )
        self._compute_results_for_thermodynamic_variables = (
            compute_results_for_thermodynamic_variables.with_backend(self._backend)
        )
        self._compute_dwdz_for_divergence_damping = (
            compute_dwdz_for_divergence_damping.with_backend(self._backend)
        )
        self._copy_cell_kdim_field_to_vp = copy_cell_kdim_field_to_vp.with_backend(self._backend)
        self._compute_rho_virtual_potential_temperatures_and_pressure_gradient = (
            compute_rho_virtual_potential_temperatures_and_pressure_gradient.with_backend(
                self._backend
            )
        )
        self._add_vertical_wind_derivative_to_divergence_damping = (
            add_vertical_wind_derivative_to_divergence_damping.with_backend(self._backend)
        )
        self._add_temporal_tendencies_to_vn_by_interpolating_between_time_levels = (
            add_temporal_tendencies_to_vn_by_interpolating_between_time_levels.with_backend(
                self._backend
            )
        )
        self._compute_graddiv2_of_vn = compute_graddiv2_of_vn.with_backend(self._backend)
        self._apply_2nd_order_divergence_damping = apply_2nd_order_divergence_damping.with_backend(
            self._backend
        )
        self._apply_weighted_2nd_and_4th_order_divergence_damping = (
            apply_weighted_2nd_and_4th_order_divergence_damping.with_backend(self._backend)
        )
        self._apply_4th_order_divergence_damping = apply_4th_order_divergence_damping.with_backend(
            self._backend
        )
        self._compute_avg_vn = compute_avg_vn.with_backend(self._backend)
        self._accumulate_prep_adv_fields = accumulate_prep_adv_fields.with_backend(self._backend)
        self._update_mass_volume_flux = update_mass_volume_flux.with_backend(self._backend)
        self._update_dynamical_exner_time_increment = (
            update_dynamical_exner_time_increment.with_backend(self._backend)
        )
        self._init_cell_kdim_field_with_zero_wp = init_cell_kdim_field_with_zero_wp.with_backend(
            self._backend
        )
        self._update_mass_flux_weighted = update_mass_flux_weighted.with_backend(self._backend)
        self._compute_z_raylfac = solve_nh_utils.compute_z_raylfac.with_backend(self._backend)
        self._predictor_stencils_2_3 = nhsolve_prog.predictor_stencils_2_3.with_backend(
            self._backend
        )
        self._predictor_stencils_4_5_6 = nhsolve_prog.predictor_stencils_4_5_6.with_backend(
            self._backend
        )
        self._compute_pressure_gradient_and_perturbed_rho_and_potential_temperatures = nhsolve_prog.compute_pressure_gradient_and_perturbed_rho_and_potential_temperatures.with_backend(
            self._backend
        )
        self._predictor_stencils_11_lower_upper = (
            nhsolve_prog.predictor_stencils_11_lower_upper.with_backend(self._backend)
        )
        self._compute_horizontal_advection_of_rho_and_theta = (
            nhsolve_prog.compute_horizontal_advection_of_rho_and_theta.with_backend(self._backend)
        )
        self._predictor_stencils_35_36 = nhsolve_prog.predictor_stencils_35_36.with_backend(
            self._backend
        )
        self._predictor_stencils_37_38 = nhsolve_prog.predictor_stencils_37_38.with_backend(
            self._backend
        )
        self._stencils_39_40 = nhsolve_prog.stencils_39_40.with_backend(self._backend)
        self._stencils_43_44_45_45b = nhsolve_prog.stencils_43_44_45_45b.with_backend(self._backend)
        self._stencils_47_48_49 = nhsolve_prog.stencils_47_48_49.with_backend(self._backend)
        self._stencils_61_62 = nhsolve_prog.stencils_61_62.with_backend(self._backend)
        self._en_smag_fac_for_zero_nshift = smagorinsky.en_smag_fac_for_zero_nshift.with_backend(
            self._backend
        )
        self._init_test_fields = nhsolve_prog.init_test_fields.with_backend(self._backend)
        self._stencils_42_44_45_45b = nhsolve_prog.stencils_42_44_45_45b.with_backend(self._backend)

    def init(
        self,
        grid: icon_grid.IconGrid,
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        metric_state_nonhydro: solve_nh_states.MetricStateNonHydro,
        interpolation_state: solve_nh_states.InterpolationState,
        vertical_params: v_grid.VerticalGrid,
        edge_geometry: geometry.EdgeParams,
        cell_geometry: geometry.CellParams,
        owner_mask: fa.CellField[bool],
    ):
        """
        Initialize NonHydrostatic granule with configuration.

        calculates all local fields that are used in nh_solve within the time loop
        """
        self.grid = grid
        self.config: NonHydrostaticConfig = config
        self.params: NonHydrostaticParams = params
        self.metric_state_nonhydro: solve_nh_states.MetricStateNonHydro = metric_state_nonhydro
        self.interpolation_state: solve_nh_states.InterpolationState = interpolation_state
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
            backend=self._backend,
        )
        self._allocate_local_fields()
        self._determine_local_domains()
        # TODO (magdalena) vertical nesting is only relevant in the context of
        #      horizontal nesting, since we don't support this we should remove this option
        if grid.lvert_nest:
            self.l_vert_nested = True
            self.jk_start = 1
        else:
            self.jk_start = 0

        self._en_smag_fac_for_zero_nshift(
            self.vertical_params.interface_physical_height,
            self.config.divdamp_fac,
            self.config.divdamp_fac2,
            self.config.divdamp_fac3,
            self.config.divdamp_fac4,
            self.config.divdamp_z,
            self.config.divdamp_z2,
            self.config.divdamp_z3,
            self.config.divdamp_z4,
            self.enh_divdamp_fac,
            offset_provider={"Koff": dims.KDim},
        )

        self.p_test_run = True
        self._initialized = True

    @property
    def initialized(self):
        return self._initialized

    def _allocate_local_fields(self):
        self.z_exner_ex_pr = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, is_halfdim=True, grid=self.grid, backend=self._backend
        )
        self.z_exner_ic = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, is_halfdim=True, grid=self.grid, backend=self._backend
        )
        self.z_dexner_dz_c_1 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_theta_v_pr_ic = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, is_halfdim=True, grid=self.grid, backend=self._backend
        )
        self.z_th_ddz_exner_c = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_rth_pr_1 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_rth_pr_2 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_grad_rth_1 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_grad_rth_2 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_grad_rth_3 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_grad_rth_4 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_dexner_dz_c_2 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_hydro_corr = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_vn_avg = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_theta_v_fl_e = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_flxdiv_mass = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_flxdiv_theta = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_rho_v = field_alloc.allocate_zero_field(
            dims.VertexDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_theta_v_v = field_alloc.allocate_zero_field(
            dims.VertexDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_graddiv2_vn = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.k_field = field_alloc.allocate_indices(
            dims.KDim, grid=self.grid, is_halfdim=True, backend=self._backend
        )
        self.z_w_concorr_me = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self.grid, backend=self._backend
        )
        self.z_hydro_corr_horizontal = field_alloc.allocate_zero_field(
            dims.EdgeDim, grid=self.grid, backend=self._backend
        )
        self.z_raylfac = field_alloc.allocate_zero_field(
            dims.KDim, grid=self.grid, backend=self._backend
        )
        self.enh_divdamp_fac = field_alloc.allocate_zero_field(
            dims.KDim, grid=self.grid, backend=self._backend
        )
        self._bdy_divdamp = field_alloc.allocate_zero_field(
            dims.KDim, grid=self.grid, backend=self._backend
        )
        self.scal_divdamp = field_alloc.allocate_zero_field(
            dims.KDim, grid=self.grid, backend=self._backend
        )
        self.intermediate_fields = IntermediateFields.allocate(self.grid, backend=self._backend)

    def _determine_local_domains(self):
        vertex_domain = h_grid.domain(dims.VertexDim)
        cell_domain = h_grid.domain(dims.CellDim)
        cell_halo_level_2 = cell_domain(h_grid.Zone.HALO_LEVEL_2)
        edge_domain = h_grid.domain(dims.EdgeDim)
        edge_halo_level_2 = edge_domain(h_grid.Zone.HALO_LEVEL_2)

        self._start_cell_lateral_boundary = self.grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY)
        )
        self._start_cell_lateral_boundary_level_3 = self.grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        self._start_cell_nudging = self.grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._start_cell_local = self.grid.start_index(cell_domain(h_grid.Zone.LOCAL))
        self._start_cell_halo = self.grid.start_index(cell_domain(h_grid.Zone.HALO))
        self._start_cell_halo_level_2 = self.grid.start_index(cell_halo_level_2)
        self._end_cell_lateral_boundary_level_4 = self.grid.end_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        self._end_cell_nudging = self.grid.end_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self.grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._end_cell_halo = self.grid.end_index(cell_domain(h_grid.Zone.HALO))
        self._end_cell_halo_level_2 = self.grid.end_index(cell_halo_level_2)
        self._end_cell_end = self.grid.end_index(cell_domain(h_grid.Zone.END))

        self._start_edge_lateral_boundary = self.grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY)
        )
        self._start_edge_lateral_boundary_level_5 = self.grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._start_edge_lateral_boundary_level_7 = self.grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
        )
        self._start_edge_nudging_level_2 = self.grid.start_index(
            edge_domain(h_grid.Zone.NUDGING_LEVEL_2)
        )

        self._start_edge_halo_level_2 = self.grid.start_index(edge_halo_level_2)

        self._end_cell_lateral_boundary_level_4 = self.grid.end_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        self._end_edge_nudging = self.grid.end_index(edge_domain(h_grid.Zone.NUDGING))
        self._end_edge_local = self.grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        self._end_edge_halo = self.grid.end_index(edge_domain(h_grid.Zone.HALO))
        self._end_edge_halo_level_2 = self.grid.end_index(edge_halo_level_2)
        self._end_edge_end = self.grid.end_index(edge_domain(h_grid.Zone.END))

        self._start_vertex_lateral_boundary_level_2 = self.grid.start_index(
            vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._end_vertex_halo = self.grid.end_index(vertex_domain(h_grid.Zone.HALO))

    def set_timelevels(self, nnow, nnew):
        #  Set time levels of ddt_adv fields for call to velocity_tendencies
        if self.config.itime_scheme == TimeSteppingScheme.MOST_EFFICIENT:
            self.ntl1 = nnow
            self.ntl2 = nnew
        else:
            self.ntl1 = 0
            self.ntl2 = 0

    def time_step(
        self,
        diagnostic_state_nh: solve_nh_states.DiagnosticStateNonHydro,
        prognostic_state_ls: list[prognostics.PrognosticState],
        prep_adv: solve_nh_states.PrepAdvection,
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

        # # TODO: abishekg7 move this to tests
        if self.p_test_run:
            self._init_test_fields(
                self.intermediate_fields.z_rho_e,
                self.intermediate_fields.z_theta_v_e,
                self.intermediate_fields.z_dwdz_dd,
                self.intermediate_fields.z_graddiv_vn,
                self._start_edge_lateral_boundary,
                self._end_edge_local,
                self._start_cell_lateral_boundary,
                self._end_cell_end,
                vertical_start=gtx.int32(0),
                vertical_end=self.grid.num_levels,
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

        if self.grid.limited_area:
            self._compute_theta_and_exner(
                bdy_halo_c=self.metric_state_nonhydro.bdy_halo_c,
                rho=prognostic_state_ls[nnew].rho,
                theta_v=prognostic_state_ls[nnew].theta_v,
                exner=prognostic_state_ls[nnew].exner,
                rd_o_cvd=self.params.rd_o_cvd,
                rd_o_p0ref=self.params.rd_o_p0ref,
                horizontal_start=self._start_cell_local,
                horizontal_end=self._end_cell_end,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

            self._compute_exner_from_rhotheta(
                rho=prognostic_state_ls[nnew].rho,
                theta_v=prognostic_state_ls[nnew].theta_v,
                exner=prognostic_state_ls[nnew].exner,
                rd_o_cvd=self.params.rd_o_cvd,
                rd_o_p0ref=self.params.rd_o_p0ref,
                horizontal_start=self._start_cell_lateral_boundary,
                horizontal_end=self._end_cell_lateral_boundary_level_4,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        self._update_theta_v(
            mask_prog_halo_c=self.metric_state_nonhydro.mask_prog_halo_c,
            rho_now=prognostic_state_ls[nnow].rho,
            theta_v_now=prognostic_state_ls[nnow].theta_v,
            exner_new=prognostic_state_ls[nnew].exner,
            exner_now=prognostic_state_ls[nnow].exner,
            rho_new=prognostic_state_ls[nnew].rho,
            theta_v_new=prognostic_state_ls[nnew].theta_v,
            cvd_o_rd=self.params.cvd_o_rd,
            horizontal_start=self._start_cell_halo,
            horizontal_end=self._end_cell_end,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

    # flake8: noqa: C901
    def run_predictor_step(
        self,
        diagnostic_state_nh: solve_nh_states.DiagnosticStateNonHydro,
        prognostic_state: list[prognostics.PrognosticState],
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
            if self.config.itime_scheme == TimeSteppingScheme.MOST_EFFICIENT and not l_init:
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

        #  Precompute Rayleigh damping factor
        self._compute_z_raylfac(
            rayleigh_w=self.metric_state_nonhydro.rayleigh_w,
            dtime=dtime,
            z_raylfac=self.z_raylfac,
            offset_provider={},
        )

        # initialize nest boundary points of z_rth_pr with zero
        if self.grid.limited_area:
            self._init_two_cell_kdim_fields_with_zero_vp(
                cell_kdim_field_with_zero_vp_1=self.z_rth_pr_1,
                cell_kdim_field_with_zero_vp_2=self.z_rth_pr_2,
                horizontal_start=self._start_cell_lateral_boundary,
                horizontal_end=self._end_cell_end,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        self._predictor_stencils_2_3(
            exner_exfac=self.metric_state_nonhydro.exner_exfac,
            exner=prognostic_state[nnow].exner,
            exner_ref_mc=self.metric_state_nonhydro.exner_ref_mc,
            exner_pr=diagnostic_state_nh.exner_pr,
            z_exner_ex_pr=self.z_exner_ex_pr,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider={},
        )

        if self.config.igradp_method == HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
            self._predictor_stencils_4_5_6(
                wgtfacq_c_dsl=self.metric_state_nonhydro.wgtfacq_c,
                z_exner_ex_pr=self.z_exner_ex_pr,
                z_exner_ic=self.z_exner_ic,
                wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                z_dexner_dz_c_1=self.z_dexner_dz_c_1,
                horizontal_start=self._start_cell_lateral_boundary_level_3,
                horizontal_end=self._end_cell_halo,
                vertical_start=max(1, self.vertical_params.nflatlev),
                vertical_end=self.grid.num_levels + 1,
                offset_provider=self.grid.offset_providers,
            )

            if self.vertical_params.nflatlev == 1:
                # Perturbation Exner pressure on top half level
                raise NotImplementedError("nflatlev=1 not implemented")

        self._compute_pressure_gradient_and_perturbed_rho_and_potential_temperatures(
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
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        # Perturbation theta at top and surface levels
        self._predictor_stencils_11_lower_upper(
            wgtfacq_c_dsl=self.metric_state_nonhydro.wgtfacq_c,
            z_rth_pr=self.z_rth_pr_2,
            theta_ref_ic=self.metric_state_nonhydro.theta_ref_ic,
            z_theta_v_pr_ic=self.z_theta_v_pr_ic,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            k_field=self.k_field,
            nlev=self.grid.num_levels,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        if self.config.igradp_method == HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
            # Second vertical derivative of perturbation Exner pressure (hydrostatic approximation)
            self._compute_approx_of_2nd_vertical_derivative_of_exner(
                z_theta_v_pr_ic=self.z_theta_v_pr_ic,
                d2dexdz2_fac1_mc=self.metric_state_nonhydro.d2dexdz2_fac1_mc,
                d2dexdz2_fac2_mc=self.metric_state_nonhydro.d2dexdz2_fac2_mc,
                z_rth_pr_2=self.z_rth_pr_2,
                z_dexner_dz_c_2=self.z_dexner_dz_c_2,
                horizontal_start=self._start_cell_lateral_boundary_level_3,
                horizontal_end=self._end_cell_halo,
                vertical_start=self.vertical_params.nflat_gradp,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        # Add computation of z_grad_rth (perturbation density and virtual potential temperature at main levels)
        # at outer halo points: needed for correct calculation of the upwind gradients for Miura scheme

        self._compute_perturbation_of_rho_and_theta(
            rho=prognostic_state[nnow].rho,
            rho_ref_mc=self.metric_state_nonhydro.rho_ref_mc,
            theta_v=prognostic_state[nnow].theta_v,
            theta_ref_mc=self.metric_state_nonhydro.theta_ref_mc,
            z_rth_pr_1=self.z_rth_pr_1,
            z_rth_pr_2=self.z_rth_pr_2,
            horizontal_start=self._start_cell_halo_level_2,
            horizontal_end=self._end_cell_halo_level_2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        # Compute rho and theta at edges for horizontal flux divergence term
        if self.config.iadv_rhotheta == RhoThetaAdvectionType.SIMPLE:
            self._mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                p_cell_in=prognostic_state[nnow].rho,
                c_intp=self.interpolation_state.c_intp,
                p_vert_out=self.z_rho_v,
                horizontal_start=self._start_vertex_lateral_boundary_level_2,
                horizontal_end=self._end_vertex_halo,
                vertical_start=0,
                vertical_end=self.grid.num_levels,  # UBOUND(p_cell_in,2)
                offset_provider=self.grid.offset_providers,
            )
            self._mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                p_cell_in=prognostic_state[nnow].theta_v,
                c_intp=self.interpolation_state.c_intp,
                p_vert_out=self.z_theta_v_v,
                horizontal_start=self._start_vertex_lateral_boundary_level_2,
                horizontal_end=self._end_vertex_halo,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
        elif self.config.iadv_rhotheta == RhoThetaAdvectionType.MIURA:
            # Compute Green-Gauss gradients for rho and theta
            self._mo_math_gradients_grad_green_gauss_cell_dsl(
                p_grad_1_u=self.z_grad_rth_1,
                p_grad_1_v=self.z_grad_rth_2,
                p_grad_2_u=self.z_grad_rth_3,
                p_grad_2_v=self.z_grad_rth_4,
                p_ccpr1=self.z_rth_pr_1,
                p_ccpr2=self.z_rth_pr_2,
                geofac_grg_x=self.interpolation_state.geofac_grg_x,
                geofac_grg_y=self.interpolation_state.geofac_grg_y,
                horizontal_start=self._start_cell_lateral_boundary_level_3,
                horizontal_end=self._end_cell_halo,
                vertical_start=0,
                vertical_end=self.grid.num_levels,  # UBOUND(p_ccpr,2)
                offset_provider=self.grid.offset_providers,
            )
        if self.config.iadv_rhotheta <= 2:
            self._init_two_edge_kdim_fields_with_zero_wp(
                edge_kdim_field_with_zero_wp_1=z_fields.z_rho_e,
                edge_kdim_field_with_zero_wp_2=z_fields.z_theta_v_e,
                horizontal_start=self._start_edge_halo_level_2,
                horizontal_end=self._end_edge_halo_level_2,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
            # initialize also nest boundary points with zero
            if self.grid.limited_area:
                self._init_two_edge_kdim_fields_with_zero_wp(
                    edge_kdim_field_with_zero_wp_1=z_fields.z_rho_e,
                    edge_kdim_field_with_zero_wp_2=z_fields.z_theta_v_e,
                    horizontal_start=self._start_edge_lateral_boundary,
                    horizontal_end=self._end_edge_halo,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )
            if self.config.iadv_rhotheta == RhoThetaAdvectionType.MIURA:
                # Compute upwind-biased values for rho and theta starting from centered differences
                # Note: the length of the backward trajectory should be 0.5*dtime*(vn,vt) in order to arrive
                # at a second-order accurate FV discretization, but twice the length is needed for numerical stability

                self._compute_horizontal_advection_of_rho_and_theta(
                    p_vn=prognostic_state[nnow].vn,
                    p_vt=diagnostic_state_nh.vt,
                    pos_on_tplane_e_1=self.interpolation_state.pos_on_tplane_e_1,
                    pos_on_tplane_e_2=self.interpolation_state.pos_on_tplane_e_2,
                    primal_normal_cell_1=self.edge_geometry.primal_normal_cell[0],
                    dual_normal_cell_1=self.edge_geometry.dual_normal_cell[0],
                    primal_normal_cell_2=self.edge_geometry.primal_normal_cell[1],
                    dual_normal_cell_2=self.edge_geometry.dual_normal_cell[1],
                    p_dthalf=(0.5 * dtime),
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
                    horizontal_start=self._start_edge_lateral_boundary_level_7,
                    horizontal_end=self._end_edge_halo,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider=self.grid.offset_providers,
                )

        # Remaining computations at edge points
        self._compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates(
            inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
            z_exner_ex_pr=self.z_exner_ex_pr,
            z_gradh_exner=z_fields.z_gradh_exner,
            horizontal_start=self._start_edge_nudging_level_2,
            horizontal_end=self._end_edge_local,
            vertical_start=0,
            vertical_end=self.vertical_params.nflatlev,
            offset_provider=self.grid.offset_providers,
        )

        if self.config.igradp_method == HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
            # horizontal gradient of Exner pressure, including metric correction
            # horizontal gradient of Exner pressure, Taylor-expansion-based reconstruction

            self._compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates(
                inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                z_exner_ex_pr=self.z_exner_ex_pr,
                ddxn_z_full=self.metric_state_nonhydro.ddxn_z_full,
                c_lin_e=self.interpolation_state.c_lin_e,
                z_dexner_dz_c_1=self.z_dexner_dz_c_1,
                z_gradh_exner=z_fields.z_gradh_exner,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=self.vertical_params.nflatlev,
                vertical_end=gtx.int32(self.vertical_params.nflat_gradp + 1),
                offset_provider=self.grid.offset_providers,
            )

            self._compute_horizontal_gradient_of_exner_pressure_for_multiple_levels(
                inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                z_exner_ex_pr=self.z_exner_ex_pr,
                zdiff_gradp=self.metric_state_nonhydro.zdiff_gradp,
                ikoffset=self.metric_state_nonhydro.vertoffset_gradp,
                z_dexner_dz_c_1=self.z_dexner_dz_c_1,
                z_dexner_dz_c_2=self.z_dexner_dz_c_2,
                z_gradh_exner=z_fields.z_gradh_exner,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=gtx.int32(self.vertical_params.nflat_gradp + 1),
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
        # compute hydrostatically approximated correction term that replaces downward extrapolation
        if self.config.igradp_method == HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
            self._compute_hydrostatic_correction_term(
                theta_v=prognostic_state[nnow].theta_v,
                ikoffset=self.metric_state_nonhydro.vertoffset_gradp,
                zdiff_gradp=self.metric_state_nonhydro.zdiff_gradp,
                theta_v_ic=diagnostic_state_nh.theta_v_ic,
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
                z_hydro_corr=self.z_hydro_corr,
                grav_o_cpd=self.params.grav_o_cpd,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=self.grid.num_levels - 1,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
        lowest_level = self.grid.num_levels - 1
        hydro_corr_horizontal = gtx.as_field(
            (dims.EdgeDim,),
            self.z_hydro_corr.ndarray[:, lowest_level],
            allocator=self._backend.allocator,
        )

        if self.config.igradp_method == HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
            self._apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure(
                ipeidx_dsl=self.metric_state_nonhydro.ipeidx_dsl,
                pg_exdist=self.metric_state_nonhydro.pg_exdist,
                z_hydro_corr=hydro_corr_horizontal,
                z_gradh_exner=z_fields.z_gradh_exner,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_end,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        self._add_temporal_tendencies_to_vn(
            vn_nnow=prognostic_state[nnow].vn,
            ddt_vn_apc_ntl1=diagnostic_state_nh.ddt_vn_apc_pc[self.ntl1],
            ddt_vn_phy=diagnostic_state_nh.ddt_vn_phy,
            z_theta_v_e=z_fields.z_theta_v_e,
            z_gradh_exner=z_fields.z_gradh_exner,
            vn_nnew=prognostic_state[nnew].vn,
            dtime=dtime,
            cpd=constants.CPD,
            horizontal_start=self._start_edge_nudging_level_2,
            horizontal_end=self._end_edge_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        if self.config.is_iau_active:
            self._add_analysis_increments_to_vn(
                vn_incr=diagnostic_state_nh.vn_incr,
                vn=prognostic_state[nnew].vn,
                iau_wgt_dyn=self.config.iau_wgt_dyn,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        if self.grid.limited_area:
            self._compute_vn_on_lateral_boundary(
                grf_tend_vn=diagnostic_state_nh.grf_tend_vn,
                vn_now=prognostic_state[nnow].vn,
                vn_new=prognostic_state[nnew].vn,
                dtime=dtime,
                horizontal_start=self._start_edge_lateral_boundary,
                horizontal_end=self._end_edge_nudging,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
        log.debug("exchanging prognostic field 'vn' and local field 'z_rho_e'")
        self._exchange.exchange_and_wait(dims.EdgeDim, prognostic_state[nnew].vn, z_fields.z_rho_e)

        self._compute_avg_vn_and_graddiv_vn_and_vt(
            e_flx_avg=self.interpolation_state.e_flx_avg,
            vn=prognostic_state[nnew].vn,
            geofac_grdiv=self.interpolation_state.geofac_grdiv,
            rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
            z_vn_avg=self.z_vn_avg,
            z_graddiv_vn=z_fields.z_graddiv_vn,
            vt=diagnostic_state_nh.vt,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        self._compute_mass_flux(
            z_rho_e=z_fields.z_rho_e,
            z_vn_avg=self.z_vn_avg,
            ddqz_z_full_e=self.metric_state_nonhydro.ddqz_z_full_e,
            z_theta_v_e=z_fields.z_theta_v_e,
            mass_fl_e=diagnostic_state_nh.mass_fl_e,
            z_theta_v_fl_e=self.z_theta_v_fl_e,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        self._predictor_stencils_35_36(
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
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        if not self.l_vert_nested:
            self._predictor_stencils_37_38(
                vn=prognostic_state[nnew].vn,
                vt=diagnostic_state_nh.vt,
                vn_ie=diagnostic_state_nh.vn_ie,
                z_vt_ie=z_fields.z_vt_ie,
                z_kin_hor_e=z_fields.z_kin_hor_e,
                wgtfacq_e_dsl=self.metric_state_nonhydro.wgtfacq_e,
                horizontal_start=self._start_edge_lateral_boundary_level_5,
                horizontal_end=self._end_edge_halo_level_2,
                vertical_start=0,
                vertical_end=self.grid.num_levels + 1,
                offset_provider=self.grid.offset_providers,
            )

        self._stencils_39_40(
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            z_w_concorr_me=self.z_w_concorr_me,
            wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
            wgtfacq_c_dsl=self.metric_state_nonhydro.wgtfacq_c,
            w_concorr_c=diagnostic_state_nh.w_concorr_c,
            k_field=self.k_field,
            nflatlev_startindex_plus1=gtx.int32(self.vertical_params.nflatlev + 1),
            nlev=self.grid.num_levels,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        self._compute_divergence_of_fluxes_of_rho_and_theta(
            geofac_div=self.interpolation_state.geofac_div,
            mass_fl_e=diagnostic_state_nh.mass_fl_e,
            z_theta_v_fl_e=self.z_theta_v_fl_e,
            z_flxdiv_mass=self.z_flxdiv_mass,
            z_flxdiv_theta=self.z_flxdiv_theta,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        self._stencils_43_44_45_45b(
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
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider={},
        )

        if not self.l_vert_nested:
            self._init_two_cell_kdim_fields_with_zero_wp(
                cell_kdim_field_with_zero_wp_1=prognostic_state[nnew].w,
                cell_kdim_field_with_zero_wp_2=z_fields.z_contr_w_fl_l,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=0,
                vertical_end=1,
                offset_provider={},
            )
        self._stencils_47_48_49(
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
            dtime=dtime,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        if self.config.is_iau_active:
            self._add_analysis_increments_from_data_assimilation(
                z_fields.z_rho_expl,
                z_fields.z_exner_expl,
                diagnostic_state_nh.rho_incr,
                diagnostic_state_nh.exner_incr,
                self.config.iau_wgt_dyn,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        self._solve_tridiagonal_matrix_for_w_forward_sweep(
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
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        self._solve_tridiagonal_matrix_for_w_back_substitution(
            z_q=z_fields.z_q,
            w=prognostic_state[nnew].w,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        if self.config.rayleigh_type == constants.RayleighType.KLEMP:
            self._apply_rayleigh_damping_mechanism(
                z_raylfac=self.z_raylfac,
                w_1=prognostic_state[nnew].w_1,
                w=prognostic_state[nnew].w,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=1,
                vertical_end=gtx.int32(
                    self.vertical_params.end_index_of_damping_layer + 1
                ),  # +1 since Fortran includes boundaries
                offset_provider={},
            )

        self._compute_results_for_thermodynamic_variables(
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
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=gtx.int32(self.jk_start),
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        # compute dw/dz for divergence damping term
        if self.config.divdamp_type >= 3:
            self._compute_dwdz_for_divergence_damping(
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                w=prognostic_state[nnew].w,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                z_dwdz_dd=z_fields.z_dwdz_dd,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=self.params.kstart_dd3d,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        if at_first_substep:
            self._copy_cell_kdim_field_to_vp(
                field=prognostic_state[nnow].exner,
                field_copy=diagnostic_state_nh.exner_dyn_incr,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=self.vertical_params.kstart_moist,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        if self.grid.limited_area:
            self._stencils_61_62(
                rho_now=prognostic_state[nnow].rho,
                grf_tend_rho=diagnostic_state_nh.grf_tend_rho,
                theta_v_now=prognostic_state[nnow].theta_v,
                grf_tend_thv=diagnostic_state_nh.grf_tend_thv,
                w_now=prognostic_state[nnow].w,
                grf_tend_w=diagnostic_state_nh.grf_tend_w,
                rho_new=prognostic_state[nnew].rho,
                exner_new=prognostic_state[nnew].exner,
                w_new=prognostic_state[nnew].w,
                dtime=dtime,
                horizontal_start=self._start_cell_lateral_boundary,
                horizontal_end=self._end_cell_lateral_boundary_level_4,
                vertical_start=0,
                vertical_end=gtx.int32(self.grid.num_levels + 1),
                offset_provider={},
            )

        if self.config.divdamp_type >= 3:
            self._compute_dwdz_for_divergence_damping(
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                w=prognostic_state[nnew].w,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                z_dwdz_dd=z_fields.z_dwdz_dd,
                horizontal_start=self._start_cell_lateral_boundary,
                horizontal_end=self._end_cell_lateral_boundary_level_4,
                vertical_start=self.params.kstart_dd3d,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )
            log.debug("exchanging prognostic field 'w' and local field 'z_dwdz_dd'")
            self._exchange.exchange_and_wait(
                dims.CellDim, prognostic_state[nnew].w, z_fields.z_dwdz_dd
            )
        else:
            log.debug("exchanging prognostic field 'w'")
            self._exchange.exchange_and_wait(dims.CellDim, prognostic_state[nnew].w)

    def run_corrector_step(
        self,
        diagnostic_state_nh: solve_nh_states.DiagnosticStateNonHydro,
        prognostic_state: list[prognostics.PrognosticState],
        z_fields: IntermediateFields,
        divdamp_fac_o2: float,
        prep_adv: solve_nh_states.PrepAdvection,
        dtime: float,
        nnew: int,
        nnow: int,
        lclean_mflx: bool,
        lprep_adv: bool,
        at_last_substep: bool,
    ):
        log.info(
            f"running corrector step: dtime = {dtime}, prep_adv = {lprep_adv},  "
            f"divdamp_fac_o2 = {divdamp_fac_o2} clean_mfxl= {lclean_mflx}  "
        )

        # TODO (magdalena) is it correct to to use a config parameter here? the actual number of substeps can vary dynmically...
        #                  should this config parameter exist at all in SolveNonHydro?
        # Inverse value of ndyn_substeps for tracer advection precomputations
        r_nsubsteps = 1.0 / self.config.ndyn_substeps_var

        # scaling factor for second-order divergence damping: divdamp_fac_o2*delta_x**2
        # delta_x**2 is approximated by the mean cell area
        # Coefficient for reduced fourth-order divergence d
        scal_divdamp_o2 = divdamp_fac_o2 * self.cell_params.mean_cell_area

        solve_nh_utils._calculate_divdamp_fields(
            self.enh_divdamp_fac,
            gtx.int32(self.config.divdamp_order),
            self.cell_params.mean_cell_area,
            divdamp_fac_o2,
            self.config.nudge_max_coeff,
            constants.DBL_EPS,
            out=(self.scal_divdamp, self._bdy_divdamp),
            offset_provider={},
        )

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

        self._compute_z_raylfac(
            self.metric_state_nonhydro.rayleigh_w,
            dtime,
            self.z_raylfac,
            offset_provider={},
        )
        log.debug(f"corrector: start stencil 10")
        self._compute_rho_virtual_potential_temperatures_and_pressure_gradient(
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
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_local,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        log.debug(f"corrector: start stencil 17")
        self._add_vertical_wind_derivative_to_divergence_damping(
            hmask_dd3d=self.metric_state_nonhydro.hmask_dd3d,
            scalfac_dd3d=self.metric_state_nonhydro.scalfac_dd3d,
            inv_dual_edge_length=self.edge_geometry.inverse_dual_edge_lengths,
            z_dwdz_dd=z_fields.z_dwdz_dd,
            z_graddiv_vn=z_fields.z_graddiv_vn,
            horizontal_start=self._start_edge_lateral_boundary_level_7,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=self.params.kstart_dd3d,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        if self.config.itime_scheme == TimeSteppingScheme.MOST_EFFICIENT:
            log.debug(f"corrector: start stencil 23")
            self._add_temporal_tendencies_to_vn_by_interpolating_between_time_levels(
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
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        if (
            self.config.divdamp_order == DivergenceDampingOrder.COMBINED
            or self.config.divdamp_order == DivergenceDampingOrder.FOURTH_ORDER
        ):
            # verified for e-10
            log.debug(f"corrector start stencil 25")
            self._compute_graddiv2_of_vn(
                geofac_grdiv=self.interpolation_state.geofac_grdiv,
                z_graddiv_vn=z_fields.z_graddiv_vn,
                z_graddiv2_vn=self.z_graddiv2_vn,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider=self.grid.offset_providers,
            )

        if (
            self.config.divdamp_order == DivergenceDampingOrder.COMBINED
            and scal_divdamp_o2 > 1.0e-6
        ):
            log.debug(f"corrector: start stencil 26")
            self._apply_2nd_order_divergence_damping(
                z_graddiv_vn=z_fields.z_graddiv_vn,
                vn=prognostic_state[nnew].vn,
                scal_divdamp_o2=scal_divdamp_o2,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        # TODO: this does not get accessed in FORTRAN
        if (
            self.config.divdamp_order == DivergenceDampingOrder.COMBINED
            and divdamp_fac_o2 <= 4 * self.config.divdamp_fac
        ):
            if self.grid.limited_area:
                log.debug("corrector: start stencil 27")
                self._apply_weighted_2nd_and_4th_order_divergence_damping(
                    scal_divdamp=self.scal_divdamp,
                    bdy_divdamp=self._bdy_divdamp,
                    nudgecoeff_e=self.interpolation_state.nudgecoeff_e,
                    z_graddiv2_vn=self.z_graddiv2_vn,
                    vn=prognostic_state[nnew].vn,
                    horizontal_start=self._start_edge_nudging_level_2,
                    horizontal_end=self._end_edge_local,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )
            else:
                log.debug("corrector start stencil 4th order divdamp")
                self._apply_4th_order_divergence_damping(
                    scal_divdamp=self.scal_divdamp,
                    z_graddiv2_vn=self.z_graddiv2_vn,
                    vn=prognostic_state[nnew].vn,
                    horizontal_start=self._start_edge_nudging_level_2,
                    horizontal_end=self._end_edge_local,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )

        # TODO: this does not get accessed in FORTRAN
        if self.config.is_iau_active:
            log.debug("corrector start stencil 28")
            self._add_analysis_increments_to_vn(
                diagnostic_state_nh.vn_incr,
                prognostic_state[nnew].vn,
                self.config.iau_wgt_dyn,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
        log.debug("exchanging prognostic field 'vn'")
        self._exchange.exchange_and_wait(dims.EdgeDim, (prognostic_state[nnew].vn))
        log.debug("corrector: start stencil 31")
        self._compute_avg_vn(
            e_flx_avg=self.interpolation_state.e_flx_avg,
            vn=prognostic_state[nnew].vn,
            z_vn_avg=self.z_vn_avg,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        log.debug("corrector: start stencil 32")
        self._compute_mass_flux(
            z_rho_e=z_fields.z_rho_e,
            z_vn_avg=self.z_vn_avg,
            ddqz_z_full_e=self.metric_state_nonhydro.ddqz_z_full_e,
            z_theta_v_e=z_fields.z_theta_v_e,
            mass_fl_e=diagnostic_state_nh.mass_fl_e,
            z_theta_v_fl_e=self.z_theta_v_fl_e,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        if lprep_adv:  # Preparations for tracer advection
            log.debug("corrector: doing prep advection")
            if lclean_mflx:
                log.debug("corrector: start stencil 33")
                self._init_two_edge_kdim_fields_with_zero_wp(
                    edge_kdim_field_with_zero_wp_1=prep_adv.vn_traj,
                    edge_kdim_field_with_zero_wp_2=prep_adv.mass_flx_me,
                    horizontal_start=self._start_edge_lateral_boundary,
                    horizontal_end=self._end_edge_end,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )
            log.debug(f"corrector: start stencil 34")
            self._accumulate_prep_adv_fields(
                z_vn_avg=self.z_vn_avg,
                mass_fl_e=diagnostic_state_nh.mass_fl_e,
                vn_traj=prep_adv.vn_traj,
                mass_flx_me=prep_adv.mass_flx_me,
                r_nsubsteps=r_nsubsteps,
                horizontal_start=self._start_edge_lateral_boundary_level_5,
                horizontal_end=self._end_edge_halo_level_2,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

        # verified for e-9
        log.debug(f"corrector: start stencil 41")
        self._compute_divergence_of_fluxes_of_rho_and_theta(
            geofac_div=self.interpolation_state.geofac_div,
            mass_fl_e=diagnostic_state_nh.mass_fl_e,
            z_theta_v_fl_e=self.z_theta_v_fl_e,
            z_flxdiv_mass=self.z_flxdiv_mass,
            z_flxdiv_theta=self.z_flxdiv_theta,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        if self.config.itime_scheme == TimeSteppingScheme.MOST_EFFICIENT:
            log.debug(f"corrector start stencil 42 44 45 45b")
            self._stencils_42_44_45_45b(
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
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels + 1,
                offset_provider={},
            )
        else:
            log.debug(f"corrector start stencil 43 44 45 45b")
            self._stencils_43_44_45_45b(
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
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels + 1,
                offset_provider={},
            )
        if not self.l_vert_nested:
            self._init_two_cell_kdim_fields_with_zero_wp(
                cell_kdim_field_with_zero_wp_1=prognostic_state[nnew].w,
                cell_kdim_field_with_zero_wp_2=z_fields.z_contr_w_fl_l,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=0,
                vertical_end=0,
                offset_provider={},
            )

        log.debug(f"corrector start stencil 47 48 49")
        self._stencils_47_48_49(
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
            dtime=dtime,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        # TODO: this is not tested in green line so far
        if self.config.is_iau_active:
            log.debug(f"corrector start stencil 50")
            self._add_analysis_increments_from_data_assimilation(
                z_rho_expl=z_fields.z_rho_expl,
                z_exner_expl=z_fields.z_exner_expl,
                rho_incr=diagnostic_state_nh.rho_incr,
                exner_incr=diagnostic_state_nh.exner_incr,
                iau_wgt_dyn=self.config.iau_wgt_dyn,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
        log.debug(f"corrector start stencil 52")
        self._solve_tridiagonal_matrix_for_w_forward_sweep(
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
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )
        log.debug(f"corrector start stencil 53")
        self._solve_tridiagonal_matrix_for_w_back_substitution(
            z_q=z_fields.z_q,
            w=prognostic_state[nnew].w,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        if self.config.rayleigh_type == constants.RayleighType.KLEMP:
            log.debug(f"corrector start stencil 54")
            self._apply_rayleigh_damping_mechanism(
                z_raylfac=self.z_raylfac,
                w_1=prognostic_state[nnew].w_1,
                w=prognostic_state[nnew].w,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=1,
                vertical_end=gtx.int32(
                    self.vertical_params.end_index_of_damping_layer + 1
                ),  # +1 since Fortran includes boundaries
                offset_provider={},
            )
        log.debug(f"corrector start stencil 55")
        self._compute_results_for_thermodynamic_variables(
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
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=gtx.int32(self.jk_start),
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )

        if lprep_adv:
            if lclean_mflx:
                log.debug(f"corrector set prep_adv.mass_flx_ic to zero")
                self._init_two_cell_kdim_fields_with_zero_wp(
                    prep_adv.mass_flx_ic,
                    prep_adv.vol_flx_ic,
                    horizontal_start=self._start_cell_nudging,
                    horizontal_end=self._end_cell_local,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )
        log.debug(f"corrector start stencil 58")
        self._update_mass_volume_flux(
            z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
            rho_ic=diagnostic_state_nh.rho_ic,
            vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
            w=prognostic_state[nnew].w,
            mass_flx_ic=prep_adv.mass_flx_ic,
            vol_flx_ic=prep_adv.vol_flx_ic,
            r_nsubsteps=r_nsubsteps,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )
        if at_last_substep:
            self._update_dynamical_exner_time_increment(
                exner=prognostic_state[nnew].exner,
                ddt_exner_phy=diagnostic_state_nh.ddt_exner_phy,
                exner_dyn_incr=diagnostic_state_nh.exner_dyn_incr,
                ndyn_substeps_var=float(self.config.ndyn_substeps_var),
                dtime=dtime,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=self.vertical_params.kstart_moist,
                vertical_end=gtx.int32(self.grid.num_levels),
                offset_provider={},
            )

        if lprep_adv:
            if lclean_mflx:
                log.debug(f"corrector set prep_adv.mass_flx_ic to zero")
                self._init_cell_kdim_field_with_zero_wp(
                    field_with_zero_wp=prep_adv.mass_flx_ic,
                    horizontal_start=self._start_cell_lateral_boundary,
                    horizontal_end=self._end_cell_nudging,
                    vertical_start=0,
                    vertical_end=self.grid.num_levels + 1,
                    offset_provider={},
                )
            log.debug(f" corrector: start stencil 65")
            self._update_mass_flux_weighted(
                rho_ic=diagnostic_state_nh.rho_ic,
                vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
                vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
                w_now=prognostic_state[nnow].w,
                w_new=prognostic_state[nnew].w,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                mass_flx_ic=prep_adv.mass_flx_ic,
                r_nsubsteps=r_nsubsteps,
                horizontal_start=self._start_cell_lateral_boundary,
                horizontal_end=self._end_cell_nudging,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )
            log.debug("exchange prognostic fields 'rho' , 'exner', 'w'")
            self._exchange.exchange_and_wait(
                dims.CellDim,
                prognostic_state[nnew].rho,
                prognostic_state[nnew].exner,
                prognostic_state[nnew].w,
            )
