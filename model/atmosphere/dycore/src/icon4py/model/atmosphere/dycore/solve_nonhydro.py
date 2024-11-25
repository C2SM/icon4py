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

import icon4py.model.atmosphere.dycore.solve_nonhydro_stencils as nhsolve_stencils
import icon4py.model.common.grid.states as grid_states
from gt4py.next import backend
from icon4py.model.common import constants
from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_wp import (
    init_cell_kdim_field_with_zero_wp,
)

from icon4py.model.atmosphere.dycore.stencils.accumulate_prep_adv_fields import (
    accumulate_prep_adv_fields,
)
from icon4py.model.atmosphere.dycore.stencils.add_analysis_increments_from_data_assimilation import (
    add_analysis_increments_from_data_assimilation,
)
from icon4py.model.atmosphere.dycore.stencils.add_analysis_increments_to_vn import (
    add_analysis_increments_to_vn,
)
from icon4py.model.atmosphere.dycore.stencils.add_temporal_tendencies_to_vn import (
    add_temporal_tendencies_to_vn,
)
from icon4py.model.atmosphere.dycore.stencils.add_temporal_tendencies_to_vn_by_interpolating_between_time_levels import (
    add_temporal_tendencies_to_vn_by_interpolating_between_time_levels,
)
from icon4py.model.atmosphere.dycore.stencils.add_vertical_wind_derivative_to_divergence_damping import (
    add_vertical_wind_derivative_to_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.apply_2nd_order_divergence_damping import (
    apply_2nd_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.apply_4th_order_divergence_damping import (
    apply_4th_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure import (
    apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure,
)
from icon4py.model.atmosphere.dycore.stencils.apply_rayleigh_damping_mechanism import (
    apply_rayleigh_damping_mechanism,
)
from icon4py.model.atmosphere.dycore.stencils.apply_weighted_2nd_and_4th_order_divergence_damping import (
    apply_weighted_2nd_and_4th_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.compute_approx_of_2nd_vertical_derivative_of_exner import (
    compute_approx_of_2nd_vertical_derivative_of_exner,
)
from icon4py.model.atmosphere.dycore.stencils.compute_avg_vn import compute_avg_vn
from icon4py.model.atmosphere.dycore.stencils.compute_avg_vn_and_graddiv_vn_and_vt import (
    compute_avg_vn_and_graddiv_vn_and_vt,
)
from icon4py.model.atmosphere.dycore.stencils.compute_divergence_of_fluxes_of_rho_and_theta import (
    compute_divergence_of_fluxes_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.stencils.compute_dwdz_for_divergence_damping import (
    compute_dwdz_for_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.compute_exner_from_rhotheta import (
    compute_exner_from_rhotheta,
)
from icon4py.model.atmosphere.dycore.stencils.compute_graddiv2_of_vn import (
    compute_graddiv2_of_vn,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates import (
    compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates import (
    compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_gradient_of_exner_pressure_for_multiple_levels import (
    compute_horizontal_gradient_of_exner_pressure_for_multiple_levels,
)
from icon4py.model.atmosphere.dycore.stencils.compute_hydrostatic_correction_term import (
    compute_hydrostatic_correction_term,
)
from icon4py.model.atmosphere.dycore.stencils.compute_mass_flux import compute_mass_flux
from icon4py.model.atmosphere.dycore.stencils.compute_perturbation_of_rho_and_theta import (
    compute_perturbation_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.stencils.compute_results_for_thermodynamic_variables import (
    compute_results_for_thermodynamic_variables,
)
from icon4py.model.atmosphere.dycore.stencils.compute_rho_virtual_potential_temperatures_and_pressure_gradient import (
    compute_rho_virtual_potential_temperatures_and_pressure_gradient,
)
from icon4py.model.atmosphere.dycore.stencils.compute_theta_and_exner import (
    compute_theta_and_exner,
)
from icon4py.model.atmosphere.dycore.stencils.compute_vn_on_lateral_boundary import (
    compute_vn_on_lateral_boundary,
)
from icon4py.model.atmosphere.dycore.stencils.copy_cell_kdim_field_to_vp import (
    copy_cell_kdim_field_to_vp,
)
from icon4py.model.atmosphere.dycore.stencils.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.atmosphere.dycore.stencils.mo_math_gradients_grad_green_gauss_cell_dsl import (
    mo_math_gradients_grad_green_gauss_cell_dsl,
)
from icon4py.model.atmosphere.dycore.stencils.init_two_cell_kdim_fields_with_zero_vp import (
    init_two_cell_kdim_fields_with_zero_vp,
)
from icon4py.model.atmosphere.dycore.stencils.init_two_cell_kdim_fields_with_zero_wp import (
    init_two_cell_kdim_fields_with_zero_wp,
)
from icon4py.model.atmosphere.dycore.stencils.init_two_edge_kdim_fields_with_zero_wp import (
    init_two_edge_kdim_fields_with_zero_wp,
)
from icon4py.model.atmosphere.dycore.stencils.solve_tridiagonal_matrix_for_w_back_substitution import (
    solve_tridiagonal_matrix_for_w_back_substitution,
)
from icon4py.model.atmosphere.dycore.stencils.solve_tridiagonal_matrix_for_w_forward_sweep import (
    solve_tridiagonal_matrix_for_w_forward_sweep,
)
from icon4py.model.atmosphere.dycore import (
    dycore_states,
    dycore_utils,
)
from icon4py.model.atmosphere.dycore.stencils.update_dynamical_exner_time_increment import (
    update_dynamical_exner_time_increment,
)
from icon4py.model.atmosphere.dycore.stencils.update_mass_volume_flux import (
    update_mass_volume_flux,
)
from icon4py.model.atmosphere.dycore.stencils.update_mass_flux_weighted import (
    update_mass_flux_weighted,
)
from icon4py.model.atmosphere.dycore.stencils.update_theta_v import update_theta_v
from icon4py.model.atmosphere.dycore.velocity_advection import (
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
        grid: icon_grid.IconGrid,
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        metric_state_nonhydro: dycore_states.MetricStateNonHydro,
        interpolation_state: dycore_states.InterpolationState,
        vertical_params: v_grid.VerticalGrid,
        edge_geometry: grid_states.EdgeParams,
        cell_geometry: grid_states.CellParams,
        owner_mask: fa.CellField[bool],
        backend: backend.Backend,
        exchange: decomposition.ExchangeRuntime = decomposition.SingleNodeExchange(),
    ):
        self._exchange = exchange
        self._backend = backend

        self._grid = grid
        self._config = config
        self._params = params
        self._metric_state_nonhydro = metric_state_nonhydro
        self._interpolation_state = interpolation_state
        self._vertical_params = vertical_params
        self._edge_geometry = edge_geometry
        self._cell_params = cell_geometry

        self.enh_divdamp_fac: Optional[fa.KField[float]] = None
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
        self._compute_z_raylfac = dycore_utils.compute_z_raylfac.with_backend(self._backend)
        self._predictor_stencils_2_3 = nhsolve_stencils.predictor_stencils_2_3.with_backend(
            self._backend
        )
        self._predictor_stencils_4_5_6 = nhsolve_stencils.predictor_stencils_4_5_6.with_backend(
            self._backend
        )
        self._compute_pressure_gradient_and_perturbed_rho_and_potential_temperatures = nhsolve_stencils.compute_pressure_gradient_and_perturbed_rho_and_potential_temperatures.with_backend(
            self._backend
        )
        self._predictor_stencils_11_lower_upper = (
            nhsolve_stencils.predictor_stencils_11_lower_upper.with_backend(self._backend)
        )
        self._compute_horizontal_advection_of_rho_and_theta = (
            nhsolve_stencils.compute_horizontal_advection_of_rho_and_theta.with_backend(
                self._backend
            )
        )
        self._predictor_stencils_35_36 = nhsolve_stencils.predictor_stencils_35_36.with_backend(
            self._backend
        )
        self._predictor_stencils_37_38 = nhsolve_stencils.predictor_stencils_37_38.with_backend(
            self._backend
        )
        self._stencils_39_40 = nhsolve_stencils.stencils_39_40.with_backend(self._backend)
        self._stencils_43_44_45_45b = nhsolve_stencils.stencils_43_44_45_45b.with_backend(
            self._backend
        )
        self._stencils_47_48_49 = nhsolve_stencils.stencils_47_48_49.with_backend(self._backend)
        self._stencils_61_62 = nhsolve_stencils.stencils_61_62.with_backend(self._backend)
        self._en_smag_fac_for_zero_nshift = smagorinsky.en_smag_fac_for_zero_nshift.with_backend(
            self._backend
        )
        self._init_test_fields = nhsolve_stencils.init_test_fields.with_backend(self._backend)
        self._stencils_42_44_45_45b = nhsolve_stencils.stencils_42_44_45_45b.with_backend(
            self._backend
        )

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
        self.l_vert_nested: bool = False
        if grid.lvert_nest:
            self.l_vert_nested = True
            self.jk_start = 1
        else:
            self.jk_start = 0

        self._en_smag_fac_for_zero_nshift(
            self._vertical_params.interface_physical_height,
            self._config.divdamp_fac,
            self._config.divdamp_fac2,
            self._config.divdamp_fac3,
            self._config.divdamp_fac4,
            self._config.divdamp_z,
            self._config.divdamp_z2,
            self._config.divdamp_z3,
            self._config.divdamp_z4,
            self.enh_divdamp_fac,
            offset_provider={"Koff": dims.KDim},
        )

        self.p_test_run = True

    def _allocate_local_fields(self):
        self.z_exner_ex_pr = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, is_halfdim=True, grid=self._grid, backend=self._backend
        )
        self.z_exner_ic = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, is_halfdim=True, grid=self._grid, backend=self._backend
        )
        self.z_dexner_dz_c_1 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_theta_v_pr_ic = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, is_halfdim=True, grid=self._grid, backend=self._backend
        )
        self.z_th_ddz_exner_c = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_rth_pr_1 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_rth_pr_2 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_grad_rth_1 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_grad_rth_2 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_grad_rth_3 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_grad_rth_4 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_dexner_dz_c_2 = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_hydro_corr = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_vn_avg = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_theta_v_fl_e = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_flxdiv_mass = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_flxdiv_theta = field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_rho_v = field_alloc.allocate_zero_field(
            dims.VertexDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_theta_v_v = field_alloc.allocate_zero_field(
            dims.VertexDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_graddiv2_vn = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.k_field = field_alloc.allocate_indices(
            dims.KDim, grid=self._grid, backend=self._backend, is_halfdim=True
        )
        self.z_w_concorr_me = field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=self._grid, backend=self._backend
        )
        self.z_hydro_corr_horizontal = field_alloc.allocate_zero_field(
            dims.EdgeDim, grid=self._grid, backend=self._backend
        )
        self.z_raylfac = field_alloc.allocate_zero_field(
            dims.KDim, grid=self._grid, backend=self._backend
        )
        self.enh_divdamp_fac = field_alloc.allocate_zero_field(
            dims.KDim, grid=self._grid, backend=self._backend
        )
        self._bdy_divdamp = field_alloc.allocate_zero_field(
            dims.KDim, grid=self._grid, backend=self._backend
        )
        self.scal_divdamp = field_alloc.allocate_zero_field(
            dims.KDim, grid=self._grid, backend=self._backend
        )
        self.intermediate_fields = IntermediateFields.allocate(
            grid=self._grid, backend=self._backend
        )

    def _determine_local_domains(self):
        vertex_domain = h_grid.domain(dims.VertexDim)
        cell_domain = h_grid.domain(dims.CellDim)
        cell_halo_level_2 = cell_domain(h_grid.Zone.HALO_LEVEL_2)
        edge_domain = h_grid.domain(dims.EdgeDim)
        edge_halo_level_2 = edge_domain(h_grid.Zone.HALO_LEVEL_2)

        self._start_cell_lateral_boundary = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY)
        )
        self._start_cell_lateral_boundary_level_3 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._start_cell_local = self._grid.start_index(cell_domain(h_grid.Zone.LOCAL))
        self._start_cell_halo = self._grid.start_index(cell_domain(h_grid.Zone.HALO))
        self._start_cell_halo_level_2 = self._grid.start_index(cell_halo_level_2)
        self._end_cell_lateral_boundary_level_4 = self._grid.end_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        self._end_cell_nudging = self._grid.end_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._end_cell_halo = self._grid.end_index(cell_domain(h_grid.Zone.HALO))
        self._end_cell_halo_level_2 = self._grid.end_index(cell_halo_level_2)
        self._end_cell_end = self._grid.end_index(cell_domain(h_grid.Zone.END))

        self._start_edge_lateral_boundary = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY)
        )
        self._start_edge_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._start_edge_lateral_boundary_level_7 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
        )
        self._start_edge_nudging_level_2 = self._grid.start_index(
            edge_domain(h_grid.Zone.NUDGING_LEVEL_2)
        )

        self._start_edge_halo_level_2 = self._grid.start_index(edge_halo_level_2)

        self._end_cell_lateral_boundary_level_4 = self._grid.end_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        self._end_edge_nudging = self._grid.end_index(edge_domain(h_grid.Zone.NUDGING))
        self._end_edge_local = self._grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        self._end_edge_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))
        self._end_edge_halo_level_2 = self._grid.end_index(edge_halo_level_2)
        self._end_edge_end = self._grid.end_index(edge_domain(h_grid.Zone.END))

        self._start_vertex_lateral_boundary_level_2 = self._grid.start_index(
            vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._end_vertex_halo = self._grid.end_index(vertex_domain(h_grid.Zone.HALO))

    def set_timelevels(self, nnow, nnew):
        #  Set time levels of ddt_adv fields for call to velocity_tendencies
        if self._config.itime_scheme == TimeSteppingScheme.MOST_EFFICIENT:
            self.ntl1 = nnow
            self.ntl2 = nnew
        else:
            self.ntl1 = 0
            self.ntl2 = 0

    def time_step(
        self,
        diagnostic_state_nh: dycore_states.DiagnosticStateNonHydro,
        prognostic_state_ls: list[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
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
                vertical_end=self._grid.num_levels,
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

        if self._grid.limited_area:
            """
            theta_v (0:nlev-1):
                Update virtual temperature at full levels (cell center) at only halo cells in the boundary interpolation zone by equating it to exner.
            exner (0:nlev-1):
                Update exner function at full levels (cell center) at only halo cells in the boundary interpolation zone using the equation of state (see eq.3.9 in ICON tutorial 2023).
                exner = (rd * rho * exner / p0ref) ^ (rd / cvd)
            """
            self._compute_theta_and_exner(
                bdy_halo_c=self._metric_state_nonhydro.bdy_halo_c,
                rho=prognostic_state_ls[nnew].rho,
                theta_v=prognostic_state_ls[nnew].theta_v,
                exner=prognostic_state_ls[nnew].exner,
                rd_o_cvd=self._params.rd_o_cvd,
                rd_o_p0ref=self._params.rd_o_p0ref,
                horizontal_start=self._start_cell_local,
                horizontal_end=self._end_cell_end,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider={},
            )

            """
            theta_v (0:nlev-1):
                Update virtual temperature at full levels (cell center) at only halo cells in the boundary interpolation zone by equating it to exner.
            exner (0:nlev-1):
                Update exner function at full levels (cell center) at only halo cells in the boundary interpolation zone using the equation of state (see eq.3.9 in ICON tutorial 2023).
                exner = (rd * rho * exner / p0ref) ^ (rd / cvd)
            """
            self._compute_exner_from_rhotheta(
                rho=prognostic_state_ls[nnew].rho,
                theta_v=prognostic_state_ls[nnew].theta_v,
                exner=prognostic_state_ls[nnew].exner,
                rd_o_cvd=self._params.rd_o_cvd,
                rd_o_p0ref=self._params.rd_o_p0ref,
                horizontal_start=self._start_cell_lateral_boundary,
                horizontal_end=self._end_cell_lateral_boundary_level_4,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider={},
            )

        """
        theta_v (0:nlev-1):
            Update virtual temperature at full levels (cell center) at only halo cells in the boundary interpolation zone from the equation of state (see eqs. 3.22 and 3.23 in ICON tutorial 2023).
            rho^{n+1} theta_v^{n+1} = rho^{n} theta_v^{n} + ( cvd * rho^{n} * theta_v^{n} ) / ( rd * pi^{n} ) ( pi^{n+1} - pi^{n} )
        """
        self._update_theta_v(
            mask_prog_halo_c=self._metric_state_nonhydro.mask_prog_halo_c,
            rho_now=prognostic_state_ls[nnow].rho,
            theta_v_now=prognostic_state_ls[nnow].theta_v,
            exner_new=prognostic_state_ls[nnew].exner,
            exner_now=prognostic_state_ls[nnow].exner,
            rho_new=prognostic_state_ls[nnew].rho,
            theta_v_new=prognostic_state_ls[nnew].theta_v,
            cvd_o_rd=self._params.cvd_o_rd,
            horizontal_start=self._start_cell_halo,
            horizontal_end=self._end_cell_end,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider={},
        )

    # flake8: noqa: C901
    def run_predictor_step(
        self,
        diagnostic_state_nh: dycore_states.DiagnosticStateNonHydro,
        prognostic_state: list[prognostics.PrognosticState],
        z_fields: IntermediateFields,
        dtime: float,
        l_recompute: bool,
        l_init: bool,
        at_first_substep: bool,
        nnow: int,
        nnew: int,
    ):
        """
        Runs the predictor step of the non-hydrostatic solver.
        """

        log.info(
            f"running predictor step: dtime = {dtime}, init = {l_init}, recompute = {l_recompute} "
        )
        if l_init or l_recompute:
            if self._config.itime_scheme == TimeSteppingScheme.MOST_EFFICIENT and not l_init:
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
                cell_areas=self._cell_params.area,
            )

        #  Precompute Rayleigh damping factor
        self._compute_z_raylfac(
            rayleigh_w=self._metric_state_nonhydro.rayleigh_w,
            dtime=dtime,
            z_raylfac=self.z_raylfac,
            offset_provider={},
        )

        # initialize nest boundary points of z_rth_pr with zero
        if self._grid.limited_area:
            self._init_two_cell_kdim_fields_with_zero_vp(
                cell_kdim_field_with_zero_vp_1=self.z_rth_pr_1,
                cell_kdim_field_with_zero_vp_2=self.z_rth_pr_2,
                horizontal_start=self._start_cell_lateral_boundary,
                horizontal_end=self._end_cell_end,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider={},
            )

        # scidoc:
        # Outputs:
        #  - z_exner_ex_pr :
        #     $$
        #     \exnerprime{\ntilde}{\c}{\k} = (1 + \WtimeExner) \exnerprime{\n}{\c}{\k} - \WtimeExner \exnerprime{\n-1}{\c}{\k}, \k \in [0, \nlev) \\
        #     \exnerprime{\ntilde}{\c}{\nlev} = 0
        #     $$
        #     Compute the temporal extrapolation of perturbed exner function
        #     using the time backward scheme (see the |ICONTutorial| page 74).
        #     This variable has nlev+1 levels even though it is defined on full levels.
        #  - exner_pr :
        #     $$
        #     \exnerprime{\n-1}{\c}{\k} = \exnerprime{\ntilde}{\c}{\k}
        #     $$
        #     Store the perturbed exner function from the previous time step.
        #
        # Inputs:
        #  - $\WtimeExner$ : exner_exfac
        #  - $\exnerprime{\n}{\c}{\k}$ : exner - exner_ref_mc
        #  - $\exnerprime{\n-1}{\c}{\k}$ : exner_pr
        #
        self._predictor_stencils_2_3(
            exner_exfac=self._metric_state_nonhydro.exner_exfac,
            exner=prognostic_state[nnow].exner,
            exner_ref_mc=self._metric_state_nonhydro.exner_ref_mc,
            exner_pr=diagnostic_state_nh.exner_pr,
            z_exner_ex_pr=self.z_exner_ex_pr,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self._grid.num_levels + 1,
            offset_provider={},
        )

        if self._config.igradp_method == HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
            # scidoc:
            # Outputs:
            #  - z_exner_ic :
            #     $$
            #     \exnerprime{\ntilde}{\c}{\k-1/2} = \Wlev \exnerprime{\ntilde}{\c}{\k} + (1 - \Wlev) \exnerprime{\ntilde}{\c}{\k-1}, \quad \k \in [\max(1,\nflatlev), \nlev) \\
            #     \exnerprime{\ntilde}{\c}{\nlev-1/2} = \sum_{\k=\nlev-1}^{\nlev-3} \Wlev_{\k} \exnerprime{\ntilde}{\c}{\k}
            #     $$
            #     Interpolate the perturbation exner from full to half levels.
            #     The ground level is based on quadratic extrapolation (with
            #     hydrostatic assumption?).
            #  - z_dexner_dz_c_1 :
            #     $$
            #     \pdz{\exnerprime{\ntilde}{\c}{\k}} \approx \frac{\exnerprime{\ntilde}{\c}{\k-1/2} - \exnerprime{\ntilde}{\c}{\k+1/2}}{\Dz{\k}}, \quad \k \in [\max(1,\nflatlev), \nlev]
            #     $$
            #     Use the interpolated values to compute the vertical derivative
            #     of perturbation exner at full levels.
            #
            # Inputs:
            #  - $\Wlev$ : wgtfac_c
            #  - $\Wlev_{\k}$ : wgtfacq_c
            #  - $\exnerprime{\ntilde}{\c}{\k}$ : z_exner_ex_pr
            #  - $\exnerprime{\ntilde}{\c}{\k\pm1/2}$ : z_exner_ic
            #  - $1 / \Dz{\k}$ : inv_ddqz_z_full
            #
            self._predictor_stencils_4_5_6(
                wgtfacq_c_dsl=self._metric_state_nonhydro.wgtfacq_c,
                z_exner_ex_pr=self.z_exner_ex_pr,
                z_exner_ic=self.z_exner_ic,
                wgtfac_c=self._metric_state_nonhydro.wgtfac_c,
                inv_ddqz_z_full=self._metric_state_nonhydro.inv_ddqz_z_full,
                z_dexner_dz_c_1=self.z_dexner_dz_c_1,
                horizontal_start=self._start_cell_lateral_boundary_level_3,
                horizontal_end=self._end_cell_halo,
                vertical_start=max(1, self._vertical_params.nflatlev),
                vertical_end=self._grid.num_levels + 1,
                offset_provider=self._grid.offset_providers,
            )

            if self._vertical_params.nflatlev == 1:
                # Perturbation Exner pressure on top half level
                raise NotImplementedError("nflatlev=1 not implemented")

        """
        rho_ic & theta_v_ic (1:nlev-1):
            Compute rho and virtual temperature at half levels. rho and virtual
            temperature at model top boundary and ground are not updated.
        z_rth_pr_1 (0:nlev-1):
            Compute perturbed rho at full levels (cell center).
        z_rth_pr_2 (0:nlev-1):
            Compute perturbed virtual temperature at full levels (cell center).

        $$
        \rho_{k}^{\prime^\tilde{n}} = \hat{\rho_{k}^{\tilde{n}}} - \rho_{0\ k} \\
        $$

        z_theta_v_pr_ic (1:nlev-1):
            Compute the perturbed virtual temperature from z_rth_pr_2 at half levels.

        $$
        \theta_{v\ k-1/2}^{\prime \tilde{n}} = \nu \pi_k^{\prime\tilde{n}} + (1 - \nu) \pi_{k-1}^{\prime\tilde{n}} \\
        \nu = \text{wgtfa_c}
        $$

        z_th_ddz_exner_c (1:nlev-1):
            (see eq. 3.19 in icon tutorial 2023) at half levels (cell center) is
            also computed. Its value at the model top is not updated. No ground
            value.
            $$
            theta_v' dpi_0/dz + eta_expl theta_v dpi'/dz
            $$
            dpi_0/dz is d_exner_dz_ref_ic.
            $$
            eta_impl = 0.5 + vwind_offctr = vwind_impl_wgt
            eta_expl = 1.0 - eta_impl = vwind_expl_wgt
            $$
        """
        self._compute_pressure_gradient_and_perturbed_rho_and_potential_temperatures(
            rho=prognostic_state[nnow].rho,
            rho_ref_mc=self._metric_state_nonhydro.rho_ref_mc,
            theta_v=prognostic_state[nnow].theta_v,
            theta_ref_mc=self._metric_state_nonhydro.theta_ref_mc,
            rho_ic=diagnostic_state_nh.rho_ic,
            z_rth_pr_1=self.z_rth_pr_1,
            z_rth_pr_2=self.z_rth_pr_2,
            wgtfac_c=self._metric_state_nonhydro.wgtfac_c,
            vwind_expl_wgt=self._metric_state_nonhydro.vwind_expl_wgt,
            exner_pr=diagnostic_state_nh.exner_pr,
            d_exner_dz_ref_ic=self._metric_state_nonhydro.d_exner_dz_ref_ic,
            ddqz_z_half=self._metric_state_nonhydro.ddqz_z_half,
            z_theta_v_pr_ic=self.z_theta_v_pr_ic,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            z_th_ddz_exner_c=self.z_th_ddz_exner_c,
            k_field=self.k_field,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )

        """
        z_theta_v_pr_ic (0, nlev):
            Perturbed theta_v at half level at the model top is set to zero.
            Perturbed theta_v at half level at the ground level is computed by quadratic interpolation (hydrostatic assumption?) in the same way as z_exner_ic.
        theta_v_ic (nlev):
            virtual temperature at half level at the ground level is computed by adding theta_ref_ic to z_theta_v_pr_ic.
        """
        # Perturbation theta at top and surface levels
        self._predictor_stencils_11_lower_upper(
            wgtfacq_c_dsl=self._metric_state_nonhydro.wgtfacq_c,
            z_rth_pr=self.z_rth_pr_2,
            theta_ref_ic=self._metric_state_nonhydro.theta_ref_ic,
            z_theta_v_pr_ic=self.z_theta_v_pr_ic,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            k_field=self.k_field,
            nlev=self._grid.num_levels,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self._grid.num_levels + 1,
            offset_provider=self._grid.offset_providers,
        )

        if self._config.igradp_method == HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
            # scidoc:
            # Outputs:
            #  - z_dexner_dz_c_2 :
            #     $$
            #     \frac{1}{2}\pdzz{\exnerprime{\ntilde}{\c}{\k}} = \frac{1}{2} \left( \pdz{\vpotempprime{\n}{\c}{\k}} \vpotempref{\c}{\k} \ddz{\presref{\c}{\k}} - \vpotempprime{\n}{\c}{\k} \ddz{\frac{1}{\vpotempref{\c}{\k}} \ddz{\presref{\c}{\k}}} \right), \quad \k \in [\nflatgradp, \nlev) \\
            #     \ddz{\presref{\c}{\k}} = -\frac{g \cpd}{\vpotempref{\c}{\k}}
            #     $$
            #     Compute second vertical derivative of perturbed exner function.
            #     This second vertical derivative is approximated by hydrostatic
            #     approximation (see eqs. 13 and 7 in |ICONSteepSlopePressurePaper|).
            #     Note that, in $\ddz{\frac{1}{\vpotempref{\c}{\k}} \ddz{\presref{\c}{\k}}}$,
            #     it makes use of eq. 15 in |ICONSteepSlopePressurePaper| for the reference state
            #     of temperature when computing $\ddz{\vpotempref{\c}{\k}}$.
            #     The vertical derivative of perturbed virtual potential temperature
            #     on RHS is computed explicitly in this stencil by taking the
            #     difference between neighboring half levels (coefficient is included
            #     in d2dexdz2_fac1_mc).
            #     $\nflatgradp$ is the maximum height index at which the height of
            #     the center of an edge lies within two neighboring cells.
            #
            # Inputs:
            #  - $\vpotempprime{\n}{\c}{\k-1/2}$ : z_theta_v_pr_ic
            #  - $\frac{1}{dz \vpotempref{\c}{\k}} \ddz{\presref{\c}{\k}}$ : d2dexdz2_fac1_mc
            #  - $\ddz{\frac{1}{\vpotempref{\c}{\k}} \ddz{\presref{\c}{\k}}}$ : d2dexdz2_fac2_mc
            #  - $\vpotempprime{\n}{\c}{\k}$ : z_rth_pr_2
            #
            self._compute_approx_of_2nd_vertical_derivative_of_exner(
                z_theta_v_pr_ic=self.z_theta_v_pr_ic,
                d2dexdz2_fac1_mc=self._metric_state_nonhydro.d2dexdz2_fac1_mc,
                d2dexdz2_fac2_mc=self._metric_state_nonhydro.d2dexdz2_fac2_mc,
                z_rth_pr_2=self.z_rth_pr_2,
                z_dexner_dz_c_2=self.z_dexner_dz_c_2,
                horizontal_start=self._start_cell_lateral_boundary_level_3,
                horizontal_end=self._end_cell_halo,
                vertical_start=self._vertical_params.nflat_gradp,
                vertical_end=self._grid.num_levels,
                offset_provider=self._grid.offset_providers,
            )

        """
        z_rth_pr_1 (0:nlev-1):
            Compute perturbed rho at full levels (cell center), which is equal to rho - rho_ref_mc.
        z_rth_pr_2 (0:nlev-1):
            Compute perturbed virtual temperature at full levels (cell center), which is equal to theta_v - theta_ref_mc.
        """
        # Add computation of z_grad_rth (perturbation density and virtual potential temperature at main levels)
        # at outer halo points: needed for correct calculation of the upwind gradients for Miura scheme
        self._compute_perturbation_of_rho_and_theta(
            rho=prognostic_state[nnow].rho,
            rho_ref_mc=self._metric_state_nonhydro.rho_ref_mc,
            theta_v=prognostic_state[nnow].theta_v,
            theta_ref_mc=self._metric_state_nonhydro.theta_ref_mc,
            z_rth_pr_1=self.z_rth_pr_1,
            z_rth_pr_2=self.z_rth_pr_2,
            horizontal_start=self._start_cell_halo_level_2,
            horizontal_end=self._end_cell_halo_level_2,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider={},
        )

        # Compute rho and theta at edges for horizontal flux divergence term
        if self._config.iadv_rhotheta == RhoThetaAdvectionType.SIMPLE:
            """
            z_rho_v (0:nlev-1):
                Compute the density at cell vertices at full levels by simple area-weighted interpolation.
            """
            self._mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                p_cell_in=prognostic_state[nnow].rho,
                c_intp=self._interpolation_state.c_intp,
                p_vert_out=self.z_rho_v,
                horizontal_start=self._start_vertex_lateral_boundary_level_2,
                horizontal_end=self._end_vertex_halo,
                vertical_start=0,
                vertical_end=self._grid.num_levels,  # UBOUND(p_cell_in,2)
                offset_provider=self._grid.offset_providers,
            )
            """
            z_theta_v_v (0:nlev-1):
                Compute the virtual temperature at cell vertices at full levels by simple area-weighted interpolation.
            """
            self._mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                p_cell_in=prognostic_state[nnow].theta_v,
                c_intp=self._interpolation_state.c_intp,
                p_vert_out=self.z_theta_v_v,
                horizontal_start=self._start_vertex_lateral_boundary_level_2,
                horizontal_end=self._end_vertex_halo,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider=self._grid.offset_providers,
            )
        elif self._config.iadv_rhotheta == RhoThetaAdvectionType.MIURA:
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
            self._mo_math_gradients_grad_green_gauss_cell_dsl(
                p_grad_1_u=self.z_grad_rth_1,
                p_grad_1_v=self.z_grad_rth_2,
                p_grad_2_u=self.z_grad_rth_3,
                p_grad_2_v=self.z_grad_rth_4,
                p_ccpr1=self.z_rth_pr_1,
                p_ccpr2=self.z_rth_pr_2,
                geofac_grg_x=self._interpolation_state.geofac_grg_x,
                geofac_grg_y=self._interpolation_state.geofac_grg_y,
                horizontal_start=self._start_cell_lateral_boundary_level_3,
                horizontal_end=self._end_cell_halo,
                vertical_start=0,
                vertical_end=self._grid.num_levels,  # UBOUND(p_ccpr,2)
                offset_provider=self._grid.offset_providers,
            )
        if self._config.iadv_rhotheta <= 2:
            self._init_two_edge_kdim_fields_with_zero_wp(
                edge_kdim_field_with_zero_wp_1=z_fields.z_rho_e,
                edge_kdim_field_with_zero_wp_2=z_fields.z_theta_v_e,
                horizontal_start=self._start_edge_halo_level_2,
                horizontal_end=self._end_edge_halo_level_2,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider={},
            )
            # initialize also nest boundary points with zero
            if self._grid.limited_area:
                self._init_two_edge_kdim_fields_with_zero_wp(
                    edge_kdim_field_with_zero_wp_1=z_fields.z_rho_e,
                    edge_kdim_field_with_zero_wp_2=z_fields.z_theta_v_e,
                    horizontal_start=self._start_edge_lateral_boundary,
                    horizontal_end=self._end_edge_halo,
                    vertical_start=0,
                    vertical_end=self._grid.num_levels,
                    offset_provider={},
                )
            if self._config.iadv_rhotheta == RhoThetaAdvectionType.MIURA:
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
                self._compute_horizontal_advection_of_rho_and_theta(
                    p_vn=prognostic_state[nnow].vn,
                    p_vt=diagnostic_state_nh.vt,
                    pos_on_tplane_e_1=self._interpolation_state.pos_on_tplane_e_1,
                    pos_on_tplane_e_2=self._interpolation_state.pos_on_tplane_e_2,
                    primal_normal_cell_1=self._edge_geometry.primal_normal_cell[0],
                    dual_normal_cell_1=self._edge_geometry.dual_normal_cell[0],
                    primal_normal_cell_2=self._edge_geometry.primal_normal_cell[1],
                    dual_normal_cell_2=self._edge_geometry.dual_normal_cell[1],
                    p_dthalf=(0.5 * dtime),
                    rho_ref_me=self._metric_state_nonhydro.rho_ref_me,
                    theta_ref_me=self._metric_state_nonhydro.theta_ref_me,
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
                    vertical_end=self._grid.num_levels,
                    offset_provider=self._grid.offset_providers,
                )

        # scidoc:
        # Outputs:
        #  - z_gradh_exner :
        #     $$
        #     \exnergradh{\ntilde}{\e}{\k} = \Cgrad \Gradn_{\offProv{e2c}} \exnerprime{\ntilde}{\c}{\k}, \quad \k \in [0, \nflatlev)
        #     $$
        #     Compute the horizontal gradient of temporal extrapolation of
        #     perturbed exner function on flat levels, unaffected by the terrain
        #     following deformation.
        #
        # Inputs:
        #  - $\exnerprime{\ntilde}{\c}{\k}$ : z_exner_ex_pr
        #  - $\Cgrad$ : inv_dual_edge_length
        #
        self._compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates(
            inv_dual_edge_length=self._edge_geometry.inverse_dual_edge_lengths,
            z_exner_ex_pr=self.z_exner_ex_pr,
            z_gradh_exner=z_fields.z_gradh_exner,
            horizontal_start=self._start_edge_nudging_level_2,
            horizontal_end=self._end_edge_local,
            vertical_start=0,
            vertical_end=self._vertical_params.nflatlev,
            offset_provider=self._grid.offset_providers,
        )

        if self._config.igradp_method == HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
            """
            z_gradh_exner (flat_lev:flat_gradp):
                Compute the horizontal gradient (at constant height) of temporal extrapolation of perturbed exner function at full levels (edge center) by simple first order scheme.
                By coordinate transformation (x, y, z) <-> (x, y, eta), dpi/dn |z = dpi/dn |s + dh/dn |s dpi/dz
                dpi/dz is previously computed z_dexner_dz_c_1.
                dh/dn | s is ddxn_z_full, it is the horizontal gradient across neighboring cells at constant eta at full levels.
            """
            # horizontal gradient of Exner pressure, including metric correction
            # horizontal gradient of Exner pressure, Taylor-expansion-based reconstruction
            self._compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates(
                inv_dual_edge_length=self._edge_geometry.inverse_dual_edge_lengths,
                z_exner_ex_pr=self.z_exner_ex_pr,
                ddxn_z_full=self._metric_state_nonhydro.ddxn_z_full,
                c_lin_e=self._interpolation_state.c_lin_e,
                z_dexner_dz_c_1=self.z_dexner_dz_c_1,
                z_gradh_exner=z_fields.z_gradh_exner,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=self._vertical_params.nflatlev,
                vertical_end=gtx.int32(self._vertical_params.nflat_gradp + 1),
                offset_provider=self._grid.offset_providers,
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
            self._compute_horizontal_gradient_of_exner_pressure_for_multiple_levels(
                inv_dual_edge_length=self._edge_geometry.inverse_dual_edge_lengths,
                z_exner_ex_pr=self.z_exner_ex_pr,
                zdiff_gradp=self._metric_state_nonhydro.zdiff_gradp,
                ikoffset=self._metric_state_nonhydro.vertoffset_gradp,
                z_dexner_dz_c_1=self.z_dexner_dz_c_1,
                z_dexner_dz_c_2=self.z_dexner_dz_c_2,
                z_gradh_exner=z_fields.z_gradh_exner,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=gtx.int32(self._vertical_params.nflat_gradp + 1),
                vertical_end=self._grid.num_levels,
                offset_provider=self._grid.offset_providers,
            )
        # compute hydrostatically approximated correction term that replaces downward extrapolation
        if self._config.igradp_method == HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
            """
            z_hydro_corr (nlev-1):
                z_hydro_corr = g/(cpd theta_v^2) dtheta_v/dn (h_k - h_k*)
                Compute the hydrostatic correction term (see the last term in eq. 10 or 9 in Günther et al. 2012) at full levels (edge center).
                This is only computed for the last or bottom most level because all edge centers which have a neighboring cell center inside terrain
                beyond a certain limit (see last paragraph for discussion on page 3724) use the same correction term at k* level in eq. 10 in Günther
                et al. 2012.
                Note that the vertoffset_gradp and zdiff_gradp are recomputed for those special edges in mo_vertical_grid.f90.
            """
            self._compute_hydrostatic_correction_term(
                theta_v=prognostic_state[nnow].theta_v,
                ikoffset=self._metric_state_nonhydro.vertoffset_gradp,
                zdiff_gradp=self._metric_state_nonhydro.zdiff_gradp,
                theta_v_ic=diagnostic_state_nh.theta_v_ic,
                inv_ddqz_z_full=self._metric_state_nonhydro.inv_ddqz_z_full,
                inv_dual_edge_length=self._edge_geometry.inverse_dual_edge_lengths,
                z_hydro_corr=self.z_hydro_corr,
                grav_o_cpd=self._params.grav_o_cpd,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=self._grid.num_levels - 1,
                vertical_end=self._grid.num_levels,
                offset_provider=self._grid.offset_providers,
            )
        # TODO (Nikki) check when merging fused stencil
        lowest_level = self._grid.num_levels - 1
        hydro_corr_horizontal = gtx.as_field(
            (dims.EdgeDim,),
            self.z_hydro_corr.ndarray[:, lowest_level],
            allocator=self._backend.allocator,
        )

        if self._config.igradp_method == HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
            # scidoc:
            # Outputs:
            #  - z_gradh_exner :
            #     $$
            #     \exnergradh{\ntilde}{\e}{\k} = \exnergradh{\ntilde}{\e}{\k} + \exnhydrocorr{\e} (h_k - h_{k^*}), \quad \e \in \IDXpg \\
            #     $$
            #     Apply the hydrostatic correction term to the horizontal
            #     gradient (at constant height) of the temporal extrapolation of
            #     perturbed exner function.
            #     This is only applied to edges for which the adjacent cell
            #     center (horizontally, not terrain-following) would be
            #     underground, i.e. edges in the $\IDXpg$ set.
            #
            # Inputs:
            #  - $\exnergradh{\ntilde}{\e}{\k}$ : z_gradh_exner
            #  - $\exnhydrocorr{\e}$ : hydro_corr_horizontal
            #  - $(h_k - h_{k^*})$ : pg_exdist
            #  - $\IDXpg$ : ipeidx_dsl
            #
            self._apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure(
                ipeidx_dsl=self._metric_state_nonhydro.ipeidx_dsl,
                pg_exdist=self._metric_state_nonhydro.pg_exdist,
                z_hydro_corr=hydro_corr_horizontal,
                z_gradh_exner=z_fields.z_gradh_exner,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_end,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider={},
            )

        # scidoc:
        # Outputs:
        #  - vn :
        #     $$
        #     \vn{\n+1^*}{\e}{\k} = \vn{\n}{\e}{\k} - \Dt \left( \advvn{\n}{\e}{\k} + \cpd \vpotemp{\n}{\e}{\k} \exnergradh{\ntilde}{\e}{\k} \right)
        #     $$
        #     Update the normal wind speed with the advection and pressure
        #     gradient terms.
        # 
        # Inputs:
        #  - $\vn{\n}{\e}{\k}$ : vn
        #  - $\Dt$ : dtime
        #  - $\advvn{\n}{\e}{\k}$ : ddt_vn_apc_pc[self.ntl1]
        #  - $\vpotemp{\n}{\e}{\k}$ : z_theta_v_e
        #  - $\pdxn{\exnerprime{\ntilde}{}{}}_{\e\,\k}$ : z_gradh_exner
        #  - $\cpd$ : CPD
        # 
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
            vertical_end=self._grid.num_levels,
            offset_provider={},
        )

        if self._config.is_iau_active:
            self._add_analysis_increments_to_vn(
                vn_incr=diagnostic_state_nh.vn_incr,
                vn=prognostic_state[nnew].vn,
                iau_wgt_dyn=self._config.iau_wgt_dyn,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider={},
            )

        if self._grid.limited_area:
            """
            vn (0:nlev-1):
                Add boundary velocity tendendy to the normal velocity.
                vn = vn + dt * grf_tend_vn
            """
            self._compute_vn_on_lateral_boundary(
                grf_tend_vn=diagnostic_state_nh.grf_tend_vn,
                vn_now=prognostic_state[nnow].vn,
                vn_new=prognostic_state[nnew].vn,
                dtime=dtime,
                horizontal_start=self._start_edge_lateral_boundary,
                horizontal_end=self._end_edge_nudging,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider={},
            )
        log.debug("exchanging prognostic field 'vn' and local field 'z_rho_e'")
        self._exchange.exchange_and_wait(dims.EdgeDim, prognostic_state[nnew].vn, z_fields.z_rho_e)

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
        self._compute_avg_vn_and_graddiv_vn_and_vt(
            e_flx_avg=self._interpolation_state.e_flx_avg,
            vn=prognostic_state[nnew].vn,
            geofac_grdiv=self._interpolation_state.geofac_grdiv,
            rbf_vec_coeff_e=self._interpolation_state.rbf_vec_coeff_e,
            z_vn_avg=self.z_vn_avg,
            z_graddiv_vn=z_fields.z_graddiv_vn,
            vt=diagnostic_state_nh.vt,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )

        """
        mass_fl_e (0:nlev-1):
            Compute the mass flux at full levels (edge center) by multiplying density with averaged normal velocity (z_vn_avg) computed above.
        z_theta_v_fl_e (0:nlev-1):
            Compute the energy (theta_v * mass) flux by multiplying density with averaged normal velocity (z_vn_avg) computed above.
        """
        self._compute_mass_flux(
            z_rho_e=z_fields.z_rho_e,
            z_vn_avg=self.z_vn_avg,
            ddqz_z_full_e=self._metric_state_nonhydro.ddqz_z_full_e,
            z_theta_v_e=z_fields.z_theta_v_e,
            mass_fl_e=diagnostic_state_nh.mass_fl_e,
            z_theta_v_fl_e=self.z_theta_v_fl_e,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
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
        self._predictor_stencils_35_36(
            vn=prognostic_state[nnew].vn,
            ddxn_z_full=self._metric_state_nonhydro.ddxn_z_full,
            ddxt_z_full=self._metric_state_nonhydro.ddxt_z_full,
            vt=diagnostic_state_nh.vt,
            z_w_concorr_me=self.z_w_concorr_me,
            wgtfac_e=self._metric_state_nonhydro.wgtfac_e,
            vn_ie=diagnostic_state_nh.vn_ie,
            z_vt_ie=z_fields.z_vt_ie,
            z_kin_hor_e=z_fields.z_kin_hor_e,
            k_field=self.k_field,
            nflatlev_startindex=self._vertical_params.nflatlev,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
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
                #---------------  z4
                #       z3'
                #---------------  z3
                #       z2'
                #---------------  z2
                #       z1'
                #---------------  z1 (surface)
                #///////////////
                The three reference points for extrapolation are at z2, z2', and z3'. Value at z1 is
                then obtained by quadratic interpolation polynomial based on these three points.
            """
            self._predictor_stencils_37_38(
                vn=prognostic_state[nnew].vn,
                vt=diagnostic_state_nh.vt,
                vn_ie=diagnostic_state_nh.vn_ie,
                z_vt_ie=z_fields.z_vt_ie,
                z_kin_hor_e=z_fields.z_kin_hor_e,
                wgtfacq_e_dsl=self._metric_state_nonhydro.wgtfacq_e,
                horizontal_start=self._start_edge_lateral_boundary_level_5,
                horizontal_end=self._end_edge_halo_level_2,
                vertical_start=0,
                vertical_end=self._grid.num_levels + 1,
                offset_provider=self._grid.offset_providers,
            )

        """
        w_concorr_c (flat_lev+1:nlev-1):
            Interpolate contravariant correction at edge center from full levels, which is
            z_w_concorr_me computed above, to half levels using simple linear interpolation.
        w_concorr_c (nlev):
            Compute contravariant correction at ground level (cell center) by quadratic extrapolation. z_w_concorr_me needs to be first
            linearly interpolated to cell center.
            #---------------  z4
            #       z3'
            #---------------  z3
            #       z2'
            #---------------  z2
            #       z1'
            #---------------  z1 (surface)
            #///////////////
            The three reference points for extrapolation are at z2, z2', and z3'. Value at z1 is
            then obtained by quadratic interpolation polynomial based on these three points.
        """
        self._stencils_39_40(
            e_bln_c_s=self._interpolation_state.e_bln_c_s,
            z_w_concorr_me=self.z_w_concorr_me,
            wgtfac_c=self._metric_state_nonhydro.wgtfac_c,
            wgtfacq_c_dsl=self._metric_state_nonhydro.wgtfacq_c,
            w_concorr_c=diagnostic_state_nh.w_concorr_c,
            k_field=self.k_field,
            nflatlev_startindex_plus1=gtx.int32(self._vertical_params.nflatlev + 1),
            nlev=self._grid.num_levels,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_halo,
            vertical_start=0,
            vertical_end=self._grid.num_levels + 1,
            offset_provider=self._grid.offset_providers,
        )

        """
        z_flxdiv_mass (0:nlev-1):
            Compute the divergence of mass flux at full levels (cell center) by Gauss theorem.
        z_flxdiv_theta (0:nlev-1):
            Compute the divergence of energy (theta_v * mass) flux at full levels (cell center) by Gauss theorem.
        """
        self._compute_divergence_of_fluxes_of_rho_and_theta(
            geofac_div=self._interpolation_state.geofac_div,
            mass_fl_e=diagnostic_state_nh.mass_fl_e,
            z_theta_v_fl_e=self.z_theta_v_fl_e,
            z_flxdiv_mass=self.z_flxdiv_mass,
            z_flxdiv_theta=self.z_flxdiv_theta,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )

        """
        z_w_expl (1:nlev-1):
            Compute the explicit term in vertical momentum equation at half levels (cell center). See the first equation below eq. 3.25 in ICON tutorial 2023.
            z_w_expl = advection of w + cpd theta' dpi0/dz + cpd theta (1 - eta_impl) dpi'/dz @ k+1/2 level
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
        self._stencils_43_44_45_45b(
            z_w_expl=z_fields.z_w_expl,
            w_nnow=prognostic_state[nnow].w,
            ddt_w_adv_ntl1=diagnostic_state_nh.ddt_w_adv_pc[self.ntl1],
            z_th_ddz_exner_c=self.z_th_ddz_exner_c,
            z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
            rho_ic=diagnostic_state_nh.rho_ic,
            w_concorr_c=diagnostic_state_nh.w_concorr_c,
            vwind_expl_wgt=self._metric_state_nonhydro.vwind_expl_wgt,
            z_beta=z_fields.z_beta,
            exner_nnow=prognostic_state[nnow].exner,
            rho_nnow=prognostic_state[nnow].rho,
            theta_v_nnow=prognostic_state[nnow].theta_v,
            inv_ddqz_z_full=self._metric_state_nonhydro.inv_ddqz_z_full,
            z_alpha=z_fields.z_alpha,
            vwind_impl_wgt=self._metric_state_nonhydro.vwind_impl_wgt,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            z_q=z_fields.z_q,
            k_field=self.k_field,
            rd=constants.RD,
            cvd=constants.CVD,
            dtime=dtime,
            cpd=constants.CPD,
            nlev=self._grid.num_levels,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=0,
            vertical_end=self._grid.num_levels + 1,
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
            self._init_two_cell_kdim_fields_with_zero_wp(
                cell_kdim_field_with_zero_wp_1=prognostic_state[nnew].w,
                cell_kdim_field_with_zero_wp_2=z_fields.z_contr_w_fl_l,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
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
        self._stencils_47_48_49(
            w_nnew=prognostic_state[nnew].w,
            z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
            w_concorr_c=diagnostic_state_nh.w_concorr_c,
            z_rho_expl=z_fields.z_rho_expl,
            z_exner_expl=z_fields.z_exner_expl,
            rho_nnow=prognostic_state[nnow].rho,
            inv_ddqz_z_full=self._metric_state_nonhydro.inv_ddqz_z_full,
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
            vertical_end=self._grid.num_levels + 1,
            offset_provider=self._grid.offset_providers,
        )

        if self._config.is_iau_active:
            self._add_analysis_increments_from_data_assimilation(
                z_fields.z_rho_expl,
                z_fields.z_exner_expl,
                diagnostic_state_nh.rho_incr,
                diagnostic_state_nh.exner_incr,
                self._config.iau_wgt_dyn,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider={},
            )

        """
        w (1:nlev-1):
            Update intermediate vertical velocity by forward sweep (RHS of the equation).
        z_q (1:nlev-1):
            Update intermediate upper element of tridiagonal matrix by forward sweep.
            During the forward seep, the middle element is normalized to 1.
        """
        self._solve_tridiagonal_matrix_for_w_forward_sweep(
            vwind_impl_wgt=self._metric_state_nonhydro.vwind_impl_wgt,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            ddqz_z_half=self._metric_state_nonhydro.ddqz_z_half,
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
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )

        """
        w (1:nlev-1):
            Compute the vertical velocity by backward sweep. Model top and ground level are not updated.
            w_{k-1/2} = w_{k-1/2} + w_{k+1/2} * z_q_{k-1/2}
        """
        self._solve_tridiagonal_matrix_for_w_back_substitution(
            z_q=z_fields.z_q,
            w=prognostic_state[nnew].w,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=1,
            vertical_end=self._grid.num_levels,
            offset_provider={},
        )

        if self._config.rayleigh_type == constants.RayleighType.KLEMP:
            """
            w (1:damp_nlev):
                Compute the rayleigh damping of vertical velocity at half levels (cell center).
                w_{k-1/2} = Rayleigh_damping_coeff w_{k-1/2} + (1 - Rayleigh_damping_coeff) w_{-1/2}, where w_{-1/2} is model top vertical velocity. It is zero.
                Rayleigh_damping_coeff is represented by z_raylfac.
            """
            self._apply_rayleigh_damping_mechanism(
                z_raylfac=self.z_raylfac,
                w_1=prognostic_state[nnew].w_1,
                w=prognostic_state[nnew].w,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=1,
                vertical_end=gtx.int32(
                    self._vertical_params.end_index_of_damping_layer + 1
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
        self._compute_results_for_thermodynamic_variables(
            z_rho_expl=z_fields.z_rho_expl,
            vwind_impl_wgt=self._metric_state_nonhydro.vwind_impl_wgt,
            inv_ddqz_z_full=self._metric_state_nonhydro.inv_ddqz_z_full,
            rho_ic=diagnostic_state_nh.rho_ic,
            w=prognostic_state[nnew].w,
            z_exner_expl=z_fields.z_exner_expl,
            exner_ref_mc=self._metric_state_nonhydro.exner_ref_mc,
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
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )

        # compute dw/dz for divergence damping term
        if self._config.divdamp_type >= 3:
            """
            z_dwdz_dd (dd3d_lev:nlev-1):
                Compute vertical derivative of vertical velocity at full levels (cell center).
                z_dwdz_dḍ_{k} = ( w_{k-1/2} - w_{k+1/2} ) / dz_{k} - ( contravariant_correction_{k-1/2} - contravariant_correction_{k+1/2} ) / dz_{k}
                contravariant_correction is precomputed by w_concorr_c at half levels.
            """
            self._compute_dwdz_for_divergence_damping(
                inv_ddqz_z_full=self._metric_state_nonhydro.inv_ddqz_z_full,
                w=prognostic_state[nnew].w,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                z_dwdz_dd=z_fields.z_dwdz_dd,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=self._params.kstart_dd3d,
                vertical_end=self._grid.num_levels,
                offset_provider=self._grid.offset_providers,
            )

        if at_first_substep:
            self._copy_cell_kdim_field_to_vp(
                field=prognostic_state[nnow].exner,
                field_copy=diagnostic_state_nh.exner_dyn_incr,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=self._vertical_params.kstart_moist,
                vertical_end=self._grid.num_levels,
                offset_provider={},
            )

        if self._grid.limited_area:
            """
            rho (0:nlev-1):
                Add the boundary tendency to the density at full levels (cell center) for limited area simulations.
            exner (0:nlev-1):
                Add the boundary tendency to the exner function at full levels (cell center) for limited area simulations.
            w (0:nlev):
                Add the boundary tendency to the vertical velocity at full levels (cell center) for limited area simulations.
            """
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
                vertical_end=gtx.int32(self._grid.num_levels + 1),
                offset_provider={},
            )

        if self._config.divdamp_type >= 3:
            """
            z_dwdz_dd (dd3d_lev:nlev-1):
                Compute vertical derivative of vertical velocity at full levels (cell center).
                z_dwdz_dḍ_{k} = ( w_{k-1/2} - w_{k+1/2} ) / dz_{k} - ( contravariant_correction_{k-1/2} - contravariant_correction_{k+1/2} ) / dz_{k}
                contravariant_correction is precomputed by w_concorr_c at half levels.
            """
            self._compute_dwdz_for_divergence_damping(
                inv_ddqz_z_full=self._metric_state_nonhydro.inv_ddqz_z_full,
                w=prognostic_state[nnew].w,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                z_dwdz_dd=z_fields.z_dwdz_dd,
                horizontal_start=self._start_cell_lateral_boundary,
                horizontal_end=self._end_cell_lateral_boundary_level_4,
                vertical_start=self._params.kstart_dd3d,
                vertical_end=self._grid.num_levels,
                offset_provider=self._grid.offset_providers,
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
        diagnostic_state_nh: dycore_states.DiagnosticStateNonHydro,
        prognostic_state: list[prognostics.PrognosticState],
        z_fields: IntermediateFields,
        divdamp_fac_o2: float,
        prep_adv: dycore_states.PrepAdvection,
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
        r_nsubsteps = 1.0 / self._config.ndyn_substeps_var

        # scaling factor for second-order divergence damping: divdamp_fac_o2*delta_x**2
        # delta_x**2 is approximated by the mean cell area
        # Coefficient for reduced fourth-order divergence d
        scal_divdamp_o2 = divdamp_fac_o2 * self._cell_params.mean_cell_area

        dycore_utils._calculate_divdamp_fields(
            self.enh_divdamp_fac,
            gtx.int32(self._config.divdamp_order),
            self._cell_params.mean_cell_area,
            divdamp_fac_o2,
            self._config.nudge_max_coeff,
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
            cell_areas=self._cell_params.area,
        )

        nvar = nnew

        self._compute_z_raylfac(
            self._metric_state_nonhydro.rayleigh_w,
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
        self._compute_rho_virtual_potential_temperatures_and_pressure_gradient(
            w=prognostic_state[nnew].w,
            w_concorr_c=diagnostic_state_nh.w_concorr_c,
            ddqz_z_half=self._metric_state_nonhydro.ddqz_z_half,
            rho_now=prognostic_state[nnow].rho,
            rho_var=prognostic_state[nvar].rho,
            theta_now=prognostic_state[nnow].theta_v,
            theta_var=prognostic_state[nvar].theta_v,
            wgtfac_c=self._metric_state_nonhydro.wgtfac_c,
            theta_ref_mc=self._metric_state_nonhydro.theta_ref_mc,
            vwind_expl_wgt=self._metric_state_nonhydro.vwind_expl_wgt,
            exner_pr=diagnostic_state_nh.exner_pr,
            d_exner_dz_ref_ic=self._metric_state_nonhydro.d_exner_dz_ref_ic,
            rho_ic=diagnostic_state_nh.rho_ic,
            z_theta_v_pr_ic=self.z_theta_v_pr_ic,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            z_th_ddz_exner_c=self.z_th_ddz_exner_c,
            dtime=dtime,
            wgt_nnow_rth=self._params.wgt_nnow_rth,
            wgt_nnew_rth=self._params.wgt_nnew_rth,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_local,
            vertical_start=1,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )

        log.debug(f"corrector: start stencil 17")
        """
        z_graddiv_vn (dd3d_lev:nlev-1):
            Add vertical wind derivative to the normal gradient of divergence at full levels (edge center).
            z_graddiv_vn_{k} = z_graddiv_vn_{k} + scalfac_dd3d_{k} d2w_{k}/dzdn
        """
        self._add_vertical_wind_derivative_to_divergence_damping(
            hmask_dd3d=self._metric_state_nonhydro.hmask_dd3d,
            scalfac_dd3d=self._metric_state_nonhydro.scalfac_dd3d,
            inv_dual_edge_length=self._edge_geometry.inverse_dual_edge_lengths,
            z_dwdz_dd=z_fields.z_dwdz_dd,
            z_graddiv_vn=z_fields.z_graddiv_vn,
            horizontal_start=self._start_edge_lateral_boundary_level_7,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=self._params.kstart_dd3d,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )

        if self._config.itime_scheme == TimeSteppingScheme.MOST_EFFICIENT:
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
            self._add_temporal_tendencies_to_vn_by_interpolating_between_time_levels(
                vn_nnow=prognostic_state[nnow].vn,
                ddt_vn_apc_ntl1=diagnostic_state_nh.ddt_vn_apc_pc[self.ntl1],
                ddt_vn_apc_ntl2=diagnostic_state_nh.ddt_vn_apc_pc[self.ntl2],
                ddt_vn_phy=diagnostic_state_nh.ddt_vn_phy,
                z_theta_v_e=z_fields.z_theta_v_e,
                z_gradh_exner=z_fields.z_gradh_exner,
                vn_nnew=prognostic_state[nnew].vn,
                dtime=dtime,
                wgt_nnow_vel=self._params.wgt_nnow_vel,
                wgt_nnew_vel=self._params.wgt_nnew_vel,
                cpd=constants.CPD,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider={},
            )

        if (
            self._config.divdamp_order == DivergenceDampingOrder.COMBINED
            or self._config.divdamp_order == DivergenceDampingOrder.FOURTH_ORDER
        ):
            # verified for e-10
            log.debug(f"corrector start stencil 25")
            """
            z_graddiv2_vn (0:nlev-1):
                Compute the double laplacian of vn at full levels (edge center).
            """
            self._compute_graddiv2_of_vn(
                geofac_grdiv=self._interpolation_state.geofac_grdiv,
                z_graddiv_vn=z_fields.z_graddiv_vn,
                z_graddiv2_vn=self.z_graddiv2_vn,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider=self._grid.offset_providers,
            )

        if (
            self._config.divdamp_order == DivergenceDampingOrder.COMBINED
            and scal_divdamp_o2 > 1.0e-6
        ):
            log.debug(f"corrector: start stencil 26")
            """
            vn (0:nlev-1):
                Apply the divergence damping to vn at full levels (edge center).
                vn = vn + scal_divdamp_o2 * Del(normal_direction) Div(vn)
            """
            self._apply_2nd_order_divergence_damping(
                z_graddiv_vn=z_fields.z_graddiv_vn,
                vn=prognostic_state[nnew].vn,
                scal_divdamp_o2=scal_divdamp_o2,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider={},
            )

        # TODO: this does not get accessed in FORTRAN
        if (
            self._config.divdamp_order == DivergenceDampingOrder.COMBINED
            and divdamp_fac_o2 <= 4 * self._config.divdamp_fac
        ):
            if self._grid.limited_area:
                log.debug("corrector: start stencil 27")
                """
                vn (0:nlev-1):
                    Apply the higher order divergence damping to vn at full levels (edge center).
                    vn = vn + (scal_divdamp + bdy_divdamp * nudgecoeff_e) * Del(normal_direction) Div( Del(normal_direction) Div(vn) )
                """
                self._apply_weighted_2nd_and_4th_order_divergence_damping(
                    scal_divdamp=self.scal_divdamp,
                    bdy_divdamp=self._bdy_divdamp,
                    nudgecoeff_e=self._interpolation_state.nudgecoeff_e,
                    z_graddiv2_vn=self.z_graddiv2_vn,
                    vn=prognostic_state[nnew].vn,
                    horizontal_start=self._start_edge_nudging_level_2,
                    horizontal_end=self._end_edge_local,
                    vertical_start=0,
                    vertical_end=self._grid.num_levels,
                    offset_provider={},
                )
            else:
                log.debug("corrector start stencil 4th order divdamp")
                """
                vn (0:nlev-1):
                    Apply the higher order divergence damping to vn at full levels (edge center).
                    vn = vn + scal_divdamp * Del(normal_direction) Div( Del(normal_direction) Div(vn) )
                """
                self._apply_4th_order_divergence_damping(
                    scal_divdamp=self.scal_divdamp,
                    z_graddiv2_vn=self.z_graddiv2_vn,
                    vn=prognostic_state[nnew].vn,
                    horizontal_start=self._start_edge_nudging_level_2,
                    horizontal_end=self._end_edge_local,
                    vertical_start=0,
                    vertical_end=self._grid.num_levels,
                    offset_provider={},
                )

        # TODO: this does not get accessed in FORTRAN
        if self._config.is_iau_active:
            log.debug("corrector start stencil 28")
            self._add_analysis_increments_to_vn(
                diagnostic_state_nh.vn_incr,
                prognostic_state[nnew].vn,
                self._config.iau_wgt_dyn,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider={},
            )
        log.debug("exchanging prognostic field 'vn'")
        self._exchange.exchange_and_wait(dims.EdgeDim, (prognostic_state[nnew].vn))
        log.debug("corrector: start stencil 31")
        """
        z_vn_avg (0:nlev-1):
            Compute the averaged normal velocity at full levels (edge center).
            TODO (Chia Rui): Fill in details about how the coefficients are computed.
        """
        self._compute_avg_vn(
            e_flx_avg=self._interpolation_state.e_flx_avg,
            vn=prognostic_state[nnew].vn,
            z_vn_avg=self.z_vn_avg,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )

        log.debug("corrector: start stencil 32")
        """
        mass_fl_e (0:nlev-1):
            Compute the mass flux at full levels (edge center) by multiplying density with averaged normal velocity (z_vn_avg) computed above.
        z_theta_v_fl_e (0:nlev-1):
            Compute the energy (theta_v * mass) flux by multiplying density with averaged normal velocity (z_vn_avg) computed above.
        """
        self._compute_mass_flux(
            z_rho_e=z_fields.z_rho_e,
            z_vn_avg=self.z_vn_avg,
            ddqz_z_full_e=self._metric_state_nonhydro.ddqz_z_full_e,
            z_theta_v_e=z_fields.z_theta_v_e,
            mass_fl_e=diagnostic_state_nh.mass_fl_e,
            z_theta_v_fl_e=self.z_theta_v_fl_e,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
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
                    vertical_end=self._grid.num_levels,
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
                vertical_end=self._grid.num_levels,
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
        self._compute_divergence_of_fluxes_of_rho_and_theta(
            geofac_div=self._interpolation_state.geofac_div,
            mass_fl_e=diagnostic_state_nh.mass_fl_e,
            z_theta_v_fl_e=self.z_theta_v_fl_e,
            z_flxdiv_mass=self.z_flxdiv_mass,
            z_flxdiv_theta=self.z_flxdiv_theta,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )

        if self._config.itime_scheme == TimeSteppingScheme.MOST_EFFICIENT:
            log.debug(f"corrector start stencil 42 44 45 45b")
            """
            z_w_expl (1:nlev-1):
                Compute the explicit term in vertical momentum equation at half levels (cell center). See the first equation below eq. 3.25 in ICON tutorial 2023.
                z_w_expl = advection of w + cpd theta' dpi0/dz + cpd theta (1 - eta_impl) dpi'/dz @ k+1/2 level
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
            self._stencils_42_44_45_45b(
                z_w_expl=z_fields.z_w_expl,
                w_nnow=prognostic_state[nnow].w,
                ddt_w_adv_ntl1=diagnostic_state_nh.ddt_w_adv_pc[self.ntl1],
                ddt_w_adv_ntl2=diagnostic_state_nh.ddt_w_adv_pc[self.ntl2],
                z_th_ddz_exner_c=self.z_th_ddz_exner_c,
                z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
                rho_ic=diagnostic_state_nh.rho_ic,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                vwind_expl_wgt=self._metric_state_nonhydro.vwind_expl_wgt,
                z_beta=z_fields.z_beta,
                exner_nnow=prognostic_state[nnow].exner,
                rho_nnow=prognostic_state[nnow].rho,
                theta_v_nnow=prognostic_state[nnow].theta_v,
                inv_ddqz_z_full=self._metric_state_nonhydro.inv_ddqz_z_full,
                z_alpha=z_fields.z_alpha,
                vwind_impl_wgt=self._metric_state_nonhydro.vwind_impl_wgt,
                theta_v_ic=diagnostic_state_nh.theta_v_ic,
                z_q=z_fields.z_q,
                k_field=self.k_field,
                rd=constants.RD,
                cvd=constants.CVD,
                dtime=dtime,
                cpd=constants.CPD,
                wgt_nnow_vel=self._params.wgt_nnow_vel,
                wgt_nnew_vel=self._params.wgt_nnew_vel,
                nlev=self._grid.num_levels,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=0,
                vertical_end=self._grid.num_levels + 1,
                offset_provider={},
            )
        else:
            log.debug(f"corrector start stencil 43 44 45 45b")
            """
            z_w_expl (1:nlev-1):
                Compute the explicit term in vertical momentum equation at half levels (cell center). See the first equation below eq. 3.25 in ICON tutorial 2023.
                z_w_expl = advection of w + cpd theta' dpi0/dz + cpd theta (1 - eta_impl) dpi'/dz @ k+1/2 level
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
            self._stencils_43_44_45_45b(
                z_w_expl=z_fields.z_w_expl,
                w_nnow=prognostic_state[nnow].w,
                ddt_w_adv_ntl1=diagnostic_state_nh.ddt_w_adv_pc[self.ntl1],
                z_th_ddz_exner_c=self.z_th_ddz_exner_c,
                z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
                rho_ic=diagnostic_state_nh.rho_ic,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                vwind_expl_wgt=self._metric_state_nonhydro.vwind_expl_wgt,
                z_beta=z_fields.z_beta,
                exner_nnow=prognostic_state[nnow].exner,
                rho_nnow=prognostic_state[nnow].rho,
                theta_v_nnow=prognostic_state[nnow].theta_v,
                inv_ddqz_z_full=self._metric_state_nonhydro.inv_ddqz_z_full,
                z_alpha=z_fields.z_alpha,
                vwind_impl_wgt=self._metric_state_nonhydro.vwind_impl_wgt,
                theta_v_ic=diagnostic_state_nh.theta_v_ic,
                z_q=z_fields.z_q,
                k_field=self.k_field,
                rd=constants.RD,
                cvd=constants.CVD,
                dtime=dtime,
                cpd=constants.CPD,
                nlev=self._grid.num_levels,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=0,
                vertical_end=self._grid.num_levels + 1,
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
        self._stencils_47_48_49(
            w_nnew=prognostic_state[nnew].w,
            z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
            w_concorr_c=diagnostic_state_nh.w_concorr_c,
            z_rho_expl=z_fields.z_rho_expl,
            z_exner_expl=z_fields.z_exner_expl,
            rho_nnow=prognostic_state[nnow].rho,
            inv_ddqz_z_full=self._metric_state_nonhydro.inv_ddqz_z_full,
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
            vertical_end=self._grid.num_levels + 1,
            offset_provider=self._grid.offset_providers,
        )

        # TODO: this is not tested in green line so far
        if self._config.is_iau_active:
            log.debug(f"corrector start stencil 50")
            self._add_analysis_increments_from_data_assimilation(
                z_rho_expl=z_fields.z_rho_expl,
                z_exner_expl=z_fields.z_exner_expl,
                rho_incr=diagnostic_state_nh.rho_incr,
                exner_incr=diagnostic_state_nh.exner_incr,
                iau_wgt_dyn=self._config.iau_wgt_dyn,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
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
        self._solve_tridiagonal_matrix_for_w_forward_sweep(
            vwind_impl_wgt=self._metric_state_nonhydro.vwind_impl_wgt,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            ddqz_z_half=self._metric_state_nonhydro.ddqz_z_half,
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
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
        )
        log.debug(f"corrector start stencil 53")
        """
        w (1:nlev-1):
            Compute the vertical velocity by backward sweep. Model top and ground level are not updated.
            w_{k-1/2} = w_{k-1/2} + w_{k+1/2} * z_q_{k-1/2}
        """
        self._solve_tridiagonal_matrix_for_w_back_substitution(
            z_q=z_fields.z_q,
            w=prognostic_state[nnew].w,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=1,
            vertical_end=self._grid.num_levels,
            offset_provider={},
        )

        if self._config.rayleigh_type == constants.RayleighType.KLEMP:
            log.debug(f"corrector start stencil 54")
            """
            w (1:damp_nlev):
                Compute the rayleigh damping of vertical velocity at half levels (cell center).
                w_{k-1/2} = Rayleigh_damping_coeff w_{k-1/2} + (1 - Rayleigh_damping_coeff) w_{-1/2}, where w_{-1/2} is model top vertical velocity. It is zero.
                Rayleigh_damping_coeff is represented by z_raylfac.
            """
            self._apply_rayleigh_damping_mechanism(
                z_raylfac=self.z_raylfac,
                w_1=prognostic_state[nnew].w_1,
                w=prognostic_state[nnew].w,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=1,
                vertical_end=gtx.int32(
                    self._vertical_params.end_index_of_damping_layer + 1
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
        self._compute_results_for_thermodynamic_variables(
            z_rho_expl=z_fields.z_rho_expl,
            vwind_impl_wgt=self._metric_state_nonhydro.vwind_impl_wgt,
            inv_ddqz_z_full=self._metric_state_nonhydro.inv_ddqz_z_full,
            rho_ic=diagnostic_state_nh.rho_ic,
            w=prognostic_state[nnew].w,
            z_exner_expl=z_fields.z_exner_expl,
            exner_ref_mc=self._metric_state_nonhydro.exner_ref_mc,
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
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.offset_providers,
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
                    vertical_end=self._grid.num_levels,
                    offset_provider={},
                )
        log.debug(f"corrector start stencil 58")
        self._update_mass_volume_flux(
            z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
            rho_ic=diagnostic_state_nh.rho_ic,
            vwind_impl_wgt=self._metric_state_nonhydro.vwind_impl_wgt,
            w=prognostic_state[nnew].w,
            mass_flx_ic=prep_adv.mass_flx_ic,
            vol_flx_ic=prep_adv.vol_flx_ic,
            r_nsubsteps=r_nsubsteps,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=1,
            vertical_end=self._grid.num_levels,
            offset_provider={},
        )
        if at_last_substep:
            self._update_dynamical_exner_time_increment(
                exner=prognostic_state[nnew].exner,
                ddt_exner_phy=diagnostic_state_nh.ddt_exner_phy,
                exner_dyn_incr=diagnostic_state_nh.exner_dyn_incr,
                ndyn_substeps_var=float(self._config.ndyn_substeps_var),
                dtime=dtime,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=self._vertical_params.kstart_moist,
                vertical_end=gtx.int32(self._grid.num_levels),
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
                    vertical_end=self._grid.num_levels + 1,
                    offset_provider={},
                )
            log.debug(f" corrector: start stencil 65")
            self._update_mass_flux_weighted(
                rho_ic=diagnostic_state_nh.rho_ic,
                vwind_expl_wgt=self._metric_state_nonhydro.vwind_expl_wgt,
                vwind_impl_wgt=self._metric_state_nonhydro.vwind_impl_wgt,
                w_now=prognostic_state[nnow].w,
                w_new=prognostic_state[nnew].w,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                mass_flx_ic=prep_adv.mass_flx_ic,
                r_nsubsteps=r_nsubsteps,
                horizontal_start=self._start_cell_lateral_boundary,
                horizontal_end=self._end_cell_nudging,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider={},
            )
            log.debug("exchange prognostic fields 'rho' , 'exner', 'w'")
            self._exchange.exchange_and_wait(
                dims.CellDim,
                prognostic_state[nnew].rho,
                prognostic_state[nnew].exner,
                prognostic_state[nnew].w,
            )
